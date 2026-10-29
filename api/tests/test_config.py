"""api tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test api.tests.test_config
"""
import hashlib
import json
import math
import os
import shutil
import tempfile
import unittest
import zipfile
from datetime import datetime, time
from datetime import timezone as dt_timezone
from io import BytesIO
from unittest import mock

import cv2
import numpy as np
from django.conf import settings
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, TransactionTestCase, override_settings
from django.utils.functional import SimpleLazyObject
from rest_framework import status
from rest_framework.test import APIClient

from api.models import ScheduleRule, SlideshowConfig, resolve_active_slideshow_config
from api.throttles import AccessKeyMediaThrottle
from face_manager.models import Face, Person
from filepopulator.models import GeocodeCache, ImageFile, Directory, VideoFile
from filepopulator.scripts import create_image_file

from api.tests import FaceFixtureMixin, ApiTestCase, _tiny_jpeg_bytes, ensure_sentinel_people


class StatsAndParametersTests(ApiTestCase):
    def test_parameters_endpoint_requires_key_or_auth(self):
        resp = self.anon_client.get("/api/parameters/")
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)

        resp = self.client.get("/api/parameters/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_stats_endpoint_requires_auth(self):
        resp = self.anon_client.get("/api/server_stats/")
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_stats_endpoint_with_data(self):
        img = self.make_image()
        img.isProcessed = True
        img.save()
        self.make_face(img)
        resp = self.client.get("/api/server_stats/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(resp.data["num_imgs"], 1)
        self.assertGreaterEqual(resp.data["num_faces"], 1)

    def test_stats_endpoint_by_collection_breakdown(self):
        # A real second PhotoCollection, scoped to one fixture subdirectory
        # via root_directory -- an image under it resolves (longest-prefix
        # match) to this collection instead of falling back to Default.
        import os
        from filepopulator.models import PhotoCollection

        second = PhotoCollection.objects.create(
            name='Second Collection',
            root_directory=os.path.join(settings.FILEPOPULATOR_VAL_DIRECTORY, 'naming', 'good'),
        )

        # Falls back to Default -- doesn't match 'second's root_directory.
        default_img = self.make_image('orientation/2_4.jpg')
        default_img.isProcessed = True
        default_img.save()
        self.make_face(default_img)

        # Resolves to 'second' (longest-prefix match against its root).
        second_img = self.make_image('naming/good/1.JPG')
        second_img.isProcessed = True
        second_img.save()
        self.make_face(second_img)

        resp = self.client.get("/api/server_stats/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

        by_name = {row['name']: row for row in resp.data['by_collection']}
        self.assertIn('Default', by_name)
        self.assertIn('Second Collection', by_name)

        # Second Collection has exactly the one image/face placed in it,
        # not the Default collection's image too.
        self.assertEqual(by_name['Second Collection']['num_imgs'], 1)
        self.assertEqual(by_name['Second Collection']['num_faces'], 1)
        self.assertFalse(by_name['Second Collection']['is_default'])
        self.assertTrue(by_name['Default']['is_default'])

        # Global totals are unaffected -- still reflect everything, same
        # as before this feature existed.
        self.assertGreaterEqual(resp.data['num_imgs'], 2)
        self.assertGreaterEqual(resp.data['num_faces'], 2)

    def test_stats_endpoint_by_collection_video_breakdown(self):
        # Videos scope via VideoFile.directory__collection (no M2M
        # "visible from" the way images have -- see
        # _compute_video_stats' own docstring), relying on the real fix
        # to create_video_file()'s directory resolution (2026-10-21).
        from filepopulator.models import Directory, PhotoCollection, VideoFile

        second = PhotoCollection.objects.create(
            name='Video Collection', root_directory='/videos/stats_video_test',
        )
        second_dir = Directory.objects.create(dir_path='/videos/stats_video_test', collection=second)
        VideoFile.objects.create(
            filename='/videos/stats_video_test/a.mp4', directory=second_dir,
            width=100, height=100, duration_seconds=3600, isProcessed=True,
        )

        default_collection = PhotoCollection.objects.get(is_default=True)
        default_dir = Directory.objects.create(
            dir_path='/videos/stats_video_default_test', collection=default_collection,
        )
        VideoFile.objects.create(
            filename='/videos/stats_video_default_test/b.mp4', directory=default_dir,
            width=100, height=100, duration_seconds=100, isProcessed=False,
        )

        resp = self.client.get("/api/server_stats/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

        by_name = {row['name']: row for row in resp.data['by_collection']}
        self.assertEqual(by_name['Video Collection']['num_videos'], 1)
        self.assertEqual(by_name['Video Collection']['num_videos_processed'], 1)
        self.assertEqual(by_name['Video Collection']['percent_video_processed'], '100.00%')

        # The default-collection video (b.mp4, unprocessed) shows up in
        # Default's own breakdown, not Video Collection's.
        self.assertIn('Default', by_name)
        default_filenames_total = by_name['Default']['num_videos']
        self.assertGreaterEqual(default_filenames_total, 1)

    def test_stats_endpoint_video_fields_with_no_videos(self):
        """No VideoFile rows at all -- percentages must not divide by zero."""
        img = self.make_image()
        self.make_face(img)
        resp = self.client.get("/api/server_stats/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data["num_videos"], 0)
        self.assertEqual(resp.data["num_videos_processed"], 0)
        self.assertEqual(resp.data["percent_video_processed"], "0.00%")
        self.assertEqual(resp.data["percent_video_length_processed"], "0.00%")
        self.assertEqual(resp.data["total_video_length"], "00:00:00")
        self.assertEqual(resp.data["unprocessed_video_length"], "00:00:00")

    def test_stats_endpoint_video_fields_with_data(self):
        img = self.make_image()
        self.make_face(img)
        directory = Directory.objects.create(dir_path="/videos/stats_test")
        VideoFile.objects.create(
            filename="/videos/stats_test/a.mp4", directory=directory,
            width=100, height=100, duration_seconds=3700,  # 1h 00m 00s
            isProcessed=True,
        )
        VideoFile.objects.create(
            filename="/videos/stats_test/b.mp4", directory=directory,
            width=100, height=100, duration_seconds=90000,  # 1d 01h 00m 00s
            isProcessed=False,
        )
        VideoFile.objects.create(
            filename="/videos/stats_test/c.mp4", directory=directory,
            width=100, height=100, duration_seconds=5, isProcessed=True,
            face_extraction_failed=True, face_extraction_error="RuntimeError: boom",
        )
        resp = self.client.get("/api/server_stats/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data["num_videos"], 3)
        self.assertEqual(resp.data["num_videos_processed"], 2)
        self.assertEqual(resp.data["percent_video_processed"], "66.67%")
        self.assertEqual(resp.data["total_video_length"], "1d 02:01:45")
        self.assertEqual(resp.data["unprocessed_video_length"], "1d 01:00:00")
        # processed = 3705 / total (93705) = 3.954...%
        self.assertEqual(resp.data["percent_video_length_processed"], "3.95%")
        self.assertEqual(resp.data["num_videos_failed"], 1)


class CameraGpsBackfillConfigApiTests(ApiTestCase):
    """Backend for the camera-opt-in toggle page (mirrors the existing
    nearest-metro-override tool's split -- this repo provides the API,
    the actual page is built elsewhere). See filepopulator/
    owntracks_gps.py's own docstring for why this is scoped per-camera
    at all (OwnTracks only reflects one person's location)."""

    def test_requires_auth(self):
        resp = self.anon_client.get("/api/camera_gps_backfill_config/")
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_list_groups_by_model_not_make(self):
        # Real case this was built around: the same physical device can
        # write different camera_make spellings across firmware/OS
        # versions (confirmed: 'LGE'/'LG Electronics' both for
        # 'LM-G710') -- grouping by (make, model) together would
        # silently fragment one device's real count into two rows.
        img1 = self.make_image("naming/good/1.JPG")
        ImageFile.objects.filter(pk=img1.pk).update(
            camera_make='LGE', camera_model='LM-G710', gps_lat_decimal=-999,
        )
        img2 = self.make_image("naming/good/2.jpg")
        ImageFile.objects.filter(pk=img2.pk).update(
            camera_make='LG Electronics', camera_model='LM-G710', gps_lat_decimal=40.0,
        )

        resp = self.client.get("/api/camera_gps_backfill_config/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        rows = [r for r in resp.data if r['camera_model'] == 'LM-G710']
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['total'], 2)
        self.assertEqual(row['missing_gps'], 1)
        self.assertEqual(sorted(row['camera_make_examples']), ['LG Electronics', 'LGE'])
        self.assertFalse(row['enabled'])

    def test_list_reflects_enabled_state(self):
        from filepopulator.models import CameraGpsBackfillConfig

        img = self.make_image("naming/good/1.JPG")
        ImageFile.objects.filter(pk=img.pk).update(camera_model='Test Camera XYZ')
        CameraGpsBackfillConfig.objects.create(camera_model='Test Camera XYZ', enabled=True)

        resp = self.client.get("/api/camera_gps_backfill_config/")
        row = next(r for r in resp.data if r['camera_model'] == 'Test Camera XYZ')
        self.assertTrue(row['enabled'])

    def test_set_enabled_upserts_new_camera(self):
        from filepopulator.models import CameraGpsBackfillConfig

        self.assertFalse(CameraGpsBackfillConfig.objects.filter(camera_model='Brand New Camera').exists())
        resp = self.client.post(
            "/api/camera_gps_backfill_config/set_enabled/",
            {"camera_model": "Brand New Camera", "enabled": True},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        config = CameraGpsBackfillConfig.objects.get(camera_model='Brand New Camera')
        self.assertTrue(config.enabled)

    def test_set_enabled_can_flip_back_off(self):
        from filepopulator.models import CameraGpsBackfillConfig

        CameraGpsBackfillConfig.objects.create(camera_model='Already On', enabled=True)
        resp = self.client.post(
            "/api/camera_gps_backfill_config/set_enabled/",
            {"camera_model": "Already On", "enabled": False},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        config = CameraGpsBackfillConfig.objects.get(camera_model='Already On')
        self.assertFalse(config.enabled)

    def test_set_enabled_rejects_missing_fields(self):
        resp = self.client.post(
            "/api/camera_gps_backfill_config/set_enabled/",
            {"camera_model": "Incomplete"},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_set_enabled_rejects_non_boolean_enabled(self):
        resp = self.client.post(
            "/api/camera_gps_backfill_config/set_enabled/",
            {"camera_model": "Bad Type", "enabled": "yes"},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)


class WindowedClassificationConfigApiTests(ApiTestCase):
    """Backend for the per-person windowed-classification toggle page --
    see face_manager.models.Person.use_windowed_classification's own
    docstring for what this controls (a manually-curated per-person
    override, not a general default, per the real investigation that
    motivated it)."""

    def test_requires_auth(self):
        resp = self.anon_client.get("/api/windowed_classification_config/")
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_list_excludes_small_galleries_by_default(self):
        image = self.make_image()
        small_person = Person.objects.create(person_name="Tiny Gallery")
        for _ in range(3):
            self.make_face(image, declared_name=small_person, validated=True)

        resp = self.client.get("/api/windowed_classification_config/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        names = [row['person_name'] for row in resp.data]
        self.assertNotIn("Tiny Gallery", names)

    def test_list_includes_large_galleries_with_current_state(self):
        image = self.make_image()
        big_person = Person.objects.create(
            person_name="Big Gallery", use_windowed_classification=True,
            classification_window_days=90,
        )
        for _ in range(55):
            self.make_face(image, declared_name=big_person, validated=True)

        resp = self.client.get("/api/windowed_classification_config/")
        row = next(r for r in resp.data if r['person_name'] == "Big Gallery")
        self.assertEqual(row['confirmed_count'], 55)
        self.assertTrue(row['use_windowed_classification'])
        self.assertEqual(row['classification_window_days'], 90)

    def test_list_includes_already_enabled_small_gallery(self):
        # An enabled person shouldn't vanish from the list just because
        # their gallery is below MIN_GALLERY_SIZE -- otherwise turning it
        # back off would require already knowing their id out-of-band.
        image = self.make_image()
        small_enabled = Person.objects.create(
            person_name="Small But Enabled", use_windowed_classification=True,
        )
        self.make_face(image, declared_name=small_enabled, validated=True)

        resp = self.client.get("/api/windowed_classification_config/")
        names = [row['person_name'] for row in resp.data]
        self.assertIn("Small But Enabled", names)

    def test_set_windowed_enables_and_sets_window_days(self):
        person = Person.objects.create(person_name="To Enable")
        resp = self.client.post(
            "/api/windowed_classification_config/set_windowed/",
            {"person_id": person.id, "enabled": True, "window_days": 60},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        person.refresh_from_db()
        self.assertTrue(person.use_windowed_classification)
        self.assertEqual(person.classification_window_days, 60)

    def test_set_windowed_can_disable(self):
        person = Person.objects.create(
            person_name="To Disable", use_windowed_classification=True,
        )
        resp = self.client.post(
            "/api/windowed_classification_config/set_windowed/",
            {"person_id": person.id, "enabled": False},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        person.refresh_from_db()
        self.assertFalse(person.use_windowed_classification)

    def test_set_windowed_omitting_window_days_leaves_it_unchanged(self):
        person = Person.objects.create(
            person_name="Keep Window", classification_window_days=45,
        )
        resp = self.client.post(
            "/api/windowed_classification_config/set_windowed/",
            {"person_id": person.id, "enabled": True},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        person.refresh_from_db()
        self.assertEqual(person.classification_window_days, 45)

    def test_set_windowed_rejects_unknown_person(self):
        resp = self.client.post(
            "/api/windowed_classification_config/set_windowed/",
            {"person_id": 999999, "enabled": True},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_set_windowed_rejects_non_positive_window_days(self):
        person = Person.objects.create(person_name="Bad Window")
        resp = self.client.post(
            "/api/windowed_classification_config/set_windowed/",
            {"person_id": person.id, "enabled": True, "window_days": 0},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_set_windowed_rejects_missing_fields(self):
        resp = self.client.post(
            "/api/windowed_classification_config/set_windowed/",
            {"enabled": True},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)


class PhotoCollectionConfigApiTests(ApiTestCase):
    """Backend for a frontend settings page letting a human tune a
    PhotoCollection's detection floors -- see filepopulator.models.
    PhotoCollection.min_face_det_score/min_face_relative_size's own
    docstring for what these gate."""

    def _make_collection(self, **kwargs):
        from filepopulator.models import PhotoCollection
        import uuid
        defaults = {'name': f'Test Collection {uuid.uuid4()}', 'root_directory': f'/tmp/{uuid.uuid4()}'}
        defaults.update(kwargs)
        return PhotoCollection.objects.create(**defaults)

    def test_requires_auth(self):
        resp = self.anon_client.get("/api/photo_collection_config/")
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_list_includes_current_floors(self):
        collection = self._make_collection(min_face_det_score=0.6, min_face_relative_size=0.05)
        resp = self.client.get("/api/photo_collection_config/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        row = next(r for r in resp.data if r['collection_id'] == collection.id)
        self.assertEqual(row['name'], collection.name)
        self.assertEqual(row['root_directory'], collection.root_directory)
        self.assertFalse(row['is_default'])
        self.assertAlmostEqual(row['min_face_det_score'], 0.6)
        self.assertAlmostEqual(row['min_face_relative_size'], 0.05)

    def test_list_includes_unset_floors_as_null(self):
        collection = self._make_collection()
        resp = self.client.get("/api/photo_collection_config/")
        row = next(r for r in resp.data if r['collection_id'] == collection.id)
        self.assertIsNone(row['min_face_det_score'])
        self.assertIsNone(row['min_face_relative_size'])

    def test_set_floors_updates_both(self):
        collection = self._make_collection()
        resp = self.client.post(
            "/api/photo_collection_config/set_floors/",
            {"collection_id": collection.id, "min_face_det_score": 0.6, "min_face_relative_size": 0.04},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        collection.refresh_from_db()
        self.assertAlmostEqual(collection.min_face_det_score, 0.6)
        self.assertAlmostEqual(collection.min_face_relative_size, 0.04)

    def test_set_floors_updates_only_the_field_given(self):
        collection = self._make_collection(min_face_det_score=0.6, min_face_relative_size=0.04)
        resp = self.client.post(
            "/api/photo_collection_config/set_floors/",
            {"collection_id": collection.id, "min_face_det_score": 0.7},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        collection.refresh_from_db()
        self.assertAlmostEqual(collection.min_face_det_score, 0.7)
        self.assertAlmostEqual(collection.min_face_relative_size, 0.04)

    def test_set_floors_null_clears_a_floor(self):
        collection = self._make_collection(min_face_det_score=0.6, min_face_relative_size=0.04)
        resp = self.client.post(
            "/api/photo_collection_config/set_floors/",
            {"collection_id": collection.id, "min_face_det_score": None},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        collection.refresh_from_db()
        self.assertIsNone(collection.min_face_det_score)
        self.assertAlmostEqual(collection.min_face_relative_size, 0.04)

    def test_set_floors_rejects_out_of_range_value(self):
        collection = self._make_collection()
        resp = self.client.post(
            "/api/photo_collection_config/set_floors/",
            {"collection_id": collection.id, "min_face_det_score": 1.5},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_set_floors_rejects_non_numeric_value(self):
        collection = self._make_collection()
        resp = self.client.post(
            "/api/photo_collection_config/set_floors/",
            {"collection_id": collection.id, "min_face_det_score": "high"},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_set_floors_rejects_unknown_collection(self):
        resp = self.client.post(
            "/api/photo_collection_config/set_floors/",
            {"collection_id": 999999, "min_face_det_score": 0.6},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_set_floors_rejects_missing_collection_id(self):
        resp = self.client.post(
            "/api/photo_collection_config/set_floors/",
            {"min_face_det_score": 0.6},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)


class SlideshowConfigModelTests(ApiTestCase):
    """api/models.py: SlideshowConfig.is_default exclusivity and
    ScheduleRule.contains()/resolve_active_slideshow_config()."""

    def test_setting_a_new_default_demotes_the_previous_one(self):
        first = SlideshowConfig.objects.create(name='First', is_default=True)
        second = SlideshowConfig.objects.create(name='Second', is_default=True)
        first.refresh_from_db()
        self.assertFalse(first.is_default)
        self.assertTrue(second.is_default)

    def test_non_default_configs_can_coexist(self):
        SlideshowConfig.objects.create(name='A')
        SlideshowConfig.objects.create(name='B')
        self.assertEqual(SlideshowConfig.objects.filter(is_default=True).count(), 0)

    def test_rule_contains_simple_same_day_window(self):
        config = SlideshowConfig.objects.create(name='Daytime')
        rule = ScheduleRule.objects.create(
            config=config, days='MON,TUE,WED,THU,FRI',
            start_time=time(8, 0), end_time=time(17, 0),
        )
        monday_morning = datetime(2026, 10, 19, 9, 0, tzinfo=dt_timezone.utc)  # a Monday
        monday_evening = datetime(2026, 10, 19, 20, 0, tzinfo=dt_timezone.utc)
        saturday_morning = datetime(2026, 10, 24, 9, 0, tzinfo=dt_timezone.utc)
        self.assertTrue(rule.contains(monday_morning))
        self.assertFalse(rule.contains(monday_evening))
        self.assertFalse(rule.contains(saturday_morning))

    def test_rule_contains_overnight_window_wraps_past_midnight(self):
        config = SlideshowConfig.objects.create(name='Overnight')
        rule = ScheduleRule.objects.create(
            config=config, days='FRI,SAT',
            start_time=time(22, 0), end_time=time(6, 0),
        )
        # Friday 23:00 and Saturday 02:00 both fall inside a Fri 22:00 ->
        # Sat 06:00 overnight window; Saturday 10:00 does not.
        friday_late = datetime(2026, 10, 23, 23, 0, tzinfo=dt_timezone.utc)
        saturday_early = datetime(2026, 10, 24, 2, 0, tzinfo=dt_timezone.utc)
        saturday_midday = datetime(2026, 10, 24, 10, 0, tzinfo=dt_timezone.utc)
        self.assertTrue(rule.contains(friday_late))
        self.assertTrue(rule.contains(saturday_early))
        self.assertFalse(rule.contains(saturday_midday))

    def test_resolve_prefers_matching_rule_over_default(self):
        default_config = SlideshowConfig.objects.create(name='Default', is_default=True)
        scheduled_config = SlideshowConfig.objects.create(name='Scheduled')
        rule = ScheduleRule.objects.create(
            config=scheduled_config, days='MON,TUE,WED,THU,FRI,SAT,SUN',
            start_time=time(0, 0), end_time=time(23, 59, 59),
        )
        resolved_config, resolved_rule = resolve_active_slideshow_config(
            now=datetime(2026, 10, 19, 12, 0, tzinfo=dt_timezone.utc)
        )
        self.assertEqual(resolved_config, scheduled_config)
        self.assertEqual(resolved_rule, rule)

    def test_resolve_falls_back_to_default_when_nothing_matches(self):
        default_config = SlideshowConfig.objects.create(name='Default', is_default=True)
        resolved_config, resolved_rule = resolve_active_slideshow_config(
            now=datetime(2026, 10, 19, 12, 0, tzinfo=dt_timezone.utc)
        )
        self.assertEqual(resolved_config, default_config)
        self.assertIsNone(resolved_rule)

    def test_resolve_returns_none_with_nothing_configured_at_all(self):
        resolved_config, resolved_rule = resolve_active_slideshow_config(
            now=datetime(2026, 10, 19, 12, 0, tzinfo=dt_timezone.utc)
        )
        self.assertIsNone(resolved_config)
        self.assertIsNone(resolved_rule)

    def test_disabled_rule_is_never_matched(self):
        default_config = SlideshowConfig.objects.create(name='Default', is_default=True)
        other_config = SlideshowConfig.objects.create(name='Other')
        ScheduleRule.objects.create(
            config=other_config, days='MON,TUE,WED,THU,FRI,SAT,SUN',
            start_time=time(0, 0), end_time=time(23, 59, 59), enabled=False,
        )
        resolved_config, resolved_rule = resolve_active_slideshow_config(
            now=datetime(2026, 10, 19, 12, 0, tzinfo=dt_timezone.utc)
        )
        self.assertEqual(resolved_config, default_config)
        self.assertIsNone(resolved_rule)

    def test_overlapping_rules_resolved_by_priority_then_recency(self):
        low_config = SlideshowConfig.objects.create(name='Low')
        high_config = SlideshowConfig.objects.create(name='High')
        ScheduleRule.objects.create(
            config=low_config, days='MON,TUE,WED,THU,FRI,SAT,SUN',
            start_time=time(0, 0), end_time=time(23, 59, 59), priority=0,
        )
        high_rule = ScheduleRule.objects.create(
            config=high_config, days='MON,TUE,WED,THU,FRI,SAT,SUN',
            start_time=time(0, 0), end_time=time(23, 59, 59), priority=10,
        )
        resolved_config, resolved_rule = resolve_active_slideshow_config(
            now=datetime(2026, 10, 19, 12, 0, tzinfo=dt_timezone.utc)
        )
        self.assertEqual(resolved_config, high_config)
        self.assertEqual(resolved_rule, high_rule)


class SlideshowConfigViewSetTests(ApiTestCase):
    """CRUD endpoints (/api/slideshow_configs/, /api/schedule_rules/) -
    session-authenticated only, same as ImageViewSet etc."""

    def test_anonymous_cannot_list_configs(self):
        resp = self.anon_client.get('/api/slideshow_configs/')
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_slideshow_key_alone_is_not_enough_to_manage_configs(self):
        # Deliberately stricter than the slideshow-facing endpoints:
        # the slideshow key authenticates the Pi's *display*, not
        # someone editing what it shows.
        resp = self.anon_client.get(
            '/api/slideshow_configs/', HTTP_X_SLIDESHOW_KEY=settings.SLIDESHOW_API_KEY
        )
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_authenticated_user_can_create_and_list_a_config(self):
        resp = self.client.post('/api/slideshow_configs/', {
            'name': 'Holiday Show', 'people': 'Alice,Bob', 'year_start': 2020,
        })
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED, resp.data)
        resp = self.client.get('/api/slideshow_configs/')
        names = [c['name'] for c in resp.data['results']] if 'results' in resp.data else [
            c['name'] for c in resp.data
        ]
        self.assertIn('Holiday Show', names)

    def test_year_start_after_year_end_is_rejected(self):
        resp = self.client.post('/api/slideshow_configs/', {
            'name': 'Bad Range', 'year_start': 2020, 'year_end': 2010,
        })
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_create_schedule_rule_normalizes_and_validates_days(self):
        config = SlideshowConfig.objects.create(name='Weekday Mornings')
        resp = self.client.post('/api/schedule_rules/', {
            'config': config.id, 'days': 'fri, Mon , wed',
            'start_time': '08:00:00', 'end_time': '12:00:00',
        })
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED, resp.data)
        self.assertEqual(resp.data['days'], 'MON,WED,FRI')

    def test_schedule_rule_rejects_invalid_day_abbreviation(self):
        config = SlideshowConfig.objects.create(name='Oops')
        resp = self.client.post('/api/schedule_rules/', {
            'config': config.id, 'days': 'MONDAY',
            'start_time': '08:00:00', 'end_time': '12:00:00',
        })
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)


class ActiveSlideshowConfigViewTests(ApiTestCase):
    """GET /api/slideshow/active_config/ - slideshow-key reachable,
    same permission class as filteredImagesView."""

    def test_requires_slideshow_key_or_session(self):
        resp = self.anon_client.get('/api/slideshow/active_config/')
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)

    def test_returns_unfiltered_defaults_with_nothing_configured(self):
        resp = self.anon_client.get(
            '/api/slideshow/active_config/', HTTP_X_SLIDESHOW_KEY=settings.SLIDESHOW_API_KEY
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIsNone(resp.data['config_id'])
        self.assertEqual(resp.data['people'], '')

    def test_returns_resolved_matching_config(self):
        config = SlideshowConfig.objects.create(
            name='Grandparents', people='Nana,Pop', year_start=2015, slide_len=10,
        )
        ScheduleRule.objects.create(
            config=config, days='MON,TUE,WED,THU,FRI,SAT,SUN',
            start_time=time(0, 0), end_time=time(23, 59, 59),
        )
        resp = self.anon_client.get(
            '/api/slideshow/active_config/', HTTP_X_SLIDESHOW_KEY=settings.SLIDESHOW_API_KEY
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data['config_id'], config.id)
        self.assertEqual(resp.data['config_name'], 'Grandparents')
        self.assertIsNotNone(resp.data['rule_id'])
        self.assertEqual(resp.data['people'], 'Nana,Pop')
        self.assertEqual(resp.data['slide_len'], 10)



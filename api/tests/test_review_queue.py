"""api tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test api.tests.test_review_queue
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


class IgnoreReviewFlaggedPartitionTests(ApiTestCase):
    # The frontend's ".ignore" main unlabeled screen and its "Flagged for
    # review" subordinate row (PersonParamView's face_poss `flagged`
    # param) are meant to be a complementary partition of the same
    # poss_ident1 candidates - a face flagged via the mobile app's
    # ignore-review flow (mobile_review_hidden=True) should show up in
    # exactly one of the two, never both and never neither.
    def setUp(self):
        super().setUp()
        self.image = self.make_image()
        self.ignore = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        self.flagged_face = self.make_face(
            self.image, declared_name=blank, poss_ident1=self.ignore,
            weight_1=0.9, mobile_review_hidden=True,
        )
        self.plain_face = self.make_face(
            self.image, declared_name=blank, poss_ident1=self.ignore,
            weight_1=0.5, mobile_review_hidden=False,
        )
        self.never_reviewed_face = self.make_face(
            self.image, declared_name=blank, poss_ident1=self.ignore,
            weight_1=0.3,
        )

    def test_default_face_poss_excludes_flagged(self):
        resp = self.client.get(f"/api/paginate_obj_ids/{self.ignore.id}/face_poss")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        id_list = json.loads(resp.content)["id_list"]
        self.assertNotIn(self.flagged_face.id, id_list)
        self.assertIn(self.plain_face.id, id_list)
        self.assertIn(self.never_reviewed_face.id, id_list)

    def test_flagged_true_returns_only_flagged(self):
        resp = self.client.get(f"/api/paginate_obj_ids/{self.ignore.id}/face_poss?flagged=true")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        id_list = json.loads(resp.content)["id_list"]
        self.assertEqual(id_list, [self.flagged_face.id])

    def test_flagged_and_unflagged_partition_the_full_poss_ident1_set(self):
        # Explicit invariant, not just implied by the two tests above:
        # every face with poss_ident1=.ignore appears in exactly one of
        # the two queries, and their union is the whole set with no
        # overlap - queried straight against the DB (num_possibilities is
        # itself a live query too now - see IgnoreReviewFlaggedCountTests
        # below - but this test independently confirms the partition
        # itself, regardless of how that count is computed).
        main_ids = set(json.loads(
            self.client.get(f"/api/paginate_obj_ids/{self.ignore.id}/face_poss").content
        )["id_list"])
        flagged_ids = set(json.loads(
            self.client.get(f"/api/paginate_obj_ids/{self.ignore.id}/face_poss?flagged=true").content
        )["id_list"])
        all_poss_ident1_ids = set(
            Face.objects.filter(poss_ident1=self.ignore).values_list("id", flat=True)
        )

        self.assertEqual(main_ids & flagged_ids, set(), "a face should never appear in both")
        self.assertEqual(main_ids | flagged_ids, all_poss_ident1_ids, "union should cover every poss_ident1 face")

    def test_real_person_still_shows_a_mobile_review_hidden_candidate(self):
        # Regression test: mobile_review_hidden is set not just by the
        # .ignore-review flow, but also by the mobile app's plain "Skip"
        # action on *any* real person's suggestion queue
        # (HideFromMobile/LabelingGroupsView) - a real person has no
        # "Flagged for review" view the way .ignore does, so excluding
        # mobile_review_hidden=True faces from their face_poss query made
        # a skipped candidate disappear from the gallery entirely (while
        # still counting toward the sidebar's num_possibilities) -
        # reported by the user 2026-09-02 as "a person shows one
        # unlabeled face but the gallery is empty."
        real_person = Person.objects.create(person_name="Garrett Egan Test")
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        skipped_face = self.make_face(
            self.image, declared_name=blank, poss_ident1=real_person,
            weight_1=0.2, mobile_review_hidden=True,
        )
        resp = self.client.get(f"/api/paginate_obj_ids/{real_person.id}/face_poss")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        id_list = json.loads(resp.content)["id_list"]
        self.assertIn(skipped_face.id, id_list)


class IgnoreReviewFlaggedUnverifiedPartitionTests(ApiTestCase):
    # A face's mobile_review_hidden flag survives being manually confirmed
    # as .ignore (declared_name set, poss_identN cleared) - nothing in
    # Face.associate_person/reject_association clears it (checked). The
    # verify screen's main ".ignore" query and its "Flagged & unverified"
    # subordinate row (PersonParamView's face_declared/do_only_unverified
    # branch, `flagged` param) are meant to be a complementary partition of
    # .ignore's unverified-and-declared faces, same shape as
    # IgnoreReviewFlaggedPartitionTests above but one step further along
    # (already confirmed, not just proposed).
    def setUp(self):
        super().setUp()
        self.image = self.make_image()
        self.ignore = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        self.flagged_unverified_face = self.make_face(
            self.image, declared_name=self.ignore, validated=False, mobile_review_hidden=True,
        )
        self.plain_unverified_face = self.make_face(
            self.image, declared_name=self.ignore, validated=False, mobile_review_hidden=False,
        )
        self.never_reviewed_unverified_face = self.make_face(
            self.image, declared_name=self.ignore, validated=False,
        )
        # Already verified, and flagged - should show up in neither the
        # main verify query nor the flagged one, both of which require
        # validated=False.
        self.flagged_but_verified_face = self.make_face(
            self.image, declared_name=self.ignore, validated=True, mobile_review_hidden=True,
        )

    def test_default_only_unverified_excludes_flagged(self):
        resp = self.client.get(f"/api/paginate_obj_ids/{self.ignore.id}/face_declared?only_unverified=true")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        id_list = json.loads(resp.content)["id_list"]
        self.assertNotIn(self.flagged_unverified_face.id, id_list)
        self.assertIn(self.plain_unverified_face.id, id_list)
        self.assertIn(self.never_reviewed_unverified_face.id, id_list)
        self.assertNotIn(self.flagged_but_verified_face.id, id_list)

    def test_flagged_true_returns_only_flagged_unverified(self):
        resp = self.client.get(
            f"/api/paginate_obj_ids/{self.ignore.id}/face_declared?only_unverified=true&flagged=true"
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        id_list = json.loads(resp.content)["id_list"]
        self.assertEqual(id_list, [self.flagged_unverified_face.id])

    def test_flagged_and_unflagged_partition_the_full_unverified_set(self):
        main_ids = set(json.loads(
            self.client.get(f"/api/paginate_obj_ids/{self.ignore.id}/face_declared?only_unverified=true").content
        )["id_list"])
        flagged_ids = set(json.loads(
            self.client.get(
                f"/api/paginate_obj_ids/{self.ignore.id}/face_declared?only_unverified=true&flagged=true"
            ).content
        )["id_list"])
        all_unverified_ids = set(
            Face.objects.filter(declared_name=self.ignore, validated=False).values_list("id", flat=True)
        )
        self.assertEqual(main_ids & flagged_ids, set(), "a face should never appear in both")
        self.assertEqual(main_ids | flagged_ids, all_unverified_ids, "union should cover every unverified face")

    def test_flagged_true_is_a_no_op_for_a_real_person(self):
        # Scoped to .ignore, same as the face_poss branch's own gate - a
        # real person's unverified faces shouldn't be filtered by this
        # mobile-only flag.
        real_person = Person.objects.create(person_name="Flagged Unverified Real Person")
        real_face = self.make_face(
            self.image, declared_name=real_person, validated=False, mobile_review_hidden=True,
        )
        resp = self.client.get(
            f"/api/paginate_obj_ids/{real_person.id}/face_declared?only_unverified=true&flagged=true"
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        id_list = json.loads(resp.content)["id_list"]
        self.assertIn(real_face.id, id_list)

    def test_default_face_declared_without_only_unverified_is_unaffected(self):
        # The flagged partition only applies to the do_only_unverified
        # branch - plain browsing of .ignore's full declared gallery
        # (verify screen off) should still show every confirmed face
        # regardless of the flag, same as before this feature.
        resp = self.client.get(f"/api/paginate_obj_ids/{self.ignore.id}/face_declared")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        id_list = json.loads(resp.content)["id_list"]
        self.assertIn(self.flagged_unverified_face.id, id_list)
        self.assertIn(self.flagged_but_verified_face.id, id_list)


@override_settings(MEDIA_ROOT="/tmp/api_test_media")
class IgnoreReviewFlaggedUnverifiedCountTests(FaceFixtureMixin, TransactionTestCase):
    # Same threading/visibility reasoning as IgnoreReviewFlaggedCountTests
    # above - num_unverified_faces comes from compute_live_face_counts,
    # farmed out to worker threads on separate DB connections, so this
    # needs TransactionTestCase (real commits) rather than a plain
    # TestCase's rolled-back transaction.
    def setUp(self):
        ensure_sentinel_people()
        self.user = User.objects.create_user(username="tester_flagged_unverified_count", password="pw123456")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

        self.image = self.make_image()
        self.ignore = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        self.make_face(
            self.image, declared_name=self.ignore, validated=False, mobile_review_hidden=True,
        )

    def test_person_list_num_unverified_excludes_flagged_for_ignore(self):
        self.make_face(
            self.image, declared_name=self.ignore, validated=False, mobile_review_hidden=False,
        )
        self.make_face(
            self.image, declared_name=self.ignore, validated=False, mobile_review_hidden=False,
        )
        resp = self.client.get("/api/person_list/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        results = json.loads(resp.content)["results"]
        ignore_dict = next(p for p in results if p["id"] == self.ignore.id)
        self.assertEqual(ignore_dict["num_unverified_faces"], 2)
        self.assertEqual(ignore_dict["num_review_flagged_unverified"], 1)


@override_settings(MEDIA_ROOT="/tmp/api_test_media")
class IgnoreReviewFlaggedCountTests(FaceFixtureMixin, TransactionTestCase):
    # PersonListView farms its per-person work out to worker threads
    # (api/views.py), each on its own DB connection - a plain TestCase's
    # rolled-back-at-teardown transaction is only visible on the
    # connection that opened it, so fixture rows created under a regular
    # ApiTestCase are invisible to those worker threads' queries. Needs
    # TransactionTestCase (real commits) instead, just for this one
    # count - IgnoreReviewFlaggedPartitionTests above covers the
    # non-threaded PersonParamView query the same fixture shape feeds.
    def setUp(self):
        ensure_sentinel_people()
        self.user = User.objects.create_user(username="tester_flagged_count", password="pw123456")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

        self.image = self.make_image()
        self.ignore = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        self.make_face(
            self.image, declared_name=blank, poss_ident1=self.ignore,
            weight_1=0.9, mobile_review_hidden=True,
        )

    def test_person_list_num_possibilities_excludes_flagged_for_ignore(self):
        # .ignore's live num_possibilities count includes every
        # poss_ident1 candidate regardless of the flag - the sidebar's
        # main-screen count needs the same exclusion the query above
        # applies, or it would overcount relative to what the main
        # screen actually shows. setUp already created 1 flagged
        # candidate; add 2 more unflagged ones so the live total is 3.
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        self.make_face(
            self.image, declared_name=blank, poss_ident1=self.ignore,
            weight_1=0.5, mobile_review_hidden=False,
        )
        self.make_face(
            self.image, declared_name=blank, poss_ident1=self.ignore,
            weight_1=0.4, mobile_review_hidden=False,
        )
        resp = self.client.get("/api/person_list/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        results = json.loads(resp.content)["results"]
        ignore_dict = next(p for p in results if p["id"] == self.ignore.id)
        self.assertEqual(ignore_dict["num_possibilities"], 2)
        self.assertEqual(ignore_dict["num_review_flagged"], 1)


class GeocodeReviewTests(ApiTestCase):
    # GeocodeCache is keyed per-coordinate (~11m rounding), not per place -
    # these tests exercise the grouping (api/geocode_views.py) that turns
    # that into a "one row per place" review table, and the two actions
    # (validate-as-is / correct) that write back to every coordinate in a
    # group at once.
    def setUp(self):
        super().setUp()
        self.image = self.make_image()
        # Pisa-area coordinate that resolves (wrongly, per the real bug
        # this feature is for) to Livorno instead of Florence - two
        # distinct coordinates sharing the same (locality, country, metro)
        # triple, so they should collapse into one review-table row.
        self.pisa_1 = GeocodeCache.objects.create(
            lat=43.7000, lon=10.4000, locality='Pisa', country='Italy',
            nearest_metro_name='Livorno', nearest_metro_distance_km=20.0,
        )
        self.pisa_2 = GeocodeCache.objects.create(
            lat=43.7050, lon=10.4050, locality='Pisa', country='Italy',
            nearest_metro_name='Livorno', nearest_metro_distance_km=21.0,
        )
        # A different, unrelated locality that happens to ALSO resolve to
        # "Livorno" as its nearest metro - must stay a separate group, and
        # must NOT be touched by a correction scoped to the Pisa group.
        self.other_livorno_area = GeocodeCache.objects.create(
            lat=43.55, lon=10.30, locality='Collesalvetti', country='Italy',
            nearest_metro_name='Livorno', nearest_metro_distance_km=8.0,
        )
        self.image.geocode = self.pisa_1
        self.image.save(update_fields=['geocode'])

    def _pisa_group_payload(self, **overrides):
        payload = {'locality': 'Pisa', 'country': 'Italy', 'metro_name': 'Livorno'}
        payload.update(overrides)
        return payload

    def test_list_groups_by_locality_country_metro_not_by_metro_alone(self):
        resp = self.client.get('/api/geocode_review/')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = json.loads(resp.content)
        self.assertEqual(data['metrics']['total'], 2)  # Pisa group + Collesalvetti group
        self.assertEqual(data['metrics']['validated'], 0)

        pisa_row = next(r for r in data['results'] if r['locality'] == 'Pisa')
        self.assertEqual(pisa_row['metro_name'], 'Livorno')
        self.assertEqual(pisa_row['num_coords'], 2)
        self.assertEqual(pisa_row['num_images'], 1)
        self.assertFalse(pisa_row['metro_validated'])

        other_row = next(r for r in data['results'] if r['locality'] == 'Collesalvetti')
        self.assertEqual(other_row['num_coords'], 1)

    def test_unknown_locality_sinks_below_known_pending_but_above_validated(self):
        # A row with no precise locality at all (a failed/empty reverse-
        # geocode - see CLAUDE.md) has nothing to usefully compare its
        # metro pick against, so it sinks below known-but-pending rows -
        # but it's still unresolved work, unlike an already-validated row,
        # so it stays above those.
        GeocodeCache.objects.create(
            lat=10.0, lon=10.0, locality=None, country=None,
            nearest_metro_name='Nowhereville', nearest_metro_distance_km=5.0,
        )
        self.other_livorno_area.metro_validated = True
        self.other_livorno_area.save(update_fields=['metro_validated'])

        resp = self.client.get('/api/geocode_review/')
        data = json.loads(resp.content)
        localities = [r['locality'] for r in data['results']]
        self.assertEqual(localities, ['Pisa', None, 'Collesalvetti'])

    def test_us_precise_city_and_metro_area_expose_state(self):
        GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, locality='Bothell', state='Washington', country='United States',
            nearest_metro_name='Seattle', nearest_metro_distance_km=15.0,
        )
        resp = self.client.get('/api/geocode_review/')
        data = json.loads(resp.content)
        row = next(r for r in data['results'] if r['locality'] == 'Bothell')
        self.assertEqual(row['state'], 'Washington')
        self.assertEqual(row['metro_state'], 'WA')
        # Non-US metro names don't get a (meaningless, numeric) state.
        pisa_row = next(r for r in data['results'] if r['locality'] == 'Pisa')
        self.assertIsNone(pisa_row['metro_state'])

    def test_search_places_includes_state_for_us_only(self):
        resp = self.client.get('/api/geocode_review/search_places/', {'q': 'seattle'})
        data = json.loads(resp.content)
        seattle = next(r for r in data['results'] if r['name'] == 'Seattle')
        self.assertEqual(seattle['state'], 'WA')

        resp2 = self.client.get('/api/geocode_review/search_places/', {'q': 'florence'})
        data2 = json.loads(resp2.content)
        florence = next(r for r in data2['results'] if r['name'] == 'Florence' and r['country_code'] == 'IT')
        self.assertIsNone(florence['state'])

    def test_correct_response_includes_metro_state_when_available(self):
        resp = self.client.patch(
            '/api/geocode_review/action/',
            self._pisa_group_payload(action='correct', metro_name_correction='Seattle'),
            format='json',
        )
        data = json.loads(resp.content)
        self.assertEqual(data['metro_state'], 'WA')

    def test_validate_marks_every_row_in_the_group_without_changing_the_name(self):
        resp = self.client.patch(
            '/api/geocode_review/action/', self._pisa_group_payload(action='validate'),
            format='json',
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

        self.pisa_1.refresh_from_db()
        self.pisa_2.refresh_from_db()
        self.other_livorno_area.refresh_from_db()
        self.assertTrue(self.pisa_1.metro_validated)
        self.assertTrue(self.pisa_2.metro_validated)
        self.assertFalse(self.pisa_1.metro_override)
        self.assertEqual(self.pisa_1.nearest_metro_name, 'Livorno')
        # Not part of the Pisa group - untouched.
        self.assertFalse(self.other_livorno_area.metro_validated)

    def test_correct_via_major_places_gazetteer_updates_every_row_in_group_only(self):
        resp = self.client.patch(
            '/api/geocode_review/action/',
            self._pisa_group_payload(action='correct', metro_name_correction='Florence'),
            format='json',
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = json.loads(resp.content)
        self.assertEqual(data['metro_name'], 'Florence')
        self.assertGreater(data['metro_distance_km'], 20)  # Florence is further than Livorno

        self.pisa_1.refresh_from_db()
        self.pisa_2.refresh_from_db()
        self.other_livorno_area.refresh_from_db()
        self.assertEqual(self.pisa_1.nearest_metro_name, 'Florence')
        self.assertEqual(self.pisa_2.nearest_metro_name, 'Florence')
        self.assertTrue(self.pisa_1.metro_override)
        self.assertTrue(self.pisa_1.metro_validated)
        # Distance recomputed per-row, not copied from row 1 to row 2 -
        # the two Pisa coordinates aren't identical.
        self.assertNotEqual(self.pisa_1.nearest_metro_distance_km, self.pisa_2.nearest_metro_distance_km)
        # The unrelated Collesalvetti/Livorno group is untouched by a
        # correction scoped to the Pisa group, even though it shares the
        # same OLD metro name - the exact corruption this feature must
        # avoid (see api/geocode_views.py's _group_filter comment).
        self.assertEqual(self.other_livorno_area.nearest_metro_name, 'Livorno')
        self.assertFalse(self.other_livorno_area.metro_override)

    def test_correct_falls_back_to_nominatim_for_a_place_not_in_the_gazetteer(self):
        fake_location = mock.Mock()
        fake_location.raw = {'address': {'country': 'United States', 'state': 'Wyoming'}, 'name': 'Yellowstone National Park'}
        fake_location.latitude = 44.4280
        fake_location.longitude = -110.5885
        fake_location.address = 'Yellowstone National Park, Wyoming, United States'

        with mock.patch('filepopulator.geocode._get_nominatim_forward_geocode', return_value=lambda *a, **k: fake_location):
            resp = self.client.patch(
                '/api/geocode_review/action/',
                self._pisa_group_payload(action='correct', metro_name_correction='Yellowstone National Park'),
                format='json',
            )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = json.loads(resp.content)
        self.assertEqual(data['metro_name'], 'Yellowstone National Park')

        self.pisa_1.refresh_from_db()
        self.assertEqual(self.pisa_1.nearest_metro_name, 'Yellowstone National Park')
        self.assertTrue(self.pisa_1.metro_override)

    def test_correct_rejects_an_unrecognized_place(self):
        with mock.patch('filepopulator.geocode._get_nominatim_forward_geocode', return_value=lambda *a, **k: None):
            resp = self.client.patch(
                '/api/geocode_review/action/',
                self._pisa_group_payload(action='correct', metro_name_correction='Asdfqwertyville'),
                format='json',
            )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.pisa_1.refresh_from_db()
        self.assertEqual(self.pisa_1.nearest_metro_name, 'Livorno')  # unchanged

    def test_correct_rejects_a_us_result_with_no_resolvable_state(self):
        fake_location = mock.Mock()
        fake_location.raw = {'address': {'country': 'United States'}, 'name': 'Somewhere'}
        fake_location.latitude = 40.0
        fake_location.longitude = -100.0
        fake_location.address = 'Somewhere, United States'

        with mock.patch('filepopulator.geocode._get_nominatim_forward_geocode', return_value=lambda *a, **k: fake_location):
            resp = self.client.patch(
                '/api/geocode_review/action/',
                self._pisa_group_payload(action='correct', metro_name_correction='Somewhere'),
                format='json',
            )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_correct_surfaces_a_real_nominatim_failure_distinctly(self):
        # swallow_exceptions=False (geocode.py) means a real Nominatim
        # failure now actually raises instead of looking identical to
        # "not a real place" - this should reach the user as a distinct,
        # retry-worthy error rather than "we don't recognize that place".
        with mock.patch(
            'filepopulator.geocode._get_nominatim_forward_geocode',
            return_value=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('simulated Nominatim timeout')),
        ):
            resp = self.client.patch(
                '/api/geocode_review/action/',
                self._pisa_group_payload(action='correct', metro_name_correction='Asdfqwertyville'),
                format='json',
            )
        self.assertEqual(resp.status_code, 503)
        self.pisa_1.refresh_from_db()
        self.assertEqual(self.pisa_1.nearest_metro_name, 'Livorno')  # unchanged

    def test_correct_surfaces_a_rate_limit_with_a_specific_wait_message(self):
        # Confirmed for real 2026-09-10: Nominatim's public instance can
        # temporarily rate-limit/block a server's IP for a cooldown period
        # noticeably longer than "try again in a moment" - this needs its
        # own distinct message/status so the frontend doesn't undersell
        # how long the wait actually is, or auto-retry into it.
        from geopy.exc import GeocoderRateLimited
        with mock.patch(
            'filepopulator.geocode._get_nominatim_forward_geocode',
            return_value=lambda *a, **k: (_ for _ in ()).throw(GeocoderRateLimited('Non-successful status code 429')),
        ):
            resp = self.client.patch(
                '/api/geocode_review/action/',
                self._pisa_group_payload(action='correct', metro_name_correction='Asdfqwertyville'),
                format='json',
            )
        self.assertEqual(resp.status_code, 429)
        data = json.loads(resp.content)
        self.assertIn('rate limit', data['error'].lower())
        self.assertIn('hour', data['error'].lower())
        self.pisa_1.refresh_from_db()
        self.assertEqual(self.pisa_1.nearest_metro_name, 'Livorno')  # unchanged

    def test_list_exposes_locality_is_approximate(self):
        GeocodeCache.objects.create(
            lat=2.0, lon=2.0, locality='Nearest Named Place', locality_is_approximate=True,
            nearest_metro_name='Some Metro', nearest_metro_distance_km=50.0,
        )
        resp = self.client.get('/api/geocode_review/')
        data = json.loads(resp.content)
        row = next(r for r in data['results'] if r['locality'] == 'Nearest Named Place')
        self.assertTrue(row['locality_is_approximate'])
        # Pisa's locality came from a real precise lookup in this fixture set.
        pisa_row = next(r for r in data['results'] if r['locality'] == 'Pisa')
        self.assertFalse(pisa_row['locality_is_approximate'])

    def test_action_requires_authentication(self):
        resp = self.anon_client.patch(
            '/api/geocode_review/action/', self._pisa_group_payload(action='validate'), format='json',
        )
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_search_places_returns_matches_sorted_by_population(self):
        resp = self.client.get('/api/geocode_review/search_places/', {'q': 'florence'})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = json.loads(resp.content)
        names = [r['name'] for r in data['results']]
        self.assertIn('Florence', names)

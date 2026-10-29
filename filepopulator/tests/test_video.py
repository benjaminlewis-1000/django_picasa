"""filepopulator tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test filepopulator.tests.test_video
"""
from django.test import TestCase
from django.test import override_settings
from django.utils import timezone
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django import forms
from django.conf import settings
from django.db import connection
import psycopg2
import zlib
import hashlib
import json
import os
import binascii
import subprocess
from datetime import datetime, timedelta
from textwrap import wrap # for splitting string
import os
import shutil
import tempfile
import numpy as np
import imageio
import time
from GPSPhoto import gpsphoto
import random
from time import sleep
from PIL import Image
import unittest
from unittest import mock
import pillow_heif

from django.core.management import call_command
from django.core.management.base import CommandError

from filepopulator.models import ImageFile, Directory, DuplicateFile, FailedImageFile, PhotoCollection, guess_date_from_filename, VideoFile, FailedVideoFile, DuplicateVideoFile
from filepopulator.scripts import create_image_file, add_from_root_dir, delete_removed_photos, update_dirs_datetime, check_file_mods, delete_old_thumbnails
from filepopulator.video_scripts import create_video_file, add_videos_from_root_dir, delete_removed_videos, _parse_exif_date, compute_video_content_hash
from face_manager.models import Face, get_default_blank_person
from common.open_img_oriented import apply_exif_orientation
from filepopulator import video_scripts as _video_scripts_module
from filepopulator.test_video_cache import cached_run_ffprobe, cached_run_exiftool

from filepopulator.tests import _tiny_jpeg_bytes

_video_cache_patches = []


def setUpModule():
    _video_cache_patches.extend([
        mock.patch.object(_video_scripts_module, '_run_ffprobe', side_effect=cached_run_ffprobe),
        mock.patch.object(_video_scripts_module, '_run_exiftool', side_effect=cached_run_exiftool),
    ])
    for p in _video_cache_patches:
        p.start()


def tearDownModule():
    for p in _video_cache_patches:
        p.stop()
    _video_cache_patches.clear()



@override_settings(MEDIA_ROOT="/tmp/filepopulator_test_media")
class VideoIngestionTests(TestCase):
    """VIDEO_DIR mirrors real fixture data locally: 8 real video samples
    (one per real format found in the actual video library -- mp4, mov,
    mpg, avi, m2ts, mts, wmv, 3gp) at
    /mnt/fast_storage/appdata/django_picasa/test_suite/sample_photos/
    video_samples/, mounted read-only under /photos/video_samples. CI
    only has the single synthetic no-metadata stub from
    ci_fixtures/video_stub/ (see generate_fixtures.py) -- tests here work
    with "whatever's present" rather than assuming a specific count or
    specific metadata, so they hold in both environments.

    Metadata handling verified empirically against all 8 real samples
    (see CLAUDE.md): ffprobe alone misses creation date/camera info that
    exiftool recovers for some real camcorder-sourced files; a global
    exiftool -n flag was tried and rejected (it also breaks Make/Model
    decoding for at least one real file); dateutil.parser was tried and
    rejected as a date-parsing fallback (silently defaults the date
    portion to today() for exiftool's colon-separated date format,
    confirmed against a real file, no exception raised)."""

    VIDEO_DIR = '/photos/video_samples'

    def tearDown(self):
        VideoFile.objects.all().delete()
        FailedVideoFile.objects.all().delete()

    def _video_files(self):
        return sorted(os.listdir(self.VIDEO_DIR))

    def _duration_seconds(self, path):
        from filepopulator.video_scripts import _run_ffprobe
        probe = _run_ffprobe(path)
        return float(probe.get('format', {}).get('duration', 0) or 0)

    def _long_enough_files(self):
        # Real fixtures include at least one real clip under
        # settings.MIN_VIDEO_DURATION_SECONDS (a real ~2s digitized home
        # movie snippet), and the CI synthetic stub is 1s -- tests here
        # partition by actual measured duration rather than assuming
        # every fixture clears the bar.
        from django.conf import settings
        return [
            f for f in self._video_files()
            if self._duration_seconds(os.path.join(self.VIDEO_DIR, f)) >= settings.MIN_VIDEO_DURATION_SECONDS
        ]

    def test_ingests_fixtures_at_or_above_minimum_duration(self):
        video_files = self._video_files()
        self.assertGreaterEqual(len(video_files), 1)
        long_enough = set(self._long_enough_files())
        for filename in video_files:
            path = os.path.join(self.VIDEO_DIR, filename)
            create_video_file(path)
            if filename in long_enough:
                v = VideoFile.objects.filter(filename=path).first()
                self.assertIsNotNone(v, f"{filename} was not ingested")
                self.assertGreater(v.width, 0)
                self.assertGreater(v.height, 0)
                self.assertEqual(v.file_size_bytes, os.path.getsize(path))
                self.assertFalse(FailedVideoFile.objects.filter(filename=path).exists())
            else:
                self.assertFalse(VideoFile.objects.filter(filename=path).exists())
                failed = FailedVideoFile.objects.filter(filename=path).first()
                self.assertIsNotNone(failed, f"{filename} should have been excluded as too short")
                self.assertIn("Excluded", failed.error_message)

    def test_videos_shorter_than_minimum_are_excluded_not_ingested(self):
        # Explicit regression test for the exclusion itself, not just
        # incidental coverage from the fixture loop above.
        long_enough = set(self._long_enough_files())
        short_files = [f for f in self._video_files() if f not in long_enough]
        if not short_files:
            self.skipTest("no fixture in VIDEO_DIR is shorter than MIN_VIDEO_DURATION_SECONDS")
        path = os.path.join(self.VIDEO_DIR, short_files[0])
        create_video_file(path)
        self.assertFalse(VideoFile.objects.filter(filename=path).exists())
        self.assertTrue(FailedVideoFile.objects.filter(filename=path, error_message__icontains="Excluded").exists())

    def test_add_videos_from_root_dir_discovers_video_files(self):
        add_videos_from_root_dir([self.VIDEO_DIR])
        self.assertEqual(VideoFile.objects.count(), len(self._long_enough_files()))

    def test_rerunning_finds_nothing_new(self):
        # Regression-shaped test: a second run over the same root
        # shouldn't create duplicate rows or re-process already-known
        # files.
        add_videos_from_root_dir([self.VIDEO_DIR])
        first_count = VideoFile.objects.count()
        add_videos_from_root_dir([self.VIDEO_DIR])
        self.assertEqual(VideoFile.objects.count(), first_count)

    def test_tracked_video_root_going_empty_is_recorded_as_an_application_error(self):
        # Mirrors add_from_root_dir()'s own check (image side) -- a real,
        # exceptional signal (dropped mount, bad permissions, a renamed
        # root), distinct from a brand-new, genuinely-empty root (next
        # test), which must NOT be flagged. VIDEO_DIR itself is a
        # read-only real-fixture mount, so this copies one fixture into a
        # controlled tmp root instead of touching VIDEO_DIR directly.
        from common.models import ApplicationError
        long_enough = self._long_enough_files()
        if not long_enough:
            self.skipTest("no real fixture long enough to ingest")
        tmp_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_root, ignore_errors=True)
        shutil.copy(os.path.join(self.VIDEO_DIR, long_enough[0]), os.path.join(tmp_root, long_enough[0]))

        add_videos_from_root_dir([tmp_root])
        self.assertGreater(VideoFile.objects.count(), 0)

        os.remove(os.path.join(tmp_root, long_enough[0]))
        add_videos_from_root_dir([tmp_root])

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'filepopulator.video_scripts.add_videos_from_root_dir')
        self.assertIn(tmp_root, error.message)

    def test_brand_new_empty_video_root_is_not_recorded(self):
        from common.models import ApplicationError
        empty_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, empty_root, ignore_errors=True)
        add_videos_from_root_dir([empty_root])
        self.assertEqual(ApplicationError.objects.count(), 0)

    def test_multiple_roots_are_all_walked(self):
        # add_videos_from_root_dir() takes a LIST of roots (built by the
        # caller from every PhotoCollection's root_directory plus its
        # extra_video_roots -- see load_videos_into_db()) -- confirm
        # passing two roots actually walks both, not just the first.
        #
        # The second root's copy is BYTE-IDENTICAL to the first (same
        # fixture, copied verbatim) -- since content-hash-based duplicate
        # detection now runs at ingestion, that means it's correctly
        # recognized as a duplicate rather than given its own VideoFile
        # row (see VideoContentHashDuplicateTests). The real thing this
        # test needs to confirm -- that the second root was genuinely
        # walked and its file genuinely discovered/processed, not just
        # silently skipped -- still holds: a DuplicateVideoFile record
        # is exactly that evidence.
        other_dir = '/tmp/video_other_root'
        os.makedirs(other_dir, exist_ok=True)
        self.addCleanup(shutil.rmtree, other_dir, ignore_errors=True)
        long_enough = self._long_enough_files()
        self.assertGreaterEqual(len(long_enough), 1)
        first_file = os.path.join(self.VIDEO_DIR, long_enough[0])
        other_copy = os.path.join(other_dir, long_enough[0])
        shutil.copy(first_file, other_copy)

        add_videos_from_root_dir([self.VIDEO_DIR, other_dir])

        self.assertFalse(VideoFile.objects.filter(filename=other_copy).exists())
        self.assertTrue(DuplicateVideoFile.objects.filter(filename=other_copy).exists())
        self.assertEqual(VideoFile.objects.count(), len(long_enough))

    def test_nonexistent_file_is_recorded_as_failed(self):
        create_video_file('/photos/video_samples/does_not_exist.mp4')
        self.assertFalse(VideoFile.objects.filter(filename='/photos/video_samples/does_not_exist.mp4').exists())
        self.assertTrue(FailedVideoFile.objects.filter(filename='/photos/video_samples/does_not_exist.mp4').exists())

    def test_gps_and_camera_info_recovered_when_present(self):
        # Not every fixture (or CI's synthetic stub) has GPS/camera
        # metadata -- only assert against ones that actually do.
        found_gps = False
        found_camera = False
        for filename in self._long_enough_files():
            path = os.path.join(self.VIDEO_DIR, filename)
            create_video_file(path)
            v = VideoFile.objects.get(filename=path)
            if v.gps_lat_decimal != -999 or v.gps_lon_decimal != -999:
                found_gps = True
                self.assertGreaterEqual(v.gps_lat_decimal, -90)
                self.assertLessEqual(v.gps_lat_decimal, 90)
                self.assertGreaterEqual(v.gps_lon_decimal, -180)
                self.assertLessEqual(v.gps_lon_decimal, 180)
            if v.camera_make:
                found_camera = True
        if not found_gps and not found_camera:
            self.skipTest("no fixture in VIDEO_DIR has GPS or camera metadata (expected for CI's synthetic stub)")

    def test_backfill_command_populates_missing_file_size(self):
        from django.core.management import call_command

        long_enough = self._long_enough_files()
        if not long_enough:
            self.skipTest("no fixture in VIDEO_DIR clears MIN_VIDEO_DURATION_SECONDS")
        path = os.path.join(self.VIDEO_DIR, long_enough[0])
        create_video_file(path)
        VideoFile.objects.filter(filename=path).update(file_size_bytes=None)

        call_command('backfill_video_file_size')

        v = VideoFile.objects.get(filename=path)
        self.assertEqual(v.file_size_bytes, os.path.getsize(path))

    def test_backfill_command_dry_run_makes_no_changes(self):
        long_enough = self._long_enough_files()
        if not long_enough:
            self.skipTest("no fixture in VIDEO_DIR clears MIN_VIDEO_DURATION_SECONDS")
        from django.core.management import call_command

        path = os.path.join(self.VIDEO_DIR, long_enough[0])
        create_video_file(path)
        VideoFile.objects.filter(filename=path).update(file_size_bytes=None)

        call_command('backfill_video_file_size', '--dry-run')

        v = VideoFile.objects.get(filename=path)
        self.assertIsNone(v.file_size_bytes)

    def test_content_hash_is_populated_on_ingest(self):
        long_enough = self._long_enough_files()
        if not long_enough:
            self.skipTest("no fixture in VIDEO_DIR clears MIN_VIDEO_DURATION_SECONDS")
        path = os.path.join(self.VIDEO_DIR, long_enough[0])
        create_video_file(path)
        v = VideoFile.objects.get(filename=path)
        self.assertIsNotNone(v.content_hash)
        self.assertEqual(v.content_hash, compute_video_content_hash(path))

    def test_duplicate_content_at_new_path_is_recorded_not_reingested(self):
        long_enough = self._long_enough_files()
        if not long_enough:
            self.skipTest("no fixture in VIDEO_DIR clears MIN_VIDEO_DURATION_SECONDS")
        original_path = os.path.join(self.VIDEO_DIR, long_enough[0])
        create_video_file(original_path)
        original = VideoFile.objects.get(filename=original_path)

        dup_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, dup_dir, ignore_errors=True)
        dup_path = os.path.join(dup_dir, 'copy_' + long_enough[0])
        shutil.copy(original_path, dup_path)

        create_video_file(dup_path)

        self.assertFalse(VideoFile.objects.filter(filename=dup_path).exists())
        dup_record = DuplicateVideoFile.objects.filter(filename=dup_path).first()
        self.assertIsNotNone(dup_record)
        self.assertEqual(dup_record.original_id, original.pk)
        # No second row was created for the duplicate content.
        self.assertEqual(VideoFile.objects.filter(content_hash=original.content_hash).count(), 1)

    def test_moved_file_same_content_reuses_existing_row(self):
        long_enough = self._long_enough_files()
        if not long_enough:
            self.skipTest("no fixture in VIDEO_DIR clears MIN_VIDEO_DURATION_SECONDS")
        move_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, move_dir, ignore_errors=True)
        original_copy = os.path.join(move_dir, 'orig_' + long_enough[0])
        shutil.copy(os.path.join(self.VIDEO_DIR, long_enough[0]), original_copy)

        create_video_file(original_copy)
        original_pk = VideoFile.objects.get(filename=original_copy).pk

        new_path = os.path.join(move_dir, 'moved_' + long_enough[0])
        os.rename(original_copy, new_path)

        create_video_file(new_path)

        self.assertFalse(VideoFile.objects.filter(filename=original_copy).exists())
        moved = VideoFile.objects.filter(filename=new_path).first()
        self.assertIsNotNone(moved)
        self.assertEqual(moved.pk, original_pk)
        self.assertEqual(VideoFile.objects.count(), 1)

    def test_stale_duplicate_record_is_wiped_when_content_at_path_changes(self):
        # Real gap found 2026-10-14 (video move-robustness review): unlike
        # create_image_file(), create_video_file() had no guard for "a
        # stale DuplicateVideoFile record already sits at this exact
        # path" -- if that path's real content later genuinely changes,
        # the old record would just sit there stale, pointing at
        # whatever used to be the "original". Not a data-loss risk (no
        # VideoFile row ever coexists with a live DuplicateVideoFile at
        # the same path), just confusing bookkeeping left unresolved.
        long_enough = self._long_enough_files()
        if len(long_enough) < 2:
            self.skipTest("need at least 2 distinct real fixtures in VIDEO_DIR")

        work_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, work_dir, ignore_errors=True)
        original_path = os.path.join(work_dir, 'orig_' + long_enough[0])
        shutil.copy(os.path.join(self.VIDEO_DIR, long_enough[0]), original_path)
        create_video_file(original_path)
        original = VideoFile.objects.get(filename=original_path)

        # Record it as a duplicate of `original`, matching the real
        # shape of the stale-bookkeeping scenario.
        changing_path = os.path.join(work_dir, 'changing_' + long_enough[0])
        shutil.copy(original_path, changing_path)
        create_video_file(changing_path)
        self.assertTrue(DuplicateVideoFile.objects.filter(filename=changing_path).exists())

        # The real content at that path now genuinely changes.
        os.remove(changing_path)
        shutil.copy(os.path.join(self.VIDEO_DIR, long_enough[1]), changing_path)

        create_video_file(changing_path)

        # The stale duplicate record is gone (wiped, not left pointing at
        # the wrong "original" forever) -- and a later rescan of this
        # same path can genuinely ingest it fresh.
        self.assertFalse(DuplicateVideoFile.objects.filter(filename=changing_path).exists())
        self.assertFalse(VideoFile.objects.filter(filename=changing_path).exists())
        create_video_file(changing_path)
        self.assertTrue(VideoFile.objects.filter(filename=changing_path).exists())


class VideoDirectoryCollectionTests(TestCase):
    """create_video_file() resolves+sets Directory.collection on a newly
    created video directory (2026-10-21 fix) -- previously a plain
    get_or_create with no collection resolution at all, leaving every
    video Directory permanently NULL. Mirrors PhotoCollectionTests'
    image-side setup/assertions, since VideoFile.directory is the SAME
    Directory model ImageFile uses -- no new field was needed, just the
    missing resolve_for_path() call."""

    def setUp(self):
        self.default_collection = PhotoCollection.objects.get(is_default=True)
        self.second_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.second_root, ignore_errors=True)
        self.second_collection = PhotoCollection.objects.create(
            name='Second Video Collection', root_directory=self.second_root,
        )

    def tearDown(self):
        VideoFile.objects.all().delete()
        FailedVideoFile.objects.all().delete()
        Directory.objects.exclude(collection=self.default_collection).delete()

    def _a_long_enough_fixture(self):
        from django.conf import settings
        video_dir = '/photos/video_samples'
        for filename in sorted(os.listdir(video_dir)):
            path = os.path.join(video_dir, filename)
            from filepopulator.video_scripts import _run_ffprobe
            probe = _run_ffprobe(path)
            duration = float(probe.get('format', {}).get('duration', 0) or 0)
            if duration >= settings.MIN_VIDEO_DURATION_SECONDS:
                return path
        self.skipTest("No fixture video clears MIN_VIDEO_DURATION_SECONDS in this environment.")

    def test_new_video_under_second_root_gets_that_collection(self):
        fixture = self._a_long_enough_fixture()
        dest = os.path.join(self.second_root, os.path.basename(fixture))
        shutil.copy(fixture, dest)

        create_video_file(dest)

        video = VideoFile.objects.get(filename=dest)
        self.assertEqual(video.directory.collection_id, self.second_collection.pk)

    def test_video_under_unregistered_path_falls_back_to_default(self):
        fixture = self._a_long_enough_fixture()
        unregistered_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, unregistered_dir, ignore_errors=True)
        dest = os.path.join(unregistered_dir, os.path.basename(fixture))
        shutil.copy(fixture, dest)

        create_video_file(dest)

        video = VideoFile.objects.get(filename=dest)
        self.assertEqual(video.directory.collection_id, self.default_collection.pk)

    def test_backfill_resolves_preexisting_null_video_directory(self):
        # Simulates a Directory created before the fix existed -- plain
        # get_or_create, no collection ever resolved.
        pre_existing_dir = Directory.objects.create(dir_path=self.second_root)
        self.assertIsNone(pre_existing_dir.collection)

        fixture = self._a_long_enough_fixture()
        dest = os.path.join(self.second_root, os.path.basename(fixture))
        shutil.copy(fixture, dest)
        video = VideoFile.objects.create(
            filename=dest, directory=pre_existing_dir, width=100, height=100, duration_seconds=10,
        )

        call_command('backfill_video_directory_collection')

        pre_existing_dir.refresh_from_db()
        self.assertEqual(pre_existing_dir.collection_id, self.second_collection.pk)

    def test_backfill_dry_run_does_not_write(self):
        pre_existing_dir = Directory.objects.create(dir_path=self.second_root)
        VideoFile.objects.create(
            filename=os.path.join(self.second_root, 'a.mp4'), directory=pre_existing_dir,
            width=100, height=100, duration_seconds=10,
        )

        call_command('backfill_video_directory_collection', '--dry-run')

        pre_existing_dir.refresh_from_db()
        self.assertIsNone(pre_existing_dir.collection)

    def test_backfill_does_not_touch_image_only_directories(self):
        # A Directory with no VideoFile at all (NULL collection for some
        # other, unrelated reason) must not be touched by this video-
        # specific backfill.
        image_only_dir = Directory.objects.create(dir_path=tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, image_only_dir.dir_path, ignore_errors=True)
        self.assertIsNone(image_only_dir.collection)

        call_command('backfill_video_directory_collection')

        image_only_dir.refresh_from_db()
        self.assertIsNone(image_only_dir.collection)


class VideoCrossCollectionDuplicateTests(TestCase):
    """create_video_file()'s content_hash-based duplicate detection
    already prevented a second VideoFile row for the same real content
    landing in a second collection -- these confirm the real gap that
    was missing until 2026-10-21: that the primary becomes queryable as
    "visible from" the second collection too (VideoFile.collections),
    not just left attributed to its own original home. Exact mirror of
    PhotoCollectionTests' equivalent image-side test."""

    def setUp(self):
        self.default_collection = PhotoCollection.objects.get(is_default=True)
        self.second_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.second_root, ignore_errors=True)
        self.second_collection = PhotoCollection.objects.create(
            name='Second Video Collection', root_directory=self.second_root,
        )

    def tearDown(self):
        VideoFile.objects.all().delete()
        FailedVideoFile.objects.all().delete()
        DuplicateVideoFile.objects.all().delete()
        Directory.objects.exclude(collection=self.default_collection).delete()

    def _a_long_enough_fixture(self):
        from django.conf import settings
        video_dir = '/photos/video_samples'
        for filename in sorted(os.listdir(video_dir)):
            path = os.path.join(video_dir, filename)
            from filepopulator.video_scripts import _run_ffprobe
            probe = _run_ffprobe(path)
            duration = float(probe.get('format', {}).get('duration', 0) or 0)
            if duration >= settings.MIN_VIDEO_DURATION_SECONDS:
                return path
        self.skipTest("No fixture video clears MIN_VIDEO_DURATION_SECONDS in this environment.")

    def test_duplicate_found_under_second_collection_cross_links_the_original(self):
        fixture = self._a_long_enough_fixture()

        original_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, original_dir, ignore_errors=True)
        original_path = os.path.join(original_dir, os.path.basename(fixture))
        shutil.copy(fixture, original_path)
        create_video_file(original_path)
        original = VideoFile.objects.get(filename=original_path)

        # Confirm its home really did resolve to the default collection
        # (original_dir was never registered as any collection's root).
        self.assertEqual(original.directory.collection_id, self.default_collection.pk)
        self.assertIn(self.default_collection, original.collections.all())
        self.assertNotIn(self.second_collection, original.collections.all())

        dup_path = os.path.join(self.second_root, 'duplicate_' + os.path.basename(fixture))
        shutil.copy(fixture, dup_path)
        create_video_file(dup_path)

        # No second VideoFile row -- same content, recorded as a
        # DuplicateVideoFile instead.
        self.assertFalse(VideoFile.objects.filter(filename=dup_path).exists())
        dup = DuplicateVideoFile.objects.get(filename=dup_path)
        self.assertEqual(dup.original_id, original.pk)
        self.assertEqual(dup.collection_id, self.second_collection.pk)

        # The ORIGINAL (whose home is still the default collection) is
        # now also visible from the second collection.
        original.refresh_from_db()
        self.assertIn(self.default_collection, original.collections.all())
        self.assertIn(self.second_collection, original.collections.all())


@override_settings(VIDEO_TRANSCODE_ROOT='/tmp/video_transcode_test')
class TranscodeVideoTests(TestCase):
    """Phase 5 (transcode pipeline), added 2026-10-15. Mocks subprocess.run
    entirely (real ffmpeg encoding is slow and this is about the
    control-flow/bookkeeping around it, not ffmpeg itself) -- a fake
    success writes a dummy file at the expected .tmp path (since the
    real code checks os.path.isfile() on it) before returning
    returncode=0."""

    def setUp(self):
        from filepopulator.video_scripts import _sharded_transcode_path
        self.transcode_root = settings.VIDEO_TRANSCODE_ROOT
        if os.path.exists(self.transcode_root):
            shutil.rmtree(self.transcode_root)
        os.makedirs(self.transcode_root)
        self._sharded_transcode_path = _sharded_transcode_path

    def tearDown(self):
        shutil.rmtree(self.transcode_root, ignore_errors=True)
        VideoFile.objects.all().delete()

    _UNSET = object()

    def _make_video(self, codec='hevc', content_hash=_UNSET, **overrides):
        directory, _ = Directory.objects.get_or_create(dir_path='/tmp/transcode_test_videos')
        if content_hash is self._UNSET:
            content_hash = binascii.hexlify(os.urandom(16)).decode()
        defaults = dict(
            filename=f'/tmp/transcode_test_videos/{overrides.pop("name", "a")}.mp4',
            directory=directory, width=320, height=240, duration_seconds=10.0,
            codec=codec, content_hash=content_hash,
        )
        defaults.update(overrides)
        return VideoFile.objects.create(**defaults)

    def _fake_success(self, tmp_path_arg_index=-1):
        def side_effect(cmd, **kwargs):
            tmp_path = cmd[tmp_path_arg_index]
            with open(tmp_path, 'wb') as f:
                f.write(b'fake mp4 bytes')
            return subprocess.CompletedProcess(cmd, 0, stdout='', stderr='')
        return side_effect

    def test_successful_transcode_sets_path_and_clears_failure_state(self):
        from filepopulator.video_scripts import transcode_video
        video = self._make_video()

        with mock.patch('filepopulator.video_scripts.subprocess.run', side_effect=self._fake_success()):
            transcode_video(video)

        video.refresh_from_db()
        self.assertFalse(video.transcode_failed)
        self.assertIsNone(video.transcode_error)
        self.assertIsNotNone(video.transcoded_path)
        expected_rel = self._sharded_transcode_path(video.content_hash)
        self.assertEqual(video.transcoded_path, expected_rel)
        out_path = os.path.join(self.transcode_root, expected_rel)
        self.assertTrue(os.path.isfile(out_path))
        # The .tmp sibling was renamed away, not left behind.
        self.assertFalse(os.path.isfile(out_path + '.tmp'))

    def test_ffmpeg_nonzero_exit_marks_failed(self):
        from filepopulator.video_scripts import transcode_video
        video = self._make_video()

        with mock.patch('filepopulator.video_scripts.subprocess.run') as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 1, stdout='', stderr='real ffmpeg error output')
            transcode_video(video)

        video.refresh_from_db()
        self.assertTrue(video.transcode_failed)
        self.assertIn('real ffmpeg error output', video.transcode_error)
        self.assertIsNone(video.transcoded_path)

    def test_ffmpeg_timeout_marks_failed(self):
        from filepopulator.video_scripts import transcode_video
        video = self._make_video()

        with mock.patch('filepopulator.video_scripts.subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd='ffmpeg', timeout=3600)
            transcode_video(video)

        video.refresh_from_db()
        self.assertTrue(video.transcode_failed)
        self.assertIn('timed out', video.transcode_error)

    def test_missing_content_hash_fails_without_calling_ffmpeg(self):
        from filepopulator.video_scripts import transcode_video
        video = self._make_video(content_hash=None)

        with mock.patch('filepopulator.video_scripts.subprocess.run') as mock_run:
            transcode_video(video)
            mock_run.assert_not_called()

        video.refresh_from_db()
        self.assertTrue(video.transcode_failed)

    def test_pending_videos_skips_native_codecs(self):
        from filepopulator.video_scripts import transcode_pending_videos
        native = self._make_video(codec='h264', name='native')
        needs_it = self._make_video(codec='hevc', name='needs_it')

        with mock.patch('filepopulator.video_scripts.subprocess.run', side_effect=self._fake_success()):
            done, failed = transcode_pending_videos()

        native.refresh_from_db()
        needs_it.refresh_from_db()
        self.assertIsNone(native.transcoded_path)
        self.assertIsNotNone(needs_it.transcoded_path)
        self.assertEqual((done, failed), (1, 0))

    def test_pending_videos_skips_already_done_or_failed(self):
        from filepopulator.video_scripts import transcode_pending_videos
        already_done = self._make_video(codec='hevc', name='done', transcoded_path='xx/y/already.mp4')
        already_failed = self._make_video(codec='hevc', name='failed', transcode_failed=True)

        with mock.patch('filepopulator.video_scripts.subprocess.run') as mock_run:
            done, failed = transcode_pending_videos()
            mock_run.assert_not_called()

        self.assertEqual((done, failed), (0, 0))

    def test_max_runtime_seconds_stops_between_videos(self):
        from filepopulator.video_scripts import transcode_pending_videos
        self._make_video(codec='hevc', name='first')
        self._make_video(codec='hevc', name='second')

        with mock.patch('filepopulator.video_scripts.subprocess.run', side_effect=self._fake_success()):
            done, failed = transcode_pending_videos(max_runtime_seconds=0)

        self.assertEqual((done, failed), (0, 0))

    def test_sharded_path_matches_thumbnail_style_convention(self):
        from filepopulator.video_scripts import _sharded_transcode_path
        content_hash = 'abcdef0123456789'
        rel = _sharded_transcode_path(content_hash)
        self.assertEqual(rel, os.path.join('ab', 'c', 'abcdef0123456789.mp4'))

    def test_deleting_video_removes_transcoded_file(self):
        from filepopulator.video_scripts import transcode_video
        video = self._make_video()
        with mock.patch('filepopulator.video_scripts.subprocess.run', side_effect=self._fake_success()):
            transcode_video(video)
        video.refresh_from_db()
        out_path = os.path.join(self.transcode_root, video.transcoded_path)
        self.assertTrue(os.path.isfile(out_path))

        video.delete()

        self.assertFalse(os.path.isfile(out_path))


class DeleteRemovedVideosTests(TestCase):
    """delete_removed_videos() -- the video-side mirror of scripts.py's
    delete_removed_photos(), built 2026-09-15 (previously a deliberate
    Phase 1 scope cut -- see CLAUDE.md) once a real case came up (a
    VTS_*-named DVD-rip .avi moved out of the library, leaving its
    VideoFile row and Face row stranded)."""

    VIDEO_DIR = '/photos/video_samples'

    def tearDown(self):
        VideoFile.objects.all().delete()

    def _long_enough_candidates(self):
        from django.conf import settings
        from filepopulator.video_scripts import _run_ffprobe

        out = []
        for filename in sorted(os.listdir(self.VIDEO_DIR)):
            src = os.path.join(self.VIDEO_DIR, filename)
            probe = _run_ffprobe(src)
            duration = float(probe.get('format', {}).get('duration', 0) or 0)
            if duration >= settings.MIN_VIDEO_DURATION_SECONDS:
                out.append(filename)
        if not out:
            self.skipTest("no fixture in VIDEO_DIR clears MIN_VIDEO_DURATION_SECONDS")
        return out

    def _a_real_fixture_copy(self):
        # Cycles through distinct real fixtures on repeated calls (the
        # real dev/test fixture dir has 8) so two calls in the same test
        # produce genuinely different file content -- needed since
        # content-hash-based duplicate detection (2026-09-22) now
        # recognizes two byte-identical copies as the same video and
        # only ever creates one VideoFile row for them. Falls back to
        # appending random trailing bytes if fewer distinct fixtures
        # exist than calls made (e.g. CI's single synthetic stub) --
        # verified safe against a real synthetic fixture: appending
        # bytes after a container's own real data doesn't disturb
        # ffprobe's ability to read it.
        candidates = self._long_enough_candidates()
        count = getattr(self, '_fixture_copy_count', 0)
        filename = candidates[count % len(candidates)]
        self._fixture_copy_count = count + 1

        src = os.path.join(self.VIDEO_DIR, filename)
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        dest = os.path.join(tmp_dir, filename)
        shutil.copy(src, dest)
        if count >= len(candidates):
            with open(dest, 'ab') as f:
                f.write(os.urandom(64))
        return dest

    def test_delete_removed_videos_removes_vanished_rows_only(self):
        # Two-phase behavior (see VideoFile.missing_since's own comment):
        # a vanished file is flagged, not deleted, on the first pass that
        # notices it, and only actually deleted once it's been missing
        # longer than MISSING_FILE_GRACE_PERIOD.
        gone_path = self._a_real_fixture_copy()
        create_video_file(gone_path)
        self.assertTrue(VideoFile.objects.filter(filename=gone_path).exists())

        staying_path = self._a_real_fixture_copy()
        create_video_file(staying_path)

        os.remove(gone_path)
        delete_removed_videos()
        self.assertTrue(VideoFile.objects.filter(filename=gone_path).exists())
        self.assertIsNotNone(VideoFile.objects.get(filename=gone_path).missing_since)

        VideoFile.objects.filter(filename=gone_path).update(
            missing_since=timezone.now() - settings.MISSING_FILE_GRACE_PERIOD - timedelta(minutes=1)
        )
        delete_removed_videos()

        self.assertFalse(VideoFile.objects.filter(filename=gone_path).exists())
        self.assertTrue(VideoFile.objects.filter(filename=staying_path).exists())

    def test_delete_removed_videos_within_grace_period_is_not_yet_deleted(self):
        gone_path = self._a_real_fixture_copy()
        create_video_file(gone_path)
        os.remove(gone_path)

        VideoFile.objects.filter(filename=gone_path).update(
            missing_since=timezone.now() - timedelta(hours=1)
        )
        delete_removed_videos()

        self.assertTrue(VideoFile.objects.filter(filename=gone_path).exists())

    def test_delete_removed_videos_reappearing_clears_missing_since(self):
        path = self._a_real_fixture_copy()
        create_video_file(path)
        VideoFile.objects.filter(filename=path).update(missing_since=timezone.now())

        delete_removed_videos()

        self.assertIsNone(VideoFile.objects.get(filename=path).missing_since)

    def test_delete_removed_videos_cleans_up_attached_face_and_thumbnail(self):
        from face_manager.models import Person

        gone_path = self._a_real_fixture_copy()
        create_video_file(gone_path)
        video = VideoFile.objects.get(filename=gone_path)

        person, _ = Person.objects.get_or_create(person_name="DeleteRemovedVideosTests Person")
        face = Face(
            source_video_file=video, declared_name=person,
            box_left=1, box_top=1, box_right=min(40, video.width - 1), box_bottom=min(40, video.height - 1),
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()
        thumb_path = face.face_thumbnail.path
        self.assertTrue(os.path.isfile(thumb_path))

        os.remove(gone_path)
        VideoFile.objects.filter(pk=video.pk).update(
            missing_since=timezone.now() - settings.MISSING_FILE_GRACE_PERIOD - timedelta(minutes=1)
        )
        delete_removed_videos()

        self.assertFalse(VideoFile.objects.filter(pk=video.pk).exists())
        self.assertFalse(Face.objects.filter(pk=face.pk).exists())
        self.assertFalse(os.path.isfile(thumb_path))


class VideoContentHashBackfillAndMergeTests(TestCase):
    """backfill_video_content_hash + merge_duplicate_videofiles -- the
    two management commands that resolve VideoFile rows ingested BEFORE
    content-hash-based duplicate detection existed (see
    VideoContentHashDuplicateTests above, which covers the same
    protection at ingestion time going forward). Real motivating case
    (2026-09-22): 5 Face rows all turned out to be the same real frame
    because 5 differently-named files were confirmed byte-for-byte
    identical, each independently face-extracted.

    Builds rows directly (not via create_video_file(), which now
    prevents this scenario from recurring) to simulate the pre-existing
    contamination these commands are meant to clean up."""

    VIDEO_DIR = '/photos/video_samples'

    def tearDown(self):
        VideoFile.objects.all().delete()

    def _a_real_fixture_copy(self):
        from django.conf import settings
        from filepopulator.video_scripts import _run_ffprobe

        for filename in sorted(os.listdir(self.VIDEO_DIR)):
            src = os.path.join(self.VIDEO_DIR, filename)
            probe = _run_ffprobe(src)
            duration = float(probe.get('format', {}).get('duration', 0) or 0)
            if duration >= settings.MIN_VIDEO_DURATION_SECONDS:
                tmp_dir = tempfile.mkdtemp()
                self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
                dest = os.path.join(tmp_dir, filename)
                shutil.copy(src, dest)
                return dest
        self.skipTest("no fixture in VIDEO_DIR clears MIN_VIDEO_DURATION_SECONDS")

    def _pre_content_hash_video(self, path):
        # Ingest normally (populates content_hash going forward), then
        # strip it back off -- simulates a row that predates the field,
        # without duplicating all of create_video_file()'s own field
        # population logic here.
        create_video_file(path)
        video = VideoFile.objects.get(filename=path)
        VideoFile.objects.filter(pk=video.pk).update(content_hash=None)
        video.refresh_from_db()
        return video

    def _ingest(self, path):
        create_video_file(path)
        return VideoFile.objects.get(filename=path)

    def _make_duplicate_row(self, path, primary, isProcessed=False):
        # Bypasses create_video_file()'s own (now-live) duplicate
        # detection deliberately -- constructs a second real row
        # directly, the same shape a row ingested BEFORE this feature
        # existed would have: identical content_hash/dimensions, a
        # genuinely different path that's still really on disk. This is
        # exactly the pre-existing contamination merge_duplicate_
        # videofiles exists to clean up.
        return VideoFile.objects.create(
            filename=path, directory=primary.directory,
            width=primary.width, height=primary.height,
            duration_seconds=primary.duration_seconds,
            content_hash=primary.content_hash, isProcessed=isProcessed,
        )

    def _attach_face(self, video, person=None):
        from face_manager.models import Person
        if person is None:
            person, _ = Person.objects.get_or_create(person_name="VideoContentHashMergeTests Person")
        face = Face(
            source_video_file=video, declared_name=person,
            box_left=1, box_top=1,
            box_right=min(40, video.width - 1), box_bottom=min(40, video.height - 1),
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()
        return face

    def test_backfill_populates_content_hash(self):
        path = self._a_real_fixture_copy()
        video = self._pre_content_hash_video(path)
        self.assertIsNone(video.content_hash)

        from django.core.management import call_command
        call_command('backfill_video_content_hash')

        video.refresh_from_db()
        self.assertEqual(video.content_hash, compute_video_content_hash(path))

    def test_backfill_dry_run_makes_no_changes(self):
        path = self._a_real_fixture_copy()
        video = self._pre_content_hash_video(path)

        from django.core.management import call_command
        call_command('backfill_video_content_hash', '--dry-run')

        video.refresh_from_db()
        self.assertIsNone(video.content_hash)

    def test_backfill_skips_file_vanished_from_disk(self):
        path = self._a_real_fixture_copy()
        video = self._pre_content_hash_video(path)
        os.remove(path)

        from django.core.management import call_command
        call_command('backfill_video_content_hash')

        video.refresh_from_db()
        self.assertIsNone(video.content_hash)

    def test_merge_transfers_faces_and_deletes_unprocessed_duplicate(self):
        # Two rows, same real content, only the primary has been
        # face-processed -- the safe case: merge should keep the
        # processed one, transfer nothing (the duplicate has no faces of
        # its own), and delete the duplicate row.
        path_a = self._a_real_fixture_copy()
        primary = self._ingest(path_a)
        path_b = path_a + '.dup.mp4'
        shutil.copy(path_a, path_b)
        self.addCleanup(lambda: os.path.exists(path_b) and os.remove(path_b))
        duplicate = self._make_duplicate_row(path_b, primary, isProcessed=False)
        VideoFile.objects.filter(pk=primary.pk).update(isProcessed=True)
        primary.refresh_from_db()
        face = self._attach_face(primary)

        from django.core.management import call_command
        call_command('merge_duplicate_videofiles', '--yes')

        self.assertFalse(VideoFile.objects.filter(pk=duplicate.pk).exists())
        self.assertTrue(VideoFile.objects.filter(pk=primary.pk).exists())
        self.assertTrue(Face.objects.filter(pk=face.pk, source_video_file=primary).exists())

    def test_merge_transfers_faces_from_unprocessed_duplicate_onto_primary(self):
        # The duplicate itself has a stray face (e.g. hand-added, or from
        # some other path never marking isProcessed) -- confirm it's
        # really transferred, not silently dropped.
        path_a = self._a_real_fixture_copy()
        primary = self._ingest(path_a)
        VideoFile.objects.filter(pk=primary.pk).update(isProcessed=True)
        primary.refresh_from_db()
        path_b = path_a + '.dup.mp4'
        shutil.copy(path_a, path_b)
        self.addCleanup(lambda: os.path.exists(path_b) and os.remove(path_b))
        duplicate = self._make_duplicate_row(path_b, primary, isProcessed=False)
        stray_face = self._attach_face(duplicate)

        from django.core.management import call_command
        call_command('merge_duplicate_videofiles', '--yes')

        self.assertFalse(VideoFile.objects.filter(pk=duplicate.pk).exists())
        stray_face.refresh_from_db()
        self.assertEqual(stray_face.source_video_file_id, primary.pk)

    def _attach_face_at(self, video, first, last, person=None, validated=False,
                         embedding=None, kps=None, det_score=None):
        from face_manager.models import Person
        if person is None:
            person, _ = Person.objects.get_or_create(person_name=settings.SOFT_IGNORE_NAME)
        face = Face(
            source_video_file=video, declared_name=person, validated=validated,
            box_left=1, box_top=1,
            box_right=min(40, video.width - 1), box_bottom=min(40, video.height - 1),
            video_first_timestamp_seconds=first, video_last_timestamp_seconds=last,
            face_encoding_512=embedding, kps=kps, det_score=det_score,
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()
        return face

    def test_merge_resolves_both_processed_duplicates_keeping_unmatched_faces(self):
        # Both rows already independently face-extracted, and the two
        # faces have no timestamps at all (can't be matched) -- this
        # used to be left alone entirely ("unresolved"); now the videos
        # still merge, and since the faces can't be confidently matched
        # to each other, BOTH survive (transferred onto the primary),
        # never silently discarded.
        path_a = self._a_real_fixture_copy()
        video_a = self._ingest(path_a)
        VideoFile.objects.filter(pk=video_a.pk).update(isProcessed=True)
        video_a.refresh_from_db()
        path_b = path_a + '.dup.mp4'
        shutil.copy(path_a, path_b)
        self.addCleanup(lambda: os.path.exists(path_b) and os.remove(path_b))
        video_b = self._make_duplicate_row(path_b, video_a, isProcessed=True)
        face_a = self._attach_face(video_a)
        face_b = self._attach_face(video_b)

        from django.core.management import call_command
        call_command('merge_duplicate_videofiles', '--yes')

        self.assertTrue(VideoFile.objects.filter(pk=video_a.pk).exists())
        self.assertFalse(VideoFile.objects.filter(pk=video_b.pk).exists())
        self.assertTrue(Face.objects.filter(pk=face_a.pk, source_video_file=video_a).exists())
        face_b.refresh_from_db()
        self.assertEqual(face_b.source_video_file_id, video_a.pk)

    def test_merge_dedupes_matched_faces_keeping_the_labeled_survivor(self):
        # Same real span on both copies (the deterministic-pipeline
        # case this feature was actually built for) -- one copy's face
        # is a real, declared identification; the other is just
        # .ignore. The declared one must survive, not be discarded
        # because the un-labeled copy happened to live on the primary.
        from face_manager.models import Person
        path_a = self._a_real_fixture_copy()
        video_a = self._ingest(path_a)
        VideoFile.objects.filter(pk=video_a.pk).update(isProcessed=True)
        video_a.refresh_from_db()
        path_b = path_a + '.dup.mp4'
        shutil.copy(path_a, path_b)
        self.addCleanup(lambda: os.path.exists(path_b) and os.remove(path_b))
        video_b = self._make_duplicate_row(path_b, video_a, isProcessed=True)

        real_person, _ = Person.objects.get_or_create(person_name="Test Real Person")
        ignore_face = self._attach_face_at(video_a, 10.0, 12.0)
        labeled_face = self._attach_face_at(video_b, 10.0, 12.0, person=real_person)

        from django.core.management import call_command
        call_command('merge_duplicate_videofiles', '--yes')

        self.assertFalse(Face.objects.filter(pk=ignore_face.pk).exists())
        labeled_face.refresh_from_db()
        self.assertEqual(labeled_face.source_video_file_id, video_a.pk)
        self.assertEqual(labeled_face.declared_name.person_name, "Test Real Person")
        self.assertEqual(Face.objects.filter(source_video_file=video_a).count(), 1)

    def test_merge_keeps_faces_whose_timestamps_dont_match_as_distinct(self):
        # Two faces at clearly different points in the clip -- these are
        # two different real detections (or the same person appearing
        # twice), not a duplicate pair, and must both survive.
        path_a = self._a_real_fixture_copy()
        video_a = self._ingest(path_a)
        VideoFile.objects.filter(pk=video_a.pk).update(isProcessed=True)
        video_a.refresh_from_db()
        path_b = path_a + '.dup.mp4'
        shutil.copy(path_a, path_b)
        self.addCleanup(lambda: os.path.exists(path_b) and os.remove(path_b))
        video_b = self._make_duplicate_row(path_b, video_a, isProcessed=True)

        face_early = self._attach_face_at(video_a, 1.0, 2.0)
        face_late = self._attach_face_at(video_b, 40.0, 41.0)

        from django.core.management import call_command
        call_command('merge_duplicate_videofiles', '--yes')

        self.assertTrue(Face.objects.filter(pk=face_early.pk, source_video_file=video_a).exists())
        face_late.refresh_from_db()
        self.assertEqual(face_late.source_video_file_id, video_a.pk)
        self.assertEqual(Face.objects.filter(source_video_file=video_a).count(), 2)

    def test_merge_dry_run_makes_no_changes(self):
        path_a = self._a_real_fixture_copy()
        primary = self._ingest(path_a)
        VideoFile.objects.filter(pk=primary.pk).update(isProcessed=True)
        primary.refresh_from_db()
        path_b = path_a + '.dup.mp4'
        shutil.copy(path_a, path_b)
        self.addCleanup(lambda: os.path.exists(path_b) and os.remove(path_b))
        duplicate = self._make_duplicate_row(path_b, primary, isProcessed=False)

        from django.core.management import call_command
        call_command('merge_duplicate_videofiles', '--dry-run')

        self.assertTrue(VideoFile.objects.filter(pk=primary.pk).exists())
        self.assertTrue(VideoFile.objects.filter(pk=duplicate.pk).exists())

    def test_merge_rerun_finds_nothing(self):
        path_a = self._a_real_fixture_copy()
        primary = self._ingest(path_a)
        VideoFile.objects.filter(pk=primary.pk).update(isProcessed=True)
        primary.refresh_from_db()
        path_b = path_a + '.dup.mp4'
        shutil.copy(path_a, path_b)
        self.addCleanup(lambda: os.path.exists(path_b) and os.remove(path_b))
        self._make_duplicate_row(path_b, primary, isProcessed=False)

        from django.core.management import call_command
        call_command('merge_duplicate_videofiles', '--yes')
        self.assertEqual(VideoFile.objects.count(), 1)

        call_command('merge_duplicate_videofiles', '--yes')
        self.assertEqual(VideoFile.objects.count(), 1)


class VideoPlaybackPathTests(TestCase):
    """compute_video_playback_path() (the single source of truth for
    VideoFile.playback_path, shared between create_video_file() and
    backfill_video_playback_path) and the backfill command itself --
    see Phase 4's "slideshow serving" write-up in CLAUDE.md."""

    def tearDown(self):
        VideoFile.objects.all().delete()

    @override_settings(PHOTO_ROOT='/photos', VIDEO_ROOT='/videos')
    def test_photo_root_path_uses_full_res_mount(self):
        from filepopulator.video_scripts import compute_video_playback_path
        url = compute_video_playback_path('/photos/some dir/clip.mp4')
        self.assertIn('/full_res/some%20dir/clip.mp4', url)
        self.assertNotIn('video_original', url)

    @override_settings(PHOTO_ROOT='/photos', VIDEO_ROOT='/videos')
    def test_video_root_path_uses_video_original_mount(self):
        from filepopulator.video_scripts import compute_video_playback_path
        url = compute_video_playback_path('/videos/Our_Home_Videos/clip.mp4')
        self.assertIn('/video_original/Our_Home_Videos/clip.mp4', url)

    def _bare_video(self, filename):
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(filename))
        return VideoFile.objects.create(
            filename=filename, directory=directory, width=100, height=100,
            playback_path=None,
        )

    def test_backfill_populates_missing_playback_path(self):
        video = self._bare_video('/videos/Our_Home_Videos/clip1.mp4')
        self.assertIsNone(video.playback_path)

        call_command('backfill_video_playback_path')

        video.refresh_from_db()
        self.assertIsNotNone(video.playback_path)
        self.assertIn('video_original', video.playback_path)

    def test_backfill_dry_run_does_not_write(self):
        video = self._bare_video('/videos/Our_Home_Videos/clip2.mp4')
        call_command('backfill_video_playback_path', '--dry-run')
        video.refresh_from_db()
        self.assertIsNone(video.playback_path)

    def test_backfill_skips_already_populated_rows(self):
        video = self._bare_video('/videos/Our_Home_Videos/clip3.mp4')
        VideoFile.objects.filter(pk=video.pk).update(playback_path='https://example.com/already/set.mp4')
        call_command('backfill_video_playback_path')
        video.refresh_from_db()
        self.assertEqual(video.playback_path, 'https://example.com/already/set.mp4')


class MatchDuplicateVideoFacesTests(unittest.TestCase):
    """Unit tests for merge_duplicate_videofiles.match_duplicate_video_faces
    and _survivor_key -- pure functions, no DB access needed. See that
    command's module docstring for the real motivating case (5 Face rows
    from 5 byte-identical video copies, matched by near-exact timestamp
    since the extraction pipeline is deterministic over identical input
    bytes)."""

    def _face(self, first, last, embedding=None):
        return {'first': first, 'last': last, 'embedding': embedding}

    def test_close_timestamps_match(self):
        from filepopulator.management.commands.merge_duplicate_videofiles import (
            match_duplicate_video_faces,
        )
        primary = [self._face(10.0, 12.0)]
        other = [self._face(10.001, 12.002)]
        matches = match_duplicate_video_faces(primary, other)
        self.assertEqual(matches, [(0, 0)])

    def test_timestamps_too_far_apart_do_not_match(self):
        from filepopulator.management.commands.merge_duplicate_videofiles import (
            match_duplicate_video_faces,
        )
        primary = [self._face(10.0, 12.0)]
        other = [self._face(40.0, 41.0)]
        matches = match_duplicate_video_faces(primary, other)
        self.assertEqual(matches, [])

    def test_missing_timestamps_never_match(self):
        # A face with no timestamps at all (e.g. a hand-added test/stub
        # row) can't be confidently matched -- must stay unmatched
        # rather than guessed at.
        from filepopulator.management.commands.merge_duplicate_videofiles import (
            match_duplicate_video_faces,
        )
        primary = [self._face(None, None)]
        other = [self._face(None, None)]
        matches = match_duplicate_video_faces(primary, other)
        self.assertEqual(matches, [])

    def test_close_timestamps_but_dissimilar_embeddings_do_not_match(self):
        # Same instant, but the embeddings disagree -- guards against
        # two different real people who happen to overlap in time
        # within tolerance. Near-orthogonal 3-d vectors for clarity.
        from filepopulator.management.commands.merge_duplicate_videofiles import (
            match_duplicate_video_faces,
        )
        primary = [self._face(10.0, 12.0, embedding=[1.0, 0.0, 0.0])]
        other = [self._face(10.0, 12.0, embedding=[0.0, 1.0, 0.0])]
        matches = match_duplicate_video_faces(primary, other)
        self.assertEqual(matches, [])

    def test_close_timestamps_and_similar_embeddings_do_match(self):
        from filepopulator.management.commands.merge_duplicate_videofiles import (
            match_duplicate_video_faces,
        )
        primary = [self._face(10.0, 12.0, embedding=[1.0, 0.0, 0.0])]
        other = [self._face(10.0, 12.0, embedding=[0.99, 0.01, 0.0])]
        matches = match_duplicate_video_faces(primary, other)
        self.assertEqual(matches, [(0, 0)])

    def test_optimal_assignment_picks_the_correct_pairing_not_first_match(self):
        # Two primary faces, two other faces, each clearly closer in
        # time to a DIFFERENT one of the two -- confirms the matcher
        # solves the whole matrix at once rather than greedily grabbing
        # whichever pair it sees first.
        from filepopulator.management.commands.merge_duplicate_videofiles import (
            match_duplicate_video_faces,
        )
        primary = [self._face(10.0, 12.0), self._face(50.0, 52.0)]
        other = [self._face(50.05, 52.05), self._face(10.05, 12.05)]
        matches = match_duplicate_video_faces(primary, other)
        self.assertEqual(set(matches), {(0, 1), (1, 0)})

    def test_survivor_key_prefers_validated(self):
        from filepopulator.management.commands.merge_duplicate_videofiles import _survivor_key
        validated = {'id': 2, 'validated': True, 'person_name': '.ignore', 'has_kps': False, 'has_det_score': False}
        unvalidated = {'id': 1, 'validated': False, 'person_name': 'Real Name', 'has_kps': True, 'has_det_score': True}
        self.assertLess(_survivor_key(validated), _survivor_key(unvalidated))

    def test_survivor_key_prefers_real_label_over_ignore(self):
        # The real motivating case: a genuine person name must beat
        # .ignore even though .ignore isn't the blank sentinel either.
        from filepopulator.management.commands.merge_duplicate_videofiles import _survivor_key
        labeled = {'id': 2, 'validated': False, 'person_name': 'Emily Wilson', 'has_kps': False, 'has_det_score': False}
        ignored = {'id': 1, 'validated': False, 'person_name': '.ignore', 'has_kps': True, 'has_det_score': True}
        self.assertLess(_survivor_key(labeled), _survivor_key(ignored))

    def test_survivor_key_prefers_kps_then_det_score_then_lowest_id(self):
        from filepopulator.management.commands.merge_duplicate_videofiles import _survivor_key
        has_kps = {'id': 5, 'validated': False, 'person_name': '.ignore', 'has_kps': True, 'has_det_score': False}
        no_kps = {'id': 1, 'validated': False, 'person_name': '.ignore', 'has_kps': False, 'has_det_score': True}
        self.assertLess(_survivor_key(has_kps), _survivor_key(no_kps))

        has_det_score = {'id': 5, 'validated': False, 'person_name': '.ignore', 'has_kps': False, 'has_det_score': True}
        no_det_score = {'id': 1, 'validated': False, 'person_name': '.ignore', 'has_kps': False, 'has_det_score': False}
        self.assertLess(_survivor_key(has_det_score), _survivor_key(no_det_score))

        lower_id = {'id': 1, 'validated': False, 'person_name': '.ignore', 'has_kps': False, 'has_det_score': False}
        higher_id = {'id': 2, 'validated': False, 'person_name': '.ignore', 'has_kps': False, 'has_det_score': False}
        self.assertLess(_survivor_key(lower_id), _survivor_key(higher_id))



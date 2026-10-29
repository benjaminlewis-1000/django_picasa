"""filepopulator tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test filepopulator.tests.test_image_formats
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


@override_settings(MEDIA_ROOT='/tmp/filepopulator_test_media')
class CorruptedImageIngestionTests(TestCase):
    """Uses the 5 real corrupted JPEGs pulled from production logs (see
    /mnt/fast_storage/appdata/django_picasa/test_suite/corrupted_images/NOTES.md),
    mounted read-only under /photos/corrupted."""

    CORRUPTED_DIR = '/photos/corrupted'

    def tearDown(self):
        for obj in ImageFile.objects.all():
            obj.delete()
        FailedImageFile.objects.all().delete()

    def test_create_image_file_tracks_never_ingested_corrupted_file(self):
        # Regression test for a fixed bug: ImageFile._generate_md5_hash()
        # used to only catch TypeError/PIL.Image.DecompressionBombError,
        # not the OSError ("image file is truncated" / "broken data
        # stream") a corrupted JPEG actually raises -- so
        # create_image_file() crashed outright instead of degrading
        # gracefully. Now the OSError is caught and, since this file was
        # never successfully ingested (no ImageFile row exists to record
        # the failure on), it's tracked in FailedImageFile instead.
        # Pick a fixture that's actually unrecoverable rather than a
        # hardcoded real-production filename -- the real fixture set (5
        # files, see NOTES.md) mounted locally has a mix of recoverable and
        # unrecoverable files (see the next test's docstring), while CI's
        # synthetic ci_fixtures/corrupted/ (2 files) are both deliberately
        # deep-truncated and always unrecoverable. Try each present file
        # until one actually lands in FailedImageFile.
        path = None
        for filename in os.listdir(self.CORRUPTED_DIR):
            candidate = os.path.join(self.CORRUPTED_DIR, filename)
            create_image_file(candidate)
            if FailedImageFile.objects.filter(filename=candidate).exists():
                path = candidate
                break
        self.assertIsNotNone(path, "no unrecoverable fixture found in CORRUPTED_DIR")

        self.assertFalse(ImageFile.objects.filter(filename=path).exists())
        failed = FailedImageFile.objects.get(filename=path)
        self.assertTrue(failed.error_message)

    def test_add_from_root_dir_tracks_and_stops_retrying_corrupted_files(self):
        # add_from_root_dir()'s per-file try/except already meant one
        # corrupted file didn't take down the whole batch. The bug was
        # that these files also never got any record at all, so they were
        # re-attempted (and re-failed) on every single scheduled
        # populate_files_from_root run, forever. Now confirms both halves
        # of the fix: every file ends up accounted for (either a real
        # ImageFile or a FailedImageFile -- some of these "corrupted"
        # fixtures are truncated shallowly enough that PIL/cv2's fallback
        # decoding actually recovers them, which is fine, not every
        # corrupted file is unrecoverable), and a second run doesn't
        # re-attempt whichever ones did fail (last_attempted_at is
        # untouched -- if it *had* retried, FailedImageFile.
        # objects.update_or_create() would have bumped it via auto_now).
        add_from_root_dir(self.CORRUPTED_DIR)

        corrupted_files = os.listdir(self.CORRUPTED_DIR)
        # >=5 locally (the real fixture set); CI's synthetic
        # ci_fixtures/corrupted/ only has 2 -- just confirm there's at least
        # something to test against, not a specific count.
        self.assertGreaterEqual(len(corrupted_files), 1)
        self.assertEqual(
            ImageFile.objects.count() + FailedImageFile.objects.count(),
            len(corrupted_files),
        )
        self.assertGreater(FailedImageFile.objects.count(), 0)
        first_pass_timestamps = {
            f.filename: f.last_attempted_at for f in FailedImageFile.objects.all()
        }

        add_from_root_dir(self.CORRUPTED_DIR)

        second_pass_timestamps = {
            f.filename: f.last_attempted_at for f in FailedImageFile.objects.all()
        }
        self.assertEqual(first_pass_timestamps, second_pass_timestamps)

    def test_previously_good_photo_that_becomes_corrupted_is_flagged_not_crashed(self):
        # The other half of the bug: a photo that was ingested fine and
        # later becomes unreadable on disk (bit rot, a bad in-place edit)
        # hits the same OSError, but through the "photo already exists"
        # branch of create_image_file() -- there's already a real
        # ImageFile row for it. That row is flagged via
        # image_load_failed/image_load_error (added for the face_manager
        # retry-forever fix) rather than losing its prior good data, and
        # dateModified is bumped so it isn't re-attempted every run either.
        tmp_dir = '/tmp/filepop_corrupt_existing_test'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        try:
            src = os.path.join(settings.FILEPOPULATOR_VAL_DIRECTORY, 'naming', 'good', '1.JPG')
            file_path = os.path.join(tmp_dir, '1.JPG')
            shutil.copy(src, file_path)
            create_image_file(file_path)
            before = ImageFile.objects.get(filename=file_path)
            self.assertFalse(before.image_load_failed)

            # Corrupt it in place (truncate) and push the mtime forward so
            # create_image_file() treats it as changed, same technique
            # ci_fixtures/generate_fixtures.py uses to build corrupted
            # fixtures.
            with open(file_path, 'rb') as f:
                data = f.read()
            with open(file_path, 'wb') as f:
                f.write(data[: len(data) - 200])
            future = time.time() + 3600
            os.utime(file_path, (future, future))

            create_image_file(file_path)

            after = ImageFile.objects.get(filename=file_path)
            self.assertEqual(before.id, after.id)
            self.assertTrue(after.image_load_failed)
            self.assertTrue(after.image_load_error)
            # Prior good data (pixel_hash, thumbnails, etc.) is untouched.
            self.assertEqual(before.pixel_hash, after.pixel_hash)
            self.assertFalse(FailedImageFile.objects.filter(filename=file_path).exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@override_settings(MEDIA_ROOT='/tmp/filepopulator_test_media')
class HeicIngestionTests(TestCase):
    """HEIC (Apple's default photo format since iOS 11) support.

    HEIC_DIR mirrors real fixture data locally: 8 real iPhone photos
    (models 12 through 17 Pro) at
    /mnt/fast_storage/appdata/django_picasa/test_suite/heic_images/,
    mounted read-only under /photos/heic_stub. CI only has the single
    synthetic no-EXIF stub from ci_fixtures/heic_stub/ (see
    generate_fixtures.py) -- tests here work with "whatever's present"
    rather than assuming a specific count or specific metadata, so they
    hold in both environments.

    Verified empirically (see CLAUDE.md) against all 8 real samples:
    pillow_heif/libheif always decodes to plain RGB, and usually reports
    EXIF orientation 1 regardless of the photo's actual portrait/
    landscape framing -- meaning it usually auto-applies any container-
    level rotation transform during decode and resets the tag to match.
    A real production file (IMG_9370.HEIC, orientation 8) showed that
    doesn't always hold, so a non-1 orientation is now rotated via the
    same apply_exif_orientation() JPEG uses rather than rejected. The
    multi-frame (Live Photo/burst) guard below still fails loudly
    (recorded via FailedImageFile/image_load_failed, not silently) since
    that case is still genuinely unsupported.
    """

    HEIC_DIR = '/photos/heic_stub'

    def tearDown(self):
        for obj in ImageFile.objects.all():
            obj.delete()
        FailedImageFile.objects.all().delete()

    def _heic_files(self):
        return [
            f for f in sorted(os.listdir(self.HEIC_DIR))
            if f.lower().endswith(('.heic', '.heif'))
        ]

    def test_ingests_every_heic_fixture_successfully(self):
        heic_files = self._heic_files()
        self.assertGreaterEqual(len(heic_files), 1)
        for filename in heic_files:
            path = os.path.join(self.HEIC_DIR, filename)
            create_image_file(path)
            img = ImageFile.objects.filter(filename=path).first()
            self.assertIsNotNone(img, f"{filename} was not ingested")
            self.assertEqual(img.orientation, 1)
            self.assertGreater(img.width, 0)
            self.assertGreater(img.height, 0)
            self.assertTrue(os.path.isfile(img.thumbnail_small.path))

    def test_add_from_root_dir_discovers_heic_files(self):
        # Regression test for a fixed bug: add_from_root_dir() used to
        # filter to `.jpg`/`.jpeg` before even building its file list, so
        # .heic files were invisible to the whole ingestion pipeline from
        # the start, not just rejected.
        add_from_root_dir(self.HEIC_DIR)
        self.assertEqual(ImageFile.objects.count(), len(self._heic_files()))

    def test_gps_decimal_conversion_is_sane_when_present(self):
        # Not every fixture (or CI's synthetic stub) has GPS data -- only
        # assert against ones that actually do.
        found_any = False
        for filename in self._heic_files():
            path = os.path.join(self.HEIC_DIR, filename)
            create_image_file(path)
            img = ImageFile.objects.get(filename=path)
            if img.gps_lat_decimal == -999 and img.gps_lon_decimal == -999:
                # -999 is ImageFile's own "no GPS data" sentinel default.
                continue
            found_any = True
            self.assertGreaterEqual(img.gps_lat_decimal, -90)
            self.assertLessEqual(img.gps_lat_decimal, 90)
            self.assertGreaterEqual(img.gps_lon_decimal, -180)
            self.assertLessEqual(img.gps_lon_decimal, 180)
        if not found_any:
            self.skipTest("no fixture in HEIC_DIR has GPS data (expected for CI's synthetic stub)")

    def test_non_one_orientation_is_rotated_and_ingests_successfully(self):
        # Regression test for a fixed bug: a non-1 HEIC orientation used
        # to raise OSError and be recorded as a FailedImageFile instead
        # of being ingested, on the (usually true, but not always, see
        # IMG_9370.HEIC in CLAUDE.md) assumption that libheif always
        # normalizes this itself. Now it's rotated via
        # apply_exif_orientation(), same as JPEG.
        path = os.path.join(self.HEIC_DIR, self._heic_files()[0])
        raw_width, raw_height = Image.open(path).size

        class FakeExif(dict):
            def get_ifd(self, tag):
                return {}

        fake_exif = FakeExif({274: 6})  # Orientation tag id = 274; 6 = 90deg CW
        with mock.patch("PIL.Image.Image.getexif", return_value=fake_exif):
            create_image_file(path)

        self.assertFalse(FailedImageFile.objects.filter(filename=path).exists())
        img = ImageFile.objects.get(filename=path)
        self.assertEqual(img.orientation, 6)
        # Orientation 6 is a 90-degree rotation, so width/height swap
        # relative to the raw (unrotated) decode.
        self.assertEqual(img.width, raw_height)
        self.assertEqual(img.height, raw_width)
        self.assertTrue(os.path.isfile(img.thumbnail_small.path))

    def test_all_orientation_codes_round_trip_correctly_on_real_photos(self):
        # Broader validation for the same fix as
        # test_non_one_orientation_is_rotated_and_ingests_successfully,
        # using real photo pixel data instead of a single synthetic
        # rotation and a single orientation code: for every real HEIC
        # fixture (all naturally orientation 1 -- already upright) and
        # every non-identity EXIF orientation code, simulate what that
        # code's "as captured" raw pixels would look like (the exact
        # mathematical inverse of apply_exif_orientation()'s own
        # transform for that code), then confirm apply_exif_orientation()
        # restores the original upright pixels exactly. This is the same
        # helper JPEG has always used -- this test exists to give HEIC's
        # newly-relaxed guard (see CLAUDE.md, IMG_9370.HEIC) the same
        # confidence for every orientation value, not just the one
        # production actually hit.
        def simulate_raw(image, orientation):
            if orientation == 2:
                return image.transpose(Image.FLIP_LEFT_RIGHT)
            if orientation == 3:
                return image.transpose(Image.ROTATE_180)
            if orientation == 4:
                return image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 5:
                return image.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)
            if orientation == 6:
                return image.transpose(Image.ROTATE_90)
            if orientation == 7:
                return image.transpose(Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 8:
                return image.transpose(Image.ROTATE_270)

        for filename in self._heic_files():
            path = os.path.join(self.HEIC_DIR, filename)
            original = Image.open(path).convert('RGB')
            orig_arr = np.array(original)
            for code in range(2, 9):
                raw = simulate_raw(original, code)
                restored = apply_exif_orientation(raw, code)
                restored_arr = np.array(restored)
                self.assertEqual(
                    restored_arr.shape, orig_arr.shape,
                    f"{filename} orientation={code}: shape mismatch after round-trip",
                )
                self.assertTrue(
                    np.array_equal(restored_arr, orig_arr),
                    f"{filename} orientation={code}: pixels did not round-trip exactly",
                )

    def test_multi_frame_guard_fails_loudly(self):
        path = os.path.join(self.HEIC_DIR, self._heic_files()[0])

        with mock.patch.object(
            pillow_heif.HeifImageFile, "n_frames",
            new_callable=mock.PropertyMock, return_value=3,
        ):
            create_image_file(path)

        self.assertFalse(ImageFile.objects.filter(filename=path).exists())
        failed = FailedImageFile.objects.get(filename=path)
        self.assertIn("frames", failed.error_message.lower())

    def test_rgba_heic_thumbnail_does_not_crash(self):
        # Regression test for a fixed bug: some real-world HEIC files
        # decode to RGBA mode (an alpha channel) even for a plain photo --
        # found via a genuine production file after HEIC support shipped,
        # unlike any of the all-RGB samples this was originally tested
        # against. _generate_thumbnail() tried to save that straight to
        # JPEG, which can't encode alpha, raising
        # "cannot write mode RGBA as JPEG".
        os.makedirs("/tmp/heic_rgba_test", exist_ok=True)
        path = "/tmp/heic_rgba_test/rgba.heic"
        arr = np.zeros((150, 200, 4), dtype=np.uint8)
        arr[:, :, 0] = 90
        arr[:, :, 1] = 160
        arr[:, :, 2] = 210
        arr[:, :, 3] = 255  # fully opaque, same as a real photo would be
        img = Image.fromarray(arr, mode="RGBA")
        heif_file = pillow_heif.from_pillow(img)
        heif_file.save(path, quality=80)
        self.addCleanup(shutil.rmtree, "/tmp/heic_rgba_test", ignore_errors=True)

        # Confirm the fixture is actually RGBA before testing our handling.
        self.assertEqual(Image.open(path).mode, "RGBA")

        create_image_file(path)

        self.assertFalse(FailedImageFile.objects.filter(filename=path).exists())
        obj = ImageFile.objects.get(filename=path)
        self.assertTrue(os.path.isfile(obj.thumbnail_small.path))

    def test_save_does_not_redecode_when_image_already_populated(self):
        # Real perf bug found 2026-09-18 while profiling why this test
        # class was so slow: ImageFile.save() used to call _init_image()
        # unconditionally, even when process_new_no_md5() (the normal
        # create_image_file() path) had already called it once on the
        # same instance -- silently doubling every HEIC decode (~0.89s
        # each, confirmed via pillow_heif.heif.BaseImage.load() tracing
        # against real fixtures; negligible for cheap formats like JPEG,
        # which is why it went unnoticed). self.image is only ever
        # assigned inside _init_image() itself, so guarding on
        # self.image already being set can never skip a genuinely-needed
        # decode -- verified separately (see CLAUDE.md) against all 8
        # real HEIC fixtures + 5 real JPEGs with byte-identical results.
        # This test targets the specific mechanism (a second
        # _init_image() call is skipped), not full-pipeline correctness.
        # Use whichever real fixture actually ingests successfully --
        # same "work with whatever's present" convention as the rest of
        # this class (a local dev fixture may include a multi-frame Live
        # Photo that's expected to fail loudly instead; CI's single
        # synthetic stub is expected to succeed).
        for filename in self._heic_files():
            path = os.path.join(self.HEIC_DIR, filename)
            call_count = {'n': 0}
            real_init_image = ImageFile._init_image

            def counting_init_image(self):
                call_count['n'] += 1
                return real_init_image(self)

            with mock.patch.object(ImageFile, '_init_image', counting_init_image):
                create_image_file(path)

            if ImageFile.objects.filter(filename=path).exists():
                self.assertEqual(
                    call_count['n'], 1,
                    "_init_image() should run exactly once per real ingestion, not twice",
                )
                return
        self.skipTest("no fixture in HEIC_DIR ingested successfully")



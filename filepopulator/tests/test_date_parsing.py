"""filepopulator tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test filepopulator.tests.test_date_parsing
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


class GuessDateFromFilenameTests(unittest.TestCase):
    """Pure-function tests for guess_date_from_filename() -- no DB needed.
    Patterns and the "no match" cases are all drawn from a real 2026-09
    survey of this library's own images with no valid EXIF date."""

    def test_android_style_compact(self):
        got = guess_date_from_filename('/photos/x/IMG_20240719_211850.jpg')
        self.assertEqual(got.isoformat(), '2024-07-19T21:18:50+00:00')

    def test_iso_space_dot_style(self):
        got = guess_date_from_filename('/photos/x/2019-04-08 21.08.22.jpg')
        self.assertEqual(got.isoformat(), '2019-04-08T21:08:22+00:00')

    def test_iso_dash_time_style(self):
        got = guess_date_from_filename('/photos/x/2016-02-06_16-18-35_000.jpeg')
        self.assertEqual(got.isoformat(), '2016-02-06T16:18:35+00:00')

    def test_epoch_ms_style(self):
        got = guess_date_from_filename('/photos/x/1697447941437.jpg')
        self.assertEqual(got.isoformat(), '2023-10-16T09:19:01.437000+00:00')

    def test_mixed_format_double_date_picks_earlier_capture_not_later_export(self):
        """Regression case for the real bug found 2026-09-01: a filename
        embedding two timestamps in DIFFERENT formats (hyphenated capture
        date, compact export date) used to silently return whichever
        pattern's priority happened to match first -- which could be the
        LATER (export) date instead of the real, earlier capture date."""
        name = 'Resized_2022-05-29_09-45-29_923_20220531_164559.jpg'
        got = guess_date_from_filename(name)
        self.assertEqual(got.isoformat(), '2022-05-29T09:45:29+00:00')

    def test_same_format_double_date_picks_earlier_one(self):
        name = 'Resized_20221217_162217_20221220_013629.jpg'
        got = guess_date_from_filename(name)
        self.assertEqual(got.isoformat(), '2022-12-17T16:22:17+00:00')

    def test_no_date_in_whatsapp_received_filename_returns_none(self):
        self.assertIsNone(guess_date_from_filename('/photos/x/received_3804555136348326.jpeg'))

    def test_no_date_in_uuid_filename_returns_none(self):
        self.assertIsNone(guess_date_from_filename(
            '/photos/x/744A2F1E-1265-4FA2-93CE-B9EBA91E2188.jpeg'))

    def test_no_date_in_plain_sequential_filename_returns_none(self):
        self.assertIsNone(guess_date_from_filename('/photos/x/IMG_6403.JPG'))

    def test_no_date_in_scan_album_filename_returns_none(self):
        self.assertIsNone(guess_date_from_filename('/photos/x/album_1 0161.jpg'))

    def test_invalid_calendar_date_is_rejected_not_crashed(self):
        # Digits that happen to line up like the pattern but aren't a
        # real date (month 99) -- must not raise, just skip this match.
        self.assertIsNone(guess_date_from_filename('/photos/x/IMG_20249919_211850.jpg'))

    def test_date_before_1990_is_rejected(self):
        self.assertIsNone(guess_date_from_filename('/photos/x/IMG_19850101_120000.jpg'))

    def test_date_in_the_future_is_rejected(self):
        self.assertIsNone(guess_date_from_filename('/photos/x/IMG_29990101_120000.jpg'))

    def test_epoch_ms_style_2014_to_2017_range_is_recognized(self):
        # Real bug, found 2026-10-28: the pattern was originally bounded
        # to a "1[5-8]" leading digit (~2017-2035), silently missing the
        # entire 2001-2017 range of real 13-digit ms-epoch timestamps --
        # confirmed against 1,815 real "Mom Phone" photos with a leading
        # "14" (2014-2017) that used to return None here despite having a
        # perfectly real, recoverable date embedded in the filename.
        got = guess_date_from_filename('/photos/x/1482109271865.jpg')
        self.assertEqual(got.isoformat(), '2016-12-19T01:01:11.865000+00:00')


class GetDateTakenFilenameFallbackTests(unittest.TestCase):
    """Integration-style coverage for ImageFile._get_date_taken()'s use of
    guess_date_from_filename() as a fallback -- constructs a bare,
    unsaved ImageFile with exifDict set directly, so no real image file
    or DB row is needed to exercise the branching logic itself."""

    def test_no_exif_falls_back_to_filename_guess(self):
        obj = ImageFile(filename='/photos/x/IMG_20200101_120000.jpg')
        obj.exifDict = None
        obj._get_date_taken()
        self.assertFalse(obj.dateTakenValid)
        self.assertEqual(obj.dateTaken.isoformat(), '2020-01-01T12:00:00+00:00')

    def test_exif_dict_present_but_no_valid_date_key_falls_back_to_filename(self):
        obj = ImageFile(filename='/photos/x/IMG_20200101_120000.jpg')
        obj.exifDict = {'Make': 'Canon'}  # no DateTime*/DateTimeOriginal/DateTimeDigitized
        obj._get_date_taken()
        self.assertFalse(obj.dateTakenValid)
        self.assertEqual(obj.dateTaken.isoformat(), '2020-01-01T12:00:00+00:00')

    def test_valid_exif_date_wins_even_if_filename_also_has_a_date(self):
        obj = ImageFile(filename='/photos/x/IMG_20200101_120000.jpg')
        obj.exifDict = {'DateTimeOriginal': '2018:06:15 08:00:00'}
        obj._get_date_taken()
        self.assertTrue(obj.dateTakenValid)
        self.assertEqual(obj.dateTaken.isoformat(), '2018-06-15T08:00:00+00:00')

    def test_no_exif_and_unparseable_filename_falls_back_to_now_unchanged(self):
        obj = ImageFile(filename='/photos/x/received_3804555136348326.jpeg')
        obj.exifDict = None
        before = timezone.now()
        obj._get_date_taken()
        after = timezone.now()
        self.assertFalse(obj.dateTakenValid)
        self.assertTrue(before <= obj.dateTaken <= after)

    def test_device_clock_reset_exif_date_is_rejected_falls_back_to_filename(self):
        # Real bug, found 2026-10-28: a genuine EXIF DateTimeOriginal
        # field containing "1900:01:01 01:01:01" -- a well-known device-
        # clock-reset default (confirmed against 2,376 real images, 1,815
        # from one phone) -- used to be accepted as a real, valid date
        # just because it parsed successfully, even though it's obviously
        # not a real digital capture date. Must be rejected the same as
        # the other garbage-string EXIF cases, falling through to the
        # filename guess instead.
        obj = ImageFile(filename='/photos/x/IMG_20200101_120000.jpg')
        obj.exifDict = {'DateTimeOriginal': '1900:01:01 01:01:01'}
        obj._get_date_taken()
        self.assertFalse(obj.dateTakenValid)
        self.assertEqual(obj.dateTaken.isoformat(), '2020-01-01T12:00:00+00:00')

    def test_filename_guess_also_sets_dateTakenUTC_not_just_dateTaken(self):
        # Real bug, found 2026-10-28: the filename-fallback branch only
        # ever set self.dateTaken (a DateTimeField) -- self.dateTakenUTC
        # (the plain float field Face.dateTakenUTC and other downstream
        # numeric-date logic actually reads) was silently left at its
        # stale/default value even when a real date was recovered.
        obj = ImageFile(filename='/photos/x/IMG_20200101_120000.jpg')
        obj.exifDict = None
        obj._get_date_taken()
        self.assertFalse(obj.dateTakenValid)
        self.assertEqual(obj.dateTaken.isoformat(), '2020-01-01T12:00:00+00:00')
        self.assertEqual(obj.dateTakenUTC, obj.dateTaken.timestamp())


@override_settings(MEDIA_ROOT="/tmp/filepopulator_test_media")
class BackfillDatesFromFilenameTests(TestCase):
    """backfill_dates_from_filename replaces the now()-placeholder
    dateTaken with a filename-derived guess for existing rows that have
    no valid EXIF date -- mirrors PhashBackfillTests' pattern of ingesting
    a real fixture image, then forcing it into the "needs backfill" state
    via .update() (bypassing ImageFile.save(), which would otherwise redo
    its own EXIF-based date detection and undo the setup)."""

    def _ingest_real_image_at(self, dest_filename, fixture_name='1.JPG'):
        val_dir = settings.FILEPOPULATOR_VAL_DIRECTORY
        src = os.path.join(val_dir, 'naming', 'good', fixture_name)
        os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
        shutil.copy(src, dest_filename)
        create_image_file(dest_filename)
        return ImageFile.objects.get(filename=dest_filename)

    def test_backfill_updates_dateTaken_for_row_with_filename_date(self):
        dest = '/tmp/date_backfill_test/IMG_20200101_120000.jpg'
        obj = self._ingest_real_image_at(dest)
        ImageFile.objects.filter(pk=obj.pk).update(dateTakenValid=False)

        call_command('backfill_dates_from_filename')

        obj.refresh_from_db()
        self.assertFalse(obj.dateTakenValid)
        self.assertEqual(obj.dateTaken.isoformat(), '2020-01-01T12:00:00+00:00')

    def test_row_with_no_filename_date_is_left_alone(self):
        dest = '/tmp/date_backfill_test/received_3804555136348326.jpeg'
        obj = self._ingest_real_image_at(dest, fixture_name='2.jpg')
        ImageFile.objects.filter(pk=obj.pk).update(dateTakenValid=False)
        original_date = ImageFile.objects.get(pk=obj.pk).dateTaken

        call_command('backfill_dates_from_filename')

        obj.refresh_from_db()
        self.assertEqual(obj.dateTaken, original_date)

    def test_row_with_valid_exif_is_never_touched(self):
        dest = '/tmp/date_backfill_test/IMG_20200101_120000_realexif.jpg'
        obj = self._ingest_real_image_at(dest)
        # Whatever real EXIF this fixture has (or its own now()-fallback)
        # -- the point is dateTakenValid=True rows must be skipped
        # entirely regardless of what their filename says.
        ImageFile.objects.filter(pk=obj.pk).update(dateTakenValid=True)
        original_date = ImageFile.objects.get(pk=obj.pk).dateTaken

        call_command('backfill_dates_from_filename')

        obj.refresh_from_db()
        self.assertEqual(obj.dateTaken, original_date)

    def test_dry_run_writes_nothing(self):
        dest = '/tmp/date_backfill_test/IMG_20200101_120000_dryrun.jpg'
        obj = self._ingest_real_image_at(dest)
        ImageFile.objects.filter(pk=obj.pk).update(dateTakenValid=False)
        original_date = ImageFile.objects.get(pk=obj.pk).dateTaken

        call_command('backfill_dates_from_filename', '--dry-run')

        obj.refresh_from_db()
        self.assertEqual(obj.dateTaken, original_date)


class BackfillImplausibleExifDatesTests(TestCase):
    """backfill_implausible_exif_dates -- corrects rows already written
    with a wrongly-trusted implausible (pre-1990) EXIF date, the real
    bug fixed 2026-10-28 (see _get_date_taken()'s own history). Mirrors
    BackfillDatesFromFilenameTests' pattern: ingest a real fixture, then
    force it into the "already has the bad value" state via .update()."""

    def _ingest_real_image_at(self, dest_filename, fixture_name='1.JPG'):
        val_dir = settings.FILEPOPULATOR_VAL_DIRECTORY
        src = os.path.join(val_dir, 'naming', 'good', fixture_name)
        os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
        shutil.copy(src, dest_filename)
        create_image_file(dest_filename)
        return ImageFile.objects.get(filename=dest_filename)

    def test_recovers_real_date_from_filename_when_available(self):
        dest = '/tmp/implausible_date_backfill_test/IMG_20200101_120000.jpg'
        obj = self._ingest_real_image_at(dest)
        ImageFile.objects.filter(pk=obj.pk).update(
            dateTakenValid=True, dateTakenUTC=-2208985139.0,
        )

        call_command('backfill_implausible_exif_dates')

        obj.refresh_from_db()
        self.assertFalse(obj.dateTakenValid)
        self.assertEqual(obj.dateTaken.isoformat(), '2020-01-01T12:00:00+00:00')
        self.assertEqual(obj.dateTakenUTC, obj.dateTaken.timestamp())

    def test_resets_to_unknown_placeholder_when_no_filename_date_available(self):
        dest = '/tmp/implausible_date_backfill_test/received_3804555136348326.jpeg'
        obj = self._ingest_real_image_at(dest, fixture_name='2.jpg')
        ImageFile.objects.filter(pk=obj.pk).update(
            dateTakenValid=True, dateTakenUTC=-2208985139.0,
        )

        before = timezone.now()
        call_command('backfill_implausible_exif_dates')
        after = timezone.now()

        obj.refresh_from_db()
        self.assertFalse(obj.dateTakenValid)
        self.assertTrue(before <= obj.dateTaken <= after)
        self.assertEqual(obj.dateTakenUTC, obj.dateTaken.timestamp())

    def test_row_with_a_plausible_date_is_never_touched(self):
        dest = '/tmp/implausible_date_backfill_test/IMG_20200101_120000_plausible.jpg'
        obj = self._ingest_real_image_at(dest)
        ImageFile.objects.filter(pk=obj.pk).update(dateTakenValid=True, dateTakenUTC=1600000000.0)
        original_date = ImageFile.objects.get(pk=obj.pk).dateTaken

        call_command('backfill_implausible_exif_dates')

        obj.refresh_from_db()
        self.assertEqual(obj.dateTaken, original_date)

    def test_row_already_marked_invalid_is_never_touched(self):
        # Out of scope for this command -- backfill_dates_from_filename
        # already covers the dateTakenValid=False population.
        dest = '/tmp/implausible_date_backfill_test/IMG_20200101_120000_alreadyfalse.jpg'
        obj = self._ingest_real_image_at(dest)
        ImageFile.objects.filter(pk=obj.pk).update(dateTakenValid=False, dateTakenUTC=-2208985139.0)
        original_date = ImageFile.objects.get(pk=obj.pk).dateTaken

        call_command('backfill_implausible_exif_dates')

        obj.refresh_from_db()
        self.assertEqual(obj.dateTaken, original_date)

    def test_dry_run_writes_nothing(self):
        dest = '/tmp/implausible_date_backfill_test/IMG_20200101_120000_dryrun.jpg'
        obj = self._ingest_real_image_at(dest)
        ImageFile.objects.filter(pk=obj.pk).update(
            dateTakenValid=True, dateTakenUTC=-2208985139.0,
        )

        call_command('backfill_implausible_exif_dates', '--dry-run')

        obj.refresh_from_db()
        self.assertTrue(obj.dateTakenValid)
        self.assertEqual(obj.dateTakenUTC, -2208985139.0)


class ParseExifVideoDateTests(unittest.TestCase):
    """Unit tests for video_scripts._parse_exif_date() -- no DB/file
    access needed. Covers the real parsing bug found against a real Sony
    camcorder file: dateutil.parser silently defaulted the date portion
    to today() for exiftool's 'YYYY:MM:DD ...' format instead of raising,
    which explicit strptime formats don't do."""

    def test_no_timezone_offset(self):
        date = _parse_exif_date('2025:11:27 01:31:30')
        self.assertEqual((date.year, date.month, date.day), (2025, 11, 27))
        self.assertEqual((date.hour, date.minute, date.second), (1, 31, 30))

    def test_with_timezone_offset(self):
        date = _parse_exif_date('2018:12:14 20:15:14-04:00')
        self.assertEqual((date.year, date.month, date.day), (2018, 12, 14))
        self.assertEqual(date.hour, 20)

    def test_with_timezone_offset_and_zone_name(self):
        # The exact real-world shape that broke dateutil.parser: a
        # numeric offset followed by a zone-name word ("DST").
        date = _parse_exif_date('2018:12:14 20:15:14-04:00 DST')
        self.assertEqual((date.year, date.month, date.day), (2018, 12, 14))
        self.assertEqual(date.hour, 20)

    def test_empty_or_none_returns_none(self):
        self.assertIsNone(_parse_exif_date(None))
        self.assertIsNone(_parse_exif_date(''))

    def test_garbage_string_returns_none_not_todays_date(self):
        # The actual regression: this used to silently return today's
        # date via dateutil.parser instead of None.
        result = _parse_exif_date('not a real date at all')
        self.assertIsNone(result)



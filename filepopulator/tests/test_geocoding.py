"""filepopulator tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test filepopulator.tests.test_geocoding
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


class NormalizeNullIslandGpsTests(TestCase):
    """Regression test for the normalize_null_island_gps management
    command, which cleans up ImageFile rows left with GPS (0, 0) by a
    fixed bug in _init_image(): a NaN GPS decimal conversion used to fall
    back to 0 instead of the -999 "no GPS" sentinel used everywhere else,
    so those rows look like real (if geographically absurd) GPS data
    rather than "no GPS" under a straight equality check."""

    def _make_bare_image_file(self, filename, lat, lon):
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(filename))
        obj = ImageFile(
            filename=filename,
            directory=directory,
            thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=10, height=10,
            gps_lat_decimal=lat, gps_lon_decimal=lon,
        )
        ImageFile.objects.bulk_create([obj])
        return ImageFile.objects.get(filename=filename)

    def test_normalizes_null_island_rows(self):
        null_island = self._make_bare_image_file("/tmp/null_island_test/a.jpg", 0, 0)
        real_gps = self._make_bare_image_file("/tmp/null_island_test/b.jpg", 47.62, -122.33)
        no_gps = self._make_bare_image_file("/tmp/null_island_test/c.jpg", -999, -999)

        call_command("normalize_null_island_gps", "--yes")

        null_island.refresh_from_db()
        real_gps.refresh_from_db()
        no_gps.refresh_from_db()
        self.assertEqual((null_island.gps_lat_decimal, null_island.gps_lon_decimal), (-999, -999))
        self.assertEqual((real_gps.gps_lat_decimal, real_gps.gps_lon_decimal), (47.62, -122.33))
        self.assertEqual((no_gps.gps_lat_decimal, no_gps.gps_lon_decimal), (-999, -999))

    def test_dry_run_writes_nothing(self):
        null_island = self._make_bare_image_file("/tmp/null_island_test/d.jpg", 0, 0)

        call_command("normalize_null_island_gps", "--dry-run")

        null_island.refresh_from_db()
        self.assertEqual((null_island.gps_lat_decimal, null_island.gps_lon_decimal), (0, 0))


class FindNearestMetroTests(unittest.TestCase):
    """find_nearest_metro() is pure/offline (no DB, no network) -- a plain
    unittest.TestCase rather than Django's TestCase is enough."""

    def test_suburb_maps_to_nearby_major_city(self):
        from filepopulator.geocode import find_nearest_metro
        name, distance_km = find_nearest_metro(47.7623, -122.2054)  # Bothell, WA
        self.assertIsNotNone(name)
        self.assertIn(name, ("Seattle", "Bellevue"))
        self.assertLess(distance_km, 25)

    def test_remote_location_has_no_match(self):
        from filepopulator.geocode import find_nearest_metro
        # Rural central Montana (Fergus County / Lewistown area) -- the
        # nearest sizeable city (Billings, Great Falls) is well over
        # SEARCH_RADIUS_BANDS_KM's widest band away.
        name, distance_km = find_nearest_metro(47.0623, -109.4280)
        self.assertIsNone(name)
        self.assertIsNone(distance_km)


class FindNearestNamedPlaceTests(unittest.TestCase):
    """find_nearest_named_place() is the last-resort, population-blind
    fallback used when Nominatim has nothing at all - pure/offline, same
    as FindNearestMetroTests above."""

    def test_returns_nearest_regardless_of_population(self):
        from filepopulator.geocode import find_nearest_named_place
        # Bothell itself may not be "major" enough for find_nearest_metro
        # to prefer it over Seattle/Bellevue (population-ranked within
        # the radius band) - this function doesn't rank by population at
        # all, so the literal nearest place wins even if it's small.
        name, country_code, distance_km = find_nearest_named_place(47.7623, -122.2054)
        self.assertIsNotNone(name)
        self.assertEqual(country_code, 'US')
        self.assertLess(distance_km, 25)

    def test_nothing_within_radius_returns_none(self):
        from filepopulator.geocode import find_nearest_named_place
        # Middle of the Pacific, nowhere near any place in the gazetteer
        # even at this function's generous default radius.
        name, country_code, distance_km = find_nearest_named_place(0.0, -150.0)
        self.assertIsNone(name)
        self.assertIsNone(country_code)
        self.assertIsNone(distance_km)


class NominatimRateLimiterReuseTests(unittest.TestCase):
    """Regression test for a real bug (found 2026-09-10): a fresh
    RateLimiter used to be constructed on every single call - min_delay_
    seconds is enforced via *instance* state, so a fresh instance had
    never made a call before and every coordinate in a backfill run got
    waved through with zero delay, never actually throttling anything."""

    def setUp(self):
        import filepopulator.geocode as geocode_module
        self._geocode_module = geocode_module
        self._orig_reverse = geocode_module._reverse_rate_limiter
        self._orig_forward = geocode_module._forward_rate_limiter
        geocode_module._reverse_rate_limiter = None
        geocode_module._forward_rate_limiter = None

    def tearDown(self):
        self._geocode_module._reverse_rate_limiter = self._orig_reverse
        self._geocode_module._forward_rate_limiter = self._orig_forward

    def test_reverse_limiter_is_a_reused_singleton(self):
        from filepopulator.geocode import _get_nominatim_geocode
        self.assertIs(_get_nominatim_geocode(), _get_nominatim_geocode())

    def test_forward_limiter_is_a_reused_singleton(self):
        from filepopulator.geocode import _get_nominatim_forward_geocode
        self.assertIs(_get_nominatim_forward_geocode(), _get_nominatim_forward_geocode())


class ReverseGeocodePreciseExtractionTests(unittest.TestCase):
    """reverse_geocode_precise()'s locality extraction, exercised against
    a mocked Nominatim client (patched at _get_nominatim_geocode, one
    level below reverse_geocode_precise itself) rather than mocking the
    function wholesale like BackfillGeocodingTests does - this is
    specifically testing the extraction logic those tests take as given."""

    def _location(self, address, display_name='Somewhere'):
        location = mock.Mock()
        location.raw = {'address': address, 'name': ''}
        location.address = display_name
        return location

    def test_settlement_tag_wins_over_weaker_tags(self):
        from filepopulator.geocode import reverse_geocode_precise
        address = {'city': 'Bothell', 'county': 'King County', 'country': 'United States', 'state': 'Washington'}
        with mock.patch('filepopulator.geocode._get_nominatim_geocode',
                         return_value=lambda *a, **k: self._location(address)):
            result = reverse_geocode_precise(47.7623, -122.2054)
        self.assertEqual(result['locality'], 'Bothell')

    def test_falls_back_to_county_when_no_settlement_tag(self):
        # Real production case (2026-09-10): a rural attraction/landmark
        # address has no city/town/village/hamlet/suburb tag at all, but
        # does have a county - a real, useful answer that was previously
        # discarded entirely in favor of leaving locality None.
        from filepopulator.geocode import reverse_geocode_precise
        address = {'county': 'Honolulu County', 'country': 'United States', 'state': 'Hawaii'}
        with mock.patch('filepopulator.geocode._get_nominatim_geocode',
                         return_value=lambda *a, **k: self._location(address)):
            result = reverse_geocode_precise(21.5258, -158.038)
        self.assertEqual(result['locality'], 'Honolulu County')

    def test_no_address_tags_at_all_leaves_locality_none(self):
        # Nothing left to fall back to at this level - run_geocoding_
        # backfill's own offline nearest-named-place fallback is what
        # handles this case, not reverse_geocode_precise itself.
        from filepopulator.geocode import reverse_geocode_precise
        with mock.patch('filepopulator.geocode._get_nominatim_geocode',
                         return_value=lambda *a, **k: self._location({})):
            result = reverse_geocode_precise(1.0, 1.0)
        self.assertIsNone(result['locality'])


class BackfillGeocodingTests(TestCase):
    """Exercises the backfill_geocoding management command against a
    mocked Nominatim lookup (no real network calls in tests) -- covers
    coordinate deduplication (multiple images at the same rounded spot
    share one GeocodeCache row), --dry-run, and that a failure on one
    coordinate is recorded (lookup_failed) rather than aborting the whole
    backfill for every other coordinate queued behind it."""

    def _make_bare_image_file(self, filename, lat, lon):
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(filename))
        obj = ImageFile(
            filename=filename,
            directory=directory,
            thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=10, height=10,
            gps_lat_decimal=lat, gps_lon_decimal=lon,
        )
        ImageFile.objects.bulk_create([obj])
        return ImageFile.objects.get(filename=filename)

    def test_shared_coordinate_reuses_one_cache_entry(self):
        from filepopulator.models import GeocodeCache
        img_a = self._make_bare_image_file("/tmp/geocode_test/a.jpg", 47.6062, -122.3321)
        img_b = self._make_bare_image_file("/tmp/geocode_test/b.jpg", 47.6062, -122.3321)

        fake_result = {
            'locality': 'Seattle', 'county': 'King', 'state': 'Washington',
            'country': 'United States', 'display_name': 'Seattle, WA, USA',
            'raw_response': {'address': {}},
        }
        with mock.patch(
            'filepopulator.geocode.reverse_geocode_precise',
            return_value=fake_result,
        ):
            call_command('backfill_geocoding')

        self.assertEqual(GeocodeCache.objects.count(), 1)
        img_a.refresh_from_db()
        img_b.refresh_from_db()
        self.assertIsNotNone(img_a.geocode)
        self.assertEqual(img_a.geocode_id, img_b.geocode_id)
        self.assertEqual(img_a.geocode.locality, 'Seattle')

    def test_dry_run_writes_nothing(self):
        from filepopulator.models import GeocodeCache
        self._make_bare_image_file("/tmp/geocode_test/c.jpg", 47.6062, -122.3321)

        call_command('backfill_geocoding', '--dry-run')

        self.assertEqual(GeocodeCache.objects.count(), 0)

    def test_failed_lookup_recorded_and_does_not_block_other_coordinates(self):
        from filepopulator.models import GeocodeCache
        img_fail = self._make_bare_image_file("/tmp/geocode_test/d.jpg", 47.6062, -122.3321)
        img_ok = self._make_bare_image_file("/tmp/geocode_test/e.jpg", 45.5152, -122.6784)

        fake_result = {
            'locality': 'Portland', 'county': 'Multnomah', 'state': 'Oregon',
            'country': 'United States', 'display_name': 'Portland, OR, USA',
            'raw_response': {'address': {}},
        }

        def flaky_geocode(lat, lon):
            if round(lat, 2) == 47.61:
                raise RuntimeError("simulated Nominatim failure")
            return fake_result

        with mock.patch(
            'filepopulator.geocode.reverse_geocode_precise',
            side_effect=flaky_geocode,
        ):
            call_command('backfill_geocoding')

        self.assertEqual(GeocodeCache.objects.count(), 2)
        img_fail.refresh_from_db()
        img_ok.refresh_from_db()
        self.assertTrue(img_fail.geocode.lookup_failed)
        self.assertFalse(img_ok.geocode.lookup_failed)
        self.assertEqual(img_ok.geocode.locality, 'Portland')

    def test_empty_locality_falls_back_to_nearest_named_place(self):
        # Real production bug (2026-09-10): Nominatim genuinely finding
        # nothing usable (not a failure) used to leave locality
        # permanently None. run_geocoding_backfill now falls back to the
        # offline nearest-named-place lookup and marks it approximate.
        from filepopulator.models import GeocodeCache
        img = self._make_bare_image_file("/tmp/geocode_test/f.jpg", 47.7623, -122.2054)  # near Bothell, WA

        fake_result = {
            'locality': None, 'county': None, 'state': None, 'country': None,
            'display_name': None, 'raw_response': None,
        }
        with mock.patch('filepopulator.geocode.reverse_geocode_precise', return_value=fake_result):
            call_command('backfill_geocoding')

        img.refresh_from_db()
        self.assertIsNotNone(img.geocode.locality)
        self.assertTrue(img.geocode.locality_is_approximate)
        self.assertFalse(img.geocode.lookup_failed)  # a genuine empty result, not a failure

    def test_concurrent_duplicate_coordinate_does_not_crash_the_batch(self):
        # Regression test for a real production bug (2026-08-27): the
        # recurring hourly geocode_new_images task and the one-time
        # backfill_geocoding command can run concurrently, and both
        # compute "what's missing" from the same snapshot -- if both
        # decide to geocode the same coordinate, whichever saves second
        # used to hit GeocodeCache's unique (lat, lon) constraint with no
        # handling at all, crashing the entire run for every other
        # coordinate queued behind it. Simulates the race by inserting a
        # competing row *during* our own reverse_geocode_precise() call,
        # the actual vulnerable window in production.
        from filepopulator.models import GeocodeCache
        img_race = self._make_bare_image_file("/tmp/geocode_test/race_a.jpg", 47.6062, -122.3321)
        img_ok = self._make_bare_image_file("/tmp/geocode_test/race_b.jpg", 45.5152, -122.6784)

        winner_result = {
            'locality': 'Seattle', 'county': 'King', 'state': 'Washington',
            'country': 'United States', 'display_name': 'Seattle, WA, USA',
            'raw_response': {'address': {}},
        }
        portland_result = {
            'locality': 'Portland', 'county': 'Multnomah', 'state': 'Oregon',
            'country': 'United States', 'display_name': 'Portland, OR, USA',
            'raw_response': {'address': {}},
        }

        def racy_geocode(lat, lon):
            if round(lat, 2) == 47.61:
                GeocodeCache.objects.create(lat=lat, lon=lon, locality='Concurrent Winner')
                return winner_result
            return portland_result

        with mock.patch('filepopulator.geocode.reverse_geocode_precise', side_effect=racy_geocode):
            call_command('backfill_geocoding')

        self.assertEqual(GeocodeCache.objects.count(), 2)
        img_race.refresh_from_db()
        img_ok.refresh_from_db()
        self.assertEqual(img_race.geocode.locality, 'Concurrent Winner')
        self.assertEqual(img_ok.geocode.locality, 'Portland')

    def test_nearby_uncached_coordinate_reuses_existing_precise_geocode_without_a_network_call(self):
        # A real production scenario (2026-09-10): a photo library has a
        # lot of "wandered the same block across visits" GPS noise -- two
        # coordinates ~50m apart round to different ~11m GeocodeCache
        # cells (ROUND_DECIMALS=4) but are still clearly the same place.
        # An already-cached PRECISE result within GEOCODE_REUSE_RADIUS_KM
        # should be copied directly, with zero Nominatim calls.
        from filepopulator.models import GeocodeCache
        GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, locality='Seattle', county='King',
            state='Washington', country='United States',
            display_name='Seattle, WA, USA',
        )
        # ~50m north of the cached coordinate -- well within the 150m
        # reuse radius, but a different rounded (lat, lon) cache key.
        img = self._make_bare_image_file("/tmp/geocode_test/g.jpg", 47.60665, -122.3321)

        with mock.patch('filepopulator.geocode.reverse_geocode_precise') as mock_reverse:
            call_command('backfill_geocoding')
            mock_reverse.assert_not_called()

        self.assertEqual(GeocodeCache.objects.count(), 2)
        img.refresh_from_db()
        self.assertEqual(img.geocode.locality, 'Seattle')
        self.assertEqual(img.geocode.state, 'Washington')
        self.assertFalse(img.geocode.locality_is_approximate)

    def test_far_uncached_coordinate_still_calls_nominatim(self):
        # The reuse radius is tight (150m) -- a coordinate genuinely in a
        # different area must not silently inherit a nearby cache entry's
        # locality.
        from filepopulator.models import GeocodeCache
        GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, locality='Seattle', county='King',
            state='Washington', country='United States',
            display_name='Seattle, WA, USA',
        )
        img = self._make_bare_image_file("/tmp/geocode_test/h.jpg", 45.5152, -122.6784)

        fake_result = {
            'locality': 'Portland', 'county': 'Multnomah', 'state': 'Oregon',
            'country': 'United States', 'display_name': 'Portland, OR, USA',
            'raw_response': {'address': {}},
        }
        with mock.patch('filepopulator.geocode.reverse_geocode_precise', return_value=fake_result) as mock_reverse:
            call_command('backfill_geocoding')
            mock_reverse.assert_called_once()

        img.refresh_from_db()
        self.assertEqual(img.geocode.locality, 'Portland')

    def test_approximate_existing_entry_is_not_reused_as_a_source(self):
        # An offline nearest-named-place fallback result
        # (locality_is_approximate=True) is itself already a guess -- it
        # must not be propagated to a second, different coordinate as if
        # it were a genuine precise answer.
        from filepopulator.models import GeocodeCache
        GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, locality='Some Distant Named Place',
            country='United States', locality_is_approximate=True,
        )
        img = self._make_bare_image_file("/tmp/geocode_test/i.jpg", 47.60665, -122.3321)

        fake_result = {
            'locality': 'Real Nearby Place', 'county': 'King', 'state': 'Washington',
            'country': 'United States', 'display_name': 'Real Nearby Place, WA, USA',
            'raw_response': {'address': {}},
        }
        with mock.patch('filepopulator.geocode.reverse_geocode_precise', return_value=fake_result) as mock_reverse:
            call_command('backfill_geocoding')
            mock_reverse.assert_called_once()

        img.refresh_from_db()
        self.assertEqual(img.geocode.locality, 'Real Nearby Place')

    def test_reuse_chains_within_the_same_run(self):
        # A coordinate resolved earlier in THIS run (not predating it)
        # should also be usable as a reuse source for a later coordinate
        # in the same run, not just rows that already existed beforehand.
        from filepopulator.models import GeocodeCache
        img_seed = self._make_bare_image_file("/tmp/geocode_test/j.jpg", 47.6062, -122.3321)
        img_chain = self._make_bare_image_file("/tmp/geocode_test/k.jpg", 47.60665, -122.3321)

        fake_result = {
            'locality': 'Seattle', 'county': 'King', 'state': 'Washington',
            'country': 'United States', 'display_name': 'Seattle, WA, USA',
            'raw_response': {'address': {}},
        }
        with mock.patch('filepopulator.geocode.reverse_geocode_precise', return_value=fake_result) as mock_reverse:
            call_command('backfill_geocoding')
            mock_reverse.assert_called_once()

        img_seed.refresh_from_db()
        img_chain.refresh_from_db()
        self.assertEqual(img_seed.geocode.locality, 'Seattle')
        self.assertEqual(img_chain.geocode.locality, 'Seattle')
        self.assertFalse(img_chain.geocode.locality_is_approximate)

    def test_previously_failed_coordinate_is_retried_and_recovers(self):
        # Real production scenario (2026-09-10): 80 coordinates failed
        # with a genuine 429 during the rate-limit storm this session
        # diagnosed and fixed -- before this fix, a failed row looked
        # permanently "already cached" and was never retried, even once
        # the underlying rate-limit issue was resolved.
        from filepopulator.models import GeocodeCache
        img = self._make_bare_image_file("/tmp/geocode_test/l.jpg", 47.6062, -122.3321)
        stale = GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, lookup_failed=True,
            lookup_error='Non-successful status code 429',
        )
        # A plain .update() rather than img.save() -- ImageFile.save()
        # unconditionally re-decodes the file for its pixel hash, and this
        # bare test fixture has no real file on disk.
        ImageFile.objects.filter(pk=img.pk).update(geocode=stale)

        fake_result = {
            'locality': 'Seattle', 'county': 'King', 'state': 'Washington',
            'country': 'United States', 'display_name': 'Seattle, WA, USA',
            'raw_response': {'address': {}},
        }
        with mock.patch('filepopulator.geocode.reverse_geocode_precise', return_value=fake_result):
            result = call_command('backfill_geocoding')

        # GeocodeCache row is updated in place, not duplicated.
        self.assertEqual(GeocodeCache.objects.count(), 1)
        stale.refresh_from_db()
        self.assertFalse(stale.lookup_failed)
        self.assertIsNone(stale.lookup_error)
        self.assertEqual(stale.locality, 'Seattle')
        img.refresh_from_db()
        self.assertEqual(img.geocode.locality, 'Seattle')

    def test_previously_failed_coordinate_that_fails_again_stays_flagged(self):
        from filepopulator.models import GeocodeCache
        self._make_bare_image_file("/tmp/geocode_test/m.jpg", 47.6062, -122.3321)
        GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, lookup_failed=True,
            lookup_error='Non-successful status code 429',
        )

        def still_failing(lat, lon):
            raise RuntimeError("still rate-limited")

        with mock.patch('filepopulator.geocode.reverse_geocode_precise', side_effect=still_failing):
            call_command('backfill_geocoding')

        self.assertEqual(GeocodeCache.objects.count(), 1)
        entry = GeocodeCache.objects.get(lat=47.6062, lon=-122.3321)
        self.assertTrue(entry.lookup_failed)
        self.assertEqual(entry.lookup_error, 'still rate-limited')

    def test_reuse_does_not_chain_through_a_previously_reused_result(self):
        # Real concern raised 2026-09-10: if a reused (not independently
        # Nominatim-confirmed) result could itself seed further reuse, a
        # sequence of points spaced just under GEOCODE_REUSE_RADIUS_KM
        # apart -- a realistic shape for a GPS trace along a street or
        # across a metro area -- could drift arbitrarily far from the one
        # real geocode the whole chain traces back to, each hop
        # individually "within radius" of the previous one. A must reuse
        # B, and B must reuse C, but C should never inherit A's answer if
        # C is actually outside the reuse radius of A itself.
        from filepopulator.models import GeocodeCache
        GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, locality='Seattle', county='King',
            state='Washington', country='United States',
            display_name='Seattle, WA, USA',
        )
        # ~100m north of the real anchor -- within the 150m reuse radius,
        # should reuse 'Seattle' with no Nominatim call.
        img_b = self._make_bare_image_file("/tmp/geocode_test/n.jpg", 47.6071, -122.3321)
        # ~100m further north again -- ~200m from the real anchor, so
        # OUTSIDE the reuse radius of it. If B (itself only reused, never
        # independently confirmed) were allowed to seed reuse, this would
        # wrongly inherit 'Seattle' from B. It must not.
        img_c = self._make_bare_image_file("/tmp/geocode_test/o.jpg", 47.6080, -122.3321)

        tacoma_result = {
            'locality': 'Tacoma', 'county': 'Pierce', 'state': 'Washington',
            'country': 'United States', 'display_name': 'Tacoma, WA, USA',
            'raw_response': {'address': {}},
        }
        with mock.patch('filepopulator.geocode.reverse_geocode_precise', return_value=tacoma_result) as mock_reverse:
            call_command('backfill_geocoding')
            # Exactly one real call -- for C, which is genuinely out of
            # every real anchor's reuse radius. B needed none at all.
            mock_reverse.assert_called_once()

        img_b.refresh_from_db()
        img_c.refresh_from_db()
        self.assertEqual(img_b.geocode.locality, 'Seattle')
        self.assertEqual(img_c.geocode.locality, 'Tacoma')
        self.assertNotEqual(img_c.geocode.locality, img_b.geocode.locality)

    def test_image_at_already_cached_coordinate_is_relinked_with_no_nominatim_call(self):
        # Real production bug (2026-10-21): an ImageFile whose (rounded)
        # coordinate is ALREADY fully resolved in GeocodeCache (e.g. a
        # second photo from an already-geocoded place, ingested after the
        # first photo from that spot already cached+linked it) used to
        # never get its own geocode FK set at all -- the per-coordinate
        # loop only ever revisits a coordinate that's missing from the
        # cache or lookup_failed=True, so an already-resolved coordinate
        # was never looked at again by any future run. Now a standing,
        # unconditional relink pass fixes this on every call, at zero
        # Nominatim cost.
        from filepopulator.models import GeocodeCache
        GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, locality='Seattle', county='King',
            state='Washington', country='United States',
            display_name='Seattle, WA, USA',
        )
        img = self._make_bare_image_file("/tmp/geocode_test/relink1.jpg", 47.6062, -122.3321)
        self.assertIsNone(img.geocode)

        with mock.patch('filepopulator.geocode.reverse_geocode_precise') as mock_reverse:
            call_command('backfill_geocoding')
            mock_reverse.assert_not_called()

        img.refresh_from_db()
        self.assertIsNotNone(img.geocode)
        self.assertEqual(img.geocode.locality, 'Seattle')

    def test_relink_dry_run_writes_nothing(self):
        from filepopulator.models import GeocodeCache
        GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, locality='Seattle', county='King',
            state='Washington', country='United States',
            display_name='Seattle, WA, USA',
        )
        img = self._make_bare_image_file("/tmp/geocode_test/relink2.jpg", 47.6062, -122.3321)

        call_command('backfill_geocoding', '--dry-run')

        img.refresh_from_db()
        self.assertIsNone(img.geocode)

    def test_image_at_a_failed_coordinate_is_not_relinked_goes_through_retry_instead(self):
        # A lookup_failed=True cache row is a known-bad, not a resolved
        # answer -- it must go through the normal retry path (a real
        # reverse_geocode_precise call), not be silently "relinked" to a
        # failed, contentless entry.
        from filepopulator.models import GeocodeCache
        GeocodeCache.objects.create(
            lat=47.6062, lon=-122.3321, lookup_failed=True, lookup_error='simulated prior failure',
        )
        img = self._make_bare_image_file("/tmp/geocode_test/relink3.jpg", 47.6062, -122.3321)

        fake_result = {
            'locality': 'Seattle', 'county': 'King', 'state': 'Washington',
            'country': 'United States', 'display_name': 'Seattle, WA, USA',
            'raw_response': {'address': {}},
        }
        with mock.patch('filepopulator.geocode.reverse_geocode_precise', return_value=fake_result) as mock_reverse:
            call_command('backfill_geocoding')
            mock_reverse.assert_called_once()

        img.refresh_from_db()
        self.assertIsNotNone(img.geocode)
        self.assertFalse(img.geocode.lookup_failed)
        self.assertEqual(img.geocode.locality, 'Seattle')

    def test_high_failure_rate_this_run_is_recorded_as_an_application_error(self):
        # Run-level, not per-coordinate -- a real outage/rate-limit event
        # (confirmed real, 2026-09-10) fails most or all of a run's real
        # calls at once, unlike ordinary single-coordinate noise.
        from common.models import ApplicationError
        for i in range(6):
            self._make_bare_image_file(f"/tmp/geocode_test/fail{i}.jpg", 10.0 + i, 10.0 + i)

        with mock.patch('filepopulator.geocode.reverse_geocode_precise',
                         side_effect=RuntimeError("simulated Nominatim outage")):
            call_command('backfill_geocoding')

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'filepopulator.geocode.run_geocoding_backfill')
        self.assertIn('6/6', error.message)

    def test_low_failure_rate_is_not_recorded(self):
        from common.models import ApplicationError
        for i in range(6):
            self._make_bare_image_file(f"/tmp/geocode_test/ok{i}.jpg", 20.0 + i, 20.0 + i)

        fake_result = {
            'locality': 'Somewhere', 'county': None, 'state': None,
            'country': 'United States', 'display_name': 'Somewhere, USA',
            'raw_response': {'address': {}},
        }
        with mock.patch('filepopulator.geocode.reverse_geocode_precise', return_value=fake_result):
            call_command('backfill_geocoding')

        self.assertEqual(ApplicationError.objects.count(), 0)

    def test_small_batch_below_min_attempts_is_not_recorded(self):
        # Even a 100% failure rate shouldn't flag a tiny batch -- too
        # noisy a signal below GEOCODE_FAILURE_RATE_MIN_ATTEMPTS.
        from common.models import ApplicationError
        self._make_bare_image_file("/tmp/geocode_test/tiny0.jpg", 30.0, 30.0)

        with mock.patch('filepopulator.geocode.reverse_geocode_precise',
                         side_effect=RuntimeError("simulated failure")):
            call_command('backfill_geocoding')

        self.assertEqual(ApplicationError.objects.count(), 0)


class BackfillOwnTracksGpsTaskTests(TestCase):
    """The recurring Celery task -- backfill_camera_gps() itself is
    already thoroughly tested directly (see filepopulator/tests.py's
    OwnTracks GPS backfill tests); this just confirms the task wires it
    up correctly (right default args, respects its own advisory lock,
    doesn't crash the whole task on OwnTracksConfigError)."""

    def test_calls_backfill_with_a_run_limit_and_no_camera_override(self):
        from filepopulator.tasks import backfill_owntracks_gps

        with mock.patch('filepopulator.owntracks_gps.backfill_camera_gps') as mock_backfill:
            mock_backfill.return_value = {'matched': 3, 'no_offset_tag': 1, 'no_nearby_point': 2}
            backfill_owntracks_gps()

        # No camera_models= override -- defaults to every camera currently
        # enabled in CameraGpsBackfillConfig, so a camera toggled on via
        # the frontend is picked up on the very next scheduled run.
        mock_backfill.assert_called_once()
        _, kwargs = mock_backfill.call_args
        self.assertNotIn('camera_models', kwargs)
        self.assertEqual(kwargs['limit'], 1000)

    def test_concurrent_run_is_skipped(self):
        from filepopulator.tasks import backfill_owntracks_gps

        with mock.patch('filepopulator.tasks.advisory_lock') as mock_lock, \
             mock.patch('filepopulator.owntracks_gps.backfill_camera_gps') as mock_backfill:
            mock_lock.return_value.__enter__.return_value = False
            backfill_owntracks_gps()

        mock_backfill.assert_not_called()

    def test_not_configured_is_a_no_op_not_a_crash(self):
        from filepopulator.owntracks_gps import OwnTracksConfigError
        from filepopulator.tasks import backfill_owntracks_gps

        with mock.patch('filepopulator.owntracks_gps.backfill_camera_gps') as mock_backfill:
            mock_backfill.side_effect = OwnTracksConfigError("not configured")
            backfill_owntracks_gps()  # must not raise

        mock_backfill.assert_called_once()


@override_settings(OWNTRACKS_LOCATION_API_URL='https://owntracks.example.com',
                    OWNTRACKS_LOCATION_API_KEY='test-key')
class BackfillS23GpsTests(TestCase):
    """backfill_owntracks_gps / filepopulator.owntracks_gps -- see that
    module's own docstring for the real detail this was built around:
    ImageFile.dateTaken/_get_date_taken() throws away the real UTC
    offset, so this re-reads EXIF directly rather than trusting the DB,
    trying each known per-camera offset source in turn (standard EXIF
    OffsetTimeOriginal, Nikon's MakerNotes TimeZone+DaylightSavings).
    exiftool and the HTTP call are both mocked; no real subprocess or
    network calls."""

    FIXTURE_IMG = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'ci_fixtures', 'test_imgs_filepopulate', 'gps.jpg',
    )

    def tearDown(self):
        ImageFile.objects.all().delete()

    def _make_face(self, image, person, validated):
        face = Face(
            source_image_file=image, declared_name=person,
            box_left=1, box_top=1, box_right=5, box_bottom=5, validated=validated,
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()
        return face

    def _bare_image(self, camera_model, gps_lat_decimal=-999, filename=None, dateTaken=None):
        # ImageFile.save() -> _init_image() really opens the file (PIL),
        # so this needs a real tiny image on disk, not just a DB row --
        # same reason other tests in this file copy a real fixture
        # rather than fabricate a path.
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        dest = filename or os.path.join(tmp_dir, f'{camera_model}_{ImageFile.objects.count()}.jpg')
        if filename is None:
            shutil.copy(self.FIXTURE_IMG, dest)
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(dest))
        img = ImageFile.objects.create(
            filename=dest,
            directory=directory,
            pixel_hash=f'hash{ImageFile.objects.count()}',
            file_hash=f'fhash{ImageFile.objects.count()}',
            width=100, height=100, orientation=1,
            dateAdded=timezone.now(), dateModified=timezone.now(),
            camera_model=camera_model,
            gps_lat_decimal=gps_lat_decimal,
        )
        if dateTaken is not None:
            ImageFile.objects.filter(pk=img.pk).update(dateTaken=dateTaken)
            img.refresh_from_db()
        return img

    def _exiftool_result(self, date_str=None, offset_str=None,
                          nikon_timezone=None, nikon_dst=None):
        entry = {}
        if date_str is not None:
            entry['DateTimeOriginal'] = date_str
        if offset_str is not None:
            entry['OffsetTimeOriginal'] = offset_str
        if nikon_timezone is not None:
            entry['TimeZone'] = nikon_timezone
        if nikon_dst is not None:
            entry['DaylightSavings'] = nikon_dst
        return mock.Mock(stdout=json.dumps([entry]))

    def test_nikon_timezone_and_dst_sets_gps(self):
        # Real case this was built from: a Nikon D3200 trip to Italy,
        # clock left on home World Time (-05:00) with DST on -- the
        # true effective offset is -04:00, confirmed against a real
        # OwnTracks match that resolved to Pisa, Italy.
        img = self._bare_image('NIKON D3200')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result(
                             '2026:06:14 14:28:52', nikon_timezone='-05:00', nikon_dst='Yes')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 43.718, 'longitude': 10.402},
            )
            from filepopulator.owntracks_gps import backfill_camera_gps
            backfill_camera_gps()

        called_params = mock_get.call_args.kwargs['params']
        self.assertEqual(called_params['datetime'], '2026-06-14T14:28:52-04:00')
        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, 43.718)

    def test_nikon_without_dst_uses_base_timezone_only(self):
        img = self._bare_image('NIKON D3200')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result(
                             '2026:01:14 14:28:52', nikon_timezone='-05:00', nikon_dst='No')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 1.0, 'longitude': 1.0},
            )
            from filepopulator.owntracks_gps import backfill_camera_gps
            backfill_camera_gps()

        called_params = mock_get.call_args.kwargs['params']
        self.assertEqual(called_params['datetime'], '2026-01-14T14:28:52-05:00')

    def test_exif_offset_preferred_over_nikon_tags_if_both_present(self):
        # Not a real-world case (a phone wouldn't have Nikon MakerNotes
        # tags), but confirms the precedence is deliberate, not
        # order-of-dict-insertion luck.
        from filepopulator.owntracks_gps import _read_offset_datetime
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result(
                             '2026:06:14 14:28:52', offset_str='+01:00',
                             nikon_timezone='-05:00', nikon_dst='Yes')):
            date_str, offset_str = _read_offset_datetime('/fake/path.jpg')
        self.assertEqual(offset_str, '+01:00')

    def test_unparseable_nikon_timezone_is_skipped_not_guessed(self):
        img = self._bare_image('NIKON D3200')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result(
                             '2026:06:14 14:28:52', nikon_timezone='garbage', nikon_dst='Yes')):
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps()

        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, -999)
        self.assertEqual(counts['no_offset_tag'], 1)

    def test_command_rejects_camera_model_not_in_allowlist(self):
        with self.assertRaises(CommandError):
            call_command('backfill_owntracks_gps', '--camera-model', 'Some Random Camera')

    def test_command_scopes_to_requested_camera_model_only(self):
        self._bare_image('Galaxy S23+')
        self._bare_image('NIKON D3200')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 40.1, 'longitude': -111.9},
            )
            call_command('backfill_owntracks_gps', '--camera-model', 'Galaxy S23+')

        nikon = ImageFile.objects.get(camera_model='NIKON D3200')
        self.assertEqual(nikon.gps_lat_decimal, -999)
        s23 = ImageFile.objects.get(camera_model='Galaxy S23+')
        self.assertEqual(s23.gps_lat_decimal, 40.1)

    def test_start_end_date_excludes_photos_outside_the_window(self):
        # Real motivating case: a camera with years of history predating
        # OwnTracks entirely would otherwise burn a real exiftool+HTTP
        # round trip per photo just to learn "no nearby point," every
        # single run.
        import datetime
        old = self._bare_image('NIKON D3200', dateTaken=datetime.datetime(2014, 10, 25, tzinfo=datetime.timezone.utc))
        in_range = self._bare_image('NIKON D3200', dateTaken=datetime.datetime(2026, 6, 14, tzinfo=datetime.timezone.utc))

        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result(
                             '2026:06:14 14:28:52', nikon_timezone='-05:00', nikon_dst='Yes')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 43.718, 'longitude': 10.402},
            )
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps(
                camera_models=['NIKON D3200'],
                start_date=datetime.date(2026, 6, 10), end_date=datetime.date(2026, 6, 21),
            )

        self.assertEqual(counts['matched'], 1)
        old.refresh_from_db()
        self.assertEqual(old.gps_lat_decimal, -999)
        in_range.refresh_from_db()
        self.assertEqual(in_range.gps_lat_decimal, 43.718)

    def test_only_scopes_configured_s23_camera_models(self):
        self._bare_image('Galaxy S23+')
        self._bare_image('SM-S916U')
        self._bare_image('iPhone 12 mini')  # must never be touched

        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 40.1, 'longitude': -111.9},
            )
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps()

        self.assertEqual(counts['matched'], 2)
        iphone = ImageFile.objects.get(camera_model='iPhone 12 mini')
        self.assertEqual(iphone.gps_lat_decimal, -999)

    def test_builds_offset_aware_datetime_from_fresh_exif_not_db(self):
        # The real detail this whole module exists to get right -- the
        # DB's own dateTaken is unreliable (tagged UTC regardless of the
        # real offset), so the request must reflect the file's own real
        # OffsetTimeOriginal, re-read fresh.
        self._bare_image('Galaxy S23+')

        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:06:14 15:18:10', '+01:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 43.7, 'longitude': 10.4},
            )
            from filepopulator.owntracks_gps import backfill_camera_gps
            backfill_camera_gps()

        called_params = mock_get.call_args.kwargs['params']
        self.assertEqual(called_params['datetime'], '2026-06-14T15:18:10+01:00')

    def test_200_sets_gps_and_source(self):
        img = self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 40.1, 'longitude': -111.9},
            )
            from filepopulator.owntracks_gps import backfill_camera_gps
            backfill_camera_gps()

        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, 40.1)
        self.assertEqual(img.gps_lon_decimal, -111.9)
        self.assertEqual(img.gps_source, 'owntracks')

    def test_404_leaves_gps_untouched(self):
        img = self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(status_code=404, text='nothing within 15 minutes')
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps()

        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, -999)
        self.assertEqual(counts['no_nearby_point'], 1)

    def test_401_aborts_the_whole_run(self):
        self._bare_image('Galaxy S23+')
        self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(status_code=401, text='bad key')
            from filepopulator.owntracks_gps import backfill_camera_gps, OwnTracksConfigError
            with self.assertRaises(OwnTracksConfigError):
                backfill_camera_gps()

    def test_missing_offset_tag_is_skipped_not_guessed(self):
        img = self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', None)):
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps()

        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, -999)
        self.assertEqual(counts['no_offset_tag'], 1)

    def test_dry_run_does_not_write(self):
        img = self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 40.1, 'longitude': -111.9},
            )
            from filepopulator.owntracks_gps import backfill_camera_gps
            backfill_camera_gps(dry_run=True)

        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, -999)

    def test_already_populated_rows_are_excluded(self):
        self._bare_image('Galaxy S23+', gps_lat_decimal=40.5)
        with mock.patch('filepopulator.owntracks_gps.subprocess.run') as mock_run:
            from filepopulator.owntracks_gps import backfill_camera_gps
            backfill_camera_gps()
        mock_run.assert_not_called()

    def test_failed_attempt_is_recorded_and_skipped_on_next_run(self):
        from filepopulator.models import GpsEstimationAttempt
        from filepopulator.owntracks_gps import backfill_camera_gps

        img = self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(status_code=404, text='nothing nearby')
            counts = backfill_camera_gps()
        self.assertEqual(counts['no_nearby_point'], 1)
        attempt = GpsEstimationAttempt.objects.get(image=img)
        self.assertEqual(attempt.reason, GpsEstimationAttempt.REASON_NO_NEARBY_POINT)

        # Second run, no --retry-failed -- must not even attempt exiftool.
        with mock.patch('filepopulator.owntracks_gps.subprocess.run') as mock_run:
            counts2 = backfill_camera_gps()
        mock_run.assert_not_called()
        self.assertEqual(counts2['no_nearby_point'], 0)

    def test_no_offset_tag_attempt_is_never_retried_even_with_retry_failed(self):
        from filepopulator.models import GpsEstimationAttempt
        from filepopulator.owntracks_gps import backfill_camera_gps

        img = self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', None)):
            backfill_camera_gps()
        self.assertEqual(
            GpsEstimationAttempt.objects.get(image=img).reason,
            GpsEstimationAttempt.REASON_NO_OFFSET_TAG,
        )

        with mock.patch('filepopulator.owntracks_gps.subprocess.run') as mock_run:
            backfill_camera_gps(retry_failed=True)
        mock_run.assert_not_called()

    def test_retry_failed_reattempts_no_nearby_point_and_can_succeed(self):
        from filepopulator.owntracks_gps import backfill_camera_gps

        img = self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(status_code=404, text='nothing nearby')
            backfill_camera_gps()

        # Simulates the OwnTracks service's own history growing -- the
        # exact real case this session hit.
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 40.1, 'longitude': -111.9},
            )
            counts = backfill_camera_gps(retry_failed=True)

        self.assertEqual(counts['matched'], 1)
        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, 40.1)
        from filepopulator.models import GpsEstimationAttempt
        self.assertFalse(GpsEstimationAttempt.objects.filter(image=img).exists())

    def test_missing_config_raises_clear_error(self):
        self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             override_settings(OWNTRACKS_LOCATION_API_URL=None, OWNTRACKS_LOCATION_API_KEY=None):
            from filepopulator.owntracks_gps import backfill_camera_gps, OwnTracksConfigError
            with self.assertRaises(OwnTracksConfigError):
                backfill_camera_gps()

    def test_management_command_runs_end_to_end(self):
        self._bare_image('Galaxy S23+')
        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 40.1, 'longitude': -111.9},
            )
            call_command('backfill_owntracks_gps')

    def test_validated_face_for_gps_backfill_source_person_is_eligible(self):
        # A camera NOT in CameraGpsBackfillConfig at all -- eligibility
        # here must come entirely from the verified-person path.
        from face_manager.models import Person
        img = self._bare_image('Totally Unenrolled Camera')
        person = Person.objects.create(person_name='Benjamin Lewis', gps_backfill_source=True)
        self._make_face(img, person, True)

        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 40.1, 'longitude': -111.9},
            )
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps()

        self.assertEqual(counts['matched'], 1)
        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, 40.1)

    def test_unvalidated_face_for_gps_backfill_source_person_is_not_eligible(self):
        # Only a CONFIRMED (validated) appearance is a real "this person
        # was physically here" signal -- a mere proposed/unconfirmed
        # candidate match isn't.
        from face_manager.models import Person
        img = self._bare_image('Totally Unenrolled Camera')
        person = Person.objects.create(person_name='Benjamin Lewis', gps_backfill_source=True)
        self._make_face(img, person, False)

        with mock.patch('filepopulator.owntracks_gps.subprocess.run') as mock_exiftool:
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps()

        mock_exiftool.assert_not_called()
        self.assertEqual(counts['matched'], 0)
        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, -999)

    def test_validated_face_for_non_backfill_source_person_is_not_eligible(self):
        # The person must have gps_backfill_source=True specifically --
        # a validated face for anyone else doesn't make a photo eligible.
        from face_manager.models import Person
        img = self._bare_image('Totally Unenrolled Camera')
        person = Person.objects.create(person_name='Someone Else', gps_backfill_source=False)
        self._make_face(img, person, True)

        with mock.patch('filepopulator.owntracks_gps.subprocess.run') as mock_exiftool:
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps()

        mock_exiftool.assert_not_called()
        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, -999)

    def test_include_verified_person_false_excludes_it(self):
        from face_manager.models import Person
        img = self._bare_image('Totally Unenrolled Camera')
        person = Person.objects.create(person_name='Benjamin Lewis', gps_backfill_source=True)
        self._make_face(img, person, True)

        with mock.patch('filepopulator.owntracks_gps.subprocess.run') as mock_exiftool:
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps(include_verified_person=False)

        mock_exiftool.assert_not_called()
        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, -999)

    def test_no_verified_person_management_command_flag(self):
        from face_manager.models import Person
        img = self._bare_image('Totally Unenrolled Camera')
        person = Person.objects.create(person_name='Benjamin Lewis', gps_backfill_source=True)
        self._make_face(img, person, True)

        with mock.patch('filepopulator.owntracks_gps.subprocess.run') as mock_exiftool:
            call_command('backfill_owntracks_gps', '--no-verified-person')

        mock_exiftool.assert_not_called()
        img.refresh_from_db()
        self.assertEqual(img.gps_lat_decimal, -999)

    def test_eligible_via_both_camera_and_person_is_not_double_processed(self):
        # ImageFile.objects.filter(eligibility, ...) must .distinct() --
        # an image matching BOTH the camera allowlist AND the
        # verified-person clause via more than one Face row shouldn't be
        # joined into duplicate result rows.
        from face_manager.models import Person
        img = self._bare_image('Galaxy S23+')
        person = Person.objects.create(person_name='Benjamin Lewis', gps_backfill_source=True)
        self._make_face(img, person, True)
        person2 = Person.objects.create(person_name='Also Tracked', gps_backfill_source=True)
        self._make_face(img, person2, True)

        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(
                status_code=200, json=lambda: {'latitude': 40.1, 'longitude': -111.9},
            )
            from filepopulator.owntracks_gps import backfill_camera_gps
            counts = backfill_camera_gps()

        self.assertEqual(counts['matched'], 1)

    def test_corrupted_ancient_date_is_excluded_from_the_queue(self):
        # Real production bug (2026-10-21): confirmed ImageFile.dateTaken
        # corruption (e.g. year 0102) sorted first under order_by(
        # 'dateTaken'), burning the whole run's budget on guaranteed-
        # failure rows before ever reaching a real candidate.
        # dateTakenValid=True on every one of these (it only tracks "a
        # date was parsed from somewhere," not "the date is plausible"),
        # so MIN_SANE_DATE_TAKEN_YEAR is a real, separate floor.
        import datetime
        old = self._bare_image(
            'Galaxy S23+', dateTaken=datetime.datetime(102, 10, 9, tzinfo=datetime.timezone.utc))

        with mock.patch('filepopulator.owntracks_gps.subprocess.run') as mock_exiftool:
            from filepopulator.owntracks_gps import backfill_camera_gps
            backfill_camera_gps()

        mock_exiftool.assert_not_called()
        from filepopulator.models import GpsEstimationAttempt
        self.assertFalse(GpsEstimationAttempt.objects.filter(image=old).exists())
        old.refresh_from_db()
        self.assertEqual(old.gps_lat_decimal, -999)

    def test_camera_and_person_populations_each_get_a_fair_share_of_limit(self):
        # Real production bug (2026-10-21): adding the verified-person
        # path grew the eligible queue substantially (camera-allowlist-
        # only backlog ~11k -> ~22k combined), and a single combined
        # date-ordered queue let the larger population completely starve
        # the other out of ever being reached. Each population must now
        # get its own fair share of `limit` instead.
        from face_manager.models import Person
        person = Person.objects.create(person_name='Benjamin Lewis', gps_backfill_source=True)

        camera_imgs = [self._bare_image('Galaxy S23+') for _ in range(3)]
        person_imgs = []
        for _ in range(3):
            img = self._bare_image('Totally Unenrolled Camera')
            self._make_face(img, person, True)
            person_imgs.append(img)

        with mock.patch('filepopulator.owntracks_gps.subprocess.run',
                         return_value=self._exiftool_result('2026:10:16 14:46:17', '-04:00')), \
             mock.patch('filepopulator.owntracks_gps.requests.get') as mock_get:
            mock_get.return_value = mock.Mock(status_code=404)  # "no nearby point"
            from filepopulator.owntracks_gps import backfill_camera_gps
            backfill_camera_gps(limit=2)

        from filepopulator.models import GpsEstimationAttempt
        touched_camera = GpsEstimationAttempt.objects.filter(
            image__in=camera_imgs).count()
        touched_person = GpsEstimationAttempt.objects.filter(
            image__in=person_imgs).count()
        self.assertEqual(touched_camera, 1)
        self.assertEqual(touched_person, 1)



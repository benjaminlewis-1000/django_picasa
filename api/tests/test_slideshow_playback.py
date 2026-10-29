"""api tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test api.tests.test_slideshow_playback
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


class SlideshowPaginationTests(ApiTestCase):
    """filteredImagesView used to hand back the ENTIRE filtered id list
    (and, with full_data, every id's metadata) in one response - which
    meant anyone holding the slideshow key could read the whole library
    in a single request, no scraping sophistication required. These
    lock down the replacement: every response is capped to a bounded
    page, and offset/limit walk through the rest."""

    def _make_images(self, n):
        # Direct ORM creation rather than make_image() (which is backed
        # by a handful of real fixture files - not enough for the
        # dozens-to-hundreds of images these pagination tests need).
        # ImageFile.save() opens and decodes the file at `filename` for
        # real (_init_image), so each one still needs an actual tiny
        # JPEG on disk, not just a DB row.
        from datetime import timedelta
        tmp_dir = tempfile.mkdtemp(prefix='pagination_test_')
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        directory = Directory.objects.create(dir_path=tmp_dir)
        base = datetime(2020, 1, 1, tzinfo=dt_timezone.utc)
        images = []
        for i in range(n):
            path = os.path.join(tmp_dir, f'{i}.jpg')
            with open(path, 'wb') as f:
                f.write(_tiny_jpeg_bytes())
            images.append(ImageFile.objects.create(
                filename=path, directory=directory, width=50, height=50,
                dateTaken=base + timedelta(days=i),
            ))
        return images

    def test_default_page_is_capped_even_with_a_much_larger_library(self):
        from api.views import SLIDESHOW_PAGE_SIZE_DEFAULT
        self._make_images(SLIDESHOW_PAGE_SIZE_DEFAULT + 15)
        resp = self.anon_client.get(f'/api/image_list/?key={settings.SLIDESHOW_API_KEY}&cronological')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = resp.json()
        self.assertEqual(len(data['items']), SLIDESHOW_PAGE_SIZE_DEFAULT)
        self.assertEqual(data['total_count'], SLIDESHOW_PAGE_SIZE_DEFAULT + 15)

    def test_requested_limit_is_hard_capped_regardless_of_what_client_asks_for(self):
        from api.views import SLIDESHOW_PAGE_SIZE_MAX
        self._make_images(SLIDESHOW_PAGE_SIZE_MAX + 20)
        resp = self.anon_client.get(
            f'/api/image_list/?key={settings.SLIDESHOW_API_KEY}&cronological&limit=100000'
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = resp.json()
        self.assertEqual(len(data['items']), SLIDESHOW_PAGE_SIZE_MAX)
        self.assertEqual(data['limit'], SLIDESHOW_PAGE_SIZE_MAX)

    def test_chronological_pages_are_consecutive_and_non_overlapping(self):
        images = self._make_images(10)
        resp1 = self.anon_client.get(
            f'/api/image_list/?key={settings.SLIDESHOW_API_KEY}&cronological&limit=4&offset=0'
        )
        resp2 = self.anon_client.get(
            f'/api/image_list/?key={settings.SLIDESHOW_API_KEY}&cronological&limit=4&offset=4'
        )
        ids1 = [i['id'] for i in resp1.json()['items']]
        ids2 = [i['id'] for i in resp2.json()['items']]
        expected = [img.id for img in images]
        self.assertEqual(ids1, expected[0:4])
        self.assertEqual(ids2, expected[4:8])

    def test_shuffled_pages_with_same_seed_never_repeat_an_id(self):
        images = self._make_images(20)
        resp1 = self.anon_client.get(
            f'/api/image_list/?key={settings.SLIDESHOW_API_KEY}&limit=10&offset=0'
        )
        seed = resp1.json()['seed']
        resp2 = self.anon_client.get(
            f'/api/image_list/?key={settings.SLIDESHOW_API_KEY}&limit=10&offset=10&seed={seed}'
        )
        ids1 = {i['id'] for i in resp1.json()['items']}
        ids2 = {i['id'] for i in resp2.json()['items']}
        self.assertEqual(len(ids1), 10)
        self.assertEqual(len(ids2), 10)
        self.assertEqual(ids1 & ids2, set())
        self.assertEqual(ids1 | ids2, {img.id for img in images})

    def test_not_ready_video_is_excluded_from_items_and_total_count(self):
        directory = Directory.objects.create(dir_path='/videos/pagination_test')
        VideoFile.objects.create(
            filename='/videos/pagination_test/pending.mov', directory=directory,
            width=640, height=480, duration_seconds=5, codec='hevc',
            # No transcoded_path, non-web-native codec - get_playback_url()
            # returns None, so this should never appear.
        )
        resp = self.anon_client.get(f'/api/image_list/?key={settings.SLIDESHOW_API_KEY}&cronological')
        data = resp.json()
        self.assertEqual(data['total_count'], 0)
        self.assertEqual(data['items'], [])

    def test_full_data_only_covers_the_current_page_not_the_whole_library(self):
        self._make_images(5)
        resp = self.anon_client.get(
            f'/api/image_list/?key={settings.SLIDESHOW_API_KEY}&full_data=true&cronological&limit=2'
        )
        data = resp.json()
        self.assertEqual(len(data['items']), 2)
        self.assertEqual(set(data['full_data'].keys()), {str(i['id']) for i in data['items']})
        for entry in data['full_data'].values():
            self.assertIn('sig', entry)


class SlideshowMediaSigTests(ApiTestCase):
    """KeyedImageView's 'slideshow' type and KeyedVideoView now require a
    per-id, time-boxed sig (minted by filteredImagesView) instead of the
    old shared, unscoped access_key - the actual fix for 'just increment
    the id' scraping, since a sig minted for one id is cryptographically
    useless for any other id."""

    def test_sig_minted_for_one_image_does_not_work_for_a_different_image(self):
        from api.views import _slideshow_current_sig
        # Two DISTINCT fixture files - make_image()'s default arg always
        # resolves to the same underlying row, which would make this test
        # vacuously pass (same id both times).
        image_a = self.make_image(relative_fixture="naming/good/1.JPG")
        image_b = self.make_image(relative_fixture="naming/good/2.jpg")
        sig_for_a = _slideshow_current_sig('image', image_a.id)
        resp = self.anon_client.get(f'/api/keyed_image/slideshow/?id={image_b.id}&sig={sig_for_a}')
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_correct_sig_for_the_right_image_is_accepted(self):
        from api.views import _slideshow_current_sig
        image = self.make_image()
        sig = _slideshow_current_sig('image', image.id)
        resp = self.anon_client.get(f'/api/keyed_image/slideshow/?id={image.id}&sig={sig}')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_old_global_access_key_no_longer_works_for_slideshow_type(self):
        image = self.make_image()
        resp = self.anon_client.get(
            f'/api/keyed_image/slideshow/?id={image.id}&access_key={settings.RANDOM_ACCESS_KEY}'
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_sig_outside_its_time_window_is_rejected(self):
        # NOTE: this module imports `time` from `datetime` (a time-of-day
        # type), not the stdlib `time` module - hence the `import time as
        # time_mod` below rather than a bare `import time`.
        import time as time_mod
        from api.views import _slideshow_current_sig, SLIDESHOW_TOKEN_BUCKET_SECONDS
        image = self.make_image()
        sig = _slideshow_current_sig('image', image.id)
        # Jump forward far enough that neither the current nor the
        # previous grace bucket still matches.
        future_time = time_mod.time() + 3 * SLIDESHOW_TOKEN_BUCKET_SECONDS
        with mock.patch('api.views.time.time', return_value=future_time):
            resp = self.anon_client.get(f'/api/keyed_image/slideshow/?id={image.id}&sig={sig}')
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

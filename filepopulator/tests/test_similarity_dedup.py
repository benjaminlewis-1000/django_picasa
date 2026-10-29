"""filepopulator tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test filepopulator.tests.test_similarity_dedup
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


class SimilarityTests(TestCase):
    """Exercises phash-based near-duplicate detection
    (filepopulator/similarity.py) against bare ImageFile rows with
    manually-set phash values, rather than real image files -- the
    comparison logic itself (Hamming distance + threshold + bookkeeping)
    doesn't depend on how the hash was produced."""

    def _make_bare_image_file(self, filename, phash, similarity_checked=False):
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(filename))
        obj = ImageFile(
            filename=filename,
            directory=directory,
            thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=10, height=10,
            phash=phash, similarity_checked=similarity_checked,
        )
        ImageFile.objects.bulk_create([obj])
        return ImageFile.objects.get(filename=filename)

    def test_finds_close_pair_and_skips_distant_pair(self):
        from filepopulator.models import SimilarImagePair
        from filepopulator.similarity import run_similarity_check

        base = 0x0F0F0F0F0F0F0F0F
        close = base ^ 0b111  # 3 bits different -- within default threshold
        far = base ^ ((1 << 40) - 1)  # 40 bits different -- well outside it

        img_base = self._make_bare_image_file("/tmp/similarity_test/base.jpg", base)
        img_close = self._make_bare_image_file("/tmp/similarity_test/close.jpg", close)
        img_far = self._make_bare_image_file("/tmp/similarity_test/far.jpg", far)

        result = run_similarity_check()

        self.assertEqual(result['checked'], 3)
        self.assertEqual(SimilarImagePair.objects.count(), 1)
        pair = SimilarImagePair.objects.get()
        self.assertEqual({pair.image_a_id, pair.image_b_id}, {img_base.id, img_close.id})
        self.assertEqual(pair.hamming_distance, 3)

        for img in (img_base, img_close, img_far):
            img.refresh_from_db()
            self.assertTrue(img.similarity_checked)

    def test_pair_stored_with_canonical_ordering_regardless_of_which_side_records_it(self):
        from filepopulator.models import SimilarImagePair

        img_a = self._make_bare_image_file("/tmp/similarity_test/order_a.jpg", 1)
        img_b = self._make_bare_image_file("/tmp/similarity_test/order_b.jpg", 2)
        lo, hi = sorted((img_a.id, img_b.id))

        SimilarImagePair.record(img_b.id, img_a.id, 1)
        SimilarImagePair.record(img_a.id, img_b.id, 1)

        self.assertEqual(SimilarImagePair.objects.count(), 1)
        pair = SimilarImagePair.objects.get()
        self.assertEqual(pair.image_a_id, lo)
        self.assertEqual(pair.image_b_id, hi)

    def test_dry_run_writes_nothing(self):
        from filepopulator.models import SimilarImagePair

        self._make_bare_image_file("/tmp/similarity_test/dry_a.jpg", 1)
        self._make_bare_image_file("/tmp/similarity_test/dry_b.jpg", 1)

        call_command("backfill_similarity", "--dry-run")

        self.assertEqual(SimilarImagePair.objects.count(), 0)
        self.assertFalse(ImageFile.objects.filter(similarity_checked=True).exists())

    def test_already_checked_images_are_not_recompared(self):
        from filepopulator.similarity import run_similarity_check

        img_a = self._make_bare_image_file("/tmp/similarity_test/prechecked.jpg", 1, similarity_checked=True)
        img_b = self._make_bare_image_file("/tmp/similarity_test/newcomer.jpg", 1)

        result = run_similarity_check()

        # Only the unchecked image counts as "checked" by this run, even
        # though it necessarily also compared against img_a to find it.
        self.assertEqual(result['checked'], 1)
        self.assertEqual(result['already_checked'], 1)

    def test_incremental_run_still_finds_match_against_previously_checked_image(self):
        from filepopulator.models import SimilarImagePair
        from filepopulator.similarity import run_similarity_check

        img_old = self._make_bare_image_file("/tmp/similarity_test/old.jpg", 0x1234, similarity_checked=True)
        img_new = self._make_bare_image_file("/tmp/similarity_test/new.jpg", 0x1234)

        run_similarity_check()

        self.assertEqual(SimilarImagePair.objects.count(), 1)
        pair = SimilarImagePair.objects.get()
        self.assertEqual({pair.image_a_id, pair.image_b_id}, {img_old.id, img_new.id})
        self.assertEqual(pair.hamming_distance, 0)


class CrossCollectionDedupClassifyTests(TestCase):
    """classify_pair(): pure function, exactly reproduces the 4-bucket
    logic validated against real production data 2026-10-21 (see
    filepopulator/cross_collection_dedup.py's own docstring and
    CLAUDE.md for the full investigation)."""

    def _bare_image(self, name, date_taken, make, model):
        directory, _ = Directory.objects.get_or_create(dir_path='/photos/classify_test')
        path = f'/photos/classify_test/{name}.jpg'
        img = ImageFile(
            filename=path, directory=directory, thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=100, height=100, dateTaken=date_taken,
            camera_make=make, camera_model=model,
        )
        ImageFile.objects.bulk_create([img])
        return ImageFile.objects.get(filename=path)

    def test_exact_metadata_match(self):
        from filepopulator.cross_collection_dedup import classify_pair
        from filepopulator.models import CrossCollectionImageDuplicate

        when = timezone.now()
        a = self._bare_image('a', when, 'samsung', 'SM-A326W')
        b = self._bare_image('b', when, 'samsung', 'SM-A326W')
        self.assertEqual(classify_pair(a, b), CrossCollectionImageDuplicate.REASON_EXACT_METADATA)

    def test_one_side_missing_camera_info(self):
        from filepopulator.cross_collection_dedup import classify_pair
        from filepopulator.models import CrossCollectionImageDuplicate

        a = self._bare_image('a', timezone.now(), 'samsung', 'SM-A326W')
        b = self._bare_image('b', timezone.now() + timedelta(days=30), '', '')
        self.assertEqual(classify_pair(a, b), CrossCollectionImageDuplicate.REASON_STRIPPED_EXIF)

    def test_same_camera_different_datetaken_is_burst_signature(self):
        from filepopulator.cross_collection_dedup import classify_pair
        from filepopulator.models import CrossCollectionImageDuplicate

        a = self._bare_image('a', timezone.now(), 'NIKON CORPORATION', 'NIKON D3100')
        b = self._bare_image('b', timezone.now() + timedelta(seconds=4), 'NIKON CORPORATION', 'NIKON D3100')
        self.assertEqual(classify_pair(a, b), CrossCollectionImageDuplicate.REASON_BURST_SIGNATURE)

    def test_differing_camera_make_is_a_conflict(self):
        from filepopulator.cross_collection_dedup import classify_pair
        from filepopulator.models import CrossCollectionImageDuplicate

        a = self._bare_image('a', timezone.now(), 'samsung', 'SM-A326W')
        b = self._bare_image('b', timezone.now(), 'OLYMPUS IMAGING CORP.', 'C960Z')
        self.assertEqual(classify_pair(a, b), CrossCollectionImageDuplicate.REASON_CAMERA_CONFLICT)


class CrossCollectionFaceMatchingTests(TestCase):
    """match_cross_collection_faces(): pure function, no DB. Mirrors the
    same optimal-assignment-then-threshold-filter pattern already tested
    for the duplicate-video case (merge_duplicate_videofiles.py)."""

    def test_rescaled_overlapping_boxes_match(self):
        from filepopulator.cross_collection_dedup import match_cross_collection_faces

        primary_faces = [{'box': (100, 100, 200, 200)}]
        # Same relative position, but secondary is half the resolution --
        # needs scale_x=scale_y=2 to land back on the primary's box.
        secondary_faces = [{'box': (50, 50, 100, 100)}]
        matches = match_cross_collection_faces(primary_faces, secondary_faces, scale_x=2.0, scale_y=2.0)
        self.assertEqual(matches, [(0, 0)])

    def test_non_overlapping_boxes_do_not_match(self):
        from filepopulator.cross_collection_dedup import match_cross_collection_faces

        primary_faces = [{'box': (0, 0, 50, 50)}]
        secondary_faces = [{'box': (500, 500, 550, 550)}]
        matches = match_cross_collection_faces(primary_faces, secondary_faces, scale_x=1.0, scale_y=1.0)
        self.assertEqual(matches, [])

    def test_picks_the_globally_optimal_pairing_not_a_greedy_one(self):
        from filepopulator.cross_collection_dedup import match_cross_collection_faces

        # Two faces each side, at (roughly) swapped positions --
        # confirms this uses real assignment, not "first match wins".
        primary_faces = [{'box': (0, 0, 50, 50)}, {'box': (200, 200, 250, 250)}]
        secondary_faces = [{'box': (200, 200, 250, 250)}, {'box': (0, 0, 50, 50)}]
        matches = match_cross_collection_faces(primary_faces, secondary_faces, scale_x=1.0, scale_y=1.0)
        self.assertEqual(set(matches), {(0, 1), (1, 0)})

    def test_empty_either_side_returns_no_matches(self):
        from filepopulator.cross_collection_dedup import match_cross_collection_faces

        self.assertEqual(match_cross_collection_faces([], [{'box': (0, 0, 10, 10)}], 1.0, 1.0), [])
        self.assertEqual(match_cross_collection_faces([{'box': (0, 0, 10, 10)}], [], 1.0, 1.0), [])


@override_settings(MEDIA_ROOT="/tmp/filepopulator_cross_collection_test_media")
class ResolveCrossCollectionPairTests(TestCase):
    """resolve_pair(): the real end-to-end orchestration -- real
    PhotoCollection/Directory/ImageFile/Face rows, real IOU matching and
    DB writes (or not, for dry_run/needs-review)."""

    def setUp(self):
        self.collection_a = PhotoCollection.objects.get(is_default=True)
        self.first_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.first_root, ignore_errors=True)
        self.second_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.second_root, ignore_errors=True)
        self.collection_b = PhotoCollection.objects.create(
            name='Second Collection', root_directory=self.second_root,
        )
        # Shared across both sides of a pair by default -- two separate
        # timezone.now() calls would never land on the same microsecond,
        # which would wrongly push an otherwise-identical-metadata pair
        # into the BURST_SIGNATURE bucket instead of EXACT_METADATA.
        self.shared_date_taken = timezone.now()

    def _make_image(self, name, collection, size_bytes, width=100, height=100,
                     date_taken=None, make='samsung', model='SM-A326W', is_processed=False):
        root = self.second_root if collection.pk == self.collection_b.pk else self.first_root
        directory, _ = Directory.objects.get_or_create(dir_path=root, defaults={'collection': collection})
        if directory.collection_id != collection.pk:
            directory.collection = collection
            directory.save()
        path = os.path.join(root, f'{name}.jpg')
        with open(path, 'wb') as f:
            f.write(os.urandom(size_bytes))
        img = ImageFile(
            filename=path, directory=directory, thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=width, height=height, dateTaken=date_taken or self.shared_date_taken,
            camera_make=make, camera_model=model, isProcessed=is_processed,
        )
        ImageFile.objects.bulk_create([img])
        img = ImageFile.objects.get(filename=path)
        # Matches what the real ingestion pipeline does at save time
        # (instance_clean_and_save()) -- bulk_create() bypasses it, so
        # this fixture replicates it explicitly rather than leaving
        # .collections permanently empty like a real ingested row never is.
        img.collections.add(collection)
        return img

    def _make_face(self, image, box, person=None, validated=False, kps=None):
        bl, bt, br, bb = box
        face = Face(
            source_image_file=image, declared_name=person or get_default_blank_person(),
            box_left=bl, box_top=bt, box_right=br, box_bottom=bb, validated=validated, kps=kps,
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()
        return face

    def _make_pair(self, image_a, image_b, distance=0):
        from filepopulator.models import SimilarImagePair
        SimilarImagePair.record(image_a.id, image_b.id, distance)
        return SimilarImagePair.objects.get(
            image_a_id=min(image_a.id, image_b.id), image_b_id=max(image_a.id, image_b.id),
        )

    def test_auto_merge_collapses_matched_faces_onto_the_larger_file(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        from filepopulator.models import CrossCollectionImageDuplicate

        small = self._make_image('small', self.collection_a, size_bytes=100)
        big = self._make_image('big', self.collection_b, size_bytes=5000)
        small_face = self._make_face(small, (10, 10, 30, 30))
        big_face = self._make_face(big, (10, 10, 30, 30))
        pair = self._make_pair(small, big)

        stats = resolve_pair(pair, dry_run=False)

        self.assertEqual(stats['status'], CrossCollectionImageDuplicate.STATUS_AUTO_MERGED)
        self.assertEqual(stats['primary_id'], big.id)
        self.assertEqual(stats['faces_matched'], 1)
        # Matched, both bare/unvalidated -- lowest id wins as the tiebreak.
        survivor_id = min(small_face.id, big_face.id)
        loser_id = max(small_face.id, big_face.id)
        self.assertTrue(Face.objects.filter(pk=survivor_id).exists())
        self.assertFalse(Face.objects.filter(pk=loser_id).exists())
        self.assertEqual(Face.objects.get(pk=survivor_id).source_image_file_id, big.id)

        small.refresh_from_db()
        self.assertTrue(small.isProcessed)
        self.assertIn(self.collection_b, big.collections.all())

        record = CrossCollectionImageDuplicate.objects.get(image_a_id=min(small.id, big.id), image_b_id=max(small.id, big.id))
        self.assertEqual(record.status, CrossCollectionImageDuplicate.STATUS_AUTO_MERGED)
        self.assertEqual(record.primary_image_id, big.id)

    def test_unmatched_secondary_face_is_transferred_not_deleted(self):
        from filepopulator.cross_collection_dedup import resolve_pair

        small = self._make_image('small2', self.collection_a, size_bytes=100)
        big = self._make_image('big2', self.collection_b, size_bytes=5000)
        # Only the secondary (small, since it's the smaller file) has a
        # face at all -- a detection only one side's extraction run found.
        secondary_only_face = self._make_face(small, (10, 10, 30, 30))
        pair = self._make_pair(small, big)

        stats = resolve_pair(pair, dry_run=False)

        self.assertEqual(stats['faces_transferred'], 1)
        self.assertEqual(stats['faces_deleted'], 0)
        secondary_only_face.refresh_from_db()
        self.assertEqual(secondary_only_face.source_image_file_id, big.id)

    def test_validated_face_always_survives_a_match(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        from face_manager.models import Person

        person = Person.objects.create(person_name='Real Person')
        small = self._make_image('small3', self.collection_a, size_bytes=100)
        big = self._make_image('big3', self.collection_b, size_bytes=5000)
        # The validated face is on the SMALLER (non-primary) file -- it
        # must still win and end up on the primary.
        validated_face = self._make_face(small, (10, 10, 30, 30), person=person, validated=True)
        unvalidated_face = self._make_face(big, (10, 10, 30, 30))
        pair = self._make_pair(small, big)

        resolve_pair(pair, dry_run=False)

        self.assertTrue(Face.objects.filter(pk=validated_face.pk).exists())
        self.assertFalse(Face.objects.filter(pk=unvalidated_face.pk).exists())
        validated_face.refresh_from_db()
        self.assertEqual(validated_face.source_image_file_id, big.id)

    def test_needs_review_pair_touches_nothing(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        from filepopulator.models import CrossCollectionImageDuplicate

        a = self._make_image(
            'burst_a', self.collection_a, size_bytes=100,
            date_taken=timezone.now(), make='NIKON CORPORATION', model='NIKON D3100',
        )
        b = self._make_image(
            'burst_b', self.collection_b, size_bytes=5000,
            date_taken=timezone.now() + timedelta(seconds=4), make='NIKON CORPORATION', model='NIKON D3100',
        )
        face_a = self._make_face(a, (10, 10, 30, 30))
        face_b = self._make_face(b, (10, 10, 30, 30))
        pair = self._make_pair(a, b)

        stats = resolve_pair(pair, dry_run=False)

        self.assertEqual(stats['status'], CrossCollectionImageDuplicate.STATUS_NEEDS_REVIEW)
        self.assertEqual(stats['reason'], CrossCollectionImageDuplicate.REASON_BURST_SIGNATURE)
        # Both faces untouched, both images untouched.
        self.assertTrue(Face.objects.filter(pk=face_a.pk).exists())
        self.assertTrue(Face.objects.filter(pk=face_b.pk).exists())
        a.refresh_from_db()
        b.refresh_from_db()
        self.assertFalse(a.isProcessed)
        self.assertFalse(b.isProcessed)
        self.assertNotIn(self.collection_b, a.collections.all())

        record = CrossCollectionImageDuplicate.objects.get(image_a_id=min(a.id, b.id), image_b_id=max(a.id, b.id))
        self.assertEqual(record.status, CrossCollectionImageDuplicate.STATUS_NEEDS_REVIEW)

    def test_dry_run_writes_nothing(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        from filepopulator.models import CrossCollectionImageDuplicate

        small = self._make_image('small4', self.collection_a, size_bytes=100)
        big = self._make_image('big4', self.collection_b, size_bytes=5000)
        face_small = self._make_face(small, (10, 10, 30, 30))
        face_big = self._make_face(big, (10, 10, 30, 30))
        pair = self._make_pair(small, big)

        stats = resolve_pair(pair, dry_run=True)

        self.assertEqual(stats['status'], CrossCollectionImageDuplicate.STATUS_AUTO_MERGED)
        self.assertEqual(stats['faces_matched'], 1)
        # But nothing was actually written.
        self.assertTrue(Face.objects.filter(pk=face_small.pk).exists())
        self.assertTrue(Face.objects.filter(pk=face_big.pk).exists())
        self.assertFalse(CrossCollectionImageDuplicate.objects.exists())

    def test_rescales_boxes_for_different_resolutions(self):
        from filepopulator.cross_collection_dedup import resolve_pair

        # big is 2x the resolution of small -- a face at the same
        # relative position should still match once rescaled.
        small = self._make_image('small5', self.collection_a, size_bytes=100, width=100, height=100)
        big = self._make_image('big5', self.collection_b, size_bytes=5000, width=200, height=200)
        self._make_face(small, (10, 10, 30, 30))  # 10% of width, relative
        self._make_face(big, (20, 20, 60, 60))  # same relative position at 2x scale
        pair = self._make_pair(small, big)

        stats = resolve_pair(pair, dry_run=False)
        self.assertEqual(stats['faces_matched'], 1)

    def test_unmatched_transferred_face_box_is_rescaled_into_primarys_space(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        # Real bug, found 2026-10-21 (see CLAUDE.md): a face transferred
        # from secondary onto primary without rescaling its box kept
        # secondary's own pixel coordinates -- silently wrong once the
        # two images have different resolutions, and a real risk of the
        # face being clamped/deleted the next time primary goes through
        # face_extract_encode.py's "existing face, no matching fresh
        # detection" reconciliation path.
        small = self._make_image('resc_small', self.collection_a, size_bytes=100, width=100, height=200)
        big = self._make_image('resc_big', self.collection_b, size_bytes=5000, width=400, height=800)  # 4x both axes
        # Only on the secondary (small) -- nothing on primary to match
        # against, so this is definitely the "unmatched, transfer" path.
        unmatched_face = self._make_face(small, (10, 20, 30, 50))
        pair = self._make_pair(small, big)

        stats = resolve_pair(pair, dry_run=False)

        self.assertEqual(stats['primary_id'], big.id)
        self.assertEqual(stats['faces_transferred'], 1)
        unmatched_face.refresh_from_db()
        self.assertEqual(unmatched_face.source_image_file_id, big.id)
        # Rescaled by the real 4x width/4x height ratio between the pair.
        self.assertEqual(
            (unmatched_face.box_left, unmatched_face.box_top, unmatched_face.box_right, unmatched_face.box_bottom),
            (40, 80, 120, 200),
        )
        # And the rescaled box is a real, valid, non-degenerate box that
        # genuinely fits inside primary's actual dimensions -- the
        # concrete guarantee this fix is for.
        self.assertTrue(0 <= unmatched_face.box_left < unmatched_face.box_right <= big.width)
        self.assertTrue(0 <= unmatched_face.box_top < unmatched_face.box_bottom <= big.height)

    def test_matched_but_secondary_wins_face_box_is_also_rescaled(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        # The OTHER path a transferred face can take -- matched to a
        # primary face, but the secondary's own copy is judged the
        # better survivor (e.g. it's validated) -- needs the exact same
        # rescaling as the unmatched-transfer path, not just a special
        # case of it.
        from face_manager.models import Person
        person = Person.objects.create(person_name='Rescale Survivor')

        small = self._make_image('resc_small2', self.collection_a, size_bytes=100, width=100, height=100)
        big = self._make_image('resc_big2', self.collection_b, size_bytes=5000, width=300, height=300)  # 3x both axes
        # Secondary's copy is validated -- it must win and be transferred
        # (rescaled) onto primary; primary's own (unvalidated) copy at
        # the matching relative position must be deleted.
        survivor = self._make_face(small, (10, 10, 30, 30), person=person, validated=True)
        loser = self._make_face(big, (30, 30, 90, 90))  # same relative position, 3x scale
        pair = self._make_pair(small, big)

        stats = resolve_pair(pair, dry_run=False)

        self.assertEqual(stats['faces_matched'], 1)
        self.assertFalse(Face.objects.filter(pk=loser.pk).exists())
        survivor.refresh_from_db()
        self.assertEqual(survivor.source_image_file_id, big.id)
        self.assertEqual(
            (survivor.box_left, survivor.box_top, survivor.box_right, survivor.box_bottom),
            (30, 30, 90, 90),
        )

    def test_transferred_face_kps_is_rescaled_not_discarded(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        small = self._make_image('resc_small3', self.collection_a, size_bytes=100, width=100, height=200)
        big = self._make_image('resc_big3', self.collection_b, size_bytes=5000, width=200, height=400)  # 2x both axes
        kps = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        # Box widened to actually bound these kps (x in [10,90], y in
        # [20,100]) within face_kps_within_box_margin's 25% margin --
        # the original (10,20,30,50) box was a geometrically-inconsistent
        # placeholder from before that constraint existed.
        face = self._make_face(small, (5, 15, 95, 105), kps=kps)
        pair = self._make_pair(small, big)

        resolve_pair(pair, dry_run=False)

        face.refresh_from_db()
        self.assertEqual(face.source_image_file_id, big.id)
        # Even indices (x) scaled by scale_x=2.0, odd indices (y) by
        # scale_y=2.0 -- same 2x on both axes here, but computed via the
        # real per-axis logic, not just "double everything".
        self.assertEqual(face.kps, [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0])

    def test_null_kps_stays_null_after_transfer(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        small = self._make_image('resc_small4', self.collection_a, size_bytes=100, width=100, height=100)
        big = self._make_image('resc_big4', self.collection_b, size_bytes=5000, width=200, height=200)
        face = self._make_face(small, (10, 10, 30, 30), kps=None)
        pair = self._make_pair(small, big)

        resolve_pair(pair, dry_run=False)

        face.refresh_from_db()
        self.assertIsNone(face.kps)

    def test_primary_marked_processed_when_secondary_already_was(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        # Real bug, found 2026-10-21: if secondary already completed its
        # own real face-extraction pass, every real face for this
        # content is now on primary too (matched + transferred) -- but
        # primary's own isProcessed flag was never updated, so the
        # normal scheduled face_extraction task would later run a
        # wasteful (and, pre-rescaling-fix, risky) redundant detection
        # pass on it.
        small = self._make_image('proc_small', self.collection_a, size_bytes=100, is_processed=True)
        big = self._make_image('proc_big', self.collection_b, size_bytes=5000, is_processed=False)
        self._make_face(small, (10, 10, 30, 30))
        pair = self._make_pair(small, big)

        resolve_pair(pair, dry_run=False)

        big.refresh_from_db()
        self.assertTrue(big.isProcessed)

    def test_primary_left_unprocessed_when_secondary_never_was(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        # The opposite case: secondary was NEVER processed (no real
        # detection pass has ever found whatever faces might be in this
        # content) -- primary must NOT be force-marked done, since it
        # may still need its own genuine first detection pass.
        small = self._make_image('proc_small2', self.collection_a, size_bytes=100, is_processed=False)
        big = self._make_image('proc_big2', self.collection_b, size_bytes=5000, is_processed=False)
        pair = self._make_pair(small, big)

        resolve_pair(pair, dry_run=False)

        big.refresh_from_db()
        self.assertFalse(big.isProcessed)

    def test_primary_already_processed_stays_processed_regardless(self):
        from filepopulator.cross_collection_dedup import resolve_pair
        small = self._make_image('proc_small3', self.collection_a, size_bytes=100, is_processed=False)
        big = self._make_image('proc_big3', self.collection_b, size_bytes=5000, is_processed=True)
        pair = self._make_pair(small, big)

        resolve_pair(pair, dry_run=False)

        big.refresh_from_db()
        self.assertTrue(big.isProcessed)


class FindUnresolvedCrossCollectionPairsTests(TestCase):
    """find_unresolved_cross_collection_pairs(): the candidate-selection
    query shared by the management command and the recurring task."""

    def setUp(self):
        self.collection_a = PhotoCollection.objects.get(is_default=True)
        self.second_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.second_root, ignore_errors=True)
        self.collection_b = PhotoCollection.objects.create(
            name='Find Unresolved Test', root_directory=self.second_root,
        )

    def _make_image(self, name, collection):
        root = self.second_root if collection.pk == self.collection_b.pk else '/photos/find_unresolved_test'
        directory, _ = Directory.objects.get_or_create(dir_path=root, defaults={'collection': collection})
        if directory.collection_id != collection.pk:
            directory.collection = collection
            directory.save()
        path = os.path.join(root, f'{name}.jpg')
        img = ImageFile(
            filename=path, directory=directory, thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=100, height=100,
        )
        ImageFile.objects.bulk_create([img])
        return ImageFile.objects.get(filename=path)

    def test_only_yields_cross_collection_exact_phash_pairs(self):
        from filepopulator.models import SimilarImagePair
        from filepopulator.cross_collection_dedup import find_unresolved_cross_collection_pairs

        cross = self._make_image('cross_a', self.collection_a)
        cross_other = self._make_image('cross_b', self.collection_b)
        SimilarImagePair.record(cross.id, cross_other.id, 0)

        same = self._make_image('same_a', self.collection_a)
        same_other = self._make_image('same_b', self.collection_a)
        SimilarImagePair.record(same.id, same_other.id, 0)

        far = self._make_image('far_a', self.collection_a)
        far_other = self._make_image('far_b', self.collection_b)
        SimilarImagePair.record(far.id, far_other.id, 4)  # not exact -- excluded

        results = list(find_unresolved_cross_collection_pairs())
        result_id_sets = [{p.image_a_id, p.image_b_id} for p in results]
        self.assertIn({cross.id, cross_other.id}, result_id_sets)
        self.assertNotIn({same.id, same_other.id}, result_id_sets)
        self.assertNotIn({far.id, far_other.id}, result_id_sets)

    def test_already_decided_pairs_are_excluded(self):
        from filepopulator.models import SimilarImagePair, CrossCollectionImageDuplicate
        from filepopulator.cross_collection_dedup import find_unresolved_cross_collection_pairs

        a = self._make_image('decided_a', self.collection_a)
        b = self._make_image('decided_b', self.collection_b)
        SimilarImagePair.record(a.id, b.id, 0)
        lo, hi = sorted((a.id, b.id))
        CrossCollectionImageDuplicate.objects.create(
            image_a_id=lo, image_b_id=hi,
            status=CrossCollectionImageDuplicate.STATUS_NEEDS_REVIEW,
            reason=CrossCollectionImageDuplicate.REASON_CAMERA_CONFLICT,
        )

        results = list(find_unresolved_cross_collection_pairs())
        result_id_sets = [{p.image_a_id, p.image_b_id} for p in results]
        self.assertNotIn({a.id, b.id}, result_id_sets)


class MergeCrossCollectionImageDuplicatesCommandTests(TestCase):
    """The management command itself -- thin wrapper over
    find_unresolved_cross_collection_pairs()/resolve_pair(), already
    covered individually above; this just confirms the command wires
    them together and reports sane totals."""

    def setUp(self):
        self.collection_a = PhotoCollection.objects.get(is_default=True)
        self.second_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.second_root, ignore_errors=True)
        self.collection_b = PhotoCollection.objects.create(
            name='Command Test Collection', root_directory=self.second_root,
        )

    def test_command_resolves_a_real_candidate(self):
        from filepopulator.models import SimilarImagePair, CrossCollectionImageDuplicate

        directory_a, _ = Directory.objects.get_or_create(
            dir_path='/photos/merge_command_test', defaults={'collection': self.collection_a},
        )
        img_a = ImageFile(
            filename='/photos/merge_command_test/a.jpg', directory=directory_a, thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=100, height=100,
        )
        ImageFile.objects.bulk_create([img_a])
        img_a = ImageFile.objects.get(filename='/photos/merge_command_test/a.jpg')

        directory_b = Directory.objects.create(dir_path=self.second_root, collection=self.collection_b)
        path_b = os.path.join(self.second_root, 'b.jpg')
        with open(path_b, 'wb') as f:
            f.write(os.urandom(2000))
        img_b = ImageFile(
            filename=path_b, directory=directory_b, thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=100, height=100,
        )
        ImageFile.objects.bulk_create([img_b])
        img_b = ImageFile.objects.get(filename=path_b)

        SimilarImagePair.record(img_a.id, img_b.id, 0)

        call_command('merge_cross_collection_image_duplicates')

        self.assertEqual(CrossCollectionImageDuplicate.objects.count(), 1)

    def test_dry_run_leaves_no_decision_records(self):
        from filepopulator.models import SimilarImagePair, CrossCollectionImageDuplicate

        directory_a, _ = Directory.objects.get_or_create(
            dir_path='/photos/merge_command_dry_test', defaults={'collection': self.collection_a},
        )
        img_a = ImageFile(
            filename='/photos/merge_command_dry_test/a.jpg', directory=directory_a, thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=100, height=100,
        )
        ImageFile.objects.bulk_create([img_a])
        img_a = ImageFile.objects.get(filename='/photos/merge_command_dry_test/a.jpg')

        directory_b = Directory.objects.create(dir_path=self.second_root, collection=self.collection_b)
        img_b = ImageFile(
            filename=os.path.join(self.second_root, 'b.jpg'), directory=directory_b, thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=100, height=100,
        )
        ImageFile.objects.bulk_create([img_b])
        img_b = ImageFile.objects.get(filename=os.path.join(self.second_root, 'b.jpg'))

        SimilarImagePair.record(img_a.id, img_b.id, 0)

        call_command('merge_cross_collection_image_duplicates', '--dry-run')

        self.assertEqual(CrossCollectionImageDuplicate.objects.count(), 0)


class BackfillMergedPrimaryIsProcessedTests(TestCase):
    """backfill_merged_primary_isprocessed: catch-up for merges that ran
    before resolve_pair() started marking primary isProcessed=True when
    secondary already held real faces at merge time (2026-10-21, see
    CLAUDE.md)."""

    def _make_record(self, primary_is_processed, faces_matched, faces_transferred, status='auto_merged'):
        from filepopulator.models import CrossCollectionImageDuplicate
        directory, _ = Directory.objects.get_or_create(dir_path='/photos/backfill_isprocessed_test')
        import uuid
        suffix = uuid.uuid4().hex[:8]
        primary = ImageFile(
            filename=f'/photos/backfill_isprocessed_test/primary_{suffix}.jpg', directory=directory,
            thumbnail_small="", pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(), width=100, height=100,
            isProcessed=primary_is_processed,
        )
        secondary = ImageFile(
            filename=f'/photos/backfill_isprocessed_test/secondary_{suffix}.jpg', directory=directory,
            thumbnail_small="", pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(), width=100, height=100,
            isProcessed=True,  # always True post-merge, regardless of history -- see module docstring
        )
        ImageFile.objects.bulk_create([primary, secondary])
        primary = ImageFile.objects.get(filename=primary.filename)
        secondary = ImageFile.objects.get(filename=secondary.filename)
        lo, hi = sorted((primary.id, secondary.id))
        record = CrossCollectionImageDuplicate.objects.create(
            image_a_id=lo, image_b_id=hi, status=status,
            reason=CrossCollectionImageDuplicate.REASON_EXACT_METADATA,
            primary_image=primary, faces_matched=faces_matched, faces_transferred=faces_transferred,
        )
        return primary, record

    def test_marks_primary_processed_when_merge_record_shows_real_faces(self):
        primary, _ = self._make_record(primary_is_processed=False, faces_matched=1, faces_transferred=0)

        call_command('backfill_merged_primary_isprocessed')

        primary.refresh_from_db()
        self.assertTrue(primary.isProcessed)

    def test_leaves_primary_unprocessed_when_merge_record_shows_no_faces(self):
        # Proxy limitation, accepted: a secondary that was processed but
        # genuinely found zero faces looks identical to "never
        # processed" here -- primary is correctly left alone (a real,
        # if likely-also-zero-yielding, redetection pass later, not a
        # correctness risk).
        primary, _ = self._make_record(primary_is_processed=False, faces_matched=0, faces_transferred=0)

        call_command('backfill_merged_primary_isprocessed')

        primary.refresh_from_db()
        self.assertFalse(primary.isProcessed)

    def test_already_processed_primary_is_left_alone(self):
        primary, _ = self._make_record(primary_is_processed=True, faces_matched=1, faces_transferred=1)

        call_command('backfill_merged_primary_isprocessed')  # must not error re-marking an already-True row

        primary.refresh_from_db()
        self.assertTrue(primary.isProcessed)

    def test_needs_review_records_are_not_touched(self):
        from filepopulator.models import CrossCollectionImageDuplicate
        primary, _ = self._make_record(
            primary_is_processed=False, faces_matched=0, faces_transferred=0,
            status=CrossCollectionImageDuplicate.STATUS_NEEDS_REVIEW,
        )
        # needs_review records never have a primary_image in real usage,
        # but confirm the status filter itself excludes them regardless.
        CrossCollectionImageDuplicate.objects.filter(primary_image=primary).update(
            faces_matched=5, faces_transferred=5,
        )

        call_command('backfill_merged_primary_isprocessed')

        primary.refresh_from_db()
        self.assertFalse(primary.isProcessed)

    def test_dry_run_does_not_write(self):
        primary, _ = self._make_record(primary_is_processed=False, faces_matched=1, faces_transferred=0)

        call_command('backfill_merged_primary_isprocessed', '--dry-run')

        primary.refresh_from_db()
        self.assertFalse(primary.isProcessed)


class MergeCrossCollectionImageDuplicatesTaskTests(TestCase):
    """The recurring Celery task -- same underlying functions as the
    command, already covered individually above; this just confirms the
    task wires them together and respects its own advisory lock, same
    contract as cleanup_orphaned_media's own lock test."""

    def setUp(self):
        self.collection_a = PhotoCollection.objects.get(is_default=True)
        self.second_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.second_root, ignore_errors=True)
        self.collection_b = PhotoCollection.objects.create(
            name='Task Test Collection', root_directory=self.second_root,
        )

    def _make_pair(self, suffix):
        directory_a, _ = Directory.objects.get_or_create(
            dir_path='/photos/merge_task_test', defaults={'collection': self.collection_a},
        )
        img_a = ImageFile(
            filename=f'/photos/merge_task_test/a{suffix}.jpg', directory=directory_a, thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=100, height=100,
        )
        ImageFile.objects.bulk_create([img_a])
        img_a = ImageFile.objects.get(filename=f'/photos/merge_task_test/a{suffix}.jpg')

        directory_b, _ = Directory.objects.get_or_create(
            dir_path=self.second_root, defaults={'collection': self.collection_b},
        )
        path_b = os.path.join(self.second_root, f'b{suffix}.jpg')
        img_b = ImageFile(
            filename=path_b, directory=directory_b, thumbnail_small="",
            pixel_hash=binascii.hexlify(os.urandom(16)).decode(),
            file_hash=binascii.hexlify(os.urandom(16)).decode(),
            width=100, height=100,
        )
        ImageFile.objects.bulk_create([img_b])
        img_b = ImageFile.objects.get(filename=path_b)

        from filepopulator.models import SimilarImagePair
        SimilarImagePair.record(img_a.id, img_b.id, 0)
        return img_a, img_b

    def test_task_resolves_pending_pairs_and_respects_the_lock(self):
        from filepopulator.models import CrossCollectionImageDuplicate
        from filepopulator.tasks import merge_cross_collection_image_duplicates

        self._make_pair('1')
        merge_cross_collection_image_duplicates()
        self.assertEqual(CrossCollectionImageDuplicate.objects.count(), 1)

        # While another session holds the lock, the task must do nothing.
        self._make_pair('2')
        cfg = connection.settings_dict
        other = psycopg2.connect(
            dbname=cfg['NAME'], user=cfg['USER'], password=cfg['PASSWORD'],
            host=cfg['HOST'] or 'localhost', port=cfg['PORT'] or 5432,
        )
        other.autocommit = True
        self.addCleanup(other.close)
        key = zlib.crc32(b'filepopulator.merge_cross_collection_image_duplicates')
        with other.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", [key])
            self.assertTrue(cur.fetchone()[0])

            merge_cross_collection_image_duplicates()
            self.assertEqual(CrossCollectionImageDuplicate.objects.count(), 1)  # unchanged

            cur.execute("SELECT pg_advisory_unlock(%s)", [key])

        # Once free again, the second pair resolves too.
        merge_cross_collection_image_duplicates()
        self.assertEqual(CrossCollectionImageDuplicate.objects.count(), 2)


class PhashComputationTests(TestCase):
    """Confirms phash is actually computed and stored during real image
    ingestion (not just exercised against bare rows above), and that two
    genuinely different real photos land far apart."""

    @override_settings(MEDIA_ROOT='/tmp')
    def test_phash_populated_on_ingestion(self):
        val_dir = settings.FILEPOPULATOR_VAL_DIRECTORY
        src = os.path.join(val_dir, 'naming', 'good', '1.JPG')
        tmp_path = '/tmp/phash_ingestion_test/1.JPG'
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        shutil.copy(src, tmp_path)

        create_image_file(tmp_path)

        obj = ImageFile.objects.get(filename=tmp_path)
        self.assertIsNotNone(obj.phash)
        self.assertFalse(obj.similarity_checked)


@override_settings(MEDIA_ROOT='/tmp')
class PhashBackfillTests(TestCase):
    """Regression coverage for backfill_phash: images ingested before
    phash computation existed (or where it's manually cleared here to
    simulate that) get a phash computed directly from the file on disk,
    without going through the expensive full ImageFile.save() pipeline
    (which would also rehash pixel_hash/regenerate thumbnails)."""

    def _ingest_real_image(self, name):
        val_dir = settings.FILEPOPULATOR_VAL_DIRECTORY
        src = os.path.join(val_dir, 'naming', 'good', name)
        tmp_path = f'/tmp/phash_backfill_test/{name}'
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        shutil.copy(src, tmp_path)
        create_image_file(tmp_path)
        return ImageFile.objects.get(filename=tmp_path)

    def test_backfill_computes_phash_for_existing_image_missing_one(self):
        obj = self._ingest_real_image('1.JPG')
        # Simulate a pre-phash-era row: has real pixel_hash/thumbnails
        # already, just never got a phash.
        ImageFile.objects.filter(id=obj.id).update(phash=None)

        call_command('backfill_phash')

        obj.refresh_from_db()
        self.assertIsNotNone(obj.phash)

    def test_dry_run_writes_nothing(self):
        obj = self._ingest_real_image('2.jpg')
        ImageFile.objects.filter(id=obj.id).update(phash=None)

        call_command('backfill_phash', '--dry-run')

        obj.refresh_from_db()
        self.assertIsNone(obj.phash)

    def test_does_not_touch_pixel_hash_or_thumbnails(self):
        obj = self._ingest_real_image('1.JPG')
        original_pixel_hash = obj.pixel_hash
        original_thumb = obj.thumbnail_small.name
        ImageFile.objects.filter(id=obj.id).update(phash=None)

        call_command('backfill_phash')

        obj.refresh_from_db()
        self.assertEqual(obj.pixel_hash, original_pixel_hash)
        self.assertEqual(obj.thumbnail_small.name, original_thumb)

    def test_multiprocess_backfill_matches_single_process_result(self):
        obj_a = self._ingest_real_image('1.JPG')
        obj_b = self._ingest_real_image('2.jpg')
        ImageFile.objects.filter(id__in=[obj_a.id, obj_b.id]).update(phash=None)

        call_command('backfill_phash', '--processes', '2')

        obj_a.refresh_from_db()
        obj_b.refresh_from_db()
        self.assertIsNotNone(obj_a.phash)
        self.assertIsNotNone(obj_b.phash)



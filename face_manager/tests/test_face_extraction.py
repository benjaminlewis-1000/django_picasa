"""face_manager tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test face_manager.tests.test_face_extraction
"""
import io
import os
import unittest
from contextlib import contextmanager
import zlib
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import psycopg2
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.db import connection, transaction
from django.db.utils import IntegrityError
from django.test import TestCase, TransactionTestCase, override_settings, tag

from django.core.management import call_command
from django.core.management.base import CommandError

from unittest.mock import patch, MagicMock

from face_manager.assign_faces import faceAssigner
from face_manager.face_extract_encode import FaceExtractor
from face_manager.live_counts import compute_live_face_counts
from face_manager.models import Face, Person, get_default_blank_person, clear_confirmed_ignore_face_encodings
from face_manager.pyramidal_detector import PyramidalDetector
from face_manager.test_face_cache import cached_detect
from face_manager.verification_clustering import cluster_all_unverified_faces
from filepopulator.models import Directory, DuplicateFile, ImageFile
from filepopulator.scripts import create_image_file

from face_manager.tests import make_face, make_person, make_image, make_preexisting_image_row, _tiny_jpeg_bytes, _embedding


@tag("slow")
@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class FaceExtractorCorruptedImageTests(TestCase):
    """Characterizes current behavior against the 5 real corrupted JPEGs
    pulled from production logs (see
    /mnt/fast_storage/appdata/django_picasa/test_suite/corrupted_images/NOTES.md).
    Loads the real FaceAnalysis model (not mocked, not cached -- these
    fail before reaching detection, so there's no inference cost)."""

    CORRUPTED_DIR = "/photos/corrupted"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.extractor = FaceExtractor()

    def test_corrupted_images_get_marked_processed_and_flagged(self):
        # Regression test for a fixed bug: the `except Exception: ...
        # continue` branch that catches the PIL "image file is truncated" /
        # "broken data stream" errors used to never set
        # img_obj.isProcessed = True, so these files were retried by the
        # scheduled face_extraction task on every run, forever. Now marks
        # isProcessed (stop retrying) and image_load_failed/
        # image_load_error (so the failure is recorded instead of just
        # silently no-op'd -- see the planned frontend "failed to open"
        # list in CLAUDE.md). Uses ImageFile.objects.filter(...).update()
        # rather than img_obj.save(), since save() would itself re-decode
        # the (still corrupted) image via _generate_md5_hash() and raise.
        corrupted_files = sorted(os.listdir(self.CORRUPTED_DIR))
        self.assertEqual(len(corrupted_files), 5)

        image_objs = []
        for fname in corrupted_files:
            path = os.path.join(self.CORRUPTED_DIR, fname)
            image_objs.append(make_preexisting_image_row(path))

        self.extractor.find_and_encode_faces()

        for img_obj in image_objs:
            img_obj.refresh_from_db()
            self.assertTrue(
                img_obj.isProcessed,
                f"{img_obj.filename} was not marked processed -- would be "
                "retried by face_extraction on every future run.",
            )
            self.assertTrue(img_obj.image_load_failed)
            self.assertTrue(img_obj.image_load_error)
            self.assertEqual(Face.objects.filter(source_image_file=img_obj).count(), 0)

    def test_good_image_gets_processed_and_gains_faces(self):
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/has_face_tags.jpg"
        create_image_file(path)
        img_obj = ImageFile.objects.get(filename=path)
        self.assertFalse(img_obj.isProcessed)

        self.extractor.find_and_encode_faces()

        img_obj.refresh_from_db()
        self.assertTrue(img_obj.isProcessed)
        self.assertGreaterEqual(Face.objects.filter(source_image_file=img_obj).count(), 1)

    def test_new_face_gains_kps_from_real_detection(self):
        """add_new_face() should now also store the 5-point landmarks
        InsightFace's detector produces, alongside the embedding it
        already stored -- needed so a face's embedding can later be
        exactly reproduced (see FaceExtractor.reencode_missing_faces())
        without re-detecting from scratch."""
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/has_face_tags.jpg"
        create_image_file(path)
        img_obj = ImageFile.objects.get(filename=path)

        self.extractor.find_and_encode_faces()

        faces = list(Face.objects.filter(source_image_file=img_obj))
        self.assertGreaterEqual(len(faces), 1)
        for f in faces:
            self.assertIsNotNone(f.kps, f"Face {f.id} has no kps after fresh detection")
            self.assertEqual(len(f.kps), 10)

    def test_rematched_existing_face_gains_kps_from_update_path(self):
        """update_existing_face_to_insightface() (the IOU-rematch branch,
        hit when find_and_encode_faces() runs again against an image that
        already has Face rows) should also populate kps, not just
        add_new_face()'s fresh-detection branch."""
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/has_face_tags.jpg"
        create_image_file(path)
        img_obj = ImageFile.objects.get(filename=path)

        self.extractor.find_and_encode_faces()
        first_pass_ids = set(Face.objects.filter(source_image_file=img_obj).values_list('id', flat=True))
        self.assertGreaterEqual(len(first_pass_ids), 1)

        # Clear kps to simulate a face that predates this field, then
        # re-run detection against the same (already-processed) image by
        # resetting isProcessed -- this should hit the existing-face
        # IOU-rematch path, not add_new_face(), since Face rows already
        # exist for this image.
        Face.objects.filter(id__in=first_pass_ids).update(kps=None)
        img_obj.isProcessed = False
        img_obj.save()

        self.extractor.find_and_encode_faces()

        second_pass_faces = Face.objects.filter(source_image_file=img_obj)
        self.assertEqual(
            set(second_pass_faces.values_list('id', flat=True)), first_pass_ids,
            "Re-running detection against an already-processed image should rematch "
            "existing faces, not create new ones."
        )
        for f in second_pass_faces:
            self.assertIsNotNone(f.kps, f"Face {f.id} has no kps after IOU-rematch")
            self.assertEqual(len(f.kps), 10)

    def test_failure_on_one_image_does_not_block_others(self):
        """Regression test: the IOU-matching logic in
        find_and_encode_faces() (everything after image load/detection --
        computing IOU, matching existing faces to detections, and the
        NotImplementedError/ValueError/bare-assert paths the original
        author left for cases believed unreachable) used to have no
        exception handling of its own. Any failure there -- one of those
        specific cases, or any future unforeseen edge case -- propagated
        out of the entire per-image loop and up into face_manager.tasks.
        process_faces()'s outer bare except, silently aborting the
        *entire* scheduled run: every other already-queued image would
        never get attempted either. Now wrapped so a failure on one image
        just skips it (left isProcessed=False to retry later) instead of
        blocking every other image in the batch."""
        path_a = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/has_face_tags.jpg"
        path_b = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/has_same_faces.jpg"
        create_image_file(path_a)
        create_image_file(path_b)
        img_a = ImageFile.objects.get(filename=path_a)
        img_b = ImageFile.objects.get(filename=path_b)

        original_add_new_face = FaceExtractor.add_new_face

        def flaky_add_new_face(self, insight_detected_face, img_obj, img_numpy):
            if img_obj.pk == img_a.pk:
                raise RuntimeError("simulated IOU-matching crash")
            return original_add_new_face(self, insight_detected_face, img_obj, img_numpy)

        with patch.object(FaceExtractor, "add_new_face", flaky_add_new_face):
            self.extractor.find_and_encode_faces()

        img_a.refresh_from_db()
        img_b.refresh_from_db()
        self.assertFalse(img_a.isProcessed)
        self.assertTrue(img_b.isProcessed)

    def test_skips_entirely_when_advisory_lock_already_held(self):
        # Regression test for a real production bug: find_and_encode_faces()
        # used to have no locking of its own at all -- only the Celery task
        # wrapper checked celery_app.control.inspect().active(), a
        # check-then-act race that also didn't cover a direct call like
        # this one. Two concurrent runs against the same never-before-
        # processed image each saw n_existing=0 and both inserted a Face
        # row for the same detected face -- confirmed against real
        # production data (~72% of found same-image duplicate-Face-row
        # pairs had adjacent/near-adjacent ids). Now find_and_encode_faces()
        # holds a Postgres advisory lock (common/advisory_lock.py) for its
        # whole run; while another session holds it, this call must do
        # nothing at all, not even mark the image processed.
        #
        # Advisory locks are reentrant PER SESSION, so the "other holder"
        # has to be a genuinely separate connection, not another lock
        # acquired on Django's own (shared, single) test connection.
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/has_face_tags.jpg"
        create_image_file(path)
        img_obj = ImageFile.objects.get(filename=path)

        cfg = connection.settings_dict
        other = psycopg2.connect(
            dbname=cfg['NAME'], user=cfg['USER'], password=cfg['PASSWORD'],
            host=cfg['HOST'] or 'localhost', port=cfg['PORT'] or 5432,
        )
        other.autocommit = True
        try:
            key = zlib.crc32(b'face_manager.find_and_encode_faces')
            with other.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock(%s)", [key])
                self.assertTrue(cur.fetchone()[0])

                self.extractor.find_and_encode_faces()

                img_obj.refresh_from_db()
                self.assertFalse(img_obj.isProcessed)
                self.assertEqual(Face.objects.filter(source_image_file=img_obj).count(), 0)

                cur.execute("SELECT pg_advisory_unlock(%s)", [key])
        finally:
            other.close()

        # Once free again, a normal run still works.
        self.extractor.find_and_encode_faces()
        img_obj.refresh_from_db()
        self.assertTrue(img_obj.isProcessed)


class AddNewFacePhotoCollectionGateTests(TestCase):
    """add_new_face()'s per-PhotoCollection detection floors (see
    below_collection_floor() in filepopulator/models.py). The Face row
    is ALWAYS created regardless of the floors -- box/kps/det_score/
    relative_size are always stored, so a detection is never silently
    lost -- only face_encoding_512 is conditionally withheld. Uses a
    real FaceExtractor (ONNX models loaded once via setUpClass, same
    pattern as FaceExtractorCorruptedImageTests -- add_new_face() itself
    never touches the ONNX session, only pure-Python helpers) with a
    synthetic insightface.app.common.Face (a plain dict subclass, real
    inference not needed to construct one) so the det_score/box can be
    controlled directly rather than depending on what a real image
    happens to detect."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.extractor = FaceExtractor()

    def _make_image_in_collection(self, collection, path):
        directory, _ = Directory.objects.get_or_create(
            dir_path=os.path.dirname(path), defaults={'collection': collection},
        )
        if directory.collection_id != (collection.id if collection else None):
            directory.collection = collection
            directory.save()
        image = ImageFile(
            filename=path, directory=directory, thumbnail_small="",
            width=640, height=480, pixel_hash='hash', file_hash='fhash', dateTakenUTC=0,
        )
        ImageFile.objects.bulk_create([image])
        return ImageFile.objects.get(filename=path)

    def _fake_detection(self, bbox, det_score):
        import insightface.app.common
        import numpy as np
        # Integer dtype deliberately, not float32 -- get_square_face_img()
        # slices img_numpy directly with values derived from this bbox,
        # which needs __index__-capable (integer) values. A real
        # detection's bbox is float32 too, but something upstream of
        # here evidently normalizes it before this point in production;
        # for this synthetic unit test, int is simplest and correct
        # enough (only relative proportions matter for the gate itself).
        # kps centered in the bbox (not a hardcoded (0,0) placeholder) --
        # must satisfy the DB's face_kps_within_box_margin constraint,
        # same as any real detection's kps always would.
        bl, bt, br, bb = bbox
        cx, cy = (bl + br) / 2, (bt + bb) / 2
        return insightface.app.common.Face(
            bbox=np.array(bbox, dtype=np.int64),
            kps=np.array([[cx, cy]] * 5, dtype=np.float32),
            det_score=det_score,
            embedding=np.ones(512, dtype=np.float32),  # ones, not zeros -- distinguishable from "never set"
            age=30, gender=0,
        )

    def _img_numpy(self):
        import numpy as np
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_default_collection_no_floor_keeps_everything_including_encoding(self):
        from filepopulator.models import PhotoCollection
        default = PhotoCollection.objects.get(is_default=True)
        img = self._make_image_in_collection(default, '/photos/gate_test_default/a.jpg')
        # Small box (20.8% of the image's min dimension) and a low
        # det_score -- would fail floors if any were set, but the
        # default collection has none.
        face = self._fake_detection([100, 100, 200, 200], det_score=0.1)
        self.extractor.add_new_face(face, img, self._img_numpy())
        saved = Face.objects.get(source_image_file=img)
        self.assertIsNotNone(saved.face_encoding_512)
        self.assertAlmostEqual(saved.relative_size, 100 / 480, places=3)

    def test_det_score_below_floor_still_creates_row_but_withholds_encoding(self):
        from filepopulator.models import PhotoCollection
        collection = PhotoCollection.objects.create(
            name='Low Fidelity', root_directory='/photos/gate_test_detscore', min_face_det_score=0.6,
        )
        img = self._make_image_in_collection(collection, '/photos/gate_test_detscore/a.jpg')
        face = self._fake_detection([50, 50, 400, 400], det_score=0.5)
        self.extractor.add_new_face(face, img, self._img_numpy())
        saved = Face.objects.get(source_image_file=img)
        self.assertIsNone(saved.face_encoding_512)
        self.assertAlmostEqual(saved.det_score, 0.5)
        self.assertIsNotNone(saved.kps)

    def test_det_score_at_or_above_floor_gets_encoded(self):
        from filepopulator.models import PhotoCollection
        collection = PhotoCollection.objects.create(
            name='Low Fidelity 2', root_directory='/photos/gate_test_detscore2', min_face_det_score=0.6,
        )
        img = self._make_image_in_collection(collection, '/photos/gate_test_detscore2/a.jpg')
        face = self._fake_detection([50, 50, 400, 400], det_score=0.6)
        self.extractor.add_new_face(face, img, self._img_numpy())
        saved = Face.objects.get(source_image_file=img)
        self.assertIsNotNone(saved.face_encoding_512)

    def test_relative_size_below_floor_still_creates_row_but_withholds_encoding(self):
        from filepopulator.models import PhotoCollection
        collection = PhotoCollection.objects.create(
            name='Low Fidelity 3', root_directory='/photos/gate_test_size', min_face_relative_size=0.3,
        )
        img = self._make_image_in_collection(collection, '/photos/gate_test_size/a.jpg')
        # 640x480 image, box 100x100 -> relative size = 100/480 = 0.208,
        # below the 0.3 floor.
        face = self._fake_detection([100, 100, 200, 200], det_score=0.9)
        self.extractor.add_new_face(face, img, self._img_numpy())
        saved = Face.objects.get(source_image_file=img)
        self.assertIsNone(saved.face_encoding_512)
        self.assertAlmostEqual(saved.relative_size, 100 / 480, places=3)

    def test_relative_size_at_or_above_floor_gets_encoded(self):
        from filepopulator.models import PhotoCollection
        collection = PhotoCollection.objects.create(
            name='Low Fidelity 4', root_directory='/photos/gate_test_size2', min_face_relative_size=0.1,
        )
        img = self._make_image_in_collection(collection, '/photos/gate_test_size2/a.jpg')
        # box 100x100 -> relative size 0.208, above the 0.1 floor.
        face = self._fake_detection([100, 100, 200, 200], det_score=0.9)
        self.extractor.add_new_face(face, img, self._img_numpy())
        saved = Face.objects.get(source_image_file=img)
        self.assertIsNotNone(saved.face_encoding_512)

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_above_floor_stores_lvface_encoding_too(self):
        from filepopulator.models import PhotoCollection
        default = PhotoCollection.objects.get(is_default=True)
        img = self._make_image_in_collection(default, '/photos/gate_test_lvface/a.jpg')
        face = self._fake_detection([100, 100, 200, 200], det_score=0.9)
        with patch('lvface.embed_from_image_array', return_value=np.full(512, 0.25, dtype=np.float32)) as mock_lv:
            self.extractor.add_new_face(face, img, self._img_numpy())
        saved = Face.objects.get(source_image_file=img)
        self.assertEqual(saved.face_encoding_lvface_b, [0.25] * 512)
        # input_is_bgr=False -- img_numpy here is the PIL/RGB-sourced
        # array this codebase's own extraction pipeline already uses,
        # not a cv2-decoded BGR one.
        mock_lv.assert_called_once()
        self.assertEqual(mock_lv.call_args.kwargs.get('input_is_bgr'), False)

    @override_settings(LVFACE_ENCODING_ENABLED=False)
    def test_lvface_disabled_never_attempts_encoding(self):
        # settings.LVFACE_ENCODING_ENABLED is False in production
        # (flipped back off 2026-10-26 once the original "validated
        # improvement" claim was found to rest on a flawed sample --
        # see that setting's own comment), but the disabled path must
        # still work correctly regardless of the current default --
        # exercised here via an explicit override, not relying on it.
        from filepopulator.models import PhotoCollection
        default = PhotoCollection.objects.get(is_default=True)
        img = self._make_image_in_collection(default, '/photos/gate_test_lvface_disabled/a.jpg')
        face = self._fake_detection([100, 100, 200, 200], det_score=0.9)
        with patch('lvface.embed_from_image_array') as mock_lv:
            self.extractor.add_new_face(face, img, self._img_numpy())
        mock_lv.assert_not_called()
        saved = Face.objects.get(source_image_file=img)
        self.assertIsNone(saved.face_encoding_lvface_b)
        self.assertIsNotNone(saved.face_encoding_512)  # insightface's own encoding is unaffected

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_below_floor_never_attempts_lvface_encoding(self):
        # Matches face_encoding_512's own floor gating -- a detection
        # that never gets a real insightface embedding shouldn't pay
        # LVFace-B's (much larger) inference cost either.
        from filepopulator.models import PhotoCollection
        collection = PhotoCollection.objects.create(
            name='Low Fidelity LVFace', root_directory='/photos/gate_test_lvface_skip', min_face_det_score=0.6,
        )
        img = self._make_image_in_collection(collection, '/photos/gate_test_lvface_skip/a.jpg')
        face = self._fake_detection([50, 50, 400, 400], det_score=0.5)
        with patch('lvface.embed_from_image_array') as mock_lv:
            self.extractor.add_new_face(face, img, self._img_numpy())
        mock_lv.assert_not_called()
        saved = Face.objects.get(source_image_file=img)
        self.assertIsNone(saved.face_encoding_lvface_b)

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_lvface_failure_is_non_fatal_and_leaves_insightface_encoding_intact(self):
        from filepopulator.models import PhotoCollection
        default = PhotoCollection.objects.get(is_default=True)
        img = self._make_image_in_collection(default, '/photos/gate_test_lvface_fail/a.jpg')
        face = self._fake_detection([100, 100, 200, 200], det_score=0.9)
        with patch('lvface.embed_from_image_array', side_effect=RuntimeError("boom")):
            self.extractor.add_new_face(face, img, self._img_numpy())
        saved = Face.objects.get(source_image_file=img)
        self.assertIsNone(saved.face_encoding_lvface_b)
        self.assertIsNotNone(saved.face_encoding_512)

    def test_retuning_floor_changes_outcome_for_the_same_stored_values(self):
        # The real point of NOT storing a separate flag: below_collection_
        # floor() always checks the collection's CURRENT thresholds, so
        # loosening/tightening a floor later automatically reclassifies
        # faces detected before the change, with no backfill needed.
        from filepopulator.models import PhotoCollection, below_collection_floor

        collection = PhotoCollection.objects.create(
            name='Retune Test', root_directory='/photos/gate_test_retune', min_face_relative_size=0.3,
        )
        self.assertTrue(below_collection_floor(collection, det_score=0.9, relative_size=0.208))

        collection.min_face_relative_size = 0.1
        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            collection.save()
        self.assertFalse(below_collection_floor(collection, det_score=0.9, relative_size=0.208))


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class BackfillRelativeSizeTests(TestCase):
    """backfill_relative_size: pure-arithmetic backfill, no decode/model
    inference needed, so plain ImageFile/Face rows built directly
    (bulk_create, same shape as make_preexisting_image_row) are enough --
    no real fixture images required."""

    def _make_image(self, width, height, name):
        directory, _ = Directory.objects.get_or_create(dir_path='/photos/relsize_test')
        path = f'/photos/relsize_test/{name}.jpg'
        img = ImageFile(filename=path, directory=directory, thumbnail_small="", width=width, height=height)
        ImageFile.objects.bulk_create([img])
        return ImageFile.objects.get(filename=path)

    def _make_bare_face(self, img, box, person=None):
        bl, bt, br, bb = box
        face = Face(
            source_image_file=img, declared_name=person or get_default_blank_person(),
            box_left=bl, box_top=bt, box_right=br, box_bottom=bb,
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()
        return face

    def test_backfill_computes_correct_relative_size(self):
        img = self._make_image(640, 480, 'a')
        face = self._make_bare_face(img, (100, 100, 200, 200))  # 100x100 box, min dim 480 -> 0.2083

        call_command('backfill_relative_size')

        face.refresh_from_db()
        self.assertAlmostEqual(face.relative_size, 100 / 480, places=4)

    def test_dry_run_does_not_write(self):
        img = self._make_image(640, 480, 'b')
        face = self._make_bare_face(img, (100, 100, 200, 200))

        call_command('backfill_relative_size', '--dry-run')

        face.refresh_from_db()
        self.assertIsNone(face.relative_size)

    def test_already_populated_rows_are_skipped(self):
        img = self._make_image(640, 480, 'c')
        face = self._make_bare_face(img, (100, 100, 200, 200))
        Face.objects.filter(pk=face.pk).update(relative_size=0.999)

        call_command('backfill_relative_size')

        face.refresh_from_db()
        self.assertEqual(face.relative_size, 0.999)

    def test_face_with_no_source_image_is_skipped_not_errored(self):
        # Video-sourced faces (source_image_file=None) must not crash
        # the backfill -- they're simply excluded from this query (the
        # feature this backfill supports is image-collection-only right
        # now).
        from filepopulator.models import VideoFile
        directory, _ = Directory.objects.get_or_create(dir_path='/videos/relsize_test')
        video = VideoFile.objects.create(
            filename='/videos/relsize_test/v.mp4', directory=directory, width=640, height=480,
        )
        video_face = Face(
            declared_name=get_default_blank_person(), source_video_file=video,
            box_left=1, box_top=1, box_right=10, box_bottom=10,
        )
        video_face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        video_face.save()

        call_command('backfill_relative_size')  # must not raise

        video_face.refresh_from_db()
        self.assertIsNone(video_face.relative_size)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class ClearEncodingsBelowFloorTests(TestCase):
    """clear_encodings_below_floor: the one-time sweep that actually
    applies a collection's floor to faces that were already encoded
    before the floor existed/was tightened -- add_new_face()/
    reencode_missing_faces() only ever gate a face that doesn't already
    have a usable embedding, so retuning a floor down has no visible
    effect on the existing backlog without this."""

    def _make_image(self, width, height, name, collection=None):
        from filepopulator.models import PhotoCollection
        directory, _ = Directory.objects.get_or_create(dir_path='/photos/clear_floor_test')
        if collection is not None:
            directory.collection = collection
            directory.save()
        path = f'/photos/clear_floor_test/{name}.jpg'
        img = ImageFile(filename=path, directory=directory, thumbnail_small="", width=width, height=height)
        ImageFile.objects.bulk_create([img])
        return ImageFile.objects.get(filename=path)

    def _make_face(self, img, relative_size, det_score=0.9, person=None, validated=False, encoding=None, poss_ident1=None):
        face = Face(
            source_image_file=img, declared_name=person or get_default_blank_person(),
            box_left=0, box_top=0, box_right=10, box_bottom=10,
            relative_size=relative_size, det_score=det_score, validated=validated,
            face_encoding_512=encoding if encoding is not None else ([0.1] * 512),
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        if poss_ident1 is not None:
            face.poss_ident1 = poss_ident1
            face.weight_1 = 0.5
        face.save()
        return face

    def _default_collection(self, min_face_relative_size):
        # Not testing PhotoCollection's own save()-triggered task here
        # (see PhotoCollectionFloorChangeTriggerTests) -- patch it out so
        # this setup helper doesn't enqueue a real Celery task against
        # the test broker.
        from filepopulator.models import PhotoCollection
        collection, _ = PhotoCollection.objects.get_or_create(
            is_default=True, defaults={'name': 'Default', 'root_directory': '/photos'},
        )
        collection.min_face_relative_size = min_face_relative_size
        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            collection.save()
        return collection

    def test_clears_encoding_on_unconfirmed_face_below_floor(self):
        collection = self._default_collection(min_face_relative_size=0.02)
        img = self._make_image(640, 480, 'a', collection=collection)
        face = self._make_face(img, relative_size=0.01)  # below 0.02

        call_command('clear_encodings_below_floor')

        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_512)

    def test_also_clears_stale_poss_identN_proposals(self):
        # A never-adjudicated face can still carry a live poss_ident1/
        # weight_1 proposal from a past classification run -- clearing
        # the encoding without also clearing this leaves a stale
        # proposal with nothing left to justify it (misleading, and a
        # real crash risk for anything that assumes a proposal implies a
        # usable embedding -- confirmed in production 2026-10-20).
        from face_manager.models import Person
        proposed = Person.objects.create(person_name='Proposed Candidate')
        collection = self._default_collection(min_face_relative_size=0.02)
        img = self._make_image(640, 480, 'g', collection=collection)
        face = self._make_face(img, relative_size=0.01, poss_ident1=proposed)

        call_command('clear_encodings_below_floor')

        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_512)
        self.assertIsNone(face.poss_ident1)
        self.assertEqual(face.weight_1, 0.0)

    def test_leaves_encoding_on_unconfirmed_face_at_or_above_floor(self):
        collection = self._default_collection(min_face_relative_size=0.02)
        img = self._make_image(640, 480, 'b', collection=collection)
        face = self._make_face(img, relative_size=0.05)  # above 0.02

        call_command('clear_encodings_below_floor')

        face.refresh_from_db()
        self.assertIsNotNone(face.face_encoding_512)

    def test_never_touches_a_validated_face_even_below_floor(self):
        collection = self._default_collection(min_face_relative_size=0.02)
        img = self._make_image(640, 480, 'c', collection=collection)
        face = self._make_face(img, relative_size=0.01, validated=True)

        call_command('clear_encodings_below_floor')

        face.refresh_from_db()
        self.assertIsNotNone(face.face_encoding_512)

    def test_never_touches_an_already_declared_face_even_below_floor(self):
        from face_manager.models import Person
        person = Person.objects.create(person_name='Someone Real')
        collection = self._default_collection(min_face_relative_size=0.02)
        img = self._make_image(640, 480, 'd', collection=collection)
        face = self._make_face(img, relative_size=0.01, person=person)

        call_command('clear_encodings_below_floor')

        face.refresh_from_db()
        self.assertIsNotNone(face.face_encoding_512)

    def test_dry_run_does_not_write(self):
        collection = self._default_collection(min_face_relative_size=0.02)
        img = self._make_image(640, 480, 'e', collection=collection)
        face = self._make_face(img, relative_size=0.01)

        call_command('clear_encodings_below_floor', '--dry-run')

        face.refresh_from_db()
        self.assertIsNotNone(face.face_encoding_512)

    def test_already_null_encoding_is_not_counted_as_cleared(self):
        collection = self._default_collection(min_face_relative_size=0.02)
        img = self._make_image(640, 480, 'f', collection=collection)
        self._make_face(img, relative_size=0.01, encoding=None)

        # Must not raise -- NULL encodings are excluded from the query
        # entirely (nothing to clear).
        call_command('clear_encodings_below_floor')


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class ClearStaleUnencodedProposalsTests(TestCase):
    """clear_stale_proposals_with_no_encoding: the catch-up pass for the
    backlog clear_encodings_below_collection_floor() can't reach -- a
    face already NULL-encoded before that function's own poss_identN
    clearing existed (or for any other reason) but still carrying a
    stale proposal."""

    def _make_image(self, name):
        directory, _ = Directory.objects.get_or_create(dir_path='/photos/stale_proposal_test')
        path = f'/photos/stale_proposal_test/{name}.jpg'
        img = ImageFile(filename=path, directory=directory, thumbnail_small="", width=640, height=480)
        ImageFile.objects.bulk_create([img])
        return ImageFile.objects.get(filename=path)

    def _make_face(self, img, person=None, validated=False, encoding=None, poss_ident1=None):
        face = Face(
            source_image_file=img, declared_name=person or get_default_blank_person(),
            box_left=0, box_top=0, box_right=10, box_bottom=10, validated=validated,
            face_encoding_512=encoding,
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        if poss_ident1 is not None:
            face.poss_ident1 = poss_ident1
            face.weight_1 = 0.5
        face.save()
        return face

    def test_clears_stale_proposal_on_unencoded_unconfirmed_face(self):
        from face_manager.models import Person, clear_stale_proposals_with_no_encoding
        proposed = Person.objects.create(person_name='Stale Candidate')
        img = self._make_image('a')
        face = self._make_face(img, encoding=None, poss_ident1=proposed)

        count = clear_stale_proposals_with_no_encoding()

        face.refresh_from_db()
        self.assertEqual(count, 1)
        self.assertIsNone(face.poss_ident1)
        self.assertEqual(face.weight_1, 0.0)

    def test_leaves_encoded_face_with_proposal_untouched(self):
        from face_manager.models import Person, clear_stale_proposals_with_no_encoding
        proposed = Person.objects.create(person_name='Real Candidate')
        img = self._make_image('b')
        face = self._make_face(img, encoding=[0.1] * 512, poss_ident1=proposed)

        clear_stale_proposals_with_no_encoding()

        face.refresh_from_db()
        self.assertEqual(face.poss_ident1_id, proposed.id)

    def test_leaves_unencoded_face_with_no_proposal_untouched(self):
        from face_manager.models import clear_stale_proposals_with_no_encoding
        img = self._make_image('c')
        self._make_face(img, encoding=None, poss_ident1=None)

        count = clear_stale_proposals_with_no_encoding()
        self.assertEqual(count, 0)

    def test_never_touches_a_validated_or_declared_face(self):
        from face_manager.models import Person, clear_stale_proposals_with_no_encoding
        proposed = Person.objects.create(person_name='Candidate')
        real_person = Person.objects.create(person_name='Real Person')
        img = self._make_image('d')
        validated_face = self._make_face(img, encoding=None, validated=True, poss_ident1=proposed)
        img2 = self._make_image('e')
        declared_face = self._make_face(img2, person=real_person, encoding=None, poss_ident1=proposed)

        clear_stale_proposals_with_no_encoding()

        validated_face.refresh_from_db()
        declared_face.refresh_from_db()
        self.assertEqual(validated_face.poss_ident1_id, proposed.id)
        self.assertEqual(declared_face.poss_ident1_id, proposed.id)

    def test_dry_run_does_not_write_but_returns_count(self):
        from face_manager.models import Person, clear_stale_proposals_with_no_encoding
        proposed = Person.objects.create(person_name='Dry Run Candidate')
        img = self._make_image('f')
        face = self._make_face(img, encoding=None, poss_ident1=proposed)

        count = clear_stale_proposals_with_no_encoding(dry_run=True)

        face.refresh_from_db()
        self.assertEqual(count, 1)
        self.assertEqual(face.poss_ident1_id, proposed.id)

    def test_scoped_to_a_specific_collection(self):
        from filepopulator.models import PhotoCollection
        from face_manager.models import Person, clear_stale_proposals_with_no_encoding

        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            other_collection = PhotoCollection.objects.create(
                name='Other Collection', root_directory='/photos/stale_proposal_other',
            )
        other_dir, _ = Directory.objects.get_or_create(
            dir_path='/photos/stale_proposal_other', defaults={'collection': other_collection},
        )
        other_img = ImageFile(
            filename='/photos/stale_proposal_other/a.jpg', directory=other_dir,
            thumbnail_small="", width=640, height=480,
        )
        ImageFile.objects.bulk_create([other_img])
        other_img = ImageFile.objects.get(filename='/photos/stale_proposal_other/a.jpg')

        proposed = Person.objects.create(person_name='Scoped Candidate')
        default_face = self._make_face(self._make_image('g'), encoding=None, poss_ident1=proposed)
        other_face = self._make_face(other_img, encoding=None, poss_ident1=proposed)

        count = clear_stale_proposals_with_no_encoding(collection=other_collection)

        default_face.refresh_from_db()
        other_face.refresh_from_db()
        self.assertEqual(count, 1)
        self.assertEqual(default_face.poss_ident1_id, proposed.id)  # untouched -- different collection
        self.assertIsNone(other_face.poss_ident1)

    def test_management_command_dry_run_vs_real(self):
        from face_manager.models import Person
        proposed = Person.objects.create(person_name='Command Candidate')
        img = self._make_image('h')
        face = self._make_face(img, encoding=None, poss_ident1=proposed)

        call_command('clear_stale_unencoded_proposals', '--dry-run')
        face.refresh_from_db()
        self.assertEqual(face.poss_ident1_id, proposed.id)

        call_command('clear_stale_unencoded_proposals')
        face.refresh_from_db()
        self.assertIsNone(face.poss_ident1)


class PhotoCollectionFloorChangeTriggerTests(TestCase):
    """PhotoCollection.save() fires face_manager.apply_collection_floor_
    change whenever min_face_det_score/min_face_relative_size actually
    changes on an existing row -- so retuning a floor in the frontend
    settings page applies itself automatically, with no one needing to
    remember to run clear_encodings_below_floor/reencode_missing_faces
    by hand. Only the triggering condition is tested here (mocking
    .delay() -- no real Celery/Redis/model-loading involved); the
    task's own behavior is covered by ApplyCollectionFloorChangeTaskTests."""

    def test_no_trigger_on_initial_create(self):
        from filepopulator.models import PhotoCollection
        with patch('face_manager.tasks.apply_collection_floor_change.delay') as mock_delay:
            PhotoCollection.objects.create(
                name='Trigger Create', root_directory='/photos/trigger_create', min_face_relative_size=0.1,
            )
        mock_delay.assert_not_called()

    def test_triggers_on_relative_size_floor_change(self):
        from filepopulator.models import PhotoCollection
        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            collection = PhotoCollection.objects.create(
                name='Trigger Relsize', root_directory='/photos/trigger_relsize', min_face_relative_size=0.1,
            )
        collection.min_face_relative_size = 0.2
        with patch('face_manager.tasks.apply_collection_floor_change.delay') as mock_delay:
            collection.save()
        mock_delay.assert_called_once_with(collection.pk)

    def test_triggers_on_det_score_floor_change(self):
        from filepopulator.models import PhotoCollection
        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            collection = PhotoCollection.objects.create(
                name='Trigger Detscore', root_directory='/photos/trigger_detscore', min_face_det_score=0.5,
            )
        collection.min_face_det_score = 0.6
        with patch('face_manager.tasks.apply_collection_floor_change.delay') as mock_delay:
            collection.save()
        mock_delay.assert_called_once_with(collection.pk)

    def test_no_trigger_when_floor_values_unchanged(self):
        from filepopulator.models import PhotoCollection
        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            collection = PhotoCollection.objects.create(
                name='Trigger Unchanged', root_directory='/photos/trigger_unchanged', min_face_relative_size=0.1,
            )
        with patch('face_manager.tasks.apply_collection_floor_change.delay') as mock_delay:
            collection.save()  # re-save with the same values
        mock_delay.assert_not_called()

    def test_no_trigger_on_unrelated_field_change(self):
        from filepopulator.models import PhotoCollection
        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            collection = PhotoCollection.objects.create(
                name='Trigger Unrelated', root_directory='/photos/trigger_unrelated', min_face_relative_size=0.1,
            )
        collection.name = 'Renamed'
        with patch('face_manager.tasks.apply_collection_floor_change.delay') as mock_delay:
            collection.save()
        mock_delay.assert_not_called()

    def test_triggers_again_on_a_freshly_loaded_instance(self):
        # The change-detection snapshot is taken in __init__, so it must
        # work correctly for an instance loaded fresh from the DB (via
        # .get()), not just one that's been held in memory since create().
        from filepopulator.models import PhotoCollection
        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            created = PhotoCollection.objects.create(
                name='Trigger Reload', root_directory='/photos/trigger_reload', min_face_relative_size=0.1,
            )
        reloaded = PhotoCollection.objects.get(pk=created.pk)
        reloaded.min_face_relative_size = 0.3
        with patch('face_manager.tasks.apply_collection_floor_change.delay') as mock_delay:
            reloaded.save()
        mock_delay.assert_called_once_with(reloaded.pk)


class ApplyCollectionFloorChangeTaskTests(TestCase):
    """face_manager.apply_collection_floor_change: the task itself, not
    the triggering condition (see PhotoCollectionFloorChangeTriggerTests).
    FaceExtractor is mocked out entirely -- real model loading is
    expensive and this task's own logic (scoping both calls to the one
    changed collection) doesn't depend on real inference."""

    def test_scopes_both_clear_and_reencode_to_the_changed_collection(self):
        from filepopulator.models import PhotoCollection
        from face_manager.tasks import apply_collection_floor_change

        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            collection = PhotoCollection.objects.create(
                name='Task Scope', root_directory='/photos/task_scope', min_face_relative_size=0.1,
            )

        with patch('face_manager.tasks.clear_encodings_below_collection_floor', return_value=(0, 0)) as mock_clear, \
             patch('face_manager.tasks.clear_stale_proposals_with_no_encoding', return_value=0) as mock_stale, \
             patch('face_manager.tasks.FaceExtractor') as mock_extractor_cls:
            mock_extractor = mock_extractor_cls.return_value
            apply_collection_floor_change(collection.pk)

        mock_clear.assert_called_once_with(collection=collection)
        mock_stale.assert_called_once_with(collection=collection)
        mock_extractor.reencode_missing_faces.assert_called_once_with(collection=collection)

    def test_missing_collection_is_a_no_op(self):
        from face_manager.tasks import apply_collection_floor_change

        with patch('face_manager.tasks.clear_encodings_below_collection_floor') as mock_clear, \
             patch('face_manager.tasks.clear_stale_proposals_with_no_encoding') as mock_stale, \
             patch('face_manager.tasks.FaceExtractor') as mock_extractor_cls:
            apply_collection_floor_change(999999)  # no such PhotoCollection

        mock_clear.assert_not_called()
        mock_stale.assert_not_called()
        mock_extractor_cls.assert_not_called()

    def test_concurrent_run_for_the_same_collection_is_skipped(self):
        from filepopulator.models import PhotoCollection
        from face_manager.tasks import apply_collection_floor_change

        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            collection = PhotoCollection.objects.create(
                name='Task Lock', root_directory='/photos/task_lock', min_face_relative_size=0.1,
            )

        with patch('face_manager.tasks.advisory_lock') as mock_lock, \
             patch('face_manager.tasks.clear_encodings_below_collection_floor') as mock_clear, \
             patch('face_manager.tasks.clear_stale_proposals_with_no_encoding') as mock_stale, \
             patch('face_manager.tasks.FaceExtractor') as mock_extractor_cls:
            mock_lock.return_value.__enter__.return_value = False
            apply_collection_floor_change(collection.pk)

        mock_clear.assert_not_called()
        mock_stale.assert_not_called()
        mock_extractor_cls.assert_not_called()

    def test_a_failing_step_is_recorded_as_an_application_error(self):
        from common.models import ApplicationError
        from filepopulator.models import PhotoCollection
        from face_manager.tasks import apply_collection_floor_change

        with patch('face_manager.tasks.apply_collection_floor_change.delay'):
            collection = PhotoCollection.objects.create(
                name='Task Error Logging', root_directory='/photos/task_error_logging',
                min_face_relative_size=0.1,
            )

        with patch('face_manager.tasks.clear_encodings_below_collection_floor',
                   side_effect=RuntimeError("boom")), \
             patch('face_manager.tasks.clear_stale_proposals_with_no_encoding', return_value=0), \
             patch('face_manager.tasks.FaceExtractor') as mock_extractor_cls:
            mock_extractor_cls.return_value.reencode_missing_faces.return_value = None
            apply_collection_floor_change(collection.pk)

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.tasks.apply_collection_floor_change')
        self.assertIn(str(collection.pk), error.message)
        self.assertIn('boom', error.message)


class TaskCrashLoggingTests(TestCase):
    """Regression tests for a real, confirmed bug (2026-10-21):
    process_faces/reencode_missing_faces/clear_ignored_encodings/
    assign_faces_task (assign_faces)/cluster_unverified_faces_task all used to
    wrap their entire body in a bare `except:` that discarded the real
    exception completely -- not even writing it to the log files. Now
    routed through the shared _log_task_crash() helper, which logs the
    real exception AND records it via common.error_log.log_error()."""

    def _mock_empty_inspect(self):
        mock_inspect = MagicMock()
        mock_inspect.active.return_value = {}
        return patch('face_manager.tasks.celery_app.control.inspect', return_value=mock_inspect)

    def test_log_task_crash_logs_and_records(self):
        from common.models import ApplicationError
        from face_manager.tasks import _log_task_crash
        _log_task_crash('some_task', RuntimeError("boom"))
        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.tasks.some_task')
        self.assertIn('RuntimeError', error.message)
        self.assertIn('boom', error.message)

    def test_process_faces_crash_is_recorded(self):
        from common.models import ApplicationError
        from face_manager.tasks import process_faces
        img = make_image()
        img.isProcessed = False
        img.save()

        with patch('face_manager.tasks.FaceExtractor', side_effect=RuntimeError("boom")):
            process_faces()

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.tasks.process_faces')
        self.assertIn('boom', error.message)

    def test_reencode_missing_faces_crash_is_recorded(self):
        from common.models import ApplicationError
        from face_manager.tasks import reencode_missing_faces
        with self._mock_empty_inspect(), \
             patch('face_manager.tasks.FaceExtractor', side_effect=RuntimeError("boom")):
            reencode_missing_faces()

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.tasks.reencode_missing_faces')

    def test_reencode_missing_faces_task_skips_when_parallel_backfill_active(self):
        # Regression test for a real production incident (2026-10-25):
        # the one-time parallel backfill (backfill_lvface_parallel)
        # shards the backlog under per-shard lock names, deliberately
        # different from this task's own plain lock name, so the two
        # were never mutually exclusive -- this task ran fully
        # concurrently with the parallel backfill's supervisor, doing
        # real, redundant work on the same backlog while competing with
        # it for the exact CPU/memory headroom the supervisor was
        # trying to budget around. Both now coordinate via a shared
        # marker lock (see face_manager.tasks.reencode_missing_faces'
        # own comment and backfill_lvface_parallel.py's).
        from face_manager.tasks import reencode_missing_faces
        cfg = connection.settings_dict
        other = psycopg2.connect(
            dbname=cfg['NAME'], user=cfg['USER'], password=cfg['PASSWORD'],
            host=cfg['HOST'] or 'localhost', port=cfg['PORT'] or 5432,
        )
        other.autocommit = True
        try:
            key = zlib.crc32(b'face_manager.reencode_missing_faces.parallel_backfill_active')
            with other.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock(%s)", [key])
                self.assertTrue(cur.fetchone()[0])

                with patch('face_manager.tasks.FaceExtractor') as mock_extractor_cls:
                    reencode_missing_faces()
                mock_extractor_cls.assert_not_called()

                cur.execute("SELECT pg_advisory_unlock(%s)", [key])
        finally:
            other.close()

        # Once free again, the task runs normally.
        with patch('face_manager.tasks.FaceExtractor') as mock_extractor_cls:
            reencode_missing_faces()
        mock_extractor_cls.assert_called_once()

    def test_clear_ignored_encodings_crash_is_recorded(self):
        from common.models import ApplicationError
        from face_manager.tasks import clear_ignored_encodings
        with self._mock_empty_inspect(), \
             patch('face_manager.tasks.clear_confirmed_ignore_face_encodings', side_effect=RuntimeError("boom")):
            clear_ignored_encodings()

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.tasks.clear_ignored_encodings')

    def test_assign_faces_crash_is_recorded(self):
        from common.models import ApplicationError
        from face_manager.tasks import assign_faces_task
        with self._mock_empty_inspect(), \
             patch('face_manager.tasks.faceAssigner', side_effect=RuntimeError("boom")):
            assign_faces_task()

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.tasks.assign_faces_task')

    def test_weekly_redo_all_run_records_its_own_heartbeat(self):
        from common.models import TaskHeartbeat
        from face_manager.tasks import assign_faces_task
        with self._mock_empty_inspect(), \
             patch('face_manager.tasks.faceAssigner') as mock_assigner_cls:
            assign_faces_task(redo_all=True)

        heartbeat = TaskHeartbeat.objects.get(task_name='face_manager.assign_faces.weekly_full')
        self.assertIsNotNone(heartbeat.last_started_at)
        # A clean (mocked, non-crashing) run must mark itself done, not stuck.
        self.assertIsNotNone(heartbeat.last_completed_at)
        self.assertGreaterEqual(heartbeat.last_completed_at, heartbeat.last_started_at)

    def test_hourly_run_records_a_distinctly_named_heartbeat(self):
        # Must not collide with the weekly one -- they're watched for
        # staleness/hangs independently (only the weekly one is
        # currently in common.tasks.HEARTBEAT_REGISTRY).
        from common.models import TaskHeartbeat
        from face_manager.tasks import assign_faces_task
        with self._mock_empty_inspect(), \
             patch('face_manager.tasks.faceAssigner') as mock_assigner_cls:
            assign_faces_task(redo_all=False)

        self.assertTrue(TaskHeartbeat.objects.filter(task_name='face_manager.assign_faces.hourly').exists())
        self.assertFalse(
            TaskHeartbeat.objects.filter(task_name='face_manager.assign_faces.weekly_full').exists())

    def test_crashing_run_still_marks_its_heartbeat_done_not_stuck(self):
        # A crash is a DIFFERENT, already-logged failure mode (see
        # test_assign_faces_crash_is_recorded above) -- it must not also
        # look like a hang to the heartbeat watchdog.
        from common.models import TaskHeartbeat
        from face_manager.tasks import assign_faces_task
        with self._mock_empty_inspect(), \
             patch('face_manager.tasks.faceAssigner', side_effect=RuntimeError("boom")):
            assign_faces_task(redo_all=True)

        heartbeat = TaskHeartbeat.objects.get(task_name='face_manager.assign_faces.weekly_full')
        self.assertIsNotNone(heartbeat.last_completed_at)

    def test_cluster_unverified_faces_crash_is_recorded(self):
        from common.models import ApplicationError
        from face_manager.tasks import cluster_unverified_faces_task
        with self._mock_empty_inspect(), \
             patch('face_manager.tasks.cluster_all_unverified_faces', side_effect=RuntimeError("boom")):
            cluster_unverified_faces_task()

        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.tasks.cluster_unverified_faces_task')


class UpdateListOfNoMatchingDetectsTests(TestCase):
    """Regression test for a fixed bug: update_list_of_no_matching_detects()
    only clamped each box coordinate on one side (box_left/box_top floored
    at 0, box_right/box_bottom capped at the image's current width/height),
    so a face whose stored box came from a different coordinate space than
    the image's current dimensions -- e.g. a stale detection from before
    this session's EXIF-orientation-consolidation fix, when
    face_extract_encode's decode path could disagree with filepopulator's
    about a rotated image's true width/height -- could still end up with
    box_right <= box_left (or box_bottom <= box_top) after "clamping",
    which Face.save() correctly rejects via ValidationError. That
    propagated all the way up through find_and_encode_faces() into
    face_manager.tasks.process_faces()'s bare except:, silently aborting
    the *entire* scheduled run, not just this one face -- confirmed
    against a real production case (FastFoto_0248.jpg) before fixing."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.extractor = FaceExtractor()

    def test_geometrically_invalid_face_is_deleted_not_crashed(self):
        img = make_image()
        stale_face = make_face(
            img,
            box_left=img.width + 500,
            box_top=1,
            box_right=img.width + 700,
            box_bottom=40,
        )

        self.extractor.update_list_of_no_matching_detects([stale_face])

        self.assertFalse(Face.objects.filter(pk=stale_face.pk).exists())

    def test_recoverable_out_of_bounds_face_is_clamped_and_kept(self):
        img = make_image()
        face = make_face(
            img,
            box_left=1,
            box_top=1,
            box_right=img.width + 50,  # slightly over -- should just clamp
            box_bottom=40,
        )

        self.extractor.update_list_of_no_matching_detects([face])

        face.refresh_from_db()
        self.assertEqual(face.box_right, img.width)
        self.assertTrue(face.reencoded)

    def test_box_that_only_fits_swapped_dimensions_is_deleted(self):
        """A box that doesn't fit the image's current (w, h) at all, but
        does fit the swapped (h, w) -- the signature of a face whose
        coordinates were computed against the wrong (unrotated) dimensions
        by the pre-fix decode-path disagreement -- must be deleted even
        though clamping alone wouldn't necessarily make it degenerate."""
        img = make_image()  # width=800, height=286
        self.assertGreater(img.width, img.height)
        face = make_face(
            img,
            box_left=1, box_top=1,
            box_right=200,  # fits both width interpretations
            box_bottom=500,  # > img.height (286), but < img.width (800): only fits swapped
        )

        self.extractor.update_list_of_no_matching_detects([face])

        self.assertFalse(Face.objects.filter(pk=face.pk).exists())

    def test_unreencoded_face_on_rotated_image_is_deleted_even_if_in_bounds(self):
        """An orientation 6/8 face that has never been successfully
        re-matched since creation (reencoded=False) has never had its box
        verified under the corrected decode path -- delete it outright,
        even if its box happens to still fit the image's current
        dimensions by coincidence."""
        img = make_image()
        ImageFile.objects.filter(pk=img.pk).update(orientation=8)
        img.refresh_from_db()
        face = make_face(img, box_left=1, box_top=1, box_right=40, box_bottom=40)
        self.assertFalse(face.reencoded)

        self.extractor.update_list_of_no_matching_detects([face])

        self.assertFalse(Face.objects.filter(pk=face.pk).exists())

    def test_unreencoded_face_on_unrotated_image_is_kept(self):
        """Regression guard: the orientation 6/8 + reencoded=False deletion
        must not apply to orientations where no width/height swap is
        possible -- an in-bounds, unreencoded face on an orientation-1
        image is just legitimately not yet re-matched, not stale data, and
        should be kept (clamped/marked reencoded like any other kept
        face)."""
        img = make_image()
        ImageFile.objects.filter(pk=img.pk).update(orientation=1)
        img.refresh_from_db()
        face = make_face(img, box_left=1, box_top=1, box_right=40, box_bottom=40)
        self.assertFalse(face.reencoded)

        self.extractor.update_list_of_no_matching_detects([face])

        face.refresh_from_db()
        self.assertTrue(Face.objects.filter(pk=face.pk).exists())
        self.assertTrue(face.reencoded)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class CleanupChronicallyUnmatchedFacesTests(TestCase):
    """A face on an orientation 6/8 image that's reencoded=True but still
    carries the NON_DETECTED_FACE_ENCODING sentinel has never been
    geometrically verified under the corrected decode path -- since a
    wrongly-positioned box can never match a real detection, it would
    otherwise persist forever. This command deletes those outright and
    marks their images unprocessed for redetection."""

    def test_deletes_chronically_unmatched_faces_and_marks_images_unprocessed(self):
        img = make_image()
        ImageFile.objects.filter(pk=img.pk).update(orientation=8, isProcessed=True)
        stale_face = make_face(img, reencoded=True, face_encoding_512=list(settings.NON_DETECTED_FACE_ENCODING))

        call_command("cleanup_chronically_unmatched_faces", "--yes")

        self.assertFalse(Face.objects.filter(pk=stale_face.pk).exists())
        img.refresh_from_db()
        self.assertFalse(img.isProcessed)

    def test_leaves_verified_faces_alone(self):
        img = make_image()
        ImageFile.objects.filter(pk=img.pk).update(orientation=8, isProcessed=True)
        good_face = make_face(img, reencoded=True, face_encoding_512=[0.1] * 512)

        call_command("cleanup_chronically_unmatched_faces", "--yes")

        self.assertTrue(Face.objects.filter(pk=good_face.pk).exists())
        img.refresh_from_db()
        self.assertTrue(img.isProcessed)

    def test_dry_run_writes_nothing(self):
        img = make_image()
        ImageFile.objects.filter(pk=img.pk).update(orientation=8, isProcessed=True)
        stale_face = make_face(img, reencoded=True, face_encoding_512=list(settings.NON_DETECTED_FACE_ENCODING))

        call_command("cleanup_chronically_unmatched_faces", "--dry-run")

        self.assertTrue(Face.objects.filter(pk=stale_face.pk).exists())


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class ClearConfirmedIgnoreEncodingsTests(TestCase):
    """clear_confirmed_ignore_encodings clears face_encoding_512 only for
    faces CONFIRMED (declared_name) to .ignore/.realignore -- never faces
    merely SUGGESTED as ignore (poss_ident1 set, declared_name still the
    blank sentinel), and never faces declared to a real person. Face.kps
    must survive untouched either way, since it's what lets a later
    reencode_missing_faces() pass exactly recover the embedding."""

    def setUp(self):
        self.image = make_image()
        self.ignore_person = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        self.realignore_person = Person.objects.get(person_name='.realignore')
        self.blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)

    def test_confirmed_ignore_face_gets_encoding_cleared_kps_kept(self):
        kps = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        face = make_face(
            self.image, declared_name=self.ignore_person,
            face_encoding_512=[0.5] * 512, kps=kps,
        )

        call_command("clear_confirmed_ignore_encodings", "--yes")

        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_512)
        self.assertEqual(face.kps, kps)

    def test_confirmed_realignore_face_gets_encoding_cleared(self):
        face = make_face(self.image, declared_name=self.realignore_person, face_encoding_512=[0.5] * 512)

        call_command("clear_confirmed_ignore_encodings", "--yes")

        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_512)

    def test_suggested_ignore_face_is_left_alone(self):
        """poss_ident1 == ignore_person but declared_name is still the
        blank sentinel -- this is an unreviewed classifier suggestion,
        not a human confirmation. Its embedding is exactly what a human
        still needs to review that suggestion against, so it must not be
        touched."""
        face = make_face(
            self.image, declared_name=self.blank_person,
            poss_ident1=self.ignore_person, weight_1=0.9,
            face_encoding_512=[0.5] * 512,
        )

        call_command("clear_confirmed_ignore_encodings", "--yes")

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, [0.5] * 512)

    def test_face_declared_to_real_person_is_left_alone(self):
        person = make_person("Real Person")
        face = make_face(self.image, declared_name=person, face_encoding_512=[0.5] * 512)

        call_command("clear_confirmed_ignore_encodings", "--yes")

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, [0.5] * 512)

    def test_already_cleared_face_is_a_noop(self):
        face = make_face(self.image, declared_name=self.ignore_person)
        face.face_encoding_512 = None
        face.save()

        call_command("clear_confirmed_ignore_encodings", "--yes")

        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_512)

    def test_dry_run_writes_nothing(self):
        face = make_face(self.image, declared_name=self.ignore_person, face_encoding_512=[0.5] * 512)

        call_command("clear_confirmed_ignore_encodings", "--dry-run")

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, [0.5] * 512)

    def test_shared_function_matches_command_behavior(self):
        """clear_confirmed_ignore_face_encodings() (face_manager/models.py)
        is the single source of truth the management command AND the
        ongoing face_manager.clear_ignored_encodings Celery task both call
        for the actual write -- covered directly here so a regression in
        either caller's own logic can't hide the underlying function
        breaking."""
        confirmed = make_face(self.image, declared_name=self.ignore_person, face_encoding_512=[0.5] * 512)
        suggested = make_face(
            self.image, declared_name=self.blank_person,
            poss_ident1=self.ignore_person, weight_1=0.9,
            face_encoding_512=[0.5] * 512,
        )

        updated_count = clear_confirmed_ignore_face_encodings()

        self.assertEqual(updated_count, 1)
        confirmed.refresh_from_db()
        suggested.refresh_from_db()
        self.assertIsNone(confirmed.face_encoding_512)
        self.assertEqual(suggested.face_encoding_512, [0.5] * 512)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class DedupeOverlappingFacesTests(TestCase):
    """dedupe_overlapping_faces management command -- cleans up the
    duplicate Face rows left behind by the (now-fixed, see common/
    advisory_lock.py) find_and_encode_faces() race condition: two
    concurrent runs on the same never-before-processed image could each
    insert a Face row for the same real face, pixel-identical box."""

    def setUp(self):
        self.person = make_person("Dup Person")
        self.other_person = make_person("Other Dup Person")
        self.img = make_image()

    def _duplicate_pair(self, **overrides_a):
        overrides_b = dict(overrides_a)
        a = make_face(self.img, declared_name=self.person, box_left=1, box_top=1,
                       box_right=30, box_bottom=30, **overrides_a)
        b = make_face(self.img, declared_name=self.person, box_left=1, box_top=1,
                       box_right=30, box_bottom=30, **overrides_b)
        return a, b

    def test_dry_run_reports_but_changes_nothing(self):
        a, b = self._duplicate_pair()
        call_command('dedupe_overlapping_faces', '--dry-run')
        self.assertEqual(Face.objects.filter(pk__in=[a.pk, b.pk]).count(), 2)

    def test_identical_pair_collapses_to_one_survivor(self):
        a, b = self._duplicate_pair()
        call_command('dedupe_overlapping_faces', '--yes')
        self.assertEqual(Face.objects.filter(source_image_file=self.img).count(), 1)

    def test_validated_face_is_kept_over_unvalidated(self):
        a, b = self._duplicate_pair()
        b.validated = True
        b.save()
        call_command('dedupe_overlapping_faces', '--yes')
        remaining = Face.objects.get(source_image_file=self.img)
        self.assertEqual(remaining.pk, b.pk)

    def test_labeled_face_is_kept_over_unlabeled_when_neither_validated(self):
        # Common real scenario: a human tagged one copy of a duplicate
        # pair without knowing the other copy existed, leaving it at the
        # blank sentinel forever. The labeled copy should survive.
        blank = get_default_blank_person()
        a = make_face(self.img, declared_name=blank, box_left=1, box_top=1, box_right=30, box_bottom=30)
        b = make_face(self.img, declared_name=self.person, box_left=1, box_top=1, box_right=30, box_bottom=30)
        call_command('dedupe_overlapping_faces', '--yes')
        remaining = Face.objects.get(source_image_file=self.img)
        self.assertEqual(remaining.pk, b.pk)

    def test_validated_takes_priority_over_label_when_the_validated_one_is_unlabeled(self):
        # Validated (real completed human verification) still outranks
        # "has a label" -- shouldn't actually diverge in practice (a
        # validated face should already be labeled), but the ordering
        # itself should hold regardless.
        blank = get_default_blank_person()
        a = make_face(self.img, declared_name=blank, box_left=1, box_top=1, box_right=30, box_bottom=30,
                      validated=True)
        b = make_face(self.img, declared_name=self.person, box_left=1, box_top=1, box_right=30, box_bottom=30)
        call_command('dedupe_overlapping_faces', '--yes')
        remaining = Face.objects.get(source_image_file=self.img)
        self.assertEqual(remaining.pk, a.pk)

    def test_face_with_kps_is_kept_over_one_without_when_neither_validated(self):
        a, b = self._duplicate_pair()
        b.kps = [1.0] * 10
        b.save()
        call_command('dedupe_overlapping_faces', '--yes')
        remaining = Face.objects.get(source_image_file=self.img)
        self.assertEqual(remaining.pk, b.pk)

    def test_lowest_id_is_kept_as_final_tiebreaker(self):
        a, b = self._duplicate_pair()
        call_command('dedupe_overlapping_faces', '--yes')
        remaining = Face.objects.get(source_image_file=self.img)
        self.assertEqual(remaining.pk, min(a.pk, b.pk))

    def test_deleted_faces_thumbnail_file_is_removed(self):
        a, b = self._duplicate_pair()
        loser_path = a.face_thumbnail.path if a.pk != min(a.pk, b.pk) else b.face_thumbnail.path
        self.assertTrue(os.path.exists(loser_path))
        call_command('dedupe_overlapping_faces', '--yes')
        self.assertFalse(os.path.exists(loser_path))

    def test_two_separate_duplicate_pairs_on_same_image_both_collapse(self):
        a1 = make_face(self.img, declared_name=self.person, box_left=1, box_top=1, box_right=30, box_bottom=30)
        a2 = make_face(self.img, declared_name=self.person, box_left=1, box_top=1, box_right=30, box_bottom=30)
        b1 = make_face(self.img, declared_name=self.other_person, box_left=50, box_top=50, box_right=80, box_bottom=80)
        b2 = make_face(self.img, declared_name=self.other_person, box_left=50, box_top=50, box_right=80, box_bottom=80)
        call_command('dedupe_overlapping_faces', '--yes')
        self.assertEqual(Face.objects.filter(source_image_file=self.img).count(), 2)

    def test_non_overlapping_faces_on_same_image_are_left_alone(self):
        a = make_face(self.img, declared_name=self.person, box_left=1, box_top=1, box_right=30, box_bottom=30)
        b = make_face(self.img, declared_name=self.other_person, box_left=50, box_top=50, box_right=80, box_bottom=80)
        call_command('dedupe_overlapping_faces', '--yes')
        self.assertEqual(Face.objects.filter(pk__in=[a.pk, b.pk]).count(), 2)

    def test_rerunning_after_cleanup_finds_nothing(self):
        self._duplicate_pair()
        call_command('dedupe_overlapping_faces', '--yes')
        call_command('dedupe_overlapping_faces', '--dry-run')
        self.assertEqual(Face.objects.filter(source_image_file=self.img).count(), 1)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class MergeDuplicateImageFilesTests(TestCase):
    """merge_duplicate_imagefiles management command -- cleans up the
    duplicate ImageFile rows left behind by the (now-fixed, see
    filepopulator/scripts.py's create_image_file()) missing-return bug:
    a genuine duplicate file used to get BOTH a correct DuplicateFile
    record AND a redundant ImageFile row of its own, which could carry
    real independent human work (labels, validations). Test data seeds
    the contaminated state directly via bulk_create, mirroring
    filepopulator.tests's own approach for the same reason: the normal
    create_image_file() ingestion path no longer produces this state at
    all now that it's fixed."""

    def setUp(self):
        self.person = make_person("Merge Person")
        self.primary_img = make_image()

    def _make_contaminated_duplicate(self, primary):
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(primary.filename))
        dup_filename = primary.filename + '.dup.jpg'
        ImageFile.objects.bulk_create([ImageFile(
            filename=dup_filename, directory=directory,
            pixel_hash=primary.pixel_hash, file_hash=primary.file_hash,
            width=primary.width, height=primary.height, isProcessed=False,
            thumbnail_small='',
        )])
        dup = ImageFile.objects.get(filename=dup_filename)
        DuplicateFile.objects.create(filename=dup_filename)
        return dup

    def test_face_is_transferred_from_duplicate_to_primary(self):
        dup = self._make_contaminated_duplicate(self.primary_img)
        face = make_face(dup, declared_name=self.person, validated=True,
                          box_left=1, box_top=1, box_right=30, box_bottom=30)

        call_command('merge_duplicate_imagefiles', '--yes')

        face.refresh_from_db()
        self.assertEqual(face.source_image_file_id, self.primary_img.id)
        self.assertFalse(ImageFile.objects.filter(pk=dup.pk).exists())

    def test_duplicate_face_pairs_after_transfer_collapse_preferring_validated(self):
        dup = self._make_contaminated_duplicate(self.primary_img)
        blank = get_default_blank_person()
        # A face already exists on the primary at this box, unlabeled/unvalidated.
        make_face(self.primary_img, declared_name=blank,
                  box_left=1, box_top=1, box_right=30, box_bottom=30)
        # The duplicate's copy of the SAME face was validated + labeled --
        # a human tagged it without knowing it was on a duplicate photo.
        dup_face = make_face(dup, declared_name=self.person, validated=True,
                              box_left=1, box_top=1, box_right=30, box_bottom=30)

        call_command('merge_duplicate_imagefiles', '--yes')

        remaining = Face.objects.get(source_image_file=self.primary_img)
        self.assertEqual(remaining.pk, dup_face.pk)
        self.assertTrue(remaining.validated)

    def test_unresolved_duplicate_is_left_alone_when_no_primary_found(self):
        # A DuplicateFile record with no OTHER ImageFile sharing its
        # pixel_hash -- e.g. the primary was itself separately removed.
        # Must be left alone, not guessed at or deleted.
        directory, _ = Directory.objects.get_or_create(dir_path='/tmp')
        ImageFile.objects.bulk_create([ImageFile(
            filename='/tmp/orphan_dup.jpg', directory=directory,
            pixel_hash='no_matching_hash_at_all', file_hash='x',
            width=10, height=10, isProcessed=False,
            thumbnail_small='',
        )])
        DuplicateFile.objects.create(filename='/tmp/orphan_dup.jpg')

        call_command('merge_duplicate_imagefiles', '--yes')

        self.assertTrue(ImageFile.objects.filter(filename='/tmp/orphan_dup.jpg').exists())

    def test_dry_run_changes_nothing(self):
        dup = self._make_contaminated_duplicate(self.primary_img)
        face = make_face(dup, declared_name=self.person, validated=True,
                          box_left=1, box_top=1, box_right=30, box_bottom=30)

        call_command('merge_duplicate_imagefiles', '--dry-run')

        self.assertTrue(ImageFile.objects.filter(pk=dup.pk).exists())
        face.refresh_from_db()
        self.assertEqual(face.source_image_file_id, dup.pk)

    def test_collapsed_away_persons_face_is_actually_gone(self):
        # A plain transfer with no collision doesn't change any person's
        # face count (declared_name is untouched by moving source_image_
        # file) -- this exercises the case where the post-transfer
        # collapse actually DELETES a face (the losing side's only face).
        dup = self._make_contaminated_duplicate(self.primary_img)
        losing_person = make_person("Losing Person")
        make_face(self.primary_img, declared_name=losing_person,
                  box_left=1, box_top=1, box_right=30, box_bottom=30)
        make_face(dup, declared_name=self.person, validated=True,
                  box_left=1, box_top=1, box_right=30, box_bottom=30)

        call_command('merge_duplicate_imagefiles', '--yes')

        self.assertEqual(losing_person.face_declared.count(), 0)

    def test_rerunning_after_merge_finds_nothing_more_to_do(self):
        dup = self._make_contaminated_duplicate(self.primary_img)
        make_face(dup, declared_name=self.person,
                  box_left=1, box_top=1, box_right=30, box_bottom=30)

        call_command('merge_duplicate_imagefiles', '--yes')
        call_command('merge_duplicate_imagefiles', '--dry-run')

        dup_filenames = set(DuplicateFile.objects.values_list('filename', flat=True))
        self.assertEqual(ImageFile.objects.filter(filename__in=dup_filenames).count(), 0)



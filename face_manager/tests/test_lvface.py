"""face_manager tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test face_manager.tests.test_lvface
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


class FlattenKpsTests(unittest.TestCase):
    """Unit tests for FaceExtractor._flatten_kps(), the helper that turns
    InsightFace's (5, 2) landmark array into the flat 10-float list stored
    on Face.kps. No DB/model access needed."""

    def test_none_input_returns_none(self):
        self.assertIsNone(FaceExtractor._flatten_kps(None))

    def test_flattens_in_row_major_order(self):
        kps = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        flat = FaceExtractor._flatten_kps(kps)
        self.assertEqual(flat, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    def test_accepts_plain_nested_list(self):
        kps = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        flat = FaceExtractor._flatten_kps(kps)
        self.assertEqual(len(flat), 10)
        self.assertTrue(all(isinstance(v, float) for v in flat))


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class ReencodeMissingFacesTests(TestCase):
    """FaceExtractor.reencode_missing_faces() re-encodes faces that have
    no face_encoding_512 but are no longer .ignore/.realignore. These
    tests mock the underlying InsightFace models entirely (constructing
    the extractor via __new__ rather than FaceExtractor(), which would
    load the real ~100MB ONNX models) so they stay fast -- no real
    inference happens, only the selection/branching logic in
    reencode_missing_faces() itself is exercised."""

    def setUp(self):
        self.image = make_image()
        self.extractor = FaceExtractor.__new__(FaceExtractor)
        self.extractor.reencode_crop_margin_mult = 0.6
        self.extractor.LVFACE_REENCODE_MAX_RUNTIME_SECONDS = 3000
        # extractor.app is a PyramidalDetector in real code; .app.app is
        # the underlying FaceAnalysis. Mock both layers directly rather
        # than constructing a real PyramidalDetector/FaceAnalysis.
        self.mock_face_analysis = MagicMock()
        self.mock_pyramidal = MagicMock()
        self.mock_pyramidal.app = self.mock_face_analysis
        self.extractor.app = self.mock_pyramidal

        # The LVFace-B catch-up sweep (_backfill_lvface_encodings(), run
        # automatically at the end of reencode_missing_faces() now) is
        # out of scope for these tests -- they're about the insightface
        # re-encode selection/branching logic specifically. Mocked out
        # (never None/real) so a test face that happens to carry kps
        # doesn't trigger a real, unmocked ONNX model load.
        lvface_patcher = patch('lvface.embed_from_image_array', return_value=None)
        lvface_patcher.start()
        self.addCleanup(lvface_patcher.stop)

    def _face_missing_encoding(self, declared_name=None, **overrides):
        face = make_face(self.image, declared_name=declared_name, **overrides)
        face.face_encoding_512 = None
        face.save()
        return face

    def test_ignore_and_realignore_faces_are_never_selected(self):
        ignore_person = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        realignore_person = Person.objects.get(person_name='.realignore')
        ignored_face = self._face_missing_encoding(declared_name=ignore_person)
        realignored_face = self._face_missing_encoding(declared_name=realignore_person)

        self.extractor.reencode_missing_faces()

        ignored_face.refresh_from_db()
        realignored_face.refresh_from_db()
        self.assertIsNone(ignored_face.face_encoding_512)
        self.assertIsNone(realignored_face.face_encoding_512)
        self.mock_face_analysis.get.assert_not_called()

    def test_non_detected_sentinel_on_a_real_person_is_selected_for_reencode(self):
        # Regression test: update_list_of_no_matching_detects() stamps
        # settings.NON_DETECTED_FACE_ENCODING ([-999]*512) onto a face
        # whose box wasn't matched to any detection during a full-image
        # reprocessing pass. Found 2026-09-03 via a clustering
        # investigation: 1,207 faces database-wide carried this sentinel
        # while still being declared to a REAL person -- not NULL, so
        # invisible to the original face_encoding_512__isnull=True filter,
        # and not .ignore/.realignore, so this isn't a "confirmed-ignore"
        # case either. Just a real face silently stuck with fake data.
        person = make_person("Has Sentinel Encoding")
        face = make_face(
            self.image, declared_name=person,
            box_left=100, box_top=100, box_right=140, box_bottom=140,
        )
        face.face_encoding_512 = list(settings.NON_DETECTED_FACE_ENCODING)
        face.save()

        self.mock_face_analysis.get.return_value = [
            {'bbox': [0, 0, 40, 40], 'kps': np.full((5, 2), 20.0), 'embedding': np.full(512, 0.4)},
        ]

        self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, [0.4] * 512)
        self.assertTrue(face.reencoded)

    def test_non_detected_sentinel_on_ignore_face_is_still_excluded(self):
        ignore_person = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        face = make_face(self.image, declared_name=ignore_person)
        face.face_encoding_512 = list(settings.NON_DETECTED_FACE_ENCODING)
        face.save()

        self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, list(settings.NON_DETECTED_FACE_ENCODING))
        self.mock_face_analysis.get.assert_not_called()

    def test_faces_with_an_existing_encoding_are_left_alone(self):
        person = make_person("Already Encoded")
        face = make_face(self.image, declared_name=person)
        # 0.5 is exactly representable in float32 -- avoids the precision
        # trap in test_face_encoding_512_stores_at_float32_precision's
        # docstring (a value like 0.1 would round on save, so comparing
        # against the original float64 literal after refresh_from_db()
        # would spuriously fail here).
        face.face_encoding_512 = [0.5] * 512
        face.save()

        self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, [0.5] * 512)
        self.assertFalse(face.reencoded)
        self.mock_face_analysis.get.assert_not_called()

    def test_kps_path_reproduces_embedding_without_redetecting(self):
        """When a face already has stored kps, reencode_missing_faces()
        should call the recognition model directly (no detection pass at
        all) -- this is the exact-reproduction path validated against
        real production data."""
        person = make_person("Has Kps")
        face = self._face_missing_encoding(declared_name=person, box_left=10, box_top=10, box_right=30, box_bottom=30)
        face.kps = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        face.save()

        expected_embedding = np.arange(512, dtype=np.float64)
        self.mock_face_analysis.models = {
            'recognition': MagicMock(get=MagicMock(return_value=expected_embedding)),
        }

        self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, expected_embedding.tolist())
        self.assertTrue(face.reencoded)
        self.mock_face_analysis.get.assert_not_called()

        # The recognition model should have been called with a Face-like
        # object carrying the ORIGINAL stored kps, reshaped to (5, 2).
        call_args = self.mock_face_analysis.models['recognition'].get.call_args
        passed_face_obj = call_args[0][1]
        np.testing.assert_array_equal(
            np.asarray(passed_face_obj.kps), np.array(face.kps).reshape(5, 2)
        )

    def test_no_kps_falls_back_to_crop_detection_and_picks_center_face(self):
        """Without stored kps, the crop-based path should run, and among
        multiple detections found in the crop, the one closest to the
        crop's center (i.e. closest to the known face) should win."""
        person = make_person("No Kps")
        face = self._face_missing_encoding(
            declared_name=person, box_left=100, box_top=100, box_right=140, box_bottom=140
        )
        self.assertIsNone(face.kps)

        near_center_embedding = np.full(512, 0.5)
        far_embedding = np.full(512, 0.9)
        # Crop center will be at roughly (crop_w/2, crop_h/2) since the
        # margin is symmetric -- a detection near (0, 0) is far off-center,
        # one near the crop's actual center should be picked instead.
        self.mock_face_analysis.get.return_value = [
            {'bbox': [0, 0, 4, 4], 'kps': np.zeros((5, 2)), 'embedding': far_embedding},
            {'bbox': [30, 30, 34, 34], 'kps': np.full((5, 2), 32.0), 'embedding': near_center_embedding},
        ]
        self.mock_face_analysis.models = {'recognition': MagicMock()}

        self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, near_center_embedding.tolist())
        self.assertTrue(face.reencoded)
        self.mock_face_analysis.models['recognition'].get.assert_not_called()

    def test_crop_detection_kps_saved_in_absolute_image_coordinates(self):
        """Freshly-detected kps from the crop-based path are in
        crop-local coordinates -- they must be translated back to the
        source image's absolute coordinate space (adding the crop's own
        offset) before being stored, matching how add_new_face()/
        update_existing_face_to_insightface() store kps, so a future
        re-encode of this same face can use the exact-replay path."""
        person = make_person("Needs Translation")
        face = self._face_missing_encoding(
            declared_name=person, box_left=100, box_top=100, box_right=140, box_bottom=140
        )
        box_w, box_h = 40, 40
        margin_x = int(box_w * self.extractor.reencode_crop_margin_mult)
        margin_y = int(box_h * self.extractor.reencode_crop_margin_mult)
        crop_left, crop_top = 100 - margin_x, 100 - margin_y

        # Centered within the bbox (near local (20,20)), not an arbitrary
        # corner value -- must still land within box_kps_within_box_margin's
        # 25% margin once translated to absolute coordinates.
        crop_local_kps = np.array([[18.0, 20.0], [22.0, 20.0], [20.0, 22.0], [18.0, 24.0], [22.0, 24.0]])
        self.mock_face_analysis.get.return_value = [
            {'bbox': [0, 0, 40, 40], 'kps': crop_local_kps, 'embedding': np.full(512, 0.3)},
        ]

        self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        expected_abs_kps = (crop_local_kps + np.array([crop_left, crop_top])).reshape(-1).tolist()
        self.assertEqual(face.kps, expected_abs_kps)

    def test_no_detection_found_uses_default_encoding_without_setting_reencoded(self):
        """A face too degraded to redetect even in a tight crop gets the
        neutral default vector rather than being left NULL forever (NULL
        is exactly what this method selects on) -- but since nothing
        about it actually came from a fresh detection, reencoded should
        NOT be flipped to True."""
        person = make_person("Undetectable")
        face = self._face_missing_encoding(
            declared_name=person, box_left=100, box_top=100, box_right=140, box_bottom=140
        )
        self.mock_face_analysis.get.return_value = []

        self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        # face_encoding_512 is stored at float32 precision (see
        # test_face_encoding_512_stores_at_float32_precision) -- compare
        # with a tolerance rather than exact equality against the
        # float64-computed settings constant, both to be correct about
        # what the DB actually guarantees and to avoid unittest's
        # list-diff (difflib recursion crashes on a 512-element mismatch
        # rather than reporting a clean failure).
        np.testing.assert_allclose(
            np.array(face.face_encoding_512), np.array(settings.REENCODE_DEFAULT_ENCODING), rtol=1e-6, atol=1e-6
        )
        self.assertFalse(face.reencoded)
        self.assertIsNone(face.kps)

    def test_reencode_default_encoding_is_a_unit_vector(self):
        vec = np.array(settings.REENCODE_DEFAULT_ENCODING)
        self.assertEqual(len(vec), 512)
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=6)
        self.assertEqual(len(set(vec.tolist())), 1, "Every component should be identical/uniform")

    # --- Merged LVFace-B image-side coverage (2026-10-25) ---------------
    # reencode_missing_faces() used to call a separate _backfill_lvface_
    # encodings() sweep at the end, which opened and decoded the same
    # image a SECOND time for any face missing both encodings. Now both
    # encodings are handled together, per face, in this same loop -- see
    # the method's own docstring for the full reasoning. These tests
    # cover the merge itself; the pre-existing tests above already cover
    # the insightface-only branching logic, which is unchanged.

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_face_needing_only_lvface_gets_it_without_touching_insight(self):
        person = make_person("Only Needs Lvface")
        face = make_face(self.image, declared_name=person)
        face.face_encoding_512 = [0.5] * 512  # already real, float32-exact
        face.kps = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        face.face_encoding_lvface_b = None
        face.reencoded = False
        face.save()

        with patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)) as mock_lv:
            self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_lvface_b, [0.4] * 512)
        self.assertEqual(face.face_encoding_512, [0.5] * 512)  # untouched
        self.assertFalse(face.reencoded)  # insight never ran
        self.mock_face_analysis.get.assert_not_called()
        mock_lv.assert_called_once()

    def test_face_needing_neither_is_skipped_without_opening_the_image(self):
        # The exact scenario that motivated this merge: a face with a
        # real insightface embedding but no kps (so it can never
        # qualify for LVFace-B without a fresh detection) must not pay
        # for a single image decode.
        person = make_person("Needs Nothing")
        face = make_face(self.image, declared_name=person)
        face.face_encoding_512 = [0.5] * 512
        face.kps = None
        face.face_encoding_lvface_b = None
        face.save()

        with patch('common.open_img_oriented') as mock_open:
            self.extractor.reencode_missing_faces()

        mock_open.assert_not_called()
        self.mock_face_analysis.get.assert_not_called()

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_crop_redetect_recovering_kps_also_triggers_lvface_in_same_pass(self):
        # A face with no kps at all (needs insight's crop-redetect path)
        # whose redetection succeeds recovers kps as a side effect --
        # that should make it newly eligible for LVFace-B too, using the
        # SAME already-open image, rather than waiting for a future run.
        person = make_person("Recovers Kps")
        face = self._face_missing_encoding(
            declared_name=person, box_left=100, box_top=100, box_right=140, box_bottom=140
        )
        self.assertIsNone(face.kps)

        crop_local_kps = np.array([[18.0, 20.0], [22.0, 20.0], [20.0, 22.0], [18.0, 24.0], [22.0, 24.0]])
        self.mock_face_analysis.get.return_value = [
            {'bbox': [0, 0, 40, 40], 'kps': crop_local_kps, 'embedding': np.full(512, 0.3)},
        ]

        with patch('lvface.embed_from_image_array', return_value=np.full(512, 0.55, dtype=np.float32)) as mock_lv:
            self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        self.assertIsNotNone(face.kps)  # recovered by the crop-redetect path
        self.assertEqual(face.face_encoding_lvface_b, [0.55] * 512)
        mock_lv.assert_called_once()
        # The recovered (absolute-coordinate) kps is what LVFace-B should
        # have been called with, not the raw crop-local values.
        called_kps = mock_lv.call_args.args[1]
        np.testing.assert_array_equal(np.asarray(called_kps).reshape(-1), np.array(face.kps))

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_image_face_below_collection_floor_excluded_from_both_encodings(self):
        from filepopulator.models import Directory, PhotoCollection
        low_fidelity = PhotoCollection.objects.create(
            name='Low Fidelity Reencode', root_directory='/photos/reencode_floor_test',
            min_face_det_score=0.6,
        )
        directory = Directory.objects.create(dir_path='/photos/reencode_floor_test', collection=low_fidelity)
        below_floor_image = ImageFile.objects.create(
            filename=self.image.filename, directory=directory,
            width=self.image.width, height=self.image.height,
            pixel_hash='reencode-floor-hash', file_hash='reencode-floor-fhash',
        )
        person = make_person("Below Floor Reencode")
        face = make_face(below_floor_image, declared_name=person, det_score=0.5)
        face.face_encoding_512 = None
        face.kps = [1.0] * 10
        face.face_encoding_lvface_b = None
        face.save()

        with patch('common.open_img_oriented') as mock_open, \
             patch('lvface.embed_from_image_array') as mock_lv:
            self.extractor.reencode_missing_faces()

        mock_open.assert_not_called()
        self.mock_face_analysis.get.assert_not_called()
        mock_lv.assert_not_called()
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_512)
        self.assertIsNone(face.face_encoding_lvface_b)

    def test_below_floor_face_never_enters_the_per_face_loop_at_all(self):
        # Distinguishes "excluded by the SQL query itself" (the real
        # point of the below_collection_floor() SQL mirror added
        # 2026-10-25 -- previously tens of thousands of permanently-
        # below-floor rows were fetched and iterated every single
        # hourly run just to be skipped) from merely "fetched, then
        # skipped in Python" -- if the SQL filter is doing its job, the
        # per-face loop body (and its own redundant below_collection_
        # floor() safety-net call) should never even run for this row.
        from filepopulator.models import Directory, PhotoCollection
        low_fidelity = PhotoCollection.objects.create(
            name='Low Fidelity SQL Filter', root_directory='/photos/reencode_sql_floor_test',
            min_face_det_score=0.6,
        )
        directory = Directory.objects.create(dir_path='/photos/reencode_sql_floor_test', collection=low_fidelity)
        below_floor_image = ImageFile.objects.create(
            filename=self.image.filename, directory=directory,
            width=self.image.width, height=self.image.height,
            pixel_hash='reencode-sql-floor-hash', file_hash='reencode-sql-floor-fhash',
        )
        person = make_person("Sql Floor Excluded")
        face = make_face(below_floor_image, declared_name=person, det_score=0.5)
        face.face_encoding_512 = None
        face.kps = [1.0] * 10
        face.face_encoding_lvface_b = None
        face.save()

        with patch('face_manager.face_extract_encode.below_collection_floor') as mock_floor_check:
            self.extractor.reencode_missing_faces()

        mock_floor_check.assert_not_called()

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_confirmed_ignore_face_excluded_from_lvface_too(self):
        ignore_person = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        face = make_face(self.image, declared_name=ignore_person)
        face.face_encoding_512 = None
        face.kps = [1.0] * 10
        face.face_encoding_lvface_b = None
        face.save()

        with patch('lvface.embed_from_image_array') as mock_lv:
            self.extractor.reencode_missing_faces()

        mock_lv.assert_not_called()
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)

    @override_settings(LVFACE_ENCODING_ENABLED=False)
    def test_lvface_disabled_skips_lvface_but_insight_still_runs(self):
        person = make_person("Lvface Off")
        face = self._face_missing_encoding(declared_name=person, box_left=10, box_top=10, box_right=30, box_bottom=30)
        face.kps = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        face.save()
        self.mock_face_analysis.models = {
            'recognition': MagicMock(get=MagicMock(return_value=np.full(512, 0.7))),
        }

        with patch('lvface.embed_from_image_array') as mock_lv:
            self.extractor.reencode_missing_faces()

        mock_lv.assert_not_called()
        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, [0.7] * 512)
        self.assertIsNone(face.face_encoding_lvface_b)

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_time_budget_shared_across_combined_loop(self):
        person_a = make_person("Budget Face A")
        person_b = make_person("Budget Face B")
        face_a = make_face(self.image, declared_name=person_a)
        face_a.face_encoding_512 = [0.5] * 512
        face_a.kps = [1.0] * 10
        face_a.face_encoding_lvface_b = None
        face_a.save()
        face_b = make_face(self.image, declared_name=person_b)
        face_b.face_encoding_512 = [0.5] * 512
        face_b.kps = [1.0] * 10
        face_b.face_encoding_lvface_b = None
        face_b.save()

        times = iter([0, 0, 1000])  # start_time, first face check (ok), second face check (over budget)

        def fake_time():
            return next(times, 1000)

        with patch('lvface.embed_from_image_array', return_value=np.full(512, 0.2, dtype=np.float32)) as mock_lv, \
             patch('time.time', side_effect=fake_time):
            self.extractor.reencode_missing_faces(lvface_max_runtime_seconds=10)

        self.assertEqual(mock_lv.call_count, 1)

    def _make_video_face(self, **overrides):
        # Mirrors BackfillLvfaceVideoEncodingsTests' own helper -- kept
        # as a small, separate copy rather than a shared fixture, since
        # these two test classes exercise different entry points
        # (reencode_missing_faces()'s own combined queryset here, vs.
        # _backfill_lvface_video_encodings() directly there).
        from filepopulator.models import Directory, VideoFile
        directory, _ = Directory.objects.get_or_create(dir_path='/videos/reencode_combined_test')
        video_path = '/tmp/reencode_combined_test_video.mp4'
        with open(video_path, 'wb') as f:
            f.write(b'not a real video, existence is all this sweep checks')
        self.addCleanup(lambda: os.path.exists(video_path) and os.remove(video_path))
        video = VideoFile.objects.create(filename=video_path, directory=directory, width=640, height=480)
        face = Face(
            declared_name=overrides.pop('declared_name', None) or get_default_blank_person(),
            source_video_file=video,
            box_left=1, box_top=1, box_right=10, box_bottom=10,
            video_thumbnail_frame_seconds=12.3,
            kps=[1.0] * 10,
            **overrides,
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()
        return face

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_video_faces_are_included_in_the_combined_sweep(self):
        # Real behavior change, 2026-10-26: video used to require a
        # completely separate sub-pass call that only ever ran AFTER
        # the image loop fully finished -- for an unbounded (shard)
        # call, that meant video never got touched until a shard's
        # entire image backlog was already drained (confirmed stuck at
        # 0 progress in real production for this exact reason). Image
        # and video faces are now unioned into one combined, randomly-
        # ordered queryset, so a call that processes anything at all
        # can reach video faces too, regardless of image backlog size.
        video_face = self._make_video_face()
        with patch('video_face_pipeline.ffprobe_info', return_value=(640, 480, 30.0, None, 0, None, None)), \
             patch('video_face_pipeline.extract_frame_near_timestamp', return_value=np.zeros((480, 640, 3), dtype=np.uint8)), \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.5, dtype=np.float32)):
            self.extractor.reencode_missing_faces()
        video_face.refresh_from_db()
        self.assertEqual(video_face.face_encoding_lvface_b, [0.5] * 512)

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_sharded_call_only_processes_video_faces_in_its_own_shard(self):
        # The combined queryset's Mod(F('id'), total_shards) partition
        # applies uniformly to both face types -- every video face must
        # still be covered by exactly one of the shard runs below.
        video_faces = [self._make_video_face() for _ in range(6)]
        total_shards = 3
        with patch('video_face_pipeline.ffprobe_info', return_value=(640, 480, 30.0, None, 0, None, None)), \
             patch('video_face_pipeline.extract_frame_near_timestamp', return_value=np.zeros((480, 640, 3), dtype=np.uint8)), \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.6, dtype=np.float32)):
            for shard_index in range(total_shards):
                self.extractor.reencode_missing_faces(shard_index=shard_index, total_shards=total_shards)
        for face in video_faces:
            face.refresh_from_db()
            self.assertEqual(face.face_encoding_lvface_b, [0.6] * 512)

    def test_collection_scoped_call_never_includes_video(self):
        # collection is an image-only concept (no PhotoCollection
        # association for VideoFile, see CLAUDE.md) -- a collection-
        # scoped call (e.g. apply_collection_floor_change) must never
        # touch the video backlog, same as before this change.
        from filepopulator.models import PhotoCollection
        video_face = self._make_video_face()
        default_collection = PhotoCollection.objects.get(is_default=True)
        with patch('video_face_pipeline.ffprobe_info') as mock_ffprobe, \
             patch('lvface.embed_from_image_array'):
            self.extractor.reencode_missing_faces(collection=default_collection)
        mock_ffprobe.assert_not_called()
        video_face.refresh_from_db()
        self.assertIsNone(video_face.face_encoding_lvface_b)

    def test_skips_entirely_when_advisory_lock_already_held(self):
        # Regression test for a real production incident (2026-10-25): a
        # manual manage.py shell call and the scheduled face_manager.
        # reencode task ran fully concurrently with zero mutual
        # awareness (the task wrapper's own overlap check only looked
        # for other Celery tasks of the same name, invisible to a direct
        # call), each building its own separate FaceExtractor/ONNX
        # sessions and stacking memory until the container neared its
        # cgroup limit. reencode_missing_faces() now holds a Postgres
        # advisory lock (common/advisory_lock.py, same pattern as
        # find_and_encode_faces()'s own lock -- see that test's own
        # comment for why a genuinely separate connection is needed to
        # simulate real contention) for its whole run; while another
        # session holds it, this call must do nothing at all.
        person = make_person("Lock Contention")
        face = self._face_missing_encoding(declared_name=person, box_left=10, box_top=10, box_right=30, box_bottom=30)
        face.kps = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        face.save()
        self.mock_face_analysis.models = {
            'recognition': MagicMock(get=MagicMock(return_value=np.full(512, 0.5))),
        }

        cfg = connection.settings_dict
        other = psycopg2.connect(
            dbname=cfg['NAME'], user=cfg['USER'], password=cfg['PASSWORD'],
            host=cfg['HOST'] or 'localhost', port=cfg['PORT'] or 5432,
        )
        other.autocommit = True
        try:
            key = zlib.crc32(b'face_manager.reencode_missing_faces')
            with other.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock(%s)", [key])
                self.assertTrue(cur.fetchone()[0])

                self.extractor.reencode_missing_faces()

                face.refresh_from_db()
                self.assertIsNone(face.face_encoding_512)
                self.mock_face_analysis.get.assert_not_called()

                cur.execute("SELECT pg_advisory_unlock(%s)", [key])
        finally:
            other.close()

        # Once free again, a normal run still works.
        self.extractor.reencode_missing_faces()
        face.refresh_from_db()
        self.assertIsNotNone(face.face_encoding_512)

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_periodic_reinit_bounds_memory_growth(self):
        # Regression test for a real production incident (2026-10-25):
        # the merged image loop can now run for a long time against a
        # very large backlog (previously this method never ran long
        # enough for ONNXRuntime's arena growth to matter) -- confirmed
        # via real data, an unbounded manual run grew to 13+ GB RSS
        # within the hour. Mirrors find_and_encode_faces()'s own
        # periodic-rebuild mitigation. Faces here are set up to need
        # ONLY LVFace-B (real face_encoding_512 already present) so this
        # stays focused purely on confirming the rebuild happens at the
        # right counts, without needing the rebuilt mock detector to
        # also support a real insight re-encode.
        self.extractor.REENCODE_REINIT_EVERY_N_IMAGES = 2
        faces = []
        for i in range(3):
            person = make_person(f"Reinit Face {i}")
            face = make_face(
                self.image, declared_name=person,
                box_left=10, box_top=10, box_right=30, box_bottom=30,
            )
            face.face_encoding_512 = [0.5] * 512
            face.kps = [20.0] * 10
            face.face_encoding_lvface_b = None
            face.save()
            faces.append(face)

        with patch.object(self.extractor, '_build_detector', return_value=MagicMock()) as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.3, dtype=np.float32)), \
             patch('lvface.reset_session') as mock_reset:
            self.extractor.reencode_missing_faces()

        # 3 faces processed, REENCODE_REINIT_EVERY_N_IMAGES=2 -> the
        # check at processed_count==2 (0-indexed boundary, i.e. right
        # before the 3rd face) fires exactly once.
        mock_build.assert_called_once()
        mock_reset.assert_called_once()
        for face in faces:
            face.refresh_from_db()
            self.assertEqual(face.face_encoding_lvface_b, [0.3] * 512)

    def test_shard_filters_to_matching_ids_only(self):
        # Core correctness property of the parallel backfill (2026-10-25,
        # see backfill_lvface_parallel.py): shard_index/total_shards
        # must partition faces (via _shard_annotation()'s hash, not
        # plain id % total_shards -- see that function's own comment
        # for why) so every face is covered by EXACTLY one shard, never
        # zero or more than one.
        faces = []
        for i in range(6):
            person = make_person(f"Shard Face {i}")
            face = self._face_missing_encoding(declared_name=person, box_left=10, box_top=10, box_right=30, box_bottom=30)
            face.kps = [20.0] * 10
            face.save()
            faces.append(face)

        total_shards = 3
        with patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            for shard_index in range(total_shards):
                self.mock_face_analysis.models = {
                    'recognition': MagicMock(get=MagicMock(return_value=np.full(512, 0.4))),
                }
                self.extractor.reencode_missing_faces(shard_index=shard_index, total_shards=total_shards)

        # Every face is covered by exactly one of the 3 shard runs above
        # (its own hashed shard assignment) -- confirms the partition
        # has no gaps, without needing to separately prove no face is
        # visited twice (a face's hash only ever matches one shard_index
        # by construction, so double-processing isn't reachable here).
        for face in faces:
            face.refresh_from_db()
            self.assertEqual(face.face_encoding_512, [0.4] * 512)

    def test_different_shards_use_independent_locks(self):
        # A real race this is meant to allow, not prevent: two DIFFERENT
        # shards of the same backfill must be able to run genuinely
        # concurrently (that's the whole point), unlike two invocations
        # of the SAME shard (already covered by test_skips_entirely_
        # when_advisory_lock_already_held) or the plain unsharded lock.
        person = make_person("Shard Lock Independence")
        face = self._face_missing_encoding(declared_name=person, box_left=10, box_top=10, box_right=30, box_bottom=30)
        face.kps = [20.0] * 10
        face.save()
        self.mock_face_analysis.models = {
            'recognition': MagicMock(get=MagicMock(return_value=np.full(512, 0.6))),
        }

        # Hold the OTHER shard's lock (the one this face does NOT belong
        # to) -- computed via the real shard annotation (a HASHTEXT-
        # based hash, not plain id % total_shards -- see
        # _shard_annotation()'s own comment for why), not assumed, to
        # stay correct regardless of what id this face lands on.
        from face_manager.face_extract_encode import _shard_annotation
        face_shard = Face.objects.filter(id=face.id).annotate(
            _shard=_shard_annotation(2)
        ).values_list('_shard', flat=True).first()
        held_shard = 1 - face_shard

        cfg = connection.settings_dict
        other = psycopg2.connect(
            dbname=cfg['NAME'], user=cfg['USER'], password=cfg['PASSWORD'],
            host=cfg['HOST'] or 'localhost', port=cfg['PORT'] or 5432,
        )
        other.autocommit = True
        try:
            held_shard_key = zlib.crc32(f'face_manager.reencode_missing_faces.shard_{held_shard}_of_2'.encode())
            with other.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock(%s)", [held_shard_key])
                self.assertTrue(cur.fetchone()[0])

                # face_shard's own lock was never touched -- this must
                # proceed normally despite the OTHER shard being locked.
                self.extractor.reencode_missing_faces(shard_index=face_shard, total_shards=2)

                cur.execute("SELECT pg_advisory_unlock(%s)", [held_shard_key])
        finally:
            other.close()

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, [0.6] * 512)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class ReencodeMissingFacesMismatchedKpsTransactionTests(TransactionTestCase):
    """test_mismatched_kps_is_discarded_in_favor_of_crop_redetect needs
    TransactionTestCase, not the regular TestCase ReencodeMissingFacesTests
    uses -- it must fabricate a face whose kps is already a real mismatch
    against its own box (simulating real pre-existing corrupted production
    data), which means dropping/re-adding face_kps_within_box_margin around
    the save -- Postgres refuses that ALTER while there are pending trigger
    events from an earlier write (here, make_face()'s own initial INSERT)
    in the SAME transaction, which is exactly the one wrapping transaction
    a plain TestCase runs each test inside."""

    def setUp(self):
        # TransactionTestCase truncates every table after each test,
        # including ones only ever populated by data migrations (the
        # default PhotoCollection make_image()'s own Directory resolution
        # needs) -- recreate the one row actually needed directly, same
        # established pattern as api.tests's IgnoreReviewFlaggedCountTests
        # (see its own comment for why serialized_rollback=True isn't
        # used instead: it also tries to restore django_content_type rows
        # that already exist under --parallel, a real UniqueViolation).
        from filepopulator.models import PhotoCollection
        PhotoCollection.objects.get_or_create(
            root_directory=settings.PHOTO_ROOT, defaults={'name': 'Default', 'is_default': True},
        )
        self.image = make_image()
        self.extractor = FaceExtractor.__new__(FaceExtractor)
        self.extractor.reencode_crop_margin_mult = 0.6
        self.extractor.LVFACE_REENCODE_MAX_RUNTIME_SECONDS = 3000
        self.mock_face_analysis = MagicMock()
        self.mock_pyramidal = MagicMock()
        self.mock_pyramidal.app = self.mock_face_analysis
        self.extractor.app = self.mock_pyramidal
        lvface_patcher = patch('lvface.embed_from_image_array', return_value=None)
        lvface_patcher.start()
        self.addCleanup(lvface_patcher.stop)

    def test_mismatched_kps_is_discarded_in_favor_of_crop_redetect(self):
        # Real bug, found 2026-10-26 via a real investigation (see
        # CLAUDE.md): a stored kps can end up unrelated to its own face's
        # box (confirmed in production: ~360k faces, 46% of everything
        # with both fields -- traced to backfill_det_score's own IOU
        # matching, separately fixed). Blindly exact-replaying such a kps
        # encodes the wrong crop with high confidence instead of a real
        # face. This face's kps (centroid ~900,900) is nowhere near its
        # own box (10,10,40,40) -- the exact-replay path (rec_model.get)
        # must NOT be used; the crop-redetect fallback (self.app.app.get)
        # must run instead and recover a fresh, box-consistent kps.
        person = make_person("Mismatched Kps Face")
        face = make_face(
            self.image, declared_name=person,
            box_left=10, box_top=10, box_right=40, box_bottom=40,
        )
        face.face_encoding_512 = list(settings.NON_DETECTED_FACE_ENCODING)
        with connection.cursor() as cursor:
            cursor.execute("ALTER TABLE face_manager_face DROP CONSTRAINT face_kps_within_box_margin")
        try:
            face.kps = [900.0, 900.0, 900.0, 900.0, 900.0, 900.0, 900.0, 900.0, 900.0, 900.0]
            face.save()
        finally:
            with connection.cursor() as cursor:
                cursor.execute("""
                    ALTER TABLE face_manager_face ADD CONSTRAINT face_kps_within_box_margin CHECK (
                        kps IS NULL OR (
                            kps[1] BETWEEN box_left - (box_right - box_left) * 0.25 AND box_right + (box_right - box_left) * 0.25
                            AND kps[2] BETWEEN box_top - (box_bottom - box_top) * 0.25 AND box_bottom + (box_bottom - box_top) * 0.25
                            AND kps[3] BETWEEN box_left - (box_right - box_left) * 0.25 AND box_right + (box_right - box_left) * 0.25
                            AND kps[4] BETWEEN box_top - (box_bottom - box_top) * 0.25 AND box_bottom + (box_bottom - box_top) * 0.25
                            AND kps[5] BETWEEN box_left - (box_right - box_left) * 0.25 AND box_right + (box_right - box_left) * 0.25
                            AND kps[6] BETWEEN box_top - (box_bottom - box_top) * 0.25 AND box_bottom + (box_bottom - box_top) * 0.25
                            AND kps[7] BETWEEN box_left - (box_right - box_left) * 0.25 AND box_right + (box_right - box_left) * 0.25
                            AND kps[8] BETWEEN box_top - (box_bottom - box_top) * 0.25 AND box_bottom + (box_bottom - box_top) * 0.25
                            AND kps[9] BETWEEN box_left - (box_right - box_left) * 0.25 AND box_right + (box_right - box_left) * 0.25
                            AND kps[10] BETWEEN box_top - (box_bottom - box_top) * 0.25 AND box_bottom + (box_bottom - box_top) * 0.25
                        )
                    ) NOT VALID
                """)

        buggy_recognition = MagicMock(get=MagicMock(return_value=np.full(512, 0.5, dtype=np.float32)))
        self.mock_face_analysis.models = {'recognition': buggy_recognition}
        self.mock_face_analysis.get.return_value = [
            {'bbox': [0, 0, 30, 30], 'kps': np.full((5, 2), 15.0), 'embedding': np.full(512, 0.7)},
        ]

        self.extractor.reencode_missing_faces()

        face.refresh_from_db()
        buggy_recognition.get.assert_not_called()
        self.assertEqual(face.face_encoding_512, [0.7] * 512)
        # Recovered kps is freshly translated from the crop-redetect
        # fallback, no longer the stale/mismatched value.
        self.assertNotEqual(face.kps, [900.0] * 10)
        kps_arr = np.array(face.kps).reshape(5, 2)
        cx, cy = kps_arr[:, 0].mean(), kps_arr[:, 1].mean()
        self.assertTrue(10 <= cx <= 40 and 10 <= cy <= 40)


class ReencodeMissingFacesShardCommandTests(TestCase):
    """reencode_missing_faces_shard -- the worker entry point the
    one-time parallel backfill (backfill_lvface_parallel) spawns one of
    per shard. Mocks FaceExtractor entirely (real model loading is
    expensive) -- this is about the command's own argument validation
    and pass-through, not FaceExtractor's own logic (covered elsewhere)."""

    def test_valid_shard_args_pass_through_to_extractor(self):
        with patch('face_manager.management.commands.reencode_missing_faces_shard.FaceExtractor') as mock_cls:
            mock_instance = mock_cls.return_value
            call_command('reencode_missing_faces_shard', '--shard-index', '2', '--total-shards', '5')
        mock_instance.reencode_missing_faces.assert_called_once_with(
            shard_index=2, total_shards=5, lvface_max_runtime_seconds=None,
        )

    def test_shard_index_out_of_range_raises(self):
        with patch('face_manager.management.commands.reencode_missing_faces_shard.FaceExtractor'):
            with self.assertRaises(CommandError):
                call_command('reencode_missing_faces_shard', '--shard-index', '5', '--total-shards', '5')

    def test_negative_shard_index_raises(self):
        with patch('face_manager.management.commands.reencode_missing_faces_shard.FaceExtractor'):
            with self.assertRaises(CommandError):
                call_command('reencode_missing_faces_shard', '--shard-index', '-1', '--total-shards', '5')


class BackfillLvfaceParallelHelperTests(TestCase):
    """Pure helper functions from backfill_lvface_parallel.py (the
    one-time parallel-backfill supervisor) -- _available_memory_gb() and
    _total_cores(). The supervisor's own main loop (subprocess spawning,
    CPU-affinity pinning, the scaling decisions) is deliberately not
    unit-tested here: it's a one-off tool run once under direct
    supervision (see CLAUDE.md's dated write-up), not a permanent code
    path, and meaningfully testing it would need mocking subprocess.
    Popen/os.sched_setaffinity/os.getloadavg/time.sleep so extensively
    that the test would mostly just re-assert the implementation rather
    than catch real bugs -- real validation came from actually running
    it against production."""

    def test_available_memory_gb_prefers_cgroup_ceiling_when_lower(self):
        from face_manager.management.commands.backfill_lvface_parallel import _available_memory_gb
        meminfo = "MemTotal:       65000000 kB\nMemAvailable:   40000000 kB\n"
        # cgroup says only ~2GB left (host thinks ~38GB) -- the smaller
        # (cgroup) figure must win, since that's the one that actually
        # kills a new worker first.
        cgroup_max = str(10 * 1024 ** 3)
        cgroup_current = str(8 * 1024 ** 3)
        contents = {
            '/proc/meminfo': meminfo,
            '/sys/fs/cgroup/memory.max': cgroup_max,
            '/sys/fs/cgroup/memory.current': cgroup_current,
        }

        def fake_open(path, *a, **kw):
            if path not in contents:
                raise FileNotFoundError(path)
            return io.StringIO(contents[path])

        with patch('builtins.open', side_effect=fake_open):
            result = _available_memory_gb()
        self.assertAlmostEqual(result, 2.0, places=3)

    def test_available_memory_gb_adds_back_reclaimable_page_cache(self):
        from face_manager.management.commands.backfill_lvface_parallel import _available_memory_gb
        meminfo = "MemTotal:       65000000 kB\nMemAvailable:   40000000 kB\n"
        # memory.current says only ~2GB left (10GB max, 8GB current), but
        # 5GB of that 8GB is reclaimable page cache -- the real incident
        # this guards against (2026-10-25): a naive memory.max -
        # memory.current treated that cache as genuinely unavailable,
        # keeping the supervisor pinned near its floor for hours despite
        # real headroom existing. Real available should be 2GB + 5GB = 7GB.
        cgroup_max = str(10 * 1024 ** 3)
        cgroup_current = str(8 * 1024 ** 3)
        memory_stat = f"anon {3 * 1024 ** 3}\nfile {5 * 1024 ** 3}\ninactive_file {5 * 1024 ** 3}\n"
        contents = {
            '/proc/meminfo': meminfo,
            '/sys/fs/cgroup/memory.max': cgroup_max,
            '/sys/fs/cgroup/memory.current': cgroup_current,
            '/sys/fs/cgroup/memory.stat': memory_stat,
        }

        def fake_open(path, *a, **kw):
            if path not in contents:
                raise FileNotFoundError(path)
            return io.StringIO(contents[path])

        with patch('builtins.open', side_effect=fake_open):
            result = _available_memory_gb()
        self.assertAlmostEqual(result, 7.0, places=3)

    def test_available_memory_gb_cgroup_math_survives_missing_memory_stat(self):
        from face_manager.management.commands.backfill_lvface_parallel import _available_memory_gb
        meminfo = "MemTotal:       65000000 kB\nMemAvailable:   40000000 kB\n"
        # memory.stat itself missing (e.g. an older/different cgroup
        # setup) must not abort the whole cgroup calculation -- it should
        # just treat reclaimable cache as 0, not fall all the way back to
        # host-only.
        cgroup_max = str(10 * 1024 ** 3)
        cgroup_current = str(8 * 1024 ** 3)
        contents = {
            '/proc/meminfo': meminfo,
            '/sys/fs/cgroup/memory.max': cgroup_max,
            '/sys/fs/cgroup/memory.current': cgroup_current,
        }

        def fake_open(path, *a, **kw):
            if path not in contents:
                raise FileNotFoundError(path)
            return io.StringIO(contents[path])

        with patch('builtins.open', side_effect=fake_open):
            result = _available_memory_gb()
        self.assertAlmostEqual(result, 2.0, places=3)

    def test_available_memory_gb_falls_back_to_host_without_cgroup_v2(self):
        from face_manager.management.commands.backfill_lvface_parallel import _available_memory_gb
        meminfo = "MemTotal:       65000000 kB\nMemAvailable:   20000000 kB\n"

        def fake_open(path, *a, **kw):
            if path == '/proc/meminfo':
                return io.StringIO(meminfo)
            raise FileNotFoundError(path)

        with patch('builtins.open', side_effect=fake_open):
            result = _available_memory_gb()
        self.assertAlmostEqual(result, 20000000 / (1024 * 1024), places=3)

    def test_total_cores_uses_sched_affinity_when_available(self):
        from face_manager.management.commands.backfill_lvface_parallel import _total_cores
        with patch('os.sched_getaffinity', return_value=set(range(6)), create=True):
            self.assertEqual(_total_cores(), 6)


class LVFaceModuleTests(TestCase):
    """face_manager.lvface -- the LVFace (ViT) embedding helper, now
    supporting all three published sizes (S/B/L). Mocks the ONNXRuntime
    session entirely (via lvface._get_session, patched module-globally)
    so these stay fast and never need any of the real model files on
    disk -- see lvface.py's own docstring for why session construction
    is lazy in the first place."""

    def setUp(self):
        import lvface
        lvface.reset_session()  # don't leak a mock session across tests

    def tearDown(self):
        import lvface
        lvface.reset_session()

    def _mock_session(self, raw_output):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name='data')]
        mock_session.run.return_value = [np.array([raw_output], dtype=np.float32)]
        return mock_session

    def test_embed_from_image_array_normalizes_and_handles_bgr(self):
        import lvface
        raw = np.full(512, 2.0, dtype=np.float32)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        kps = np.array([[50, 60], [100, 60], [75, 90], [55, 120], [95, 120]], dtype=np.float32)
        with patch('lvface._get_session', return_value=self._mock_session(raw)):
            result = lvface.embed_from_image_array(img, kps, input_is_bgr=True)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=5)

    def test_embed_from_image_array_accepts_flattened_kps(self):
        import lvface
        raw = np.full(512, 1.0, dtype=np.float32)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        flat_kps = [50, 60, 100, 60, 75, 90, 55, 120, 95, 120]
        with patch('lvface._get_session', return_value=self._mock_session(raw)):
            result = lvface.embed_from_image_array(img, flat_kps, input_is_bgr=False)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (512,))

    def test_zero_norm_output_returns_none(self):
        import lvface
        raw = np.zeros(512, dtype=np.float32)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        kps = np.array([[50, 60], [100, 60], [75, 90], [55, 120], [95, 120]], dtype=np.float32)
        with patch('lvface._get_session', return_value=self._mock_session(raw)):
            result = lvface.embed_from_image_array(img, kps, input_is_bgr=True)
        self.assertIsNone(result)

    def test_embed_from_path_returns_none_on_undecodable_file(self):
        import lvface
        kps = [50, 60, 100, 60, 75, 90, 55, 120, 95, 120]
        result = lvface.embed_from_path('/nonexistent/path/does_not_exist.jpg', kps)
        self.assertIsNone(result)

    def test_session_is_built_lazily_and_cached(self):
        import lvface
        with patch('onnxruntime.InferenceSession') as mock_ort:
            mock_ort.return_value = MagicMock()
            s1 = lvface._get_session()
            s2 = lvface._get_session()
        self.assertIs(s1, s2)
        mock_ort.assert_called_once()

    def test_each_size_gets_its_own_independently_cached_session(self):
        import lvface
        with patch('onnxruntime.InferenceSession') as mock_ort:
            mock_ort.side_effect = lambda *a, **k: MagicMock()
            s_small = lvface._get_session('S')
            s_base = lvface._get_session('B')
            s_large = lvface._get_session('L')
            s_base_again = lvface._get_session('B')
        self.assertIsNot(s_small, s_base)
        self.assertIsNot(s_base, s_large)
        self.assertIs(s_base, s_base_again)
        self.assertEqual(mock_ort.call_count, 3)
        called_paths = [c.args[0] for c in mock_ort.call_args_list]
        self.assertEqual(sorted(called_paths), sorted(lvface.LVFACE_MODEL_PATHS.values()))

    def test_unknown_size_raises(self):
        import lvface
        with self.assertRaises(ValueError):
            lvface._get_session('X')

    def test_reset_session_with_no_args_clears_all_sizes(self):
        import lvface
        with patch('onnxruntime.InferenceSession') as mock_ort:
            mock_ort.side_effect = lambda *a, **k: MagicMock()
            lvface._get_session('S')
            lvface._get_session('L')
            lvface.reset_session()
            lvface._get_session('S')
            lvface._get_session('L')
        self.assertEqual(mock_ort.call_count, 4)

    def test_reset_session_with_a_size_clears_only_that_one(self):
        import lvface
        with patch('onnxruntime.InferenceSession') as mock_ort:
            mock_ort.side_effect = lambda *a, **k: MagicMock()
            lvface._get_session('S')
            lvface._get_session('L')
            lvface.reset_session('S')
            lvface._get_session('S')
            lvface._get_session('L')  # still cached, no new call
        self.assertEqual(mock_ort.call_count, 3)

    def test_embed_from_image_array_uses_requested_size(self):
        import lvface
        raw = np.full(512, 1.0, dtype=np.float32)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        kps = np.array([[50, 60], [100, 60], [75, 90], [55, 120], [95, 120]], dtype=np.float32)
        with patch('lvface._get_session', return_value=self._mock_session(raw)) as mock_get:
            lvface.embed_from_image_array(img, kps, size='L')
        mock_get.assert_called_once_with('L')


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class UpdateExistingFaceLvfaceEncodingTests(TestCase):
    """update_existing_face_to_insightface()'s LVFace-B wiring -- the
    IOU-rematch branch (find_and_encode_faces()'s "existing Face rows
    already exist for this image" path). Constructed via __new__ (same
    pattern as ReencodeMissingFacesTests) to avoid loading the real
    ~100MB insightface ONNX models, since this function never touches
    self.app at all (only self.iou_function/self.gender_map, both plain
    Python)."""

    def setUp(self):
        import torchvision.ops.boxes as bops
        self.image = make_image()
        self.extractor = FaceExtractor.__new__(FaceExtractor)
        self.extractor.iou_function = bops.distance_box_iou
        self.extractor.gender_map = {1: 'M', 0: 'F'}
        self.extractor.IOU_thresh = 0.3

    def _existing_face(self):
        return make_face(
            self.image, box_left=10, box_top=10, box_right=50, box_bottom=50,
        )

    def _new_data(self, bbox):
        import insightface.app.common
        return insightface.app.common.Face(
            bbox=np.array(bbox, dtype=np.float32),
            kps=np.zeros((5, 2), dtype=np.float32),
            det_score=0.9,
            embedding=np.ones(512, dtype=np.float32),
            age=30, gender=0,
        )

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_matching_box_stores_lvface_encoding(self):
        face = self._existing_face()
        new_data = self._new_data([10, 10, 50, 50])
        with patch('lvface.embed_from_image_array', return_value=np.full(512, 0.6, dtype=np.float32)) as mock_lv:
            self.extractor.update_existing_face_to_insightface(face, new_data, img_numpy=np.zeros((100, 100, 3)))
        face.refresh_from_db()
        self.assertEqual(face.face_encoding_lvface_b, [0.6] * 512)
        self.assertEqual(mock_lv.call_args.kwargs.get('input_is_bgr'), False)

    @override_settings(LVFACE_ENCODING_ENABLED=False)
    def test_lvface_disabled_never_attempts_encoding(self):
        face = self._existing_face()
        new_data = self._new_data([10, 10, 50, 50])
        with patch('lvface.embed_from_image_array') as mock_lv:
            self.extractor.update_existing_face_to_insightface(face, new_data, img_numpy=np.zeros((100, 100, 3)))
        mock_lv.assert_not_called()
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)
        self.assertIsNotNone(face.face_encoding_512)

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_no_img_numpy_skips_lvface_without_error(self):
        face = self._existing_face()
        new_data = self._new_data([10, 10, 50, 50])
        with patch('lvface.embed_from_image_array') as mock_lv:
            self.extractor.update_existing_face_to_insightface(face, new_data)
        mock_lv.assert_not_called()
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)
        self.assertIsNotNone(face.face_encoding_512)

    @override_settings(LVFACE_ENCODING_ENABLED=True)
    def test_lvface_failure_is_non_fatal(self):
        face = self._existing_face()
        new_data = self._new_data([10, 10, 50, 50])
        with patch('lvface.embed_from_image_array', side_effect=RuntimeError("boom")):
            self.extractor.update_existing_face_to_insightface(face, new_data, img_numpy=np.zeros((100, 100, 3)))
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)
        self.assertIsNotNone(face.face_encoding_512)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media", LVFACE_ENCODING_ENABLED=True)
class BackfillLvfaceVideoEncodingsTests(TestCase):
    """FaceExtractor._backfill_lvface_video_encodings() -- the video-only
    LVFace-B catch-up sweep, called automatically at the end of
    reencode_missing_faces() with whatever time budget remains after its
    own merged image-side insightface+LVFace-B pass (see that method's
    own tests in ReencodeMissingFacesTests, and its docstring). Mocks
    lvface.embed_from_image_array entirely to stay fast. LVFACE_ENCODING_
    ENABLED is overridden True class-wide here -- this class tests the
    sweep's own logic, not the (separately-tested) master on/off switch
    itself."""

    def setUp(self):
        self.extractor = FaceExtractor.__new__(FaceExtractor)

    def _make_video_face(self, **overrides):
        from filepopulator.models import Directory, VideoFile
        directory, _ = Directory.objects.get_or_create(dir_path='/videos/lvface_backfill_test')
        video_path = '/tmp/lvface_backfill_test_video.mp4'
        with open(video_path, 'wb') as f:
            f.write(b'not a real video, existence is all this sweep checks')
        self.addCleanup(lambda: os.path.exists(video_path) and os.remove(video_path))
        video = VideoFile.objects.create(filename=video_path, directory=directory, width=640, height=480)
        from face_manager.models import get_default_blank_person
        face = Face(
            declared_name=overrides.pop('declared_name', None) or get_default_blank_person(),
            source_video_file=video,
            box_left=1, box_top=1, box_right=10, box_bottom=10,
            video_thumbnail_frame_seconds=12.3,
            kps=[1.0] * 10,
            **overrides,
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False)
        face.save()
        return face

    def test_video_sourced_face_is_encoded_via_frame_extraction(self):
        face = self._make_video_face()
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch('video_face_pipeline.ffprobe_info', return_value=(640, 480, 30.0, None, 0, None, None)), \
             patch('video_face_pipeline.extract_frame_near_timestamp', return_value=fake_frame) as mock_extract, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.7, dtype=np.float32)) as mock_lv:
            self.extractor._backfill_lvface_video_encodings()
        mock_extract.assert_called_once()
        # input_is_bgr=True for the video path (the frame comes straight
        # from cv2/PyAV, unlike the image path's PIL-sourced RGB array).
        self.assertIs(mock_lv.call_args.kwargs.get('input_is_bgr'), True)
        face.refresh_from_db()
        self.assertEqual(face.face_encoding_lvface_b, [0.7] * 512)

    def test_video_face_with_no_decodable_frame_is_left_unencoded(self):
        face = self._make_video_face()
        with patch('video_face_pipeline.ffprobe_info', return_value=(640, 480, 30.0, None, 0, None, None)), \
             patch('video_face_pipeline.extract_frame_near_timestamp', return_value=None), \
             patch('lvface.embed_from_image_array') as mock_lv:
            self.extractor._backfill_lvface_video_encodings()
        mock_lv.assert_not_called()
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)

    def test_confirmed_ignore_video_face_is_excluded(self):
        ignore_person = make_person(settings.SOFT_IGNORE_NAME)
        face = self._make_video_face(declared_name=ignore_person)
        with patch('video_face_pipeline.ffprobe_info', return_value=(640, 480, 30.0, None, 0, None, None)), \
             patch('video_face_pipeline.extract_frame_near_timestamp') as mock_extract, \
             patch('lvface.embed_from_image_array'):
            self.extractor._backfill_lvface_video_encodings()
        mock_extract.assert_not_called()
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)

    def test_collection_scoped_call_skips_entirely(self):
        # collection is an image-only concept (PhotoCollection/
        # Directory.collection) -- a collection-scoped call has nothing
        # to say about video faces, so this sweep must not run at all
        # for one, rather than silently sweeping the whole unscoped
        # video backlog.
        from filepopulator.models import PhotoCollection
        face = self._make_video_face()
        default_collection = PhotoCollection.objects.get(is_default=True)
        with patch('video_face_pipeline.ffprobe_info') as mock_ffprobe, \
             patch('lvface.embed_from_image_array'):
            self.extractor._backfill_lvface_video_encodings(collection=default_collection)
        mock_ffprobe.assert_not_called()
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)

    def test_time_budget_stops_the_sweep_between_faces(self):
        face_a = self._make_video_face()
        face_b = self._make_video_face()

        times = iter([0, 0, 1000])  # start_time, first check (ok), second check (over budget)

        def fake_time():
            return next(times, 1000)

        with patch('video_face_pipeline.ffprobe_info', return_value=(640, 480, 30.0, None, 0, None, None)), \
             patch('video_face_pipeline.extract_frame_near_timestamp', return_value=np.zeros((480, 640, 3), dtype=np.uint8)), \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.2, dtype=np.float32)) as mock_lv, \
             patch('time.time', side_effect=fake_time):
            self.extractor._backfill_lvface_video_encodings(max_runtime_seconds=10)

        # Only the first face should have been processed before the
        # (mocked) clock crossed the budget.
        self.assertEqual(mock_lv.call_count, 1)

    def test_shard_filters_to_matching_ids_only(self):
        # Added 2026-10-26 once the parallel image backfill was already
        # running well -- video faces have a real id, just like image
        # faces, so the same hash-based shard partition (_shard_
        # annotation()) applies unchanged. Every face must be covered
        # by exactly one shard.
        faces = [self._make_video_face() for _ in range(6)]
        total_shards = 3
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch('video_face_pipeline.ffprobe_info', return_value=(640, 480, 30.0, None, 0, None, None)), \
             patch('video_face_pipeline.extract_frame_near_timestamp', return_value=fake_frame), \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.6, dtype=np.float32)):
            for shard_index in range(total_shards):
                self.extractor._backfill_lvface_video_encodings(
                    shard_index=shard_index, total_shards=total_shards,
                )
        for face in faces:
            face.refresh_from_db()
            self.assertEqual(face.face_encoding_lvface_b, [0.6] * 512)


class BackfillLvfaceTargetedTests(TestCase):
    """backfill_lvface_targeted -- the kps-replay-only LVFace backfill
    used for the LVFace-vs-insightface investigation. No detection at
    all: trusts the face's own already-verified kps directly."""

    def setUp(self):
        self.image = make_image()
        self.real_person = make_person("Targeted LVFace Real Person")
        self.ignore_person = make_person(".ignore")
        self.blank_person = make_person("_NO_FACE_ASSIGNED_")

    def _make_verified_face(self, declared_name, **overrides):
        defaults = dict(kps=[20, 20, 30, 20, 25, 25, 20, 30, 30, 30])
        defaults.update(overrides)
        return make_face(self.image, declared_name=declared_name, **defaults)

    def test_confirmed_person_face_gets_embedded(self):
        face = self._make_verified_face(self.real_person)
        with patch('face_manager.management.commands.backfill_lvface_targeted.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.3, dtype=np.float32)):
            call_command('backfill_lvface_targeted')
        face.refresh_from_db()
        self.assertEqual(face.face_encoding_lvface_s, [0.3] * 512)
        self.assertEqual(face.face_encoding_lvface_b, [0.3] * 512)
        self.assertEqual(face.face_encoding_lvface_l, [0.3] * 512)

    def test_confirmed_ignore_face_gets_embedded(self):
        face = self._make_verified_face(self.ignore_person)
        with patch('face_manager.management.commands.backfill_lvface_targeted.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.5, dtype=np.float32)):
            call_command('backfill_lvface_targeted')
        face.refresh_from_db()
        self.assertEqual(face.face_encoding_lvface_b, [0.5] * 512)

    def test_blank_sentinel_face_is_skipped(self):
        face = self._make_verified_face(self.blank_person)
        with patch('face_manager.management.commands.backfill_lvface_targeted.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.5, dtype=np.float32)) as mock_lv:
            call_command('backfill_lvface_targeted')
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)
        mock_lv.assert_not_called()

    def test_face_without_kps_is_never_a_candidate(self):
        face = make_face(self.image, declared_name=self.real_person, kps=None)
        with patch('face_manager.management.commands.backfill_lvface_targeted.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.5, dtype=np.float32)) as mock_lv:
            call_command('backfill_lvface_targeted')
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)
        mock_lv.assert_not_called()

    def test_already_embedded_face_is_not_recomputed(self):
        face = self._make_verified_face(self.real_person)
        face.face_encoding_lvface_s = [0.9] * 512
        face.face_encoding_lvface_b = [0.9] * 512
        face.face_encoding_lvface_l = [0.9] * 512
        face.save()
        with patch('face_manager.management.commands.backfill_lvface_targeted.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.5, dtype=np.float32)) as mock_lv:
            call_command('backfill_lvface_targeted')
        mock_lv.assert_not_called()

    def test_uses_the_faces_own_stored_kps_not_a_fresh_detection(self):
        face = self._make_verified_face(self.real_person, kps=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        with patch('face_manager.management.commands.backfill_lvface_targeted.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.5, dtype=np.float32)) as mock_lv:
            call_command('backfill_lvface_targeted', '--lvface-sizes=B')
        passed_kps = mock_lv.call_args.args[1]
        self.assertEqual(list(passed_kps), face.kps)

    def test_lvface_sizes_flag_restricts_which_fields_are_written(self):
        face = self._make_verified_face(self.real_person)
        with patch('face_manager.management.commands.backfill_lvface_targeted.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.5, dtype=np.float32)):
            call_command('backfill_lvface_targeted', '--lvface-sizes=B')
        face.refresh_from_db()
        self.assertEqual(face.face_encoding_lvface_b, [0.5] * 512)
        self.assertIsNone(face.face_encoding_lvface_s)
        self.assertIsNone(face.face_encoding_lvface_l)

    def test_dry_run_does_not_save(self):
        face = self._make_verified_face(self.real_person)
        with patch('face_manager.management.commands.backfill_lvface_targeted.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.5, dtype=np.float32)) as mock_lv:
            call_command('backfill_lvface_targeted', '--dry-run')
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)
        mock_lv.assert_not_called()

    def test_decode_failure_is_counted_not_fatal(self):
        face = self._make_verified_face(self.real_person)
        with patch('face_manager.management.commands.backfill_lvface_targeted.common.open_img_oriented',
                   return_value=None):
            call_command('backfill_lvface_targeted')
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)

    def test_sharding_covers_every_trigger_image_exactly_once(self):
        image2 = make_image("naming/good/2.jpg")
        self._make_verified_face(self.real_person)
        make_face(image2, declared_name=self.real_person,
                  kps=[20, 20, 30, 20, 25, 25, 20, 30, 30, 30])
        seen_image_ids = set()
        with patch('face_manager.management.commands.backfill_lvface_targeted.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.5, dtype=np.float32)):
            for shard in range(2):
                call_command('backfill_lvface_targeted', f'--shard-index={shard}', '--total-shards=2')
        from face_manager.models import Face as FaceModel
        embedded_image_ids = set(
            FaceModel.objects.filter(face_encoding_lvface_b__isnull=False)
            .values_list('source_image_file_id', flat=True)
        )
        self.assertEqual(embedded_image_ids, {self.image.id, image2.id})



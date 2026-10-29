"""face_manager tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test face_manager.tests.test_backfill_commands
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


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class BackfillDetScoreTests(TestCase):
    """backfill_det_score management command: re-detects each unlabeled
    .ignore face's source image and matches the result back to the
    stored box by IOU, storing the matched detection's own det_score.
    Mocks _build_detection_only_detector entirely (real model loading is
    expensive) and common.open_img_oriented (no real image decode
    needed) -- only the command's own IOU-matching/grouping logic is
    under test here. Uses the real FaceExtractor._flatten_kps (a pure
    static method, cheap to call directly -- doesn't instantiate
    FaceExtractor, so no model loading happens) rather than mocking it."""

    class _FakeDetection:
        def __init__(self, bbox, det_score, kps=None):
            self.bbox = np.array(bbox, dtype=np.float32)
            self.det_score = det_score
            self._kps = kps if kps is not None else [[1.0, 2.0]] * 5

        def __getitem__(self, key):
            if key == 'kps':
                return self._kps
            raise KeyError(key)

    def setUp(self):
        self.ignore_person = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        self.image = make_image()
        self.image.width, self.image.height = 1000, 1000
        self.image.save()

    def _make_ignore_face(self, box):
        face = make_face(
            self.image, declared_name=Person.objects.get(person_name=settings.BLANK_FACE_NAME),
            box_left=box[0], box_top=box[1], box_right=box[2], box_bottom=box[3],
        )
        face.set_possible_person(self.ignore_person, 1, 1.0)
        face.refresh_from_db()
        return face

    def test_matched_detection_sets_det_score_and_kps(self):
        face = self._make_ignore_face([10, 10, 50, 50])
        fake_kps = [[15.0, 15.0], [35.0, 15.0], [25.0, 25.0], [17.0, 35.0], [33.0, 35.0]]
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.91, kps=fake_kps)]
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_det_score')
        face.refresh_from_db()
        self.assertAlmostEqual(face.det_score, 0.91, places=4)
        self.assertEqual(face.kps, [v for pair in fake_kps for v in pair])

    def test_matched_face_save_does_not_open_thumbnail_file(self):
        # Real crash found running this at scale: image_list holds every
        # Face object in memory for the whole run, so nothing is ever
        # garbage-collected between saves -- Face.save() used to
        # unconditionally open the thumbnail file (to check it exists)
        # and never close it, and with thousands of faces processed in
        # one long-lived process this exhausted the OS fd limit (OSError:
        # Too many open files, hit for real at ~600/38489 images). Fixed
        # at the source (Face.save(), see FaceSaveThumbnailCheckTests
        # below) by using .path (pure filesystem-path computation) instead
        # of .file.name (which opens the file) -- this test confirms the
        # backfill command's own save() call benefits from that fix by
        # never opening the file at all, not just closing it after.
        from django.core.files.storage import Storage
        face = self._make_ignore_face([10, 10, 50, 50])
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.91)]
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build, \
             patch.object(Storage, 'open', autospec=True) as mock_storage_open:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_det_score')
        mock_storage_open.assert_not_called()

    def test_no_matching_detection_leaves_det_score_null(self):
        face = self._make_ignore_face([10, 10, 50, 50])
        # A detection far from the stored box -- shouldn't match.
        fake_dets = [self._FakeDetection([500, 500, 540, 540], 0.9)]
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_det_score')
        face.refresh_from_db()
        self.assertIsNone(face.det_score)

    def test_detection_with_box_matching_iou_but_kps_elsewhere_is_rejected(self):
        # Real bug, found 2026-10-26: a 0.3 IOU threshold is loose enough
        # that a detection whose bbox clears it against this face's
        # stored box can still carry kps that are nowhere near either --
        # confirmed in production (~360k already-corrupted faces, 46% of
        # everything with both fields). This should be treated as
        # unmatched, not silently saved.
        face = self._make_ignore_face([10, 10, 50, 50])
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.91, kps=[[900.0, 900.0]] * 5)]
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_det_score')
        face.refresh_from_db()
        self.assertIsNone(face.det_score)
        self.assertIsNone(face.kps)

    def test_dry_run_does_not_save(self):
        face = self._make_ignore_face([10, 10, 50, 50])
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.91)]
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_det_score', '--dry-run')
        face.refresh_from_db()
        self.assertIsNone(face.det_score)

    def test_multiple_faces_on_same_image_share_one_detect_call(self):
        face1 = self._make_ignore_face([10, 10, 50, 50])
        face2 = self._make_ignore_face([200, 200, 260, 260])
        fake_dets = [
            self._FakeDetection([10, 10, 50, 50], 0.80, kps=[[30.0, 30.0]] * 5),
            self._FakeDetection([200, 200, 260, 260], 0.65, kps=[[230.0, 230.0]] * 5),
        ]
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_det_score')
        self.assertEqual(mock_build.return_value.get.call_count, 1)
        face1.refresh_from_db()
        face2.refresh_from_db()
        self.assertAlmostEqual(face1.det_score, 0.80, places=4)
        self.assertAlmostEqual(face2.det_score, 0.65, places=4)

    def test_non_ignore_faces_on_a_triggered_image_are_also_backfilled(self):
        # Added 2026-09-18 per explicit request: .ignore faces still drive
        # WHICH images get processed, but a real, confirmed (non-ignore)
        # face on the same already-triggered image should also get its
        # det_score/kps backfilled for free, since the whole image is
        # already decoded and detected regardless of which face triggered
        # it -- no extra detect call needed.
        ignore_face = self._make_ignore_face([10, 10, 50, 50])
        confirmed_person = make_person("Someone Real")
        confirmed_face = make_face(
            self.image, declared_name=confirmed_person,
            box_left=200, box_top=200, box_right=260, box_bottom=260,
        )
        self.assertIsNone(confirmed_face.det_score)
        fake_dets = [
            self._FakeDetection([10, 10, 50, 50], 0.80, kps=[[30.0, 30.0]] * 5),
            self._FakeDetection([200, 200, 260, 260], 0.75, kps=[[230.0, 230.0]] * 5),
        ]
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_det_score')
        self.assertEqual(mock_build.return_value.get.call_count, 1)
        ignore_face.refresh_from_db()
        confirmed_face.refresh_from_db()
        self.assertAlmostEqual(ignore_face.det_score, 0.80, places=4)
        self.assertAlmostEqual(confirmed_face.det_score, 0.75, places=4)
        # The confirmed face's own identity is untouched -- only
        # det_score/kps were ever written.
        self.assertEqual(confirmed_face.declared_name_id, confirmed_person.id)

    def test_non_ignore_face_on_an_untriggered_image_is_not_touched(self):
        # A non-ignore face's own image is only ever processed if that
        # SAME image also has a .ignore face still needing det_score --
        # this does not become a general "backfill every face everywhere"
        # sweep.
        other_image = make_image()
        other_image.width, other_image.height = 1000, 1000
        other_image.save()
        confirmed_person = make_person("Someone Else")
        untouched_face = make_face(
            other_image, declared_name=confirmed_person,
            box_left=10, box_top=10, box_right=50, box_bottom=50,
        )
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented') as mock_open, \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = []
            call_command('backfill_det_score')
        mock_open.assert_not_called()
        untouched_face.refresh_from_db()
        self.assertIsNone(untouched_face.det_score)

    def test_already_populated_faces_are_excluded(self):
        face = self._make_ignore_face([10, 10, 50, 50])
        face.det_score = 0.5
        face.save()
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented') as mock_open, \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            call_command('backfill_det_score')
        mock_open.assert_not_called()
        mock_build.return_value.get.assert_not_called()

    def test_all_faces_flag_widens_trigger_query_beyond_ignore(self):
        # --all-faces (added 2026-09-20 per explicit request): a
        # non-ignore face on an image with NO .ignore face at all should
        # now trigger processing too, unlike the default scope (see
        # test_non_ignore_face_on_an_untriggered_image_is_not_touched
        # above, which confirms the OPPOSITE behavior without the flag).
        confirmed_person = make_person("Someone Else")
        untouched_face = make_face(
            self.image, declared_name=confirmed_person,
            box_left=10, box_top=10, box_right=50, box_bottom=50,
        )
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.88)]
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_det_score', '--all-faces')
        untouched_face.refresh_from_db()
        self.assertAlmostEqual(untouched_face.det_score, 0.88, places=4)

    def test_recheck_mismatched_leaves_already_consistent_faces_alone(self):
        face = self._make_ignore_face([10, 10, 50, 50])
        face.det_score = 0.87
        face.kps = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
        face.save()
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented') as mock_open, \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            call_command('backfill_det_score', '--recheck-mismatched')
        mock_open.assert_not_called()
        face.refresh_from_db()
        self.assertAlmostEqual(face.det_score, 0.87, places=4)
        self.assertEqual(face.kps, [30.0] * 10)

    def test_single_pass_flag_restricts_detector_to_one_pass(self):
        # cut_list=[1,3] (the default, matching the live pipeline exactly)
        # measured at ~0.10 img/s in real production -- a multi-day run
        # for the full unlabeled .ignore population. --single-pass trades
        # some recall (small/edge faces the tiled pass would catch) for
        # roughly 10x fewer detection calls per image.
        self._make_ignore_face([10, 10, 50, 50])
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = []
            call_command('backfill_det_score', '--single-pass')
        self.assertEqual(mock_build.return_value.cut_list, [1])

    def test_cut_list_flag_overrides_default(self):
        # Benchmarked 2026-09-18: cut_list=[1,2] matched exactly as well as
        # the full [1,3] on a real 20-image/123-face sample (123/123 both),
        # at half the detection calls -- the recommended middle ground.
        self._make_ignore_face([10, 10, 50, 50])
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = []
            call_command('backfill_det_score', '--cut-list', '1,2')
        self.assertEqual(mock_build.return_value.cut_list, [1, 2])

    def test_cut_list_flag_takes_precedence_over_single_pass(self):
        self._make_ignore_face([10, 10, 50, 50])
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = []
            call_command('backfill_det_score', '--single-pass', '--cut-list', '1,3')
        self.assertEqual(mock_build.return_value.cut_list, [1, 3])

    def test_per_image_timeout_skips_and_continues(self):
        # Real production hang found running this at scale: one image
        # drove CPU to 1400%+ with the DB's det_score count stuck for
        # minutes -- no specific cause identified, but the same class of
        # risk process_video_faces()'s own PER_VIDEO_TIMEOUT_SECONDS
        # guards against. A pathological image must not be able to hang
        # the whole batch indefinitely.
        import time as time_module
        import face_manager.management.commands.backfill_det_score as cmd_module
        face = self._make_ignore_face([10, 10, 50, 50])

        def slow_get(img):
            time_module.sleep(2)
            return []

        with patch.object(cmd_module, 'PER_IMAGE_TIMEOUT_SECONDS', 1), \
             patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((100, 100, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.side_effect = slow_get
            call_command('backfill_det_score')
        face.refresh_from_db()
        self.assertIsNone(face.det_score)

    @tag("slow")
    def test_detection_only_detector_produces_valid_real_scores(self):
        # Real-inference sanity check, not a re-proof of equivalence --
        # equivalence with the full FaceExtractor().app (bit-identical
        # bbox/det_score) was already confirmed manually against a real
        # 70-face production photo (see this method's own module docstring
        # / CLAUDE.md, 2026-09-18): 127.9s -> 3.2s, a 39.69x speedup, with
        # zero difference in any of the 70 detected boxes or scores. This
        # test just confirms the detection-only detector, run for real
        # (no mocks), produces sane det_score/kps output against a known
        # fixture -- catches a real regression (e.g. a bad allowed_modules
        # value silently breaking detection) without needing to reload the
        # full multi-module detector every CI run.
        import common
        from face_manager.management.commands.backfill_det_score import (
            _build_detection_only_detector,
        )
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/has_face_tags.jpg"
        img_numpy = common.open_img_oriented(path, as_numpy=True)
        app = _build_detection_only_detector()
        app.cut_list = [1, 2]
        detections = app.get(img_numpy)
        self.assertGreaterEqual(len(detections), 1)
        for d in detections:
            self.assertGreater(d.det_score, 0.0)
            self.assertLessEqual(d.det_score, 1.0)
            self.assertEqual(len(d['kps']), 5)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class BackfillDetScoreRecheckMismatchedTransactionTests(TransactionTestCase):
    """--recheck-mismatched tests that need to fabricate a face with a
    deliberately-mismatched kps as their STARTING state (simulating real
    pre-existing production corruption from before face_kps_within_box_
    margin/the pyramidal_detector.py fix existed). Needs TransactionTestCase,
    not the regular TestCase used by the rest of BackfillDetScoreTests --
    Postgres refuses an ALTER TABLE (dropping/re-adding the constraint
    around the fabricated INSERT) while there are "pending trigger
    events" from an earlier write in the SAME transaction, which is
    exactly the single wrapping transaction a plain TestCase runs each
    test inside. TransactionTestCase commits for real between statements
    instead, so the drop/insert/re-add sequence works."""

    def setUp(self):
        # TransactionTestCase flushes every table after each test,
        # including migration-seeded sentinel Person rows and the
        # default PhotoCollection -- recreate just what's actually
        # needed directly, same pattern (and the same reasoning for not
        # using serialized_rollback=True instead -- see its own comment)
        # as ReencodeMissingFacesMismatchedKpsTransactionTests.
        from filepopulator.models import PhotoCollection
        PhotoCollection.objects.get_or_create(
            root_directory=settings.PHOTO_ROOT, defaults={'name': 'Default', 'is_default': True},
        )
        for name in (settings.SOFT_IGNORE_NAME, settings.BLANK_FACE_NAME):
            if not Person.objects.filter(person_name=name).exists():
                make_person(name)
        self.ignore_person = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        self.image = make_image()

    def _make_ignore_face(self, box):
        face = make_face(
            self.image, declared_name=Person.objects.get(person_name=settings.BLANK_FACE_NAME),
            box_left=box[0], box_top=box[1], box_right=box[2], box_bottom=box[3],
        )
        face.set_possible_person(self.ignore_person, 1, 1.0)
        face.refresh_from_db()
        return face

    class _FakeDetection:
        def __init__(self, bbox, det_score, kps=None):
            self.bbox = np.array(bbox, dtype=np.float32)
            self.det_score = det_score
            self._kps = kps if kps is not None else [[1.0, 2.0]] * 5

        def __getitem__(self, key):
            if key == 'kps':
                return self._kps
            raise KeyError(key)

    @contextmanager
    def _constraint_suspended(self):
        with connection.cursor() as cursor:
            cursor.execute("ALTER TABLE face_manager_face DROP CONSTRAINT face_kps_within_box_margin")
        try:
            yield
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

    def test_recheck_mismatched_corrects_a_kps_that_doesnt_match_its_box(self):
        face = self._make_ignore_face([10, 10, 50, 50])
        with self._constraint_suspended():
            face.det_score = 0.5
            face.kps = [900.0] * 10  # nowhere near the box -- a mismatch
            face.save()
        fake_kps = [[30.0, 30.0]] * 5
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.93, kps=fake_kps)]
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_det_score', '--recheck-mismatched')
        face.refresh_from_db()
        self.assertAlmostEqual(face.det_score, 0.93, places=4)
        self.assertEqual(face.kps, [v for pair in fake_kps for v in pair])

    def test_recheck_mismatched_clears_fields_when_no_clean_match_found(self):
        face = self._make_ignore_face([10, 10, 50, 50])
        with self._constraint_suspended():
            face.det_score = 0.5
            face.kps = [900.0] * 10
            face.save()
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = []
            call_command('backfill_det_score', '--recheck-mismatched')
        face.refresh_from_db()
        self.assertIsNone(face.det_score)
        self.assertIsNone(face.kps)

    def test_recheck_mismatched_dry_run_does_not_clear_or_save(self):
        face = self._make_ignore_face([10, 10, 50, 50])
        with self._constraint_suspended():
            face.det_score = 0.5
            face.kps = [900.0] * 10
            face.save()
        with patch('face_manager.management.commands.backfill_det_score.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_det_score._build_detection_only_detector') as mock_build:
            mock_build.return_value.get.return_value = []
            call_command('backfill_det_score', '--recheck-mismatched', '--dry-run')
        face.refresh_from_db()
        self.assertAlmostEqual(face.det_score, 0.5, places=4)
        self.assertEqual(face.kps, [900.0] * 10)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class BackfillVerifiedKpsTests(TestCase):
    """backfill_verified_kps management command -- the rigorous recovery
    backfill for the kps/box translation bug (see pyramidal_detector.py,
    CLAUDE.md). Mocks _build_full_detector entirely (real model loading
    is expensive); only the command's own matching/verification/saving
    logic is under test here."""

    class _FakeDetection:
        def __init__(self, bbox, det_score, kps, embedding):
            self.bbox = np.array(bbox, dtype=np.float32)
            self.det_score = det_score
            self._kps = kps
            self._embedding = np.asarray(embedding, dtype=np.float32)

        def __getitem__(self, key):
            if key == 'kps':
                return self._kps
            if key == 'embedding':
                return self._embedding
            raise KeyError(key)

    def setUp(self):
        self.image = make_image()
        self.person = make_person("Verified Backfill Person")

    def _make_face_with_encoding(self, box, encoding, relative_size=0.5):
        face = make_face(
            self.image, declared_name=self.person,
            box_left=box[0], box_top=box[1], box_right=box[2], box_bottom=box[3],
        )
        face.face_encoding_512 = encoding
        face.relative_size = relative_size
        face.save()
        return face

    def test_high_confidence_match_saves_kps_det_score_and_lvface(self):
        stored = [0.6] * 512
        face = self._make_face_with_encoding([10, 10, 50, 50], stored)
        fresh_kps = [[30.0, 30.0]] * 5
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.9, fresh_kps, [0.6] * 512)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps')
        face.refresh_from_db()
        self.assertEqual(face.kps, [v for pair in fresh_kps for v in pair])
        self.assertAlmostEqual(face.det_score, 0.9, places=4)
        self.assertEqual(face.face_encoding_lvface_s, [0.4] * 512)
        self.assertEqual(face.face_encoding_lvface_b, [0.4] * 512)
        self.assertEqual(face.face_encoding_lvface_l, [0.4] * 512)

    def test_gray_zone_match_accepted_when_tight_iou_and_small_face(self):
        # Same embedding direction but scaled down to land in the gray
        # zone (cosine similarity ~0.8ish via a mixed vector), tight IOU
        # (identical box), and a small relative_size -- mirrors the real
        # calibration case (a small/blurry face, genuine match, 0.78
        # similarity).
        stored = [1.0] + [0.0] * 511
        fresh = [0.8] + [0.6] + [0.0] * 510  # cos ~0.8 vs stored
        face = self._make_face_with_encoding([10, 10, 50, 50], stored, relative_size=0.03)
        fresh_kps = [[30.0, 30.0]] * 5
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.85, fresh_kps, fresh)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps')
        face.refresh_from_db()
        self.assertEqual(face.kps, [v for pair in fresh_kps for v in pair])

    def test_gray_zone_match_rejected_when_face_not_small(self):
        stored = [1.0] + [0.0] * 511
        fresh = [0.8] + [0.6] + [0.0] * 510
        # Same gray-zone similarity as the accepted case above, but this
        # face is NOT small (relative_size above the gray-zone cap).
        face = self._make_face_with_encoding([10, 10, 50, 50], stored, relative_size=0.5)
        fresh_kps = [[30.0, 30.0]] * 5
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.85, fresh_kps, fresh)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps')
        face.refresh_from_db()
        self.assertIsNone(face.kps)

    def test_low_similarity_match_rejected_entirely(self):
        stored = [1.0] + [0.0] * 511
        fresh = [0.0] + [1.0] + [0.0] * 510  # orthogonal -- cos=0
        face = self._make_face_with_encoding([10, 10, 50, 50], stored, relative_size=0.03)
        fresh_kps = [[30.0, 30.0]] * 5
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.85, fresh_kps, fresh)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps')
        face.refresh_from_db()
        self.assertIsNone(face.kps)
        self.assertIsNone(face.det_score)

    def test_unverified_face_without_stored_encoding_is_saved_anyway(self):
        # Per explicit design direction: an incidental face sharing an
        # already-triggered image (e.g. unlabeled/.ignore, no stored
        # encoding to verify against) still gets its kps/det_score saved
        # -- just without the identity-verification gate, since there's
        # nothing to verify against.
        ignore_person = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        trigger_face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        incidental_face = make_face(
            self.image, declared_name=ignore_person,
            box_left=200, box_top=200, box_right=240, box_bottom=240,
        )
        self.assertIsNone(incidental_face.face_encoding_512)

        trigger_kps = [[30.0, 30.0]] * 5
        incidental_kps = [[220.0, 220.0]] * 5
        fake_dets = [
            self._FakeDetection([10, 10, 50, 50], 0.9, trigger_kps, [0.6] * 512),
            self._FakeDetection([200, 200, 240, 240], 0.8, incidental_kps, [0.1] * 512),
        ]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps')

        incidental_face.refresh_from_db()
        self.assertEqual(incidental_face.kps, [v for pair in incidental_kps for v in pair])
        self.assertAlmostEqual(incidental_face.det_score, 0.8, places=4)
        # Still unverified -- declared_name untouched, no identity claim made.
        self.assertEqual(incidental_face.declared_name_id, ignore_person.id)

    def test_geometric_rejection_even_with_high_iou(self):
        # A detection whose bbox clears IOU but whose own kps is nowhere
        # near the face's box must never be saved -- same guard as
        # backfill_det_score.py's own fix, reused here directly.
        stored = [0.6] * 512
        face = self._make_face_with_encoding([10, 10, 50, 50], stored)
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.9, [[900.0, 900.0]] * 5, [0.6] * 512)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps')
        face.refresh_from_db()
        self.assertIsNone(face.kps)

    def test_db_constraint_rejection_does_not_crash_the_whole_command(self):
        # Real production crash, 2026-10-26: _kps_matches_box()'s own
        # pre-check is deliberately looser (centroid within one box-
        # width/-height of the box's own center) than the DB's
        # face_kps_within_box_margin constraint (every one of the 5
        # points individually within a 25% margin) -- a kps with 4
        # clustered points and one wild outlier can clear the pre-check
        # (its centroid still lands near the box center) while the
        # outlier point alone fails the DB's tighter, per-point
        # requirement. An uncaught IntegrityError here killed an entire
        # shard outright in real production; must be caught and counted
        # instead, not fatal.
        stored = [0.6] * 512
        face = self._make_face_with_encoding([10, 10, 50, 50], stored)
        sneaky_kps = [[30.0, 30.0], [30.0, 30.0], [30.0, 30.0], [30.0, 30.0], [150.0, 30.0]]
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.9, sneaky_kps, [0.6] * 512)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps')  # must not raise
        face.refresh_from_db()
        self.assertIsNone(face.kps)

    def test_per_image_timeout_covers_the_matching_and_lvface_loop_not_just_detect(self):
        # Real hang, 2026-10-27: a 710-face crowd photo made the
        # matching/verification/LVFace-embedding loop AFTER detection the
        # real dominant cost (up to faces*len(lvface_sizes) separate
        # embedding calls) -- but the original alarm was canceled right
        # after app.get() returned, BEFORE that loop, so it never bounded
        # the actual slow part at all. A shard hung indefinitely on it.
        # Simulates the same shape with a tiny timeout and a slow
        # lvface.embed_from_image_array (not a slow detect) -- the
        # timeout must still fire and the command must not hang.
        import time as time_module
        face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.9, [[30.0, 30.0]] * 5, [0.6] * 512)]

        def slow_embed(*args, **kwargs):
            time_module.sleep(2)
            return np.full(512, 0.4, dtype=np.float32)

        with patch('face_manager.management.commands.backfill_verified_kps.PER_IMAGE_TIMEOUT_SECONDS', 1), \
             patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', side_effect=slow_embed):
            mock_build.return_value.get.return_value = fake_dets
            start = time_module.time()
            call_command('backfill_verified_kps')  # must not hang
            elapsed = time_module.time() - start
        self.assertLess(elapsed, 10, "timeout did not bound the LVFace-embedding loop")
        face.refresh_from_db()
        # kps/det_score were already committed (each face.save() happens
        # before the lvface loop) -- only the still-unfinished LVFace
        # writes are lost to the timeout, matching the documented
        # "partial progress on an image is retried later" behavior.
        self.assertIsNotNone(face.kps)

    def test_no_detections_leaves_face_untouched_and_counted(self):
        face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build:
            mock_build.return_value.get.return_value = []
            call_command('backfill_verified_kps')
        face.refresh_from_db()
        self.assertIsNone(face.kps)

    def test_dry_run_saves_nothing(self):
        face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.9, [[30.0, 30.0]] * 5, [0.6] * 512)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps', '--dry-run')
        face.refresh_from_db()
        self.assertIsNone(face.kps)
        self.assertIsNone(face.det_score)

    def test_sentinel_encoding_is_treated_as_untrusted(self):
        # NON_DETECTED_FACE_ENCODING is not NULL but is explicitly NOT a
        # real stored embedding -- must be treated the same as "no
        # trusted encoding" (saved without verification), not run
        # through the similarity gate (which would spuriously fail
        # against a sentinel that has nothing to do with any real face).
        # A genuinely-encoded companion face is needed on the same image
        # to trigger processing at all -- an image with ONLY a sentinel-
        # encoded face has no real encoding to verify against and is
        # correctly never selected by this command's own trigger query
        # (same as test_unverified_face_without_stored_encoding_is_saved_
        # anyway's "incidental face" shape).
        trigger_face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        face = self._make_face_with_encoding(
            [200, 200, 240, 240], list(settings.NON_DETECTED_FACE_ENCODING),
        )
        trigger_kps = [[30.0, 30.0]] * 5
        fresh_kps = [[220.0, 220.0]] * 5
        fake_dets = [
            self._FakeDetection([10, 10, 50, 50], 0.9, trigger_kps, [0.6] * 512),
            self._FakeDetection([200, 200, 240, 240], 0.9, fresh_kps, [0.1] * 512),
        ]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps')
        face.refresh_from_db()
        self.assertEqual(face.kps, [v for pair in fresh_kps for v in pair])

    def test_sharding_covers_every_trigger_image_exactly_once(self):
        # Distinct real fixture files, not 4x make_image() with its
        # default argument -- that reuses the SAME path every call (idempotent
        # on path), silently collapsing all 4 "images" onto a single shared
        # ImageFile row instead of 4 separate ones. Found via a real test
        # failure, not assumed.
        images = [
            make_image(f) for f in ("naming/good/1.JPG", "naming/good/2.jpg", "naming/good/3.JPEG", "naming/good/4.jpeg")
        ]
        faces = []
        for i, img in enumerate(images):
            f = make_face(
                img, declared_name=self.person,
                box_left=10, box_top=10, box_right=50, box_bottom=50,
            )
            f.face_encoding_512 = [0.6] * 512
            f.save()
            faces.append(f)

        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.9, [[30.0, 30.0]] * 5, [0.6] * 512)]
        total_shards = 2
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            mock_build.return_value.get.return_value = fake_dets
            for shard_index in range(total_shards):
                call_command('backfill_verified_kps', f'--shard-index={shard_index}', f'--total-shards={total_shards}')

        for f in faces:
            f.refresh_from_db()
            self.assertIsNotNone(f.kps)

    def test_person_ids_filter_restricts_to_images_with_a_targeted_person(self):
        other_person = make_person("Not The Target")
        images = [make_image(f) for f in ("naming/good/1.JPG", "naming/good/2.jpg")]
        target_face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        # move target_face onto the first image explicitly
        target_face.source_image_file = images[0]
        target_face.save()
        other_face = make_face(
            images[1], declared_name=other_person,
            box_left=10, box_top=10, box_right=50, box_bottom=50,
        )
        other_face.face_encoding_512 = [0.6] * 512
        other_face.save()

        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.9, [[30.0, 30.0]] * 5, [0.6] * 512)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps', f'--person-ids={self.person.id}')

        target_face.refresh_from_db()
        other_face.refresh_from_db()
        self.assertIsNotNone(target_face.kps)
        self.assertIsNone(other_face.kps)

    def test_person_ids_filter_also_includes_images_with_a_proposed_match(self):
        # A gallery-focused run should also pick up images where the
        # targeted person is only PROPOSED (poss_ident1), not yet
        # confirmed -- that's exactly the population a reclassification
        # investigation cares about re-encoding with LVFace-B.
        other_person = make_person("Not The Target")
        images = [make_image(f) for f in ("naming/good/1.JPG", "naming/good/2.jpg")]
        proposed_face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        proposed_face.declared_name = None
        proposed_face.poss_ident1 = self.person
        proposed_face.source_image_file = images[0]
        proposed_face.save()
        other_face = make_face(
            images[1], declared_name=other_person,
            box_left=10, box_top=10, box_right=50, box_bottom=50,
        )
        other_face.face_encoding_512 = [0.6] * 512
        other_face.save()

        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.9, [[30.0, 30.0]] * 5, [0.6] * 512)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps', f'--person-ids={self.person.id}')

        proposed_face.refresh_from_db()
        other_face.refresh_from_db()
        self.assertIsNotNone(proposed_face.kps)
        self.assertIsNone(other_face.kps)

    def test_person_ids_filter_still_backfills_bystander_faces_on_the_same_image(self):
        # --person-ids only decides WHICH IMAGES get selected -- once an
        # image is selected, detection runs over the WHOLE image
        # regardless, so every face on it (not just the targeted
        # person's) should still get kps/det_score saved. Losing this
        # would waste real compute: detection is already paid for on
        # this image, so declining to save a bystander's result for
        # free would be pure waste.
        bystander_person = make_person("Bystander")
        target_face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        bystander_face = make_face(
            self.image, declared_name=bystander_person,
            box_left=60, box_top=60, box_right=100, box_bottom=100,
        )
        fake_dets = [
            self._FakeDetection([10, 10, 50, 50], 0.9, [[30.0, 30.0]] * 5, [0.6] * 512),
            self._FakeDetection([60, 60, 100, 100], 0.85, [[80.0, 80.0]] * 5, [0.1] * 512),
        ]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps', f'--person-ids={self.person.id}', '--lvface-sizes=B')

        target_face.refresh_from_db()
        bystander_face.refresh_from_db()
        self.assertIsNotNone(target_face.kps)
        self.assertIsNotNone(bystander_face.kps)
        self.assertEqual(bystander_face.kps, [v for pair in [[80.0, 80.0]] * 5 for v in pair])
        # bystander has no stored face_encoding_512 to verify against -> unverified tier,
        # but still saved (kps/det_score AND lvface, per the unverified-faces-are-saved-too design)
        self.assertEqual(bystander_face.face_encoding_lvface_b, [0.4] * 512)

    def test_per_person_target_skips_images_once_a_person_is_satisfied(self):
        # Give self.person 5 PRE-EXISTING lvface_b faces (simulating an
        # earlier run already having satisfied part of the quota), then
        # a per-person-target of 5 should skip any NEW image whose only
        # targeted face is self.person -- the detector should never even
        # be called for it.
        for _ in range(5):
            f = make_face(self.image, declared_name=self.person,
                           box_left=1, box_top=1, box_right=20, box_bottom=20)
            f.face_encoding_lvface_b = [0.5] * 512
            f.save()

        target_face = self._make_face_with_encoding([30, 30, 50, 50], [0.6] * 512)
        fake_dets = [self._FakeDetection([30, 30, 50, 50], 0.9, [[40.0, 40.0]] * 5, [0.6] * 512)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            mock_build.return_value.get.return_value = fake_dets
            call_command(
                'backfill_verified_kps', f'--person-ids={self.person.id}',
                '--per-person-target=5',
            )
            mock_build.return_value.get.assert_not_called()

        target_face.refresh_from_db()
        self.assertIsNone(target_face.kps)

    def test_per_person_target_still_processes_an_unsatisfied_person(self):
        face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.9, [[30.0, 30.0]] * 5, [0.6] * 512)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            mock_build.return_value.get.return_value = fake_dets
            call_command(
                'backfill_verified_kps', f'--person-ids={self.person.id}',
                '--per-person-target=5',
            )
        face.refresh_from_db()
        self.assertIsNotNone(face.kps)

    def test_per_person_target_exempts_ignore_sentinel(self):
        # self._make_face_with_encoding() is what makes an image qualify
        # as a trigger image at all (a real face_encoding_512 somewhere
        # on it) -- a plain make_face() alone, as used for the 10
        # pre-existing .ignore faces below, never triggers anything.
        trigger_face = self._make_face_with_encoding([200, 200, 240, 240], [0.6] * 512)
        ignore_person = make_person('.ignore')
        for _ in range(10):
            f = make_face(self.image, declared_name=ignore_person,
                           box_left=1, box_top=1, box_right=20, box_bottom=20)
            f.face_encoding_lvface_b = [0.5] * 512
            f.save()

        ignore_face = make_face(
            self.image, declared_name=ignore_person,
            box_left=30, box_top=30, box_right=50, box_bottom=50,
        )
        fake_dets = [
            self._FakeDetection([200, 200, 240, 240], 0.9, [[220.0, 220.0]] * 5, [0.6] * 512),
            self._FakeDetection([30, 30, 50, 50], 0.85, [[40.0, 40.0]] * 5, [0.2] * 512),
        ]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            mock_build.return_value.get.return_value = fake_dets
            call_command(
                'backfill_verified_kps', f'--person-ids={ignore_person.id}',
                '--per-person-target=5',
            )
        ignore_face.refresh_from_db()
        self.assertIsNotNone(ignore_face.kps)

    def test_max_new_faces_stops_the_run_early(self):
        images = [make_image(f) for f in ("naming/good/1.JPG", "naming/good/2.jpg", "naming/good/3.JPEG")]
        faces = []
        for img in images:
            f = make_face(
                img, declared_name=self.person,
                box_left=10, box_top=10, box_right=50, box_bottom=50,
            )
            f.face_encoding_512 = [0.6] * 512
            f.save()
            faces.append(f)

        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.9, [[30.0, 30.0]] * 5, [0.6] * 512)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps', '--max-new-faces=1')

        kps_set_count = sum(1 for f in faces if Face.objects.get(id=f.id).kps is not None)
        self.assertEqual(kps_set_count, 1)

    def test_max_faces_per_image_skips_crowd_photos_without_decoding(self):
        # Real hang, 2026-10-27: a 710-face crowd photo. Checked against
        # the EXISTING face count (cheap, already fetched), before even
        # decoding -- the detector must never be called for such an image.
        target_face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        for i in range(25):
            make_face(self.image, declared_name=self.person,
                      box_left=60 + i, box_top=60, box_right=70 + i, box_bottom=70)

        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build:
            call_command('backfill_verified_kps', '--max-faces-per-image=20')
            mock_build.return_value.get.assert_not_called()

        target_face.refresh_from_db()
        self.assertIsNone(target_face.kps)

    def test_max_faces_per_image_still_processes_a_normal_photo(self):
        face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.9, [[30.0, 30.0]] * 5, [0.6] * 512)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps', '--max-faces-per-image=20')
        face.refresh_from_db()
        self.assertIsNotNone(face.kps)

    def test_max_image_pixels_skips_giant_panoramas_without_decoding(self):
        # Real hang, 2026-10-27: several library images are extreme-
        # aspect-ratio stitched panoramas (e.g. 22096x1088) that hang
        # app.get() ITSELF -- confirmed via a stuck process with 149 OS
        # threads and no open image file (past decode already), with
        # zero "TIMEOUT on" lines despite the alarm firing, since
        # app.get() is a single uninterruptible C call. Caught here via
        # width*height alone -- a panorama can have very few cataloged
        # faces, so --max-faces-per-image alone would never catch it.
        target_face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        # .update(), not .save() -- ImageFile.save() re-decodes the real
        # underlying fixture file and overwrites width/height with its
        # actual dimensions, undoing a plain attribute assignment.
        ImageFile.objects.filter(id=self.image.id).update(width=22096, height=1088)

        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build:
            # 22096*1088 = 24,040,448 -- threshold must be BELOW that.
            call_command('backfill_verified_kps', '--max-image-pixels=20000000')
            mock_build.return_value.get.assert_not_called()

        target_face.refresh_from_db()
        self.assertIsNone(target_face.kps)

    def test_max_image_pixels_still_processes_a_normal_photo(self):
        face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        ImageFile.objects.filter(id=self.image.id).update(width=4000, height=3000)
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.9, [[30.0, 30.0]] * 5, [0.6] * 512)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps', '--max-image-pixels=25000000')
        face.refresh_from_db()
        self.assertIsNotNone(face.kps)

    def test_max_aspect_ratio_skips_panoramas_pixel_count_alone_misses(self):
        # Real hang, same day: --max-image-pixels=25000000 let the real
        # 22096x1088 (24,040,448 px, UNDER 25M) offender through a
        # SECOND time, since raw pixel count can't distinguish a
        # panorama from an ordinary large photo in the same MP range --
        # aspect ratio (20.3:1 here) is the real distinguishing signal.
        target_face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        ImageFile.objects.filter(id=self.image.id).update(width=22096, height=1088)

        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build:
            # Confirm pixel count ALONE would NOT have caught this real case.
            self.assertLessEqual(22096 * 1088, 25_000_000)
            call_command('backfill_verified_kps', '--max-aspect-ratio=2.5')
            mock_build.return_value.get.assert_not_called()

        target_face.refresh_from_db()
        self.assertIsNone(target_face.kps)

    def test_max_aspect_ratio_still_processes_a_normal_photo(self):
        face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        ImageFile.objects.filter(id=self.image.id).update(width=4000, height=3000)
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.9, [[30.0, 30.0]] * 5, [0.6] * 512)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps', '--max-aspect-ratio=2.5')
        face.refresh_from_db()
        self.assertIsNotNone(face.kps)

    def test_below_collection_floor_skips_lvface_but_still_saves_kps(self):
        # Same real gate production's own pipeline already uses
        # (face_extract_encode.py) to decide whether a face is worth a
        # real embedding -- a face below its own collection's floor
        # should still get kps/det_score (cheap, already fixed), but not
        # pay for LVFace (the real dominant cost on a crowd photo full
        # of tiny background faces that would never get a real embedding
        # in production either).
        from filepopulator.models import PhotoCollection
        collection = PhotoCollection.objects.create(
            name='Floor Test', root_directory='/photos/naming', min_face_relative_size=0.5,
        )
        self.image.directory.collection = collection
        self.image.directory.save()

        face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        face.relative_size = 0.1  # below the 0.5 floor
        face.save()
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.9, [[30.0, 30.0]] * 5, [0.6] * 512)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)) as mock_lv:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps')
            mock_lv.assert_not_called()

        face.refresh_from_db()
        self.assertIsNotNone(face.kps)
        self.assertIsNone(face.face_encoding_lvface_b)

    def test_above_collection_floor_still_gets_lvface(self):
        from filepopulator.models import PhotoCollection
        collection = PhotoCollection.objects.create(
            name='Floor Test 2', root_directory='/photos/naming', min_face_relative_size=0.5,
        )
        self.image.directory.collection = collection
        self.image.directory.save()

        face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        face.relative_size = 0.9  # above the 0.5 floor
        face.save()
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.9, [[30.0, 30.0]] * 5, [0.6] * 512)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps')

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_lvface_b, [0.4] * 512)

    def test_lvface_sizes_flag_restricts_which_sizes_are_backfilled(self):
        face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.9, [[30.0, 30.0]] * 5, [0.6] * 512)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)) as mock_lv:
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps', '--lvface-sizes=B')
        face.refresh_from_db()
        self.assertEqual(face.face_encoding_lvface_b, [0.4] * 512)
        self.assertIsNone(face.face_encoding_lvface_s)
        self.assertIsNone(face.face_encoding_lvface_l)
        mock_lv.assert_called_once()
        self.assertEqual(mock_lv.call_args.kwargs.get('size'), 'B')

    def test_resuming_with_lvface_enabled_does_not_skip_a_kps_only_face(self):
        # Real bug, 2026-10-27: a face backfilled during an earlier
        # kps-only phase (--lvface-sizes '') already has kps, so the old
        # "already_done" check (kps alone) would mark its image fully
        # done and skip it forever -- even though it was never actually
        # given the LVFace fields THIS run wants. Must still be
        # (re-)visited and backfilled once LVFace is re-enabled.
        face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        face.kps = [30.0] * 10
        face.det_score = 0.9
        face.save()
        self.assertIsNone(face.face_encoding_lvface_b)

        fake_dets = [self._FakeDetection([10, 10, 50, 50], 0.9, [[30.0, 30.0]] * 5, [0.6] * 512)]
        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build, \
             patch('lvface.embed_from_image_array', return_value=np.full(512, 0.4, dtype=np.float32)):
            mock_build.return_value.get.return_value = fake_dets
            call_command('backfill_verified_kps', '--lvface-sizes=B')

        face.refresh_from_db()
        self.assertEqual(face.face_encoding_lvface_b, [0.4] * 512)

    def test_resuming_with_lvface_disabled_still_skips_a_fully_kps_done_face(self):
        # Companion case: with --lvface-sizes '' (empty), the check must
        # still reduce to the original kps-only behavior -- a face that
        # already has kps and nothing else requested stays skipped.
        face = self._make_face_with_encoding([10, 10, 50, 50], [0.6] * 512)
        face.kps = [30.0] * 10
        face.det_score = 0.9
        face.save()

        with patch('face_manager.management.commands.backfill_verified_kps.common.open_img_oriented',
                   return_value=np.zeros((1000, 1000, 3))), \
             patch('face_manager.management.commands.backfill_verified_kps._build_full_detector') as mock_build:
            call_command('backfill_verified_kps', "--lvface-sizes=")
            mock_build.return_value.get.assert_not_called()


class BackfillVideoDetScoreTests(TestCase):
    """backfill_video_det_score management command: re-extracts the
    exact frame a video face's thumbnail/box came from, redetects in a
    padded crop, and only trusts the result if it actually overlaps
    (IOU >= 0.3) the already-stored box. Mocks VideoFaceExtractor
    entirely (real model loading), plus ffprobe_info and
    extract_frame_near_timestamp (real video decode) -- only the
    command's own IOU-matching/counting logic is under test here."""

    CMD_MODULE = 'face_manager.management.commands.backfill_video_det_score'

    def setUp(self):
        from filepopulator.models import Directory, VideoFile
        directory, _ = Directory.objects.get_or_create(dir_path='/tmp/video_det_backfill_test')
        self.video = VideoFile.objects.create(
            filename='/tmp/video_det_backfill_test/a.mp4', directory=directory,
            width=320, height=240, duration_seconds=10.0,
        )
        self.blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)

    def _make_video_face(self, box=(10, 10, 50, 50)):
        face = Face(
            source_video_file=self.video, declared_name=self.blank_person,
            box_left=box[0], box_top=box[1], box_right=box[2], box_bottom=box[3],
            video_thumbnail_frame_seconds=1.5, det_score=None,
        )
        face.face_thumbnail.save(
            "thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False
        )
        face.save()
        return face

    def _ffprobe_return(self):
        # (width, height, avg_fps, field_order, rotation, color_transfer, sar_scale_width)
        return (320, 240, 30.0, 'progressive', 0, None, None)

    def test_matched_detection_sets_det_score(self):
        face = self._make_video_face(box=(10, 10, 50, 50))
        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info', return_value=self._ffprobe_return()), \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=fake_frame):
            MockExtractor.return_value._redetect_in_crop.return_value = (
                [10, 10, 50, 50], [[0, 0]] * 5, 0.87,
            )
            call_command('backfill_video_det_score')

        face.refresh_from_db()
        self.assertAlmostEqual(face.det_score, 0.87)

    def test_low_iou_match_is_rejected(self):
        face = self._make_video_face(box=(10, 10, 50, 50))
        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info', return_value=self._ffprobe_return()), \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=fake_frame):
            # Redetected box is far away -- essentially zero overlap.
            MockExtractor.return_value._redetect_in_crop.return_value = (
                [200, 200, 240, 240], [[0, 0]] * 5, 0.95,
            )
            call_command('backfill_video_det_score')

        face.refresh_from_db()
        self.assertIsNone(face.det_score)

    def test_no_frame_extracted_skips(self):
        face = self._make_video_face()

        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info', return_value=self._ffprobe_return()), \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=None):
            call_command('backfill_video_det_score')

        face.refresh_from_db()
        self.assertIsNone(face.det_score)

    def test_no_detection_in_crop_skips(self):
        face = self._make_video_face()
        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info', return_value=self._ffprobe_return()), \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=fake_frame):
            MockExtractor.return_value._redetect_in_crop.return_value = None
            call_command('backfill_video_det_score')

        face.refresh_from_db()
        self.assertIsNone(face.det_score)

    def test_dry_run_does_not_save(self):
        face = self._make_video_face(box=(10, 10, 50, 50))
        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info', return_value=self._ffprobe_return()), \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=fake_frame):
            MockExtractor.return_value._redetect_in_crop.return_value = (
                [10, 10, 50, 50], [[0, 0]] * 5, 0.87,
            )
            call_command('backfill_video_det_score', '--dry-run')

        face.refresh_from_db()
        self.assertIsNone(face.det_score)

    def test_face_with_existing_det_score_is_excluded(self):
        face = self._make_video_face()
        Face.objects.filter(pk=face.pk).update(det_score=0.75)
        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info', return_value=self._ffprobe_return()) as mock_ffprobe, \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=fake_frame):
            call_command('backfill_video_det_score')
            mock_ffprobe.assert_not_called()

        face.refresh_from_db()
        self.assertAlmostEqual(face.det_score, 0.75)

    def test_det_thresh_override_applied_to_detector(self):
        face = self._make_video_face()
        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info', return_value=self._ffprobe_return()), \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=fake_frame):
            mock_instance = MockExtractor.return_value
            mock_instance.det10g_model.det_thresh = 0.5
            mock_instance._redetect_in_crop.return_value = ([10, 10, 50, 50], [[0, 0]] * 5, 0.35)
            call_command('backfill_video_det_score', '--det-thresh', '0.3')
            self.assertEqual(mock_instance.det10g_model.det_thresh, 0.3)

        face.refresh_from_db()
        self.assertAlmostEqual(face.det_score, 0.35)

    def test_pad_pct_override_passed_through(self):
        face = self._make_video_face(box=(10, 10, 50, 50))
        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info', return_value=self._ffprobe_return()), \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=fake_frame):
            mock_instance = MockExtractor.return_value
            mock_instance._redetect_in_crop.return_value = ([10, 10, 50, 50], [[0, 0]] * 5, 0.9)
            call_command('backfill_video_det_score', '--pad-pct', '1.2')
            mock_instance._redetect_in_crop.assert_called_once_with(
                fake_frame, (10, 10, 50, 50), pad_pct=1.2,
            )


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class SyncFaceDatesFromSourceImageTests(TestCase):
    """sync_face_dates_from_source_image -- Face.dateTakenUTC is only
    ever copied from its source ImageFile.dateTakenUTC ONCE, at face-
    creation time, never kept in sync afterward. Needed alongside
    filepopulator's backfill_implausible_exif_dates.py (2026-10-28):
    correcting an image's own date does nothing for faces already
    extracted from it."""

    def setUp(self):
        self.image = make_image()
        self.person = make_person("Sync Dates Person")

    def test_stale_implausible_face_date_is_synced_to_the_image(self):
        ImageFile.objects.filter(pk=self.image.pk).update(dateTakenUTC=1600000000.0)
        face = make_face(self.image, declared_name=self.person)
        Face.objects.filter(pk=face.pk).update(dateTakenUTC=-2208985139.0)

        call_command('sync_face_dates_from_source_image')

        face.refresh_from_db()
        self.assertEqual(face.dateTakenUTC, 1600000000.0)

    def test_face_already_plausible_is_never_touched(self):
        ImageFile.objects.filter(pk=self.image.pk).update(dateTakenUTC=1600000000.0)
        face = make_face(self.image, declared_name=self.person)
        Face.objects.filter(pk=face.pk).update(dateTakenUTC=1650000000.0)

        call_command('sync_face_dates_from_source_image')

        face.refresh_from_db()
        self.assertEqual(face.dateTakenUTC, 1650000000.0)

    def test_dry_run_writes_nothing(self):
        ImageFile.objects.filter(pk=self.image.pk).update(dateTakenUTC=1600000000.0)
        face = make_face(self.image, declared_name=self.person)
        Face.objects.filter(pk=face.pk).update(dateTakenUTC=-2208985139.0)

        call_command('sync_face_dates_from_source_image', '--dry-run')

        face.refresh_from_db()
        self.assertEqual(face.dateTakenUTC, -2208985139.0)

    def test_rerun_finds_nothing(self):
        ImageFile.objects.filter(pk=self.image.pk).update(dateTakenUTC=1600000000.0)
        face = make_face(self.image, declared_name=self.person)
        Face.objects.filter(pk=face.pk).update(dateTakenUTC=-2208985139.0)

        call_command('sync_face_dates_from_source_image')
        call_command('sync_face_dates_from_source_image')  # must be a safe no-op

        face.refresh_from_db()
        self.assertEqual(face.dateTakenUTC, 1600000000.0)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class BackfillMissingFaceThumbnailsTests(TestCase):
    """backfill_missing_face_thumbnails management command: real
    incident, 2026-10-06 -- a restored-from-backup database referenced
    thumbnail files that no longer existed on disk. Re-crops from the
    already-stored box (no re-detection) using a real fixture and a real
    decode, rather than mocking -- this command's whole point is pure
    cropping math against real pixel data, so there's nothing to usefully
    mock here."""

    def setUp(self):
        self.image = make_image()
        self.person = make_person("Someone Real")

    def test_missing_thumbnail_is_regenerated_from_the_stored_box(self):
        face = make_face(self.image, declared_name=self.person)
        old_name = face.face_thumbnail.name
        os.remove(face.face_thumbnail.path)

        call_command('backfill_missing_face_thumbnails')

        face.refresh_from_db()
        self.assertTrue(os.path.isfile(face.face_thumbnail.path))
        # A fresh save() with the same filename-derivation inputs can
        # coincidentally reuse the same name if nothing else on disk
        # collides -- what actually matters is that a real file exists
        # now, not that the name necessarily changed.
        self.assertIsNotNone(face.face_thumbnail.name)
        # Classification fields must be untouched.
        self.assertEqual(face.declared_name_id, self.person.id)

    def test_existing_thumbnail_is_left_untouched(self):
        face = make_face(self.image, declared_name=self.person)
        original_name = face.face_thumbnail.name
        original_mtime = os.path.getmtime(face.face_thumbnail.path)

        call_command('backfill_missing_face_thumbnails')

        face.refresh_from_db()
        self.assertEqual(face.face_thumbnail.name, original_name)
        self.assertEqual(os.path.getmtime(face.face_thumbnail.path), original_mtime)

    def test_dry_run_does_not_write(self):
        face = make_face(self.image, declared_name=self.person)
        os.remove(face.face_thumbnail.path)

        call_command('backfill_missing_face_thumbnails', '--dry-run')

        face.refresh_from_db()
        self.assertFalse(os.path.isfile(face.face_thumbnail.path))

    def test_decode_failure_is_counted_not_fatal(self):
        face = make_face(self.image, declared_name=self.person)
        os.remove(face.face_thumbnail.path)
        with patch(
            'face_manager.management.commands.backfill_missing_face_thumbnails.common.open_img_oriented',
            side_effect=OSError('simulated decode failure'),
        ):
            call_command('backfill_missing_face_thumbnails')  # must not raise
        face.refresh_from_db()
        self.assertFalse(os.path.isfile(face.face_thumbnail.path))


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class FaceSaveThumbnailCheckTests(TestCase):
    """Face.save()'s thumbnail-existence check uses .path (a pure
    filesystem-path computation via the storage backend), not
    .file.name (which OPENS the underlying file via FieldFile._get_file()
    and is never closed by Django) -- so it stays cheap and leak-free
    regardless of update_fields, not just for a narrow save. Real crash
    this was found from: OSError: Too many open files at ~600 images
    into backfill_det_score, since that command holds every Face object
    in memory for the whole run so nothing is ever garbage-collected
    between saves. See BackfillDetScoreTests.
    test_matched_face_save_does_not_open_thumbnail_file for the same
    guarantee exercised through the real backfill command."""

    def test_save_never_opens_the_thumbnail_file(self):
        # Patches Storage.open itself (not FieldFile._get_file) since
        # property(_get_file, ...) captures that function object at
        # class-definition time -- patching the _get_file *name*
        # afterward doesn't change what the already-built property
        # descriptor actually calls. A freshly-fetched Face (matching how
        # the real backfill always encounters them) has no cached
        # FieldFile._file yet either, unlike re-saving the same in-memory
        # instance make_face() already saved once.
        from django.core.files.storage import Storage
        image = make_image()
        face_id = make_face(image).id
        for save_kwargs in [{}, {'update_fields': ['detected_age']}, {'update_fields': ['face_thumbnail']}]:
            with self.subTest(save_kwargs=save_kwargs):
                face = Face.objects.get(id=face_id)
                face.detected_age = 42
                with patch.object(Storage, 'open', autospec=True) as mock_storage_open:
                    face.save(**save_kwargs)
                mock_storage_open.assert_not_called()

    def test_missing_thumbnail_file_still_raises(self):
        image = make_image()
        face = make_face(image)
        os.remove(face.face_thumbnail.path)
        face.detected_age = 43
        with self.assertRaises(ValidationError):
            face.save()



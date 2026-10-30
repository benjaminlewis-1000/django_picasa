"""face_manager tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test face_manager.tests.test_core_models
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
class FaceKpsWithinBoxConstraintTests(TestCase):
    """DB-level structural backstop (migration 0019, see CLAUDE.md) for
    the real kps/box translation bug found 2026-10-26 -- a 25%-margin
    CHECK constraint that a wildly-mismatched kps (the actual failure
    mode, off by thousands of pixels) can never pass, while real,
    legitimately-slightly-off-box landmarks (extreme pose/expression)
    still can."""

    def setUp(self):
        self.image = make_image()

    def test_kps_wildly_outside_box_is_rejected(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                make_face(self.image, box_left=100, box_top=100, box_right=150, box_bottom=150,
                           kps=[900.0] * 10)

    def test_kps_centered_in_box_is_accepted(self):
        face = make_face(
            self.image, box_left=100, box_top=100, box_right=150, box_bottom=150,
            kps=[110, 110, 140, 110, 125, 125, 115, 140, 135, 140],
        )
        self.assertIsNotNone(face.id)

    def test_kps_slightly_outside_box_within_margin_is_accepted(self):
        # A landmark a few pixels past the box edge (plausible under an
        # extreme pose) must not be rejected -- the whole reason this
        # constraint uses a 25% margin, not a strict zero one (see its
        # own comment for the real, measured false-positive rate at
        # zero margin).
        face = make_face(
            self.image, box_left=100, box_top=100, box_right=150, box_bottom=150,
            kps=[95, 110, 140, 110, 125, 125, 115, 140, 135, 140],
        )
        self.assertIsNotNone(face.id)

    def test_null_kps_is_always_accepted(self):
        face = make_face(self.image, box_left=100, box_top=100, box_right=150, box_bottom=150)
        self.assertIsNone(face.kps)
        self.assertIsNotNone(face.id)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class PersonModelTests(TestCase):
    def test_blank_sentinel_cannot_be_deleted(self):
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        blank.delete()
        self.assertTrue(Person.objects.filter(person_name=settings.BLANK_FACE_NAME).exists())

    def test_deleting_person_removes_highlight_file(self):
        person = make_person("Deletable")
        highlight_path = person.highlight_img.path
        self.assertTrue(os.path.exists(highlight_path))
        person.delete()
        self.assertFalse(os.path.exists(highlight_path))

    def test_get_default_blank_person_returns_existing_sentinel(self):
        # Covers face_manager/models.py get_default_blank_person(). Its
        # "person doesn't exist yet" fallback branch is unreachable/broken
        # (references an undefined `sq_thumb` and uses BytesIO/ContentFile
        # without importing them) -- not exercised here since the sentinel
        # is expected to always exist; see NOTES on bootstrap gap.
        blank = get_default_blank_person()
        self.assertEqual(blank.person_name, settings.BLANK_FACE_NAME)

    def test_live_face_counts_reflect_real_assignments(self):
        # Person.num_faces/num_possibilities/num_unverified_faces used to
        # be cached IntegerFields, manually kept in sync by
        # increment_*/decrement_* methods called from Face's assignment
        # methods -- removed because that bookkeeping could silently
        # drift stale (e.g. a bulk .update() bypassing it entirely). Now
        # computed live -- see face_manager/live_counts.py.
        person = make_person("Counted")
        image = make_image()
        face = make_face(image, declared_name=person)
        counts = compute_live_face_counts([person.id])[person.id]
        self.assertEqual(counts['num_faces'], 1)
        self.assertEqual(counts['num_unverified_faces'], 1)

        face.verify_person_in_image()
        counts = compute_live_face_counts([person.id])[person.id]
        self.assertEqual(counts['num_faces'], 1)
        self.assertEqual(counts['num_unverified_faces'], 0)

        other = make_person("Other")
        face.set_possible_person(other.id, 1, 0.9)
        counts = compute_live_face_counts([other.id])[other.id]
        self.assertEqual(counts['num_possibilities'], 1)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class FaceModelTests(TestCase):
    def setUp(self):
        self.image = make_image()

    def test_face_save_rejects_inverted_box(self):
        person = make_person("Boxy")
        face = Face(
            declared_name=person,
            source_image_file=self.image,
            box_left=10,
            box_top=10,
            box_right=5,  # right < left
            box_bottom=20,
        )
        face.face_thumbnail.save("t.jpg", ContentFile(_tiny_jpeg_bytes()), save=False)
        with self.assertRaises(ValidationError):
            face.save()

    def test_face_save_rejects_missing_thumbnail_file(self):
        person = make_person("NoThumb")
        face = Face(
            declared_name=person,
            source_image_file=self.image,
            box_left=1,
            box_top=1,
            box_right=20,
            box_bottom=20,
        )
        # face_thumbnail is left unset -> face.face_thumbnail.file access
        # in Face.save() should fail validation rather than silently save.
        with self.assertRaises(Exception):
            face.save()

    def test_face_delete_removes_thumbnail_file(self):
        face = make_face(self.image)
        thumb_path = face.face_thumbnail.path
        self.assertTrue(os.path.exists(thumb_path))
        face.delete()
        self.assertFalse(os.path.exists(thumb_path))

    def test_face_delete_tolerates_already_missing_thumbnail_file(self):
        # Regression test for a fixed bug: delete() unconditionally called
        # os.remove() with no guard for the file already being gone (e.g.
        # a duplicate delete against a restored DB snapshot whose rows
        # point at the same shared media path an earlier/live run already
        # cleaned up) -- used to raise FileNotFoundError instead of still
        # removing the row.
        face = make_face(self.image)
        os.remove(face.face_thumbnail.path)
        face.delete()
        self.assertFalse(Face.objects.filter(pk=face.pk).exists())

    def test_deleting_image_cascades_to_faces(self):
        face = make_face(self.image)
        face_id = face.id
        thumb_path = face.face_thumbnail.path
        self.image.delete()
        self.assertFalse(Face.objects.filter(id=face_id).exists())
        # Regression test for a fixed bug: Face.source_image_file's
        # on_delete=CASCADE means Django's cascade-delete collector used
        # to remove cascaded Face rows via a bulk SQL DELETE, which does
        # NOT call each instance's overridden delete() -- so the DB row
        # above was always cleaned up correctly, but this thumbnail file
        # was silently left orphaned on disk. ImageFile.delete() now
        # explicitly deletes each related Face first (invoking Face's own
        # delete() override) before deleting itself.
        self.assertFalse(os.path.exists(thumb_path))

    def test_associate_person_updates_declared_name_and_weight(self):
        original = make_person("Original")
        target = make_person("Target")
        face = make_face(self.image, declared_name=original)
        face.set_possible_person(target.id, 1, 0.9)

        face.associate_person(target.id)
        face.refresh_from_db()

        self.assertEqual(face.declared_name_id, target.id)
        self.assertEqual(face.weight_1, 0.0)
        self.assertFalse(face.validated)

    def test_associate_person_clears_possible_identity(self):
        # Regression test for a fixed bug: remove_poss_ident() used to clear
        # the FK by poking `self.__dict__['poss_identN_id'] = None` directly
        # instead of `self.poss_identN = None`, so Model.save()'s FK-cache
        # reconciliation silently restored the old value. Now uses real
        # setattr()/getattr(), matching how reject_association() always did.
        original = make_person("Original2")
        target = make_person("Target2")
        face = make_face(self.image, declared_name=original)
        face.set_possible_person(target.id, 1, 0.9)

        face.associate_person(target.id)
        face.refresh_from_db()

        self.assertIsNone(face.poss_ident1)

    def test_verify_person_in_image(self):
        person = make_person("Verifiable")
        face = make_face(self.image, declared_name=person)
        face.verify_person_in_image()
        face.refresh_from_db()
        self.assertTrue(face.validated)

    def test_reset_to_pool_returns_face_to_blank_sentinel(self):
        # reset_to_pool() must land the face on the blank sentinel Person,
        # NOT NULL -- a NULL declared_name is invisible to the "Unassigned"
        # bucket and to assign_faces.py's re-classification, both of which
        # filter declared_name__person_name == settings.BLANK_FACE_NAME.
        blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        owner = make_person("PrevOwner")
        guesser = make_person("Guesser")
        face = make_face(self.image, declared_name=owner)
        face.set_possible_person(guesser.id, 1, 0.8)

        face.reset_to_pool()
        face.refresh_from_db()

        self.assertEqual(face.declared_name_id, blank.id)
        self.assertIsNotNone(face.declared_name)
        self.assertFalse(face.validated)
        self.assertIsNone(face.poss_ident1)
        self.assertEqual(owner.face_declared.count(), 0)

    def test_reject_association_removes_from_possibles(self):
        person = make_person("Rejectable")
        face = make_face(self.image)
        face.set_possible_person(person.id, 1, 0.75)
        face.reject_association(person.id)
        face.refresh_from_db()
        self.assertIsNone(face.poss_ident1)
        self.assertIn(person.id, face.rejected_fields)

    def test_add_to_rejected_fields_appends_and_dedupes_without_saving(self):
        face = make_face(self.image)
        self.assertIsNone(face.rejected_fields)

        face.add_to_rejected_fields(101)
        self.assertEqual(face.rejected_fields, [101])
        # Not saved yet - a caller doing other unsaved changes in the same
        # operation (e.g. close_assigned's "Remove from person" case) can
        # rely on a single later save() persisting both.
        face.refresh_from_db()
        self.assertIsNone(face.rejected_fields)

        face.add_to_rejected_fields(101)
        face.add_to_rejected_fields(202)
        face.add_to_rejected_fields(101)  # duplicate
        self.assertEqual(sorted(face.rejected_fields), [101, 202])

    def test_face_encoding_512_stores_at_float32_precision(self):
        """face_encoding_512 is now backed by `real` (single precision),
        not `double precision` -- insightface's embeddings are natively
        float32 (verified directly against real detections), so storing
        at float32 precision loses nothing the model didn't already lose.
        A value with real double-precision-only bits should get rounded
        to its nearest float32 representation on save, not preserved
        exactly."""
        double_only_value = 0.123456789012345  # not exactly representable in float32
        face = make_face(self.image)
        face.face_encoding_512 = [double_only_value] * 512
        face.save()
        face.refresh_from_db()

        # Compare via a float32 narrowing on both sides rather than exact
        # double equality: Postgres's `real` -> text -> Python float round
        # trip (via psycopg2) doesn't necessarily reproduce the exact same
        # double bit pattern as `float(np.float32(x))` computed directly in
        # Python -- e.g. this test observed the DB round trip coming back
        # as 0.12345679 (fewer significant digits) rather than
        # 0.12345679104328156. Narrowing both to float32 collapses that
        # text-precision noise while still genuinely verifying the stored
        # value lost precision down to float32, which is the actual point
        # of this test.
        self.assertEqual(
            np.float32(face.face_encoding_512[0]), np.float32(double_only_value)
        )
        self.assertNotEqual(face.face_encoding_512[0], double_only_value)


class FaceSetEmbeddingDualWriteTests(TestCase):
    """Face.set_embedding() -- the single write path for the dual-write
    phase of the pgvector/halfvec migration (2026-10-30, see CLAUDE.md).
    Every real call site should go through this instead of a bare
    attribute assignment, so the float32 field and its halfvec mirror
    can never silently drift apart."""

    def setUp(self):
        self.image = make_image()

    def test_sets_both_the_float32_field_and_its_halfvec_mirror(self):
        face = make_face(self.image)
        embedding = list(np.arange(512, dtype=np.float32) / 100.0)
        face.set_embedding('face_encoding_512', embedding)
        self.assertEqual(face.face_encoding_512, embedding)
        self.assertEqual(face.face_encoding_512_hv, embedding)

    def test_accepts_a_raw_numpy_array_not_just_a_list(self):
        face = make_face(self.image)
        embedding = np.arange(512, dtype=np.float32) / 100.0
        face.set_embedding('face_encoding_lvface_b', embedding)
        self.assertEqual(face.face_encoding_lvface_b, embedding.tolist())
        self.assertEqual(face.face_encoding_lvface_b_hv, embedding.tolist())

    def test_none_clears_both(self):
        face = make_face(self.image, face_encoding_512=[0.1] * 512, face_encoding_512_hv=[0.1] * 512)
        face.set_embedding('face_encoding_512', None)
        self.assertIsNone(face.face_encoding_512)
        self.assertIsNone(face.face_encoding_512_hv)

    def test_sentinel_value_is_set_on_both_unchanged(self):
        """The NON_DETECTED_FACE_ENCODING sentinel is a valid `value`
        here too -- halfvec doesn't need special-casing for it, it's
        just another vector as far as this helper is concerned."""
        face = make_face(self.image)
        face.set_embedding('face_encoding_512', settings.NON_DETECTED_FACE_ENCODING)
        self.assertEqual(face.face_encoding_512, settings.NON_DETECTED_FACE_ENCODING)
        self.assertEqual(face.face_encoding_512_hv, settings.NON_DETECTED_FACE_ENCODING)

    def test_real_save_round_trips_through_both_columns(self):
        face = make_face(self.image)
        embedding = list(np.arange(512, dtype=np.float32) / 100.0)
        face.set_embedding('face_encoding_lvface_s', embedding)
        face.save()
        face.refresh_from_db()
        self.assertEqual(face.face_encoding_lvface_s, embedding)
        self.assertIsNotNone(face.face_encoding_lvface_s_hv)


@tag("slow")
@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class PyramidalDetectorTileTranslationTests(TestCase):
    """Fast, mocked unit tests for find_raw_faces()'s tile-offset
    translation -- no real model loading. Regression coverage for a real
    bug found 2026-10-26 (see CLAUDE.md): bbox was translated from a
    tiled sub-image's own local coordinates back to the full image's,
    but kps never was, silently storing a kps thousands of pixels from
    its own face's box for any detection whose winning (NMS-selected)
    result came from a tiled (cut_dims > 1) pass -- confirmed in
    production affecting ~360k faces (46% of everything with both kps
    and box), with mismatched kps clustering unmistakably in the
    image's top-left quadrant (the signature of a small, untranslated
    local coordinate read as if it were already absolute)."""

    def _fake_face(self, bbox, kps, det_score=0.9):
        import insightface.app.common
        return insightface.app.common.Face(
            bbox=np.array(bbox, dtype=np.float32),
            kps=np.array(kps, dtype=np.float32),
            det_score=det_score,
        )

    def test_tiled_detection_kps_is_translated_to_full_image_coordinates(self):
        detector = PyramidalDetector.__new__(PyramidalDetector)
        detector.pct_overlap = 0.0
        detector.cut_list = [1, 3]
        mock_app = MagicMock()
        # A 900x900 image, cut into a 3x3 grid (300x300 tiles, no overlap
        # since pct_overlap=0), plus the required whole-image (cut_dims=1)
        # pass. Every call returns the same chip-local detection (bbox/kps
        # both near the chip's own origin) -- a fresh Face instance per
        # call, since find_raw_faces() mutates it in place and a shared
        # object would leak offsets across calls. Only the tile at
        # (x=2, y=1) is asserted on below.
        local_bbox = [10, 20, 60, 70]
        local_kps = [[20, 30], [50, 30], [35, 45], [25, 60], [45, 60]]
        mock_app.get.side_effect = lambda *a, **k: [self._fake_face(local_bbox, local_kps)]
        detector.app = mock_app

        np_image = np.zeros((900, 900, 3), dtype=np.uint8)
        overlapping_detections, box_edges = detector.find_raw_faces(np_image)

        # Tile (x=2, y=1): left_edge=600, top_edge=300. Filter on both
        # axes -- (x=2, y=0) also has bbox[0]==610 (same left_edge).
        target = [d for d in overlapping_detections if d['bbox'][0] == 610 and d['bbox'][1] == 320][0]
        self.assertEqual(list(target['bbox']), [610, 320, 660, 370])
        expected_kps = np.array(local_kps, dtype=np.float32) + np.array([600, 300])
        np.testing.assert_array_almost_equal(target['kps'], expected_kps)
        # The translated kps centroid must fall within the translated box.
        cx, cy = target['kps'][:, 0].mean(), target['kps'][:, 1].mean()
        bl, bt, br, bb = target['bbox']
        self.assertTrue(bl <= cx <= br and bt <= cy <= bb)

    def test_whole_image_pass_kps_is_unaffected(self):
        # cut_dims == 1 (the whole-image pass) needs no translation at
        # all -- confirms this fix didn't touch that branch.
        detector = PyramidalDetector.__new__(PyramidalDetector)
        detector.pct_overlap = 0.0
        detector.cut_list = [1]
        mock_app = MagicMock()
        bbox = [100, 100, 150, 150]
        kps = [[110, 110], [140, 110], [125, 125], [115, 140], [135, 140]]
        mock_app.get.return_value = [self._fake_face(bbox, kps)]
        detector.app = mock_app

        np_image = np.zeros((300, 300, 3), dtype=np.uint8)
        overlapping_detections, box_edges = detector.find_raw_faces(np_image)

        np.testing.assert_array_almost_equal(overlapping_detections[0]['kps'], np.array(kps, dtype=np.float32))


class PyramidalDetectorRealInferenceTests(TestCase):
    """Runs real insightface inference against known fixture photos.
    Uses test_face_cache so repeat runs (unchanged image + unchanged
    pyramidal_detector.py) skip the actual CPU inference."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.detector = PyramidalDetector(iou_thresh=0.3)

    def test_detects_faces_in_known_multi_face_fixture(self):
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/has_face_tags.jpg"
        faces = cached_detect(self.detector, path)
        self.assertGreaterEqual(len(faces), 1)
        for f in faces:
            self.assertEqual(len(f["embedding"]), 512)

    def test_detects_faces_in_second_known_fixture(self):
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/has_same_faces.jpg"
        faces = cached_detect(self.detector, path)
        self.assertGreaterEqual(len(faces), 1)



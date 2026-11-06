"""Tests for face_manager.rotation_correction.apply_rotation_correction().
The FaceExtractor dependency is a MagicMock throughout (only
get_square_face_img/hash_numpy_array are ever called -- neither touches
a model, see the module's own docstring) -- real model inference is
exercised separately via a real end-to-end run (see CLAUDE.md's
2026-11-05/11-06 write-up), not re-proven here. These tests cover the
orchestration wiring: box/kps transform values actually land on the
Face row, thumbnails are regenerated, embeddings are left untouched
(confirmed unnecessary -- see the module docstring), and isProcessed is
cleared so the standard face_extraction pipeline picks the image back
up.
"""
from unittest.mock import MagicMock, patch

import numpy as np
from django.test import TestCase

from face_manager.rotation_correction import apply_rotation_correction
from face_manager.rotation_correction_geometry import transform_box, transform_kps
from face_manager.tests import make_face, make_image
from filepopulator.models import ImageFile


def _mock_extractor():
    extractor = MagicMock()
    extractor.get_square_face_img.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
    extractor.hash_numpy_array.return_value = 'deadbeef1234567890'
    return extractor


class ApplyRotationCorrectionTests(TestCase):
    def setUp(self):
        self.image = make_image()
        self.old_w, self.old_h = self.image.width, self.image.height
        self.image.proposed_rotation_degrees = 90.0
        self.image.rotation_scan_state = self.image.RotationScanState.CONFIRMED
        self.image.save(update_fields=['proposed_rotation_degrees', 'rotation_scan_state'])

    def test_existing_face_box_and_kps_are_transformed(self):
        kps = [10.0, 10.0, 20.0, 10.0, 15.0, 15.0, 10.0, 20.0, 20.0, 20.0]
        face = make_face(self.image, box_left=5, box_top=5, box_right=25, box_bottom=25, kps=kps)

        extractor = _mock_extractor()
        result = apply_rotation_correction(self.image, extractor)

        face.refresh_from_db()
        expected_box = transform_box(5, 5, 25, 25, self.old_w, self.old_h, 90.0)
        self.assertEqual((face.box_left, face.box_top, face.box_right, face.box_bottom), expected_box)
        expected_kps = transform_kps(kps, self.old_w, self.old_h, 90.0)
        self.assertEqual(list(face.kps), expected_kps)
        self.assertIn(face.id, result['updated_face_ids'])

    def test_face_without_kps_still_gets_box_transformed(self):
        face = make_face(self.image, box_left=5, box_top=5, box_right=25, box_bottom=25, kps=None)

        extractor = _mock_extractor()
        apply_rotation_correction(self.image, extractor)

        face.refresh_from_db()
        self.assertEqual(
            (face.box_left, face.box_top, face.box_right, face.box_bottom),
            transform_box(5, 5, 25, 25, self.old_w, self.old_h, 90.0),
        )
        self.assertIsNone(face.kps)

    def test_embeddings_are_left_untouched(self):
        # Confirmed unnecessary (see rotation_correction.py's own
        # docstring): a global rigid rotation + the matching kps
        # transform reproduces the identical embedding, so this function
        # must never overwrite an existing stored embedding.
        kps = [10.0, 10.0, 20.0, 10.0, 15.0, 15.0, 10.0, 20.0, 20.0, 20.0]
        original_embedding = [0.5] * 512
        face = make_face(
            self.image, box_left=5, box_top=5, box_right=25, box_bottom=25, kps=kps,
            face_encoding_512=original_embedding,
        )

        extractor = _mock_extractor()
        apply_rotation_correction(self.image, extractor)

        face.refresh_from_db()
        self.assertEqual(list(face.face_encoding_512), original_embedding)

    def test_no_missed_face_redetection_is_performed(self):
        # The bespoke missed-face redetection block was removed -- see
        # the module docstring for why that's now the standard
        # face_extraction pipeline's job, triggered via isProcessed
        # below, not this function's own. A mock extractor with no
        # detection-related methods configured confirms nothing here
        # calls into detection at all.
        make_face(self.image, box_left=5, box_top=5, box_right=25, box_bottom=25, kps=None)

        extractor = _mock_extractor()
        apply_rotation_correction(self.image, extractor)

        extractor.app.get.assert_not_called()
        extractor.add_new_face.assert_not_called()

    def test_isprocessed_is_cleared_so_standard_pipeline_reconciles(self):
        make_face(self.image, box_left=5, box_top=5, box_right=25, box_bottom=25, kps=None)
        self.image.isProcessed = True
        self.image.save(update_fields=['isProcessed'])

        extractor = _mock_extractor()
        apply_rotation_correction(self.image, extractor)

        self.assertFalse(self.image.isProcessed)

    def test_image_thumbnail_small_is_regenerated_at_the_swapped_dimensions(self):
        make_face(self.image, box_left=5, box_top=5, box_right=25, box_bottom=25, kps=None)

        extractor = _mock_extractor()
        apply_rotation_correction(self.image, extractor)

        # A 90-degree correction swaps width/height; _generate_thumbnail()
        # was called against the rotated pixels (the in-memory instance's
        # width/height were updated before it ran).
        self.assertEqual(self.image.width, self.old_h)
        self.assertEqual(self.image.height, self.old_w)
        self.assertTrue(self.image.thumbnail_small.name)

    def test_a_decode_failure_raises_rather_than_silently_no_op(self):
        make_face(self.image, box_left=5, box_top=5, box_right=25, box_bottom=25, kps=None)
        extractor = _mock_extractor()
        with patch('face_manager.rotation_correction.common.open_img_oriented', return_value=None):
            with self.assertRaises(RuntimeError):
                apply_rotation_correction(self.image, extractor)


class ResetRotationScanStateForReprocessingTests(TestCase):
    """ImageFile.reset_rotation_scan_state_for_reprocessing() -- called
    alongside isProcessed=False whenever a Face's existing kps is about
    to be thrown out/redone for a reason OUTSIDE the rotation-review
    pipeline's own control (an EXIF-orientation-change reprocess,
    cleanup_chronically_unmatched_faces), per CLAUDE.md's 2026-11-06
    write-up.
    """

    def setUp(self):
        self.image = make_image(relative_fixture="naming/good/1.JPG")

    def test_not_scanned_stays_not_scanned_and_version_resets(self):
        self.image.rotation_scanned_at_version = 1
        self.image.reset_rotation_scan_state_for_reprocessing()
        self.assertEqual(self.image.rotation_scan_state, ImageFile.RotationScanState.NOT_SCANNED)
        self.assertIsNone(self.image.proposed_rotation_degrees)
        self.assertIsNone(self.image.rotation_signal_source)
        self.assertEqual(self.image.rotation_scanned_at_version, 0)

    def test_flagged_is_reset_to_not_scanned(self):
        self.image.rotation_scan_state = ImageFile.RotationScanState.FLAGGED
        self.image.proposed_rotation_degrees = 93.2
        self.image.rotation_signal_source = ImageFile.RotationSignalSource.LANDMARK_SINGLE_FACE
        self.image.rotation_scanned_at_version = 1

        self.image.reset_rotation_scan_state_for_reprocessing()

        self.assertEqual(self.image.rotation_scan_state, ImageFile.RotationScanState.NOT_SCANNED)
        self.assertIsNone(self.image.proposed_rotation_degrees)
        self.assertIsNone(self.image.rotation_signal_source)
        self.assertEqual(self.image.rotation_scanned_at_version, 0)

    def test_confirmed_is_never_touched(self):
        # A human's decision must survive reprocessing that happens to
        # touch this image for an unrelated reason -- only face_manager.
        # rotation_correction's OWN isProcessed=False step (applying that
        # same confirmation) is allowed to coexist with CONFIRMED.
        self.image.rotation_scan_state = ImageFile.RotationScanState.CONFIRMED
        self.image.proposed_rotation_degrees = 90.0
        self.image.rotation_scanned_at_version = 1

        self.image.reset_rotation_scan_state_for_reprocessing()

        self.assertEqual(self.image.rotation_scan_state, ImageFile.RotationScanState.CONFIRMED)
        self.assertEqual(self.image.proposed_rotation_degrees, 90.0)
        self.assertEqual(self.image.rotation_scanned_at_version, 1)

    def test_dismissed_not_rotated_is_never_touched(self):
        self.image.rotation_scan_state = ImageFile.RotationScanState.DISMISSED_NOT_ROTATED
        self.image.rotation_scanned_at_version = 1

        self.image.reset_rotation_scan_state_for_reprocessing()

        self.assertEqual(
            self.image.rotation_scan_state, ImageFile.RotationScanState.DISMISSED_NOT_ROTATED,
        )
        self.assertEqual(self.image.rotation_scanned_at_version, 1)

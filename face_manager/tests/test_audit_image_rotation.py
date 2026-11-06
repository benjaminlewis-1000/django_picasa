"""Tests for the pure evaluate_image() helper in face_manager.rotation_audit,
plus the scheduled audit_image_rotation_task's lock/crash-logging
wrapper (face_manager/tasks.py). Uses lightweight fake "face" objects
(just need a .kps attribute) rather than real DB Face rows for the pure
tests, since evaluate_image() itself is DB-free.
"""
import math
from types import SimpleNamespace
from unittest.mock import patch

from django.test import TestCase

from face_manager.rotation_audit import evaluate_image
from filepopulator.models import ImageFile


def _kps_for_eye_angle(angle_deg, eye_separation=50.0):
    cx, cy = 100.0, 100.0
    half = eye_separation / 2.0
    rad = math.radians(angle_deg)
    dx = half * math.cos(rad)
    dy = half * math.sin(rad)
    left_eye = (cx - dx, cy - dy)
    right_eye = (cx + dx, cy + dy)
    return [left_eye, right_eye, (cx, cy + 20), (cx - 15, cy + 35), (cx + 15, cy + 35)]


def _fake_face(angle_deg):
    return SimpleNamespace(kps=_kps_for_eye_angle(angle_deg))


class EvaluateImageTests(TestCase):
    def test_no_faces_returns_none(self):
        self.assertIsNone(evaluate_image(None, []))

    def test_single_upright_face_not_significant(self):
        outcome = evaluate_image(None, [_fake_face(2.0)])
        self.assertTrue(outcome['examined'])
        self.assertFalse(outcome['significant'])
        self.assertEqual(outcome['source'], ImageFile.RotationSignalSource.LANDMARK_SINGLE_FACE)

    def test_single_rotated_face_is_significant(self):
        outcome = evaluate_image(None, [_fake_face(90.0)])
        self.assertTrue(outcome['significant'])
        self.assertAlmostEqual(outcome['estimate'], 90.0, delta=1)
        self.assertEqual(outcome['source'], ImageFile.RotationSignalSource.LANDMARK_SINGLE_FACE)

    def test_two_agreeing_rotated_faces_flagged_multi_face(self):
        outcome = evaluate_image(None, [_fake_face(88.0), _fake_face(92.0)])
        self.assertTrue(outcome['significant'])
        self.assertAlmostEqual(outcome['estimate'], 90.0, delta=1)
        self.assertEqual(outcome['source'], ImageFile.RotationSignalSource.LANDMARK_MULTI_FACE)

    def test_two_disagreeing_faces_ambiguous(self):
        outcome = evaluate_image(None, [_fake_face(0.0), _fake_face(90.0)])
        self.assertTrue(outcome['examined'])
        self.assertIsNone(outcome['estimate'])
        self.assertFalse(outcome['significant'])

    def test_two_agreeing_upright_faces_not_significant(self):
        outcome = evaluate_image(None, [_fake_face(1.0), _fake_face(-1.0)])
        self.assertFalse(outcome['significant'])
        self.assertEqual(outcome['source'], ImageFile.RotationSignalSource.LANDMARK_MULTI_FACE)

    def test_three_faces_outlier_dropped_still_significant(self):
        outcome = evaluate_image(None, [_fake_face(88.0), _fake_face(92.0), _fake_face(-30.0)])
        self.assertTrue(outcome['significant'])
        self.assertAlmostEqual(outcome['estimate'], 90.0, delta=2)


class AuditImageRotationTaskTests(TestCase):
    """The scheduled Celery task is a thin advisory-lock + crash-logging
    wrapper around run_rotation_audit() -- same shape as this codebase's
    other scheduled tasks (e.g. apply_collection_floor_change).
    """

    def test_concurrent_run_is_skipped(self):
        from face_manager.tasks import audit_image_rotation_task

        with patch('face_manager.tasks.advisory_lock') as mock_lock, \
             patch('face_manager.tasks.run_rotation_audit') as mock_run:
            mock_lock.return_value.__enter__.return_value = False
            audit_image_rotation_task()

        mock_run.assert_not_called()

    def test_runs_and_writes_when_lock_acquired(self):
        from face_manager.tasks import audit_image_rotation_task

        with patch('face_manager.tasks.advisory_lock') as mock_lock, \
             patch('face_manager.tasks.run_rotation_audit') as mock_run:
            mock_lock.return_value.__enter__.return_value = True
            mock_run.return_value = {'examined': 0, 'counts': {}, 'updated': 0}
            audit_image_rotation_task()

        mock_run.assert_called_once_with(write=True)

    def test_a_crash_is_recorded_as_an_application_error(self):
        from common.models import ApplicationError
        from face_manager.tasks import audit_image_rotation_task

        with patch('face_manager.tasks.advisory_lock') as mock_lock, \
             patch('face_manager.tasks.run_rotation_audit', side_effect=RuntimeError("boom")):
            mock_lock.return_value.__enter__.return_value = True
            audit_image_rotation_task()

        self.assertTrue(
            ApplicationError.objects.filter(source__icontains='audit_image_rotation_task').exists()
        )


class RunRotationCorrectionTests(TestCase):
    """face_manager.rotation_correction.run_rotation_correction() -- the
    DB orchestration loop around apply_rotation_correction(), mirroring
    run_rotation_audit() above. FaceExtractor/apply_rotation_correction
    are mocked throughout -- real model inference is exercised
    separately (see CLAUDE.md's 2026-11-05 write-up), not re-proven here.
    """

    def setUp(self):
        from face_manager.tests import make_image

        # make_image() always resolves to the same ImageFile row for a
        # given fixture path (create_image_file()'s own same-content
        # dedup) -- distinct fixtures needed so these are genuinely
        # separate rows, not the same one overwritten three times.
        ScanState = ImageFile.RotationScanState
        self.confirmed = make_image(relative_fixture="naming/good/1.JPG")
        self.confirmed.rotation_scan_state = ScanState.CONFIRMED
        self.confirmed.proposed_rotation_degrees = 90.0
        self.confirmed.save(update_fields=['rotation_scan_state', 'proposed_rotation_degrees'])

        self.already_applied = make_image(relative_fixture="naming/good/2.jpg")
        self.already_applied.rotation_scan_state = ScanState.CONFIRMED
        self.already_applied.proposed_rotation_degrees = 90.0
        self.already_applied.rotation_correction_applied = True
        self.already_applied.save(update_fields=[
            'rotation_scan_state', 'proposed_rotation_degrees', 'rotation_correction_applied',
        ])

        self.flagged = make_image(relative_fixture="naming/good/3.JPEG")
        self.flagged.rotation_scan_state = ScanState.FLAGGED
        self.flagged.proposed_rotation_degrees = 90.0
        self.flagged.save(update_fields=['rotation_scan_state', 'proposed_rotation_degrees'])

    def test_no_eligible_images_is_a_cheap_no_op(self):
        from face_manager.rotation_correction import run_rotation_correction

        self.confirmed.delete()
        with patch('face_manager.rotation_correction.apply_rotation_correction') as mock_apply:
            result = run_rotation_correction()

        mock_apply.assert_not_called()
        self.assertEqual(result, {'corrected': 0, 'failed': 0, 'failures': []})

    def test_only_confirmed_and_not_yet_applied_images_are_processed(self):
        from face_manager.rotation_correction import run_rotation_correction

        with patch('face_extract_encode.FaceExtractor') as MockExtractor, \
             patch('face_manager.rotation_correction.apply_rotation_correction') as mock_apply:
            mock_apply.return_value = {'updated_face_ids': [], 'new_face_ids': []}
            result = run_rotation_correction()

        mock_apply.assert_called_once()
        self.assertEqual(mock_apply.call_args[0][0].id, self.confirmed.id)
        self.assertEqual(result['corrected'], 1)
        self.assertEqual(result['failed'], 0)

        self.confirmed.refresh_from_db()
        self.assertTrue(self.confirmed.rotation_correction_applied)

    def test_image_ids_narrows_the_queryset(self):
        from face_manager.rotation_correction import run_rotation_correction
        from face_manager.tests import make_image

        other = make_image(relative_fixture="naming/good/4.jpeg")
        other.rotation_scan_state = ImageFile.RotationScanState.CONFIRMED
        other.proposed_rotation_degrees = -90.0
        other.save(update_fields=['rotation_scan_state', 'proposed_rotation_degrees'])

        with patch('face_extract_encode.FaceExtractor'), \
             patch('face_manager.rotation_correction.apply_rotation_correction') as mock_apply:
            mock_apply.return_value = {'updated_face_ids': [], 'new_face_ids': []}
            result = run_rotation_correction(image_ids=[other.id])

        mock_apply.assert_called_once()
        self.assertEqual(mock_apply.call_args[0][0].id, other.id)
        self.assertEqual(result['corrected'], 1)

    def test_a_failure_is_recorded_and_does_not_mark_applied(self):
        from face_manager.rotation_correction import run_rotation_correction

        with patch('face_extract_encode.FaceExtractor'), \
             patch('face_manager.rotation_correction.apply_rotation_correction',
                   side_effect=RuntimeError("boom")):
            result = run_rotation_correction()

        self.assertEqual(result['corrected'], 0)
        self.assertEqual(result['failed'], 1)
        self.assertEqual(result['failures'], [(self.confirmed.id, "boom")])

        self.confirmed.refresh_from_db()
        self.assertFalse(self.confirmed.rotation_correction_applied)


class CorrectImageRotationTaskTests(TestCase):
    """The scheduled Celery task is a thin advisory-lock + crash-logging
    wrapper around run_rotation_correction(), same shape as
    AuditImageRotationTaskTests above.
    """

    def test_concurrent_run_is_skipped(self):
        from face_manager.tasks import correct_image_rotation_task

        with patch('face_manager.tasks.advisory_lock') as mock_lock, \
             patch('face_manager.tasks.run_rotation_correction') as mock_run:
            mock_lock.return_value.__enter__.return_value = False
            correct_image_rotation_task()

        mock_run.assert_not_called()

    def test_runs_when_lock_acquired(self):
        from face_manager.tasks import correct_image_rotation_task

        with patch('face_manager.tasks.advisory_lock') as mock_lock, \
             patch('face_manager.tasks.run_rotation_correction') as mock_run:
            mock_lock.return_value.__enter__.return_value = True
            mock_run.return_value = {'corrected': 0, 'failed': 0, 'failures': []}
            correct_image_rotation_task()

        mock_run.assert_called_once_with()

    def test_a_crash_is_recorded_as_an_application_error(self):
        from common.models import ApplicationError
        from face_manager.tasks import correct_image_rotation_task

        with patch('face_manager.tasks.advisory_lock') as mock_lock, \
             patch('face_manager.tasks.run_rotation_correction', side_effect=RuntimeError("boom")):
            mock_lock.return_value.__enter__.return_value = True
            correct_image_rotation_task()

        self.assertTrue(
            ApplicationError.objects.filter(source__icontains='correct_image_rotation_task').exists()
        )


class CorrectImageRotationIdsTaskTests(TestCase):
    """The on-demand (not scheduled) task dispatched directly from
    RotationReviewBulkActionView right after a human confirms -- no
    advisory lock at this level (see its own docstring for why), just
    a thin crash-logging wrapper around run_rotation_correction(image_ids=).
    """

    def test_calls_run_rotation_correction_with_the_given_ids(self):
        from face_manager.tasks import correct_image_rotation_ids_task

        with patch('face_manager.tasks.run_rotation_correction') as mock_run:
            mock_run.return_value = {'corrected': 1, 'failed': 0, 'failures': []}
            correct_image_rotation_ids_task([5, 6])

        mock_run.assert_called_once_with(image_ids=[5, 6])

    def test_a_crash_is_recorded_as_an_application_error(self):
        from common.models import ApplicationError
        from face_manager.tasks import correct_image_rotation_ids_task

        with patch('face_manager.tasks.run_rotation_correction', side_effect=RuntimeError("boom")):
            correct_image_rotation_ids_task([5])

        self.assertTrue(
            ApplicationError.objects.filter(
                source__icontains='correct_image_rotation_ids_task',
            ).exists()
        )


class RunRotationCorrectionPerImageLockTests(TestCase):
    """run_rotation_correction()'s per-image advisory lock -- lets the
    on-demand task and the scheduled safety-net sweep safely overlap on
    DIFFERENT rows, while never double-processing the SAME one. See the
    module's own comment for why this is per-image, not one shared lock.
    """

    def setUp(self):
        from face_manager.tests import make_image

        self.image = make_image(relative_fixture="naming/good/1.JPG")
        self.image.rotation_scan_state = ImageFile.RotationScanState.CONFIRMED
        self.image.proposed_rotation_degrees = 90.0
        self.image.save(update_fields=['rotation_scan_state', 'proposed_rotation_degrees'])

    def test_image_already_locked_by_another_caller_is_skipped(self):
        from face_manager.rotation_correction import run_rotation_correction

        with patch('face_extract_encode.FaceExtractor'), \
             patch('common.advisory_lock') as mock_lock, \
             patch('face_manager.rotation_correction.apply_rotation_correction') as mock_apply:
            mock_lock.return_value.__enter__.return_value = False
            result = run_rotation_correction()

        mock_apply.assert_not_called()
        self.assertEqual(result, {'corrected': 0, 'failed': 0, 'failures': []})
        self.image.refresh_from_db()
        self.assertFalse(self.image.rotation_correction_applied)

    def test_already_applied_after_acquiring_lock_is_skipped_not_reapplied(self):
        # Simulates the exact race the per-image lock exists to prevent:
        # another caller (e.g. the scheduled safety-net sweep, racing
        # against this on-demand call) finishes correcting this image in
        # the gap between this function's initial query and actually
        # acquiring the lock. The refresh-under-lock check must catch
        # that via the real DB state and skip, not blindly re-run
        # apply_rotation_correction() a second time.
        from face_manager.rotation_correction import run_rotation_correction

        class _LockEntersThenMarksAppliedElsewhere:
            def __enter__(self_inner):
                ImageFile.objects.filter(id=self.image.id).update(rotation_correction_applied=True)
                return True

            def __exit__(self_inner, *exc_info):
                return False

        with patch('face_extract_encode.FaceExtractor'), \
             patch('common.advisory_lock', return_value=_LockEntersThenMarksAppliedElsewhere()), \
             patch('face_manager.rotation_correction.apply_rotation_correction') as mock_apply:
            result = run_rotation_correction()

        mock_apply.assert_not_called()
        self.assertEqual(result['corrected'], 0)

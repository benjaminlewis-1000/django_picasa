"""Tests for face_manager.rotation_ml_audit -- the ML-classifier
counterpart to test_audit_image_rotation.py's landmark-signal tests.
Uses a cheap stub torch model (fixed logits regardless of input)
throughout -- no real checkpoint, no real inference -- so these run
fast and don't depend on a trained model existing on disk.
"""
from unittest.mock import patch

import numpy as np
from django.test import TestCase

from filepopulator.models import ImageFile


class _FixedLogitsModel:
    """A minimal stand-in for the real torch model: callable, returns a
    fixed logits tensor regardless of input, and supports .eval()/
    .load_state_dict() as no-ops so it can also stand in wherever a
    real model object is expected but never actually invoked.
    """

    def __init__(self, logits):
        import torch
        self._logits = torch.tensor([logits], dtype=torch.float32)

    def __call__(self, tensor):
        return self._logits

    def eval(self):
        return self


def _upright_confident_model():
    # class 0 (upright) picked with high confidence.
    return _FixedLogitsModel([10.0, 0.0, 0.0, 0.0])


def _rotated_90_confident_model():
    # class 1 (needs +90 undo) picked with high confidence.
    return _FixedLogitsModel([0.0, 10.0, 0.0, 0.0])


def _rotated_90_unconfident_model():
    # class 1 picked, but only barely -- below CONFIDENCE_FLOOR.
    return _FixedLogitsModel([0.1, 0.2, 0.0, 0.0])


class RotationMlScanHashTests(TestCase):
    def test_is_a_16_char_hex_string(self):
        from face_manager.rotation_ml_audit import _rotation_ml_scan_hash

        value = _rotation_ml_scan_hash()
        self.assertEqual(len(value), 16)
        int(value, 16)  # raises ValueError if not valid hex

    def test_is_deterministic_across_calls(self):
        from face_manager.rotation_ml_audit import _rotation_ml_scan_hash

        self.assertEqual(_rotation_ml_scan_hash(), _rotation_ml_scan_hash())

    def test_changes_if_module_source_changes(self):
        from face_manager.rotation_ml_audit import _rotation_ml_scan_hash

        before = _rotation_ml_scan_hash()
        with patch('inspect.getsource', return_value="x = 'a change somewhere'"):
            after = _rotation_ml_scan_hash()
        self.assertNotEqual(before, after)

    def test_missing_checkpoint_hashes_deterministically_not_an_error(self):
        from face_manager.rotation_ml_audit import _rotation_ml_scan_hash

        with patch('face_manager.rotation_ml_audit.ONNX_MODEL_PATH', '/nonexistent/path.pt'):
            value = _rotation_ml_scan_hash()
        self.assertEqual(len(value), 16)

    def test_different_checkpoint_bytes_change_the_hash(self):
        import tempfile

        from face_manager.rotation_ml_audit import _rotation_ml_scan_hash

        with tempfile.NamedTemporaryFile() as f1, tempfile.NamedTemporaryFile() as f2:
            f1.write(b'aaa')
            f1.flush()
            f2.write(b'bbb')
            f2.flush()
            with patch('face_manager.rotation_ml_audit.ONNX_MODEL_PATH', f1.name):
                hash1 = _rotation_ml_scan_hash()
            with patch('face_manager.rotation_ml_audit.ONNX_MODEL_PATH', f2.name):
                hash2 = _rotation_ml_scan_hash()
        self.assertNotEqual(hash1, hash2)


class ClassifyImageTests(TestCase):
    def test_confident_upright_prediction(self):
        from face_manager.rotation_ml_audit import classify_image

        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        predicted_class, confidence = classify_image(arr, _upright_confident_model())
        self.assertEqual(predicted_class, 0)
        self.assertGreater(confidence, 0.99)

    def test_confident_rotated_prediction(self):
        from face_manager.rotation_ml_audit import classify_image

        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        predicted_class, confidence = classify_image(arr, _rotated_90_confident_model())
        self.assertEqual(predicted_class, 1)
        self.assertGreater(confidence, 0.99)


class EvaluateImageMlTests(TestCase):
    def test_oversized_image_is_size_skipped_without_decoding(self):
        from face_manager.rotation_ml_audit import evaluate_image_ml

        with patch('face_manager.rotation_ml_audit.decode_upright') as mock_decode:
            outcome = evaluate_image_ml(
                "/does/not/matter.jpg", width=10000, height=10000, model=_upright_confident_model(),
            )
        mock_decode.assert_not_called()
        self.assertEqual(outcome['outcome'], 'size_skip')
        self.assertFalse(outcome['significant'])

    def test_extreme_aspect_ratio_is_size_skipped(self):
        from face_manager.rotation_ml_audit import evaluate_image_ml

        with patch('face_manager.rotation_ml_audit.decode_upright') as mock_decode:
            outcome = evaluate_image_ml(
                "/does/not/matter.jpg", width=3000, height=200, model=_upright_confident_model(),
            )
        mock_decode.assert_not_called()
        self.assertEqual(outcome['outcome'], 'size_skip')

    def test_missing_width_height_does_not_size_skip(self):
        from face_manager.rotation_ml_audit import evaluate_image_ml

        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        with patch('face_manager.rotation_ml_audit.decode_upright', return_value=arr):
            outcome = evaluate_image_ml(
                "/does/not/matter.jpg", width=None, height=None, model=_upright_confident_model(),
            )
        self.assertEqual(outcome['outcome'], 'classified')

    def test_decode_failure_is_reported_not_raised(self):
        from face_manager.rotation_ml_audit import evaluate_image_ml

        with patch('face_manager.rotation_ml_audit.decode_upright', return_value=None):
            outcome = evaluate_image_ml(
                "/does/not/matter.jpg", width=100, height=100, model=_upright_confident_model(),
            )
        self.assertEqual(outcome['outcome'], 'decode_failure')
        self.assertFalse(outcome['significant'])

    def test_confident_upright_is_not_significant(self):
        from face_manager.rotation_ml_audit import evaluate_image_ml

        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        with patch('face_manager.rotation_ml_audit.decode_upright', return_value=arr):
            outcome = evaluate_image_ml(
                "/does/not/matter.jpg", width=100, height=100, model=_upright_confident_model(),
            )
        self.assertEqual(outcome['outcome'], 'classified')
        self.assertFalse(outcome['significant'])

    def test_confident_rotation_is_significant_with_correct_degrees(self):
        from face_manager.rotation_ml_audit import evaluate_image_ml

        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        with patch('face_manager.rotation_ml_audit.decode_upright', return_value=arr):
            outcome = evaluate_image_ml(
                "/does/not/matter.jpg", width=100, height=100, model=_rotated_90_confident_model(),
            )
        self.assertEqual(outcome['outcome'], 'classified')
        self.assertTrue(outcome['significant'])
        self.assertEqual(outcome['degrees'], 90.0)

    def test_low_confidence_rotation_is_not_significant(self):
        # Real-world case this guards: the model leans toward "rotated"
        # but not confidently enough to trust -- must not flag.
        from face_manager.rotation_ml_audit import evaluate_image_ml

        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        with patch('face_manager.rotation_ml_audit.decode_upright', return_value=arr):
            outcome = evaluate_image_ml(
                "/does/not/matter.jpg", width=100, height=100,
                model=_rotated_90_unconfident_model(),
            )
        self.assertEqual(outcome['outcome'], 'classified')
        self.assertFalse(outcome['significant'])


class RunRotationMlAuditTests(TestCase):
    """run_rotation_ml_audit() against the real DB, with a stub model
    (no real checkpoint/inference) and decode_upright patched to avoid
    real file I/O -- these test the DB orchestration (candidate
    selection, hash-stamping, write vs. dry-run), not the model itself.
    """

    def setUp(self):
        from face_manager.tests import make_image

        self.images = []
        for fixture in ["naming/good/2.jpg", "naming/good/3.JPEG", "naming/good/4.jpeg"]:
            img = make_image(relative_fixture=fixture)
            img.isProcessed = True
            img.save(update_fields=['isProcessed'])
            self.images.append(img)

    def _decoded_arr(self, *args, **kwargs):
        return np.zeros((50, 50, 3), dtype=np.uint8)

    def test_write_false_examines_but_does_not_persist(self):
        from face_manager.rotation_ml_audit import run_rotation_ml_audit

        with patch('face_manager.rotation_ml_audit.decode_upright', side_effect=self._decoded_arr):
            result = run_rotation_ml_audit(write=False, model=_rotated_90_confident_model())

        self.assertEqual(result['examined'], 3)
        self.assertEqual(result['counts']['flagged_new'], 3)
        for img in self.images:
            img.refresh_from_db()
            self.assertEqual(img.rotation_scan_state, ImageFile.RotationScanState.NOT_SCANNED)
            self.assertEqual(img.rotation_ml_scan_hash, '')

    def test_write_true_persists_the_flag(self):
        from face_manager.rotation_ml_audit import _rotation_ml_scan_hash, run_rotation_ml_audit

        with patch('face_manager.rotation_ml_audit.decode_upright', side_effect=self._decoded_arr):
            result = run_rotation_ml_audit(write=True, model=_rotated_90_confident_model())

        self.assertEqual(result['updated'], 3)
        for img in self.images:
            img.refresh_from_db()
            self.assertEqual(img.rotation_scan_state, ImageFile.RotationScanState.FLAGGED)
            self.assertEqual(img.proposed_rotation_degrees, 90.0)
            self.assertEqual(img.rotation_signal_source, ImageFile.RotationSignalSource.ML_CLASSIFIER)
            self.assertEqual(img.rotation_ml_scan_hash, _rotation_ml_scan_hash())

    def test_confirmed_upright_stays_not_scanned_but_hash_stamped(self):
        from face_manager.rotation_ml_audit import _rotation_ml_scan_hash, run_rotation_ml_audit

        with patch('face_manager.rotation_ml_audit.decode_upright', side_effect=self._decoded_arr):
            run_rotation_ml_audit(write=True, model=_upright_confident_model())

        for img in self.images:
            img.refresh_from_db()
            self.assertEqual(img.rotation_scan_state, ImageFile.RotationScanState.NOT_SCANNED)
            self.assertIsNone(img.proposed_rotation_degrees)
            self.assertEqual(img.rotation_ml_scan_hash, _rotation_ml_scan_hash())

    def test_already_flagged_images_are_excluded_from_candidates(self):
        # Flagged by the LANDMARK path (or a prior ML run) -- either way,
        # must never be re-examined or have its flag touched.
        from face_manager.rotation_ml_audit import run_rotation_ml_audit

        self.images[0].rotation_scan_state = ImageFile.RotationScanState.FLAGGED
        self.images[0].proposed_rotation_degrees = -90.0
        self.images[0].rotation_signal_source = ImageFile.RotationSignalSource.LANDMARK_SINGLE_FACE
        self.images[0].save(update_fields=[
            'rotation_scan_state', 'proposed_rotation_degrees', 'rotation_signal_source',
        ])

        with patch('face_manager.rotation_ml_audit.decode_upright', side_effect=self._decoded_arr):
            result = run_rotation_ml_audit(write=True, model=_rotated_90_confident_model())

        self.assertEqual(result['examined'], 2)
        self.images[0].refresh_from_db()
        self.assertEqual(self.images[0].proposed_rotation_degrees, -90.0)
        self.assertEqual(
            self.images[0].rotation_signal_source, ImageFile.RotationSignalSource.LANDMARK_SINGLE_FACE,
        )

    def test_confirmed_and_dismissed_images_are_excluded_from_candidates(self):
        from face_manager.rotation_ml_audit import run_rotation_ml_audit

        self.images[0].rotation_scan_state = ImageFile.RotationScanState.CONFIRMED
        self.images[0].proposed_rotation_degrees = 180.0
        self.images[0].save(update_fields=['rotation_scan_state', 'proposed_rotation_degrees'])
        self.images[1].rotation_scan_state = ImageFile.RotationScanState.DISMISSED_NOT_ROTATED
        self.images[1].save(update_fields=['rotation_scan_state'])

        with patch('face_manager.rotation_ml_audit.decode_upright', side_effect=self._decoded_arr):
            result = run_rotation_ml_audit(write=True, model=_rotated_90_confident_model())

        self.assertEqual(result['examined'], 1)

    def test_isprocessed_false_is_excluded_from_candidates(self):
        from face_manager.rotation_ml_audit import run_rotation_ml_audit

        self.images[0].isProcessed = False
        self.images[0].save(update_fields=['isProcessed'])

        with patch('face_manager.rotation_ml_audit.decode_upright', side_effect=self._decoded_arr):
            result = run_rotation_ml_audit(write=True, model=_upright_confident_model())

        self.assertEqual(result['examined'], 2)

    def test_faceless_image_is_still_a_valid_candidate(self):
        # The whole point of this signal over the landmark one: no face/
        # kps requirement at all.
        from face_manager.rotation_ml_audit import run_rotation_ml_audit

        self.assertEqual(self.images[0].face_set.count(), 0)

        with patch('face_manager.rotation_ml_audit.decode_upright', side_effect=self._decoded_arr):
            result = run_rotation_ml_audit(write=True, model=_rotated_90_confident_model())

        self.assertEqual(result['examined'], 3)

    def test_max_images_caps_the_batch(self):
        from face_manager.rotation_ml_audit import run_rotation_ml_audit

        with patch('face_manager.rotation_ml_audit.decode_upright', side_effect=self._decoded_arr):
            result = run_rotation_ml_audit(write=True, max_images=1, model=_upright_confident_model())

        self.assertEqual(result['examined'], 1)

    def test_repeated_capped_calls_make_forward_progress(self):
        from face_manager.rotation_ml_audit import run_rotation_ml_audit

        model = _upright_confident_model()
        seen_hashes = set()
        with patch('face_manager.rotation_ml_audit.decode_upright', side_effect=self._decoded_arr):
            for _ in range(3):
                result = run_rotation_ml_audit(write=True, max_images=1, model=model)
                self.assertEqual(result['examined'], 1)
            # Backlog now exhausted.
            final = run_rotation_ml_audit(write=True, max_images=1, model=model)
        self.assertEqual(final['examined'], 0)
        for img in self.images:
            img.refresh_from_db()
            self.assertNotEqual(img.rotation_ml_scan_hash, '')
            seen_hashes.add(img.id)
        self.assertEqual(seen_hashes, {img.id for img in self.images})

    def test_size_skip_stamps_hash_without_decoding(self):
        from face_manager.rotation_ml_audit import _rotation_ml_scan_hash, run_rotation_ml_audit

        self.images[0].width = 10000
        self.images[0].height = 10000
        self.images[0].save(update_fields=['width', 'height'])

        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        with patch('face_manager.rotation_ml_audit.decode_upright', return_value=arr) as mock_decode:
            result = run_rotation_ml_audit(write=True, model=_upright_confident_model())

        self.assertEqual(result['counts']['size_skip'], 1)
        self.images[0].refresh_from_db()
        self.assertEqual(self.images[0].rotation_ml_scan_hash, _rotation_ml_scan_hash())
        # Confirms the oversized image never reached decode_upright at all.
        for call in mock_decode.call_args_list:
            self.assertNotEqual(call.args[0], self.images[0].filename)

    def test_no_model_and_no_checkpoint_raises(self):
        from face_manager.rotation_ml_audit import run_rotation_ml_audit

        with patch('face_manager.rotation_ml_audit.ONNX_MODEL_PATH', '/nonexistent/checkpoint.pt'):
            with self.assertRaises(FileNotFoundError):
                run_rotation_ml_audit(write=False)


class RunRotationMlAuditUntilDoneTests(TestCase):
    def setUp(self):
        from face_manager.tests import make_image

        self.images = []
        for fixture in ["naming/good/2.jpg", "naming/good/3.JPEG", "naming/good/4.jpeg"]:
            img = make_image(relative_fixture=fixture)
            img.isProcessed = True
            img.save(update_fields=['isProcessed'])
            self.images.append(img)

    def test_no_checkpoint_returns_immediately_without_raising(self):
        from face_manager.rotation_ml_audit import run_rotation_ml_audit_until_done

        with patch('face_manager.rotation_ml_audit.ONNX_MODEL_PATH', '/nonexistent/checkpoint.pt'):
            result = run_rotation_ml_audit_until_done()

        self.assertTrue(result['skipped_no_checkpoint'])
        self.assertEqual(result['examined'], 0)
        for img in self.images:
            img.refresh_from_db()
            self.assertEqual(img.rotation_scan_state, ImageFile.RotationScanState.NOT_SCANNED)

    def test_loops_through_the_whole_backlog_in_one_call(self):
        from face_manager.rotation_ml_audit import run_rotation_ml_audit_until_done

        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        with patch('face_manager.rotation_ml_audit.decode_upright', return_value=arr), \
             patch('face_manager.rotation_ml_audit._load_model', return_value=_upright_confident_model()):
            result = run_rotation_ml_audit_until_done(batch_size=1)

        self.assertEqual(result['examined'], 3)
        self.assertEqual(result['updated'], 3)
        self.assertEqual(result['batches'], 4)
        self.assertFalse(result['hit_time_limit'])
        self.assertFalse(result['skipped_no_checkpoint'])

    def test_stops_at_the_time_limit_with_backlog_remaining(self):
        from face_manager.rotation_ml_audit import run_rotation_ml_audit_until_done

        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        with patch('face_manager.rotation_ml_audit.decode_upright', return_value=arr), \
             patch('face_manager.rotation_ml_audit._load_model', return_value=_upright_confident_model()), \
             patch('face_manager.rotation_ml_audit.time.monotonic', side_effect=[0, 100, 100]):
            result = run_rotation_ml_audit_until_done(batch_size=1, max_runtime_seconds=1)

        self.assertEqual(result['batches'], 1)
        self.assertTrue(result['hit_time_limit'])


class AuditImageRotationMlTaskTests(TestCase):
    """The (not-yet-scheduled) Celery task -- same thin advisory-lock +
    crash-logging shape as AuditImageRotationTaskTests.
    """

    def test_concurrent_run_is_skipped(self):
        from face_manager.tasks import audit_image_rotation_ml_task

        with patch('face_manager.tasks.advisory_lock') as mock_lock, \
             patch('face_manager.tasks.run_rotation_ml_audit_until_done') as mock_run:
            mock_lock.return_value.__enter__.return_value = False
            audit_image_rotation_ml_task()

        mock_run.assert_not_called()

    def test_runs_when_lock_acquired(self):
        from face_manager.tasks import audit_image_rotation_ml_task

        with patch('face_manager.tasks.advisory_lock') as mock_lock, \
             patch('face_manager.tasks.run_rotation_ml_audit_until_done') as mock_run:
            mock_lock.return_value.__enter__.return_value = True
            mock_run.return_value = {
                'examined': 0, 'counts': {}, 'updated': 0, 'batches': 0,
                'hit_time_limit': False, 'skipped_no_checkpoint': False,
            }
            audit_image_rotation_ml_task()

        mock_run.assert_called_once_with()

    def test_skipped_no_checkpoint_does_not_crash(self):
        from face_manager.tasks import audit_image_rotation_ml_task

        with patch('face_manager.tasks.advisory_lock') as mock_lock, \
             patch('face_manager.tasks.run_rotation_ml_audit_until_done') as mock_run:
            mock_lock.return_value.__enter__.return_value = True
            mock_run.return_value = {
                'examined': 0, 'counts': {}, 'updated': 0, 'batches': 0,
                'hit_time_limit': False, 'skipped_no_checkpoint': True,
            }
            audit_image_rotation_ml_task()  # must not raise

    def test_a_crash_is_recorded_as_an_application_error(self):
        from common.models import ApplicationError
        from face_manager.tasks import audit_image_rotation_ml_task

        with patch('face_manager.tasks.advisory_lock') as mock_lock, \
             patch('face_manager.tasks.run_rotation_ml_audit_until_done', side_effect=RuntimeError("boom")):
            mock_lock.return_value.__enter__.return_value = True
            audit_image_rotation_ml_task()

        self.assertTrue(
            ApplicationError.objects.filter(source__icontains='audit_image_rotation_ml_task').exists()
        )

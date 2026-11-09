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

from face_manager.rotation_audit import _rotation_scan_logic_hash, evaluate_image
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


class RotationScanLogicHashTests(TestCase):
    """_rotation_scan_logic_hash() -- replaces a hand-maintained
    ROTATION_SCAN_LOGIC_VERSION int with sha256(source of this module +
    rotation_geometry)[:16], mirroring face_manager.test_face_cache's own
    cache-invalidation pattern (per the user's 2026-11-06 request) so
    nobody has to remember to bump a version number when the scan logic
    changes.
    """

    def test_is_a_16_char_hex_string(self):
        value = _rotation_scan_logic_hash()
        self.assertEqual(len(value), 16)
        int(value, 16)  # raises ValueError if not valid hex

    def test_is_deterministic_across_calls(self):
        self.assertEqual(_rotation_scan_logic_hash(), _rotation_scan_logic_hash())

    def test_changes_if_either_hashed_module_source_changes(self):
        before = _rotation_scan_logic_hash()
        with patch('inspect.getsource', return_value="x = 'a change somewhere in the file'"):
            after = _rotation_scan_logic_hash()
        self.assertNotEqual(before, after)
        # Patching nothing reproduces the original value -- confirms the
        # difference above came from the patch, not leftover test state.
        self.assertEqual(_rotation_scan_logic_hash(), before)


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

    def test_single_face_moderate_tilt_is_not_significant(self):
        # Real production false-positive (2026-11-06, image id 29): a
        # baby looking down at a spoon measured a genuine -25.5 degree
        # eye-tilt -- ordinary head-tilt noise, not a rotated photo, but
        # the OLD 15-degree single-face tolerance flagged it anyway and
        # proposed a wrong "correction" that visibly rotated an already-
        # upright photo sideways. A real production sweep found 91.4% of
        # all single-face-sourced flags (14,342 of 15,688) were this same
        # shape -- not near any real 90/180/270 rotation at all. Fixed by
        # raising the single-face tolerance to 70 (per the user's own
        # call), leaving the much stronger multi-face signal at 15.
        outcome = evaluate_image(None, [_fake_face(-25.5)])
        self.assertFalse(outcome['significant'])
        self.assertEqual(outcome['source'], ImageFile.RotationSignalSource.LANDMARK_SINGLE_FACE)

    def test_single_face_tolerance_boundary(self):
        just_under = evaluate_image(None, [_fake_face(69.0)])
        self.assertFalse(just_under['significant'])
        just_over = evaluate_image(None, [_fake_face(71.0)])
        self.assertTrue(just_over['significant'])

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

    def test_two_faces_with_a_genuine_shared_tilt_is_not_significant(self):
        # Real-data-informed case (2026-11-06 empirical audit, see
        # CLAUDE.md): a random 30,000-face production sample found no
        # clean bimodal structure in the multi-face distribution, but
        # confirmed a real, if thin, population of genuinely-agreeing
        # faces out past the OLD 15-degree tolerance -- e.g. a group
        # photo taken at a natural angle, or people leaning in together,
        # not an actually-rotated photo. Two faces independently agreeing
        # on ~30 degrees of shared tilt is exactly this shape: real,
        # consistent signal, but well short of an actual rotation. The
        # old 15-degree multi-face tolerance would have wrongly flagged
        # this; the new 40-degree tolerance (per the user's own call)
        # correctly leaves it alone.
        outcome = evaluate_image(None, [_fake_face(28.0), _fake_face(32.0)])
        self.assertFalse(outcome['significant'])
        self.assertEqual(outcome['source'], ImageFile.RotationSignalSource.LANDMARK_MULTI_FACE)

    def test_multi_face_tolerance_boundary(self):
        just_under = evaluate_image(None, [_fake_face(38.0), _fake_face(40.0)])
        self.assertFalse(just_under['significant'])
        just_over = evaluate_image(None, [_fake_face(40.0), _fake_face(42.0)])
        self.assertTrue(just_over['significant'])

    def test_three_faces_outlier_dropped_still_significant(self):
        outcome = evaluate_image(None, [_fake_face(88.0), _fake_face(92.0), _fake_face(-30.0)])
        self.assertTrue(outcome['significant'])
        self.assertAlmostEqual(outcome['estimate'], 90.0, delta=2)


def _flat_kps_for_eye_angle(angle_deg, eye_separation=50.0):
    return [coord for point in _kps_for_eye_angle(angle_deg, eye_separation) for coord in point]


class RunRotationAuditTests(TestCase):
    """run_rotation_audit() itself, against the real DB -- previously
    only the pure evaluate_image() helper and the task's lock/crash
    wrapper (mocking run_rotation_audit entirely) were tested, which is
    exactly the gap that let a real production incident (2026-11-06)
    through: the unbatched version loaded the ENTIRE ~271k-image backlog
    (every Face row in full, via the default prefetch_related) into
    memory in one call, measured at 25GB+ RSS and still climbing on a
    host already low on free memory before it was killed. These tests
    cover the real fix -- a bounded max_images per call, ordered by id
    so repeated calls make forward, non-overlapping progress.
    """

    def setUp(self):
        from face_manager.tests import make_face, make_image

        self.images = []
        for i, fixture in enumerate([
            "naming/good/1.JPG", "naming/good/2.jpg", "naming/good/3.JPEG",
        ]):
            img = make_image(relative_fixture=fixture)
            # Box sized to actually contain _flat_kps_for_eye_angle's
            # points (centered at (100,100), eye_separation=50) --
            # make_face()'s own tiny default box would violate the DB's
            # face_kps_within_box_margin constraint otherwise.
            make_face(img, box_left=50, box_top=50, box_right=150, box_bottom=150,
                      kps=_flat_kps_for_eye_angle(90.0))
            # _candidate_images_queryset() requires isProcessed=True (face
            # detection has actually settled) -- make_image() leaves a
            # fresh row at the real default (False); only once isProcessed
            # flips True (as the real face_extraction task does once it's
            # genuinely finished with an image) does this project consider
            # its face/kps data complete enough to audit.
            img.isProcessed = True
            img.save(update_fields=['isProcessed'])
            self.images.append(img)

    def test_write_false_examines_but_does_not_persist(self):
        from face_manager.rotation_audit import run_rotation_audit

        result = run_rotation_audit(write=False)
        self.assertEqual(result['examined'], 3)
        self.assertEqual(result['counts']['flagged_new_or_updated'], 3)
        for img in self.images:
            img.refresh_from_db()
            self.assertEqual(img.rotation_scan_state, ImageFile.RotationScanState.NOT_SCANNED)
            self.assertIsNone(img.proposed_rotation_degrees)

    def test_write_true_persists_the_flag(self):
        from face_manager.rotation_audit import _rotation_scan_logic_hash, run_rotation_audit

        result = run_rotation_audit(write=True)
        self.assertEqual(result['updated'], 3)
        for img in self.images:
            img.refresh_from_db()
            self.assertEqual(img.rotation_scan_state, ImageFile.RotationScanState.FLAGGED)
            self.assertAlmostEqual(img.proposed_rotation_degrees, 90.0, delta=1)
            self.assertEqual(img.rotation_scan_logic_hash, _rotation_scan_logic_hash())

    def test_max_images_caps_the_batch(self):
        from face_manager.rotation_audit import run_rotation_audit

        result = run_rotation_audit(write=True, max_images=1)
        self.assertEqual(result['examined'], 1)
        self.assertEqual(result['updated'], 1)

    def test_repeated_capped_calls_make_forward_progress(self):
        # Ordered by id, so three calls capped at 1 each must cover all
        # three DISTINCT images, never reprocessing the same one twice --
        # this is what lets the scheduled hourly task grind through a
        # large backlog safely over many runs instead of one unbounded
        # pass.
        from face_manager.rotation_audit import run_rotation_audit

        seen_ids = set()
        for _ in range(3):
            result = run_rotation_audit(write=True, max_images=1)
            self.assertEqual(result['examined'], 1)
            seen_ids.update(
                img.id for img in ImageFile.objects.filter(
                    rotation_scan_state=ImageFile.RotationScanState.FLAGGED,
                )
            )
        self.assertEqual(seen_ids, {img.id for img in self.images})

        # The whole eligible backlog is now exhausted -- a 4th call finds
        # nothing left (every image's rotation_scan_logic_hash now matches
        # the current code's hash, excluding it from the next scan).
        result = run_rotation_audit(write=True, max_images=1)
        self.assertEqual(result['examined'], 0)

    def test_max_images_none_removes_the_cap(self):
        from face_manager.rotation_audit import run_rotation_audit

        result = run_rotation_audit(write=False, max_images=None)
        self.assertEqual(result['examined'], 3)

    def test_no_usable_faces_still_gets_the_hash_stamped(self):
        # Real production bug (2026-11-06): a genuinely zero-face image
        # (isProcessed=True, no Face rows at all -- NOT the missing-kps
        # case covered elsewhere) never got rotation_scan_logic_hash
        # stamped, so it was re-examined on EVERY future batch forever --
        # confirmed live: one run "examined" 1,798,000 rows against a
        # ~270K-image library, almost all stuck re-visiting this same
        # population. Must be stamped like every other examined outcome.
        from face_manager.rotation_audit import _rotation_scan_logic_hash, run_rotation_audit
        from face_manager.tests import make_image

        faceless = make_image(relative_fixture="naming/good/4.jpeg")
        faceless.isProcessed = True
        faceless.save(update_fields=['isProcessed'])

        result = run_rotation_audit(write=True)
        self.assertEqual(result['counts']['no_usable_faces'], 1)
        faceless.refresh_from_db()
        self.assertEqual(faceless.rotation_scan_logic_hash, _rotation_scan_logic_hash())

        # Re-running must NOT re-examine it a second time.
        second = run_rotation_audit(write=True)
        self.assertEqual(second['counts']['no_usable_faces'], 0)

    def test_isprocessed_false_is_excluded_from_candidates(self):
        # Real production finding (2026-11-06): isProcessed=False means
        # face detection hasn't settled on this image at all yet -- it
        # must never be examined (or have rotation_scan_logic_hash
        # stamped) while that's true, so it's naturally reconsidered once
        # isProcessed flips True, rather than needing any separate
        # "retry later" bookkeeping.
        from face_manager.rotation_audit import run_rotation_audit

        self.images[0].isProcessed = False
        self.images[0].save(update_fields=['isProcessed'])

        result = run_rotation_audit(write=True)
        self.assertEqual(result['examined'], 2)
        self.images[0].refresh_from_db()
        self.assertEqual(self.images[0].rotation_scan_state, ImageFile.RotationScanState.NOT_SCANNED)
        self.assertEqual(self.images[0].rotation_scan_logic_hash, '')

    def test_a_face_still_missing_kps_excludes_the_whole_image(self):
        # Real production finding (2026-11-06): with the kps backfill
        # still in progress, an image can be isProcessed=True with SOME
        # faces missing kps -- must be excluded entirely (not evaluated
        # from only the subset of faces that happen to have kps already)
        # until every one of its faces has kps.
        from face_manager.rotation_audit import run_rotation_audit
        from face_manager.tests import make_face

        make_face(self.images[0], box_left=5, box_top=5, box_right=20, box_bottom=20, kps=None)

        result = run_rotation_audit(write=True)
        self.assertEqual(result['examined'], 2)
        self.images[0].refresh_from_db()
        self.assertEqual(self.images[0].rotation_scan_state, ImageFile.RotationScanState.NOT_SCANNED)
        self.assertEqual(self.images[0].rotation_scan_logic_hash, '')

    def test_a_face_marked_kps_unrecoverable_does_not_exclude_the_image(self):
        # 2026-11-09 (see CLAUDE.md): a face Face.kps_unrecoverable=True
        # is confirmed to never get kps at all -- "missing kps" alone
        # used to permanently exclude this image from ever being
        # evaluated again, which stopped being correct once
        # kps_unrecoverable became a real, expected permanent state
        # rather than "the backfill just hasn't reached this yet."
        # Complements the test above (a face that's STILL just missing
        # kps, not yet confirmed unrecoverable, must still block).
        from face_manager.rotation_audit import run_rotation_audit
        from face_manager.tests import make_face

        make_face(
            self.images[0], box_left=5, box_top=5, box_right=20, box_bottom=20,
            kps=None, kps_unrecoverable=True,
        )

        result = run_rotation_audit(write=True)
        self.assertEqual(result['examined'], 3)
        self.images[0].refresh_from_db()
        self.assertEqual(self.images[0].rotation_scan_state, ImageFile.RotationScanState.FLAGGED)

    def test_a_still_pending_face_still_blocks_alongside_an_unrecoverable_one(self):
        # On the same image, a genuinely-still-missing (not yet marked)
        # face must still block even when a DIFFERENT face on the same
        # image is already confirmed kps_unrecoverable -- confirms the
        # fix is a per-face carve-out, not an accidental "any
        # kps_unrecoverable face makes the whole image pass" bug.
        from face_manager.rotation_audit import run_rotation_audit
        from face_manager.tests import make_face

        make_face(
            self.images[0], box_left=5, box_top=5, box_right=20, box_bottom=20,
            kps=None, kps_unrecoverable=True,
        )
        make_face(self.images[0], box_left=25, box_top=25, box_right=40, box_bottom=40, kps=None)

        result = run_rotation_audit(write=True)
        self.assertEqual(result['examined'], 2)
        self.images[0].refresh_from_db()
        self.assertEqual(self.images[0].rotation_scan_state, ImageFile.RotationScanState.NOT_SCANNED)


class RunRotationAuditUntilDoneTests(TestCase):
    """run_rotation_audit_until_done() -- the looping wrapper used by the
    scheduled task so a large backlog is ground through within one run
    (batches are cheap, pure-geometry math) instead of one batch per
    hourly tick. Same 3-image fixture as RunRotationAuditTests above.
    """

    def setUp(self):
        from face_manager.tests import make_face, make_image

        self.images = []
        for i, fixture in enumerate([
            "naming/good/1.JPG", "naming/good/2.jpg", "naming/good/3.JPEG",
        ]):
            img = make_image(relative_fixture=fixture)
            make_face(img, box_left=50, box_top=50, box_right=150, box_bottom=150,
                      kps=_flat_kps_for_eye_angle(90.0))
            img.isProcessed = True
            img.save(update_fields=['isProcessed'])
            self.images.append(img)

    def test_loops_through_the_whole_backlog_in_one_call(self):
        from face_manager.rotation_audit import run_rotation_audit_until_done

        # batch_size=1 forces 3 real batches (one per image) within this
        # ONE call, plus a 4th confirming the backlog is now empty -- if
        # looping didn't work, only 1 of the 3 images would ever be
        # examined.
        result = run_rotation_audit_until_done(batch_size=1)
        self.assertEqual(result['examined'], 3)
        self.assertEqual(result['updated'], 3)
        self.assertEqual(result['batches'], 4)
        self.assertFalse(result['hit_time_limit'])
        for img in self.images:
            img.refresh_from_db()
            self.assertEqual(img.rotation_scan_state, ImageFile.RotationScanState.FLAGGED)

    def test_stops_at_the_time_limit_with_backlog_remaining(self):
        from face_manager.rotation_audit import run_rotation_audit_until_done

        # A time limit already expired before the first batch even runs
        # -- the loop must still complete that one batch (never a
        # zero-batch result) but then stop, correctly reporting
        # hit_time_limit=True rather than silently claiming completion.
        with patch('face_manager.rotation_audit.time.monotonic', side_effect=[0, 100, 100]):
            result = run_rotation_audit_until_done(batch_size=1, max_runtime_seconds=1)

        self.assertEqual(result['batches'], 1)
        self.assertEqual(result['examined'], 1)
        self.assertTrue(result['hit_time_limit'])

    def test_an_empty_backlog_returns_immediately_with_one_batch(self):
        from face_manager.rotation_audit import run_rotation_audit_until_done

        ImageFile.objects.all().delete()
        result = run_rotation_audit_until_done(batch_size=1)
        self.assertEqual(result['examined'], 0)
        self.assertEqual(result['batches'], 1)
        self.assertFalse(result['hit_time_limit'])


class AuditImageRotationTaskTests(TestCase):
    """The scheduled Celery task is a thin advisory-lock + crash-logging
    wrapper around run_rotation_audit_until_done() -- same shape as this
    codebase's other scheduled tasks (e.g. apply_collection_floor_change).
    """

    def test_concurrent_run_is_skipped(self):
        from face_manager.tasks import audit_image_rotation_task

        with patch('face_manager.tasks.advisory_lock') as mock_lock, \
             patch('face_manager.tasks.run_rotation_audit_until_done') as mock_run:
            mock_lock.return_value.__enter__.return_value = False
            audit_image_rotation_task()

        mock_run.assert_not_called()

    def test_runs_and_writes_when_lock_acquired(self):
        from face_manager.tasks import audit_image_rotation_task

        with patch('face_manager.tasks.advisory_lock') as mock_lock, \
             patch('face_manager.tasks.run_rotation_audit_until_done') as mock_run:
            mock_lock.return_value.__enter__.return_value = True
            mock_run.return_value = {
                'examined': 0, 'counts': {}, 'updated': 0, 'batches': 0, 'hit_time_limit': False,
            }
            audit_image_rotation_task()

        mock_run.assert_called_once_with()

    def test_a_crash_is_recorded_as_an_application_error(self):
        from common.models import ApplicationError
        from face_manager.tasks import audit_image_rotation_task

        with patch('face_manager.tasks.advisory_lock') as mock_lock, \
             patch('face_manager.tasks.run_rotation_audit_until_done', side_effect=RuntimeError("boom")):
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

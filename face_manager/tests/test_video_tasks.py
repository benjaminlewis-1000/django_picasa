"""face_manager tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test face_manager.tests.test_video_tasks
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


class ProcessVideoFacesFailureTrackingTests(TestCase):
    """face_extraction_failed/_error (filepopulator.VideoFile) -- set by
    process_video_faces()'s except branch, surfaced on /api/server_stats/
    (see api/serializers.py). Mocks VideoFaceExtractor entirely so this
    stays a fast test -- the real pipeline's own success path is already
    covered by VideoFaceExtractorRealInferenceTests above."""

    def setUp(self):
        from filepopulator.models import Directory, VideoFile
        directory = Directory.objects.create(dir_path='/videos/failure_test')
        self.video = VideoFile.objects.create(
            filename='/videos/failure_test/a.mp4', directory=directory,
            width=100, height=100, duration_seconds=5, isProcessed=False,
        )

    def test_failure_sets_tracking_fields_and_still_marks_processed(self):
        from face_manager.tasks import process_video_faces
        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.side_effect = RuntimeError("boom")
            process_video_faces()
        self.video.refresh_from_db()
        self.assertTrue(self.video.isProcessed)
        self.assertTrue(self.video.face_extraction_failed)
        self.assertIn("RuntimeError", self.video.face_extraction_error)
        self.assertIn("boom", self.video.face_extraction_error)

    def test_success_leaves_tracking_fields_unset(self):
        from face_manager.tasks import process_video_faces
        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.return_value = []
            process_video_faces()
        self.video.refresh_from_db()
        self.assertTrue(self.video.isProcessed)
        self.assertFalse(self.video.face_extraction_failed)
        self.assertIsNone(self.video.face_extraction_error)

    def test_timeout_leaves_video_unprocessed_for_a_later_retry(self):
        # Real 2026-09-17 finding: several completely normal, short videos
        # timed out purely because a concurrent CPU-heavy job was
        # competing for cores -- not a permanent problem with the file.
        # A timeout must NOT be treated like every other failure (marking
        # isProcessed=True/face_extraction_failed=True would require a
        # manual reset before it's ever retried); it should just stay in
        # the unprocessed queue for a future scheduled run to pick up
        # again once resources free up.
        from face_manager.tasks import process_video_faces, _VideoProcessingTimeout
        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.side_effect = _VideoProcessingTimeout("Video processing exceeded 1800s")
            process_video_faces()
        self.video.refresh_from_db()
        self.assertFalse(self.video.isProcessed)
        self.assertFalse(self.video.face_extraction_failed)
        self.assertIsNone(self.video.face_extraction_error)
        self.assertEqual(self.video.face_extraction_timeout_count, 1)

    def test_repeated_timeouts_eventually_marked_a_real_failure(self):
        # A genuinely-too-long video (not just resource contention) would
        # otherwise retry forever, each attempt burning a full
        # PER_VIDEO_TIMEOUT_SECONDS -- MAX_VIDEO_TIMEOUT_RETRIES caps
        # that, per the user's explicit request.
        from face_manager.tasks import process_video_faces, _VideoProcessingTimeout, MAX_VIDEO_TIMEOUT_RETRIES
        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.side_effect = _VideoProcessingTimeout("boom")
            for _ in range(MAX_VIDEO_TIMEOUT_RETRIES - 1):
                process_video_faces()
                self.video.refresh_from_db()
                self.assertFalse(self.video.isProcessed)
                self.assertFalse(self.video.face_extraction_failed)
                # isProcessed=False is required for the next call's
                # unprocessed queryset to pick this video up again.
                self.video.isProcessed = False
                self.video.save()

            process_video_faces()
        self.video.refresh_from_db()
        self.assertEqual(self.video.face_extraction_timeout_count, MAX_VIDEO_TIMEOUT_RETRIES)
        self.assertTrue(self.video.isProcessed)
        self.assertTrue(self.video.face_extraction_failed)
        self.assertIn(str(MAX_VIDEO_TIMEOUT_RETRIES), self.video.face_extraction_error)

    def test_eventual_success_resets_timeout_count(self):
        from face_manager.tasks import process_video_faces, _VideoProcessingTimeout
        self.video.face_extraction_timeout_count = 3
        self.video.save()
        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.return_value = []
            process_video_faces()
        self.video.refresh_from_db()
        self.assertEqual(self.video.face_extraction_timeout_count, 0)

    def test_real_failure_is_recorded_as_an_application_error(self):
        from common.models import ApplicationError
        from face_manager.tasks import process_video_faces
        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.side_effect = RuntimeError("boom")
            process_video_faces()
        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.tasks.process_video_faces')
        self.assertIn(self.video.filename, error.message)
        self.assertIn('boom', error.message)

    def test_exhausted_timeout_retries_is_recorded_as_an_application_error(self):
        from common.models import ApplicationError
        from face_manager.tasks import process_video_faces, _VideoProcessingTimeout, MAX_VIDEO_TIMEOUT_RETRIES
        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.side_effect = _VideoProcessingTimeout("boom")
            for _ in range(MAX_VIDEO_TIMEOUT_RETRIES):
                process_video_faces()
                self.video.refresh_from_db()
                if not self.video.face_extraction_failed:
                    self.video.isProcessed = False
                    self.video.save()
        self.assertEqual(ApplicationError.objects.count(), 1)
        error = ApplicationError.objects.get()
        self.assertEqual(error.source, 'face_manager.tasks.process_video_faces')
        self.assertIn(self.video.filename, error.message)


class VideoTimeoutBudgetTests(TestCase):
    """_video_timeout_budget_seconds()/_contention_multiplier() -- the
    duration- and contention-scaled replacement for the old flat
    PER_VIDEO_TIMEOUT_SECONDS constant (2026-11-18, per the user's own
    request)."""

    def test_budget_scales_with_duration(self):
        from face_manager.tasks import (
            _video_timeout_budget_seconds,
            PER_VIDEO_TIMEOUT_BASE_SECONDS,
            PER_VIDEO_TIMEOUT_RATE_PER_VIDEO_SECOND,
        )
        with patch('face_manager.tasks._contention_multiplier', return_value=1.0):
            short = _video_timeout_budget_seconds(10)
            long = _video_timeout_budget_seconds(3600)
        self.assertLess(short, long)
        self.assertAlmostEqual(
            long,
            PER_VIDEO_TIMEOUT_BASE_SECONDS + 3600 * PER_VIDEO_TIMEOUT_RATE_PER_VIDEO_SECOND,
        )

    def test_budget_floors_at_minimum_for_tiny_videos(self):
        from face_manager.tasks import _video_timeout_budget_seconds, PER_VIDEO_TIMEOUT_MIN_SECONDS
        with patch('face_manager.tasks._contention_multiplier', return_value=1.0):
            self.assertEqual(_video_timeout_budget_seconds(1), PER_VIDEO_TIMEOUT_MIN_SECONDS)
            self.assertEqual(_video_timeout_budget_seconds(0), PER_VIDEO_TIMEOUT_MIN_SECONDS)
            self.assertEqual(_video_timeout_budget_seconds(None), PER_VIDEO_TIMEOUT_MIN_SECONDS)

    def test_budget_caps_at_the_maximum_for_very_long_videos(self):
        from face_manager.tasks import _video_timeout_budget_seconds, PER_VIDEO_TIMEOUT_MAX_SECONDS
        with patch('face_manager.tasks._contention_multiplier', return_value=1.0):
            self.assertEqual(_video_timeout_budget_seconds(100000), PER_VIDEO_TIMEOUT_MAX_SECONDS)

    def test_five_minute_video_under_max_contention_gets_well_above_the_old_flat_timeout(self):
        # The real incident this formula exists to fix: a 5-minute video
        # (duration=300) timed out under contention against the old flat
        # 2700s timeout. Under the current formula --
        # clamp((300 + 300*3) * contention, MIN, MAX) -- max contention
        # gives it a budget comfortably above the 2700s it used to get
        # regardless of contention, and still well under the 4-hour cap.
        # (Max contention alone no longer needs to carry the whole
        # burden here -- _make_progress_callback()'s per-stage reset
        # separately protects a merely-slow-but-advancing video; this
        # test just confirms the multiplier-only math stays sane.)
        from face_manager.tasks import (
            _video_timeout_budget_seconds,
            PER_VIDEO_TIMEOUT_MIN_SECONDS,
            PER_VIDEO_TIMEOUT_MAX_SECONDS,
            PER_VIDEO_TIMEOUT_CONTENTION_MAX_MULTIPLIER,
        )
        with patch('face_manager.tasks._contention_multiplier', return_value=PER_VIDEO_TIMEOUT_CONTENTION_MAX_MULTIPLIER):
            budget = _video_timeout_budget_seconds(300)
        self.assertEqual(budget, 1200 * PER_VIDEO_TIMEOUT_CONTENTION_MAX_MULTIPLIER)
        self.assertGreater(budget, PER_VIDEO_TIMEOUT_MIN_SECONDS)
        self.assertLessEqual(budget, PER_VIDEO_TIMEOUT_MAX_SECONDS)

    def test_contention_multiplier_scales_the_whole_budget(self):
        from face_manager.tasks import _video_timeout_budget_seconds
        # duration=1000 is chosen so the raw (BASE + duration*RATE)
        # value already clears MIN_SECONDS at contention=1.0, and stays
        # under MAX_SECONDS at contention=2.0 -- otherwise either clamp
        # would distort the expected exact doubling.
        with patch('face_manager.tasks._contention_multiplier', return_value=1.0):
            base = _video_timeout_budget_seconds(1000)
        with patch('face_manager.tasks._contention_multiplier', return_value=2.0):
            contended = _video_timeout_budget_seconds(1000)
        self.assertAlmostEqual(contended, base * 2.0)

    def test_contention_multiplier_reads_load_average_relative_to_cpu_count(self):
        from face_manager.tasks import _contention_multiplier, PER_VIDEO_TIMEOUT_CONTENTION_MAX_MULTIPLIER
        with patch('os.getloadavg', return_value=(4.0, 4.0, 4.0)), patch('os.cpu_count', return_value=4):
            self.assertAlmostEqual(_contention_multiplier(), 1.0)
        with patch('os.getloadavg', return_value=(8.0, 8.0, 8.0)), patch('os.cpu_count', return_value=4):
            self.assertAlmostEqual(_contention_multiplier(), 2.0)
        with patch('os.getloadavg', return_value=(40.0, 40.0, 40.0)), patch('os.cpu_count', return_value=4):
            # Clamped -- a single extreme spike shouldn't make the
            # effective timeout unbounded.
            self.assertAlmostEqual(_contention_multiplier(), PER_VIDEO_TIMEOUT_CONTENTION_MAX_MULTIPLIER)
        with patch('os.getloadavg', return_value=(1.0, 1.0, 1.0)), patch('os.cpu_count', return_value=4):
            # Below core count never gives a DISCOUNT, only ever >= 1.0.
            self.assertAlmostEqual(_contention_multiplier(), 1.0)

    def test_contention_multiplier_falls_back_to_1_on_oserror(self):
        from face_manager.tasks import _contention_multiplier
        with patch('os.getloadavg', side_effect=OSError):
            self.assertEqual(_contention_multiplier(), 1.0)


class CheckVideoTimeoutHandlerTests(TestCase):
    """_check_video_timeout() -- the repeating signal.setitimer() handler
    that replaced a one-shot signal.alarm(). Re-derives the allowed
    budget fresh on every tick using CURRENT contention, per the user's
    own follow-up request: contention can rise mid-run, not just at the
    video's start, and a single value computed once at launch can't
    adapt to that."""

    def tearDown(self):
        from face_manager.tasks import _video_timeout_state
        _video_timeout_state.clear()

    def test_noop_when_no_video_currently_tracked(self):
        from face_manager.tasks import _check_video_timeout, _video_timeout_state
        _video_timeout_state.clear()
        _check_video_timeout(0, None)  # must not raise

    def test_raises_once_elapsed_exceeds_the_current_budget(self):
        from face_manager.tasks import _check_video_timeout, _video_timeout_state, _VideoProcessingTimeout
        _video_timeout_state.clear()
        _video_timeout_state['start'] = 1000.0
        _video_timeout_state['duration_seconds'] = 60
        with patch('face_manager.tasks.time.time', return_value=1000.0 + 50000), \
             patch('face_manager.tasks._video_timeout_budget_seconds', return_value=100.0):
            with self.assertRaises(_VideoProcessingTimeout):
                _check_video_timeout(0, None)

    def test_does_not_raise_while_still_within_budget(self):
        from face_manager.tasks import _check_video_timeout, _video_timeout_state
        _video_timeout_state.clear()
        _video_timeout_state['start'] = 1000.0
        _video_timeout_state['duration_seconds'] = 60
        with patch('face_manager.tasks.time.time', return_value=1010.0), \
             patch('face_manager.tasks._video_timeout_budget_seconds', return_value=100.0):
            _check_video_timeout(0, None)  # must not raise

    def test_budget_is_recomputed_fresh_on_every_tick(self):
        # The real motivating scenario: contention can rise mid-run, not
        # just at video start -- the handler must re-derive the budget
        # (and thus re-read current contention) on EVERY tick rather than
        # reusing one value computed once. Proven here by confirming the
        # budget function is actually re-invoked on each separate call.
        from face_manager.tasks import _check_video_timeout, _video_timeout_state
        _video_timeout_state.clear()
        _video_timeout_state['start'] = 1000.0
        _video_timeout_state['duration_seconds'] = 60
        calls = []

        def fake_budget(duration_seconds):
            calls.append(duration_seconds)
            return 100.0

        with patch('face_manager.tasks.time.time', return_value=1000.0 + 90), \
             patch('face_manager.tasks._video_timeout_budget_seconds', side_effect=fake_budget):
            _check_video_timeout(0, None)  # 90s elapsed, 100s budget: no raise
            _check_video_timeout(0, None)  # re-derived again on this tick
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls, [60, 60])


class MakeProgressCallbackTests(TestCase):
    """_make_progress_callback() -- per the user's own follow-up request
    (2026-11-18): completing a stage should reset the timeout clock, so
    a video that keeps advancing isn't penalized for the SUM of several
    genuinely-completed (if individually slow) stages exceeding one
    whole-video budget. The callback resets _video_timeout_state['start']
    on every call, which _check_video_timeout reads as "time since the
    current stage began" rather than "time since the video began"."""

    def tearDown(self):
        from face_manager.tasks import _video_timeout_state
        _video_timeout_state.clear()

    def test_calling_the_callback_resets_the_start_timestamp(self):
        from face_manager.tasks import _make_progress_callback, _video_timeout_state
        _video_timeout_state.clear()
        _video_timeout_state['start'] = 1000.0
        _video_timeout_state['duration_seconds'] = 60
        progress = _make_progress_callback()
        with patch('face_manager.tasks.time.time', return_value=5000.0):
            progress('detect_and_track')
        self.assertEqual(_video_timeout_state['start'], 5000.0)
        self.assertEqual(_video_timeout_state['stage'], 'detect_and_track')

    def test_logs_the_previous_stages_duration_and_contention_on_advance(self):
        # Per the user's own request (2026-11-18): durable, logged
        # measurements of each stage's real wall time plus the host's
        # contention level AT THAT MOMENT -- not just a bare pass/fail --
        # so a slow-but-real video can be told apart from a stuck one
        # after the fact, from the log alone.
        from face_manager.tasks import _make_progress_callback, _video_timeout_state
        _video_timeout_state.clear()
        progress = _make_progress_callback()
        with patch('face_manager.tasks.time.time', return_value=1000.0), \
             patch('face_manager.tasks.settings.LOGGER') as mock_logger:
            progress('detect_and_track')  # first call: no PRIOR stage, nothing to log yet
            mock_logger.info.assert_not_called()
        with patch('face_manager.tasks.time.time', return_value=1090.0), \
             patch('face_manager.tasks.os.getloadavg', return_value=(8.0, 8.0, 8.0)), \
             patch('face_manager.tasks.os.cpu_count', return_value=4), \
             patch('face_manager.tasks.settings.LOGGER') as mock_logger:
            progress('pool_representative_frames')  # advancing: logs detect_and_track's own 90s
            mock_logger.info.assert_called_once()
            args = mock_logger.info.call_args[0]
            self.assertIn('detect_and_track', args)
            self.assertIn(90.0, args)

    def test_a_long_running_but_still_advancing_video_never_times_out(self):
        # The real motivating scenario: each individual stage's own mid-
        # stage check stays within budget, but the stages' SUM would have
        # exceeded a pure whole-video elapsed-time budget. Checks happen
        # mid-stage (not immediately after a reset, which would be a
        # trivial 0-elapsed pass) to actually exercise the budget
        # comparison, not just confirm the reset itself.
        from face_manager.tasks import _make_progress_callback, _check_video_timeout, _video_timeout_state
        _video_timeout_state.clear()
        _video_timeout_state['start'] = 0.0
        _video_timeout_state['duration_seconds'] = 60
        progress = _make_progress_callback()
        with patch('face_manager.tasks._video_timeout_budget_seconds', return_value=100.0):
            # Four stages, each: starts, a mid-stage check at +80s (under
            # the 100s budget measured from THIS stage's own start), then
            # advances to the next stage (resetting the clock). Cumulative
            # real time is 4*80=320s -- well past 100s if ever measured
            # from video start instead of from the last reset.
            t = 0.0
            with patch('face_manager.tasks.time.time', return_value=t):
                progress('detect_and_track')
            for stage in ['pool_representative_frames', 'clustering', 'classify_and_save', None]:
                t += 80.0
                with patch('face_manager.tasks.time.time', return_value=t):
                    _check_video_timeout(0, None)  # mid-stage: must not raise
                if stage is not None:
                    with patch('face_manager.tasks.time.time', return_value=t):
                        progress(stage)  # stage finished -- advance and reset


class ProcessVideoFacesTimeoutStateWiringTests(TestCase):
    """Confirms process_video_faces() actually wires the real video's
    duration into _video_timeout_state before calling process_video(),
    and clears it afterward -- the integration point between the loop
    and the timeout-checking machinery tested in isolation above."""

    def setUp(self):
        from filepopulator.models import Directory, VideoFile
        directory = Directory.objects.create(dir_path='/videos/timeout_wiring_test')
        self.video = VideoFile.objects.create(
            filename='/videos/timeout_wiring_test/a.mp4', directory=directory,
            width=100, height=100, duration_seconds=123.0, isProcessed=False,
        )

    def test_timeout_state_reflects_the_real_video_duration_while_processing(self):
        from face_manager.tasks import process_video_faces, _video_timeout_state
        captured = {}

        def fake_process_video(video, progress_cb=None):
            captured['state'] = dict(_video_timeout_state)
            return []

        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.side_effect = fake_process_video
            process_video_faces()

        self.assertEqual(captured['state']['duration_seconds'], 123.0)
        self.assertIn('start', captured['state'])
        # Cleared once the video finishes processing.
        self.assertEqual(_video_timeout_state, {})


class ProcessVideoFacesVideoIdsScopingTests(TestCase):
    """process_video_faces(video_ids=...) -- the on-demand, frontend-
    triggered reprocess path (api.views.VideoFaceExtractionViewSet).
    Unlike the normal isProcessed=False backlog, this always forces a
    redo of exactly the named videos, regardless of their current
    isProcessed/face_extraction_failed state, and clears out any
    existing Face rows (with their thumbnails) first so a reprocess
    never leaves stale+fresh Face rows coexisting."""

    def setUp(self):
        from filepopulator.models import Directory, VideoFile
        directory = Directory.objects.create(dir_path='/videos/video_ids_test')
        self.already_processed = VideoFile.objects.create(
            filename='/videos/video_ids_test/a.mp4', directory=directory,
            width=100, height=100, duration_seconds=5, isProcessed=True,
            face_extraction_failed=True, face_extraction_error='old failure',
            face_extraction_timeout_count=2,
        )
        self.untouched = VideoFile.objects.create(
            filename='/videos/video_ids_test/b.mp4', directory=directory,
            width=100, height=100, duration_seconds=5, isProcessed=True,
        )

    def test_targeted_video_is_reprocessed_despite_already_processed(self):
        from face_manager.tasks import process_video_faces
        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.return_value = []
            process_video_faces(video_ids=[self.already_processed.id])
        self.already_processed.refresh_from_db()
        MockExtractor.return_value.process_video.assert_called_once()
        self.assertTrue(self.already_processed.isProcessed)
        self.assertFalse(self.already_processed.face_extraction_failed)
        self.assertIsNone(self.already_processed.face_extraction_error)
        self.assertEqual(self.already_processed.face_extraction_timeout_count, 0)

    def test_other_videos_are_not_touched(self):
        from face_manager.tasks import process_video_faces
        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.return_value = []
            process_video_faces(video_ids=[self.already_processed.id])
        self.assertEqual(MockExtractor.return_value.process_video.call_count, 1)
        call_arg = MockExtractor.return_value.process_video.call_args[0][0]
        self.assertEqual(call_arg.id, self.already_processed.id)

    def test_existing_face_rows_and_thumbnails_are_deleted_before_reprocessing(self):
        from face_manager.models import Face
        from face_manager.tasks import process_video_faces
        from django.core.files.base import ContentFile

        face = Face(
            source_video_file=self.already_processed,
            box_top=0, box_bottom=10, box_left=0, box_right=10,
        )
        face.face_thumbnail.save('thumb.jpg', ContentFile(b'fake-thumb-bytes'), save=False)
        face.save()
        thumb_path = face.face_thumbnail.path
        self.assertTrue(os.path.exists(thumb_path))

        with patch('video_face_pipeline.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.process_video.return_value = []
            process_video_faces(video_ids=[self.already_processed.id])

        self.assertFalse(Face.objects.filter(pk=face.pk).exists())
        self.assertFalse(os.path.exists(thumb_path))

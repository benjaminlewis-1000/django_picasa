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

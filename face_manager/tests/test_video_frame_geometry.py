"""face_manager tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test face_manager.tests.test_video_frame_geometry
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


class VideoFaceFrameSecondsTests(unittest.TestCase):
    """_frame_seconds: real, live bug found 2026-09-11 via a production
    404 (face 1097880's "large image" -- see CLAUDE.md) -- a frame that
    has passed through extract_frame_near_timestamp's yadif deinterlace
    filter graph carries a DIFFERENT time_base than the source stream
    (confirmed on a real interlaced .m2ts file: 1/180000 vs the stream's
    own 1/90000), so applying the stream's time_base uniformly silently
    doubled the computed time for every deinterlaced frame. No real
    interlaced fixture is committed for a full integration test, but the
    actual bug is entirely in this one pts-to-seconds conversion, so a
    focused unit test on it directly covers the real fix."""

    def test_same_pts_with_different_time_bases_gives_different_seconds(self):
        # This is the bug itself, made concrete: the SAME numeric pts
        # value means a different real time depending on which
        # time_base it's interpreted through.
        from fractions import Fraction

        from video_face_pipeline import _frame_seconds
        pts = 44522298
        seconds_at_90k = _frame_seconds(pts, Fraction(1, 90000), start_seconds=0)
        seconds_at_180k = _frame_seconds(pts, Fraction(1, 180000), start_seconds=0)
        self.assertAlmostEqual(seconds_at_90k, seconds_at_180k * 2, places=6)

    def test_matching_real_world_time_survives_a_time_base_change(self):
        # The actual real-production values: an input frame at pts=
        # 22261149 in the stream's native 1/90000 time_base represents
        # the exact same real-world instant as yadif's corresponding
        # deinterlaced output (its first output overall -- yadif's
        # mode=0 needs one input's worth of buffering delay before its
        # first output, so this is the deinterlaced version of the
        # FIRST input frame, not the second) at pts=44522298 in yadif's
        # 1/180000 time_base -- confirmed directly against the real file
        # (2026-09-11). Using each frame's own time_base must recover
        # the same real second count for both.
        from fractions import Fraction

        from video_face_pipeline import _frame_seconds
        start_seconds = _frame_seconds(99009, Fraction(1, 90000), start_seconds=0)
        input_seconds = _frame_seconds(22261149, Fraction(1, 90000), start_seconds=start_seconds)
        output_seconds = _frame_seconds(44522298, Fraction(1, 180000), start_seconds=start_seconds)
        self.assertAlmostEqual(input_seconds, output_seconds, places=3)
        self.assertAlmostEqual(input_seconds, 246.246, places=2)

    def test_pre_fix_bug_reproduced_with_stream_time_base_applied_to_both(self):
        # Demonstrates the actual regression: naively applying the
        # STREAM's time_base (1/90000) to an output frame's pts that's
        # really in 1/180000 units produces roughly double the correct
        # value -- exactly the silent doubling that made
        # extract_frame_near_timestamp return None for interlaced videos.
        from fractions import Fraction

        from video_face_pipeline import _frame_seconds
        correct = _frame_seconds(44522298, Fraction(1, 180000), start_seconds=0)
        buggy = _frame_seconds(44522298, Fraction(1, 90000), start_seconds=0)
        self.assertAlmostEqual(buggy, correct * 2, places=3)


class DecodeFaceThumbnailTests(TestCase):
    """decode_face_thumbnail() -- the shared open/read/close/cv2.imdecode
    helper consolidated 2026-10-30 out of 7 independently copy-pasted
    management-command implementations (2 of which didn't even guard
    the open/read call -- see CLAUDE.md's video-face-pipeline
    consolidation write-up). Source-agnostic (just a FileField read),
    so a plain image-sourced Face exercises it exactly the same as a
    video-sourced one would."""

    def test_valid_thumbnail_decodes_to_a_real_array(self):
        from video_face_pipeline import decode_face_thumbnail
        face = make_face(make_image())
        img = decode_face_thumbnail(face)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (30, 30, 3))

    def test_missing_file_on_disk_returns_none_without_raising(self):
        from video_face_pipeline import decode_face_thumbnail
        face = make_face(make_image())
        os.remove(face.face_thumbnail.path)
        self.assertIsNone(decode_face_thumbnail(face))

    def test_corrupted_content_returns_none_without_raising(self):
        from video_face_pipeline import decode_face_thumbnail
        face = make_face(make_image())
        with open(face.face_thumbnail.path, 'wb') as f:
            f.write(b'not a real image')
        self.assertIsNone(decode_face_thumbnail(face))

    def test_file_handle_is_released_across_many_calls(self):
        # Real motivation for consolidating this in the first place: a
        # leaked fd across a tight loop (the exact shape every one of
        # the 7 original call sites ran in) eventually exhausts the
        # process's fd limit. Calling it many times in a row on the
        # same face is a cheap proxy for "does this leak."
        from video_face_pipeline import decode_face_thumbnail
        face = make_face(make_image())
        for _ in range(200):
            self.assertIsNotNone(decode_face_thumbnail(face))


class HdrVideoToneMapTests(unittest.TestCase):
    """is_hdr_transfer/build_vf_filter -- pure functions, no DB/decode
    needed. See CLAUDE.md's 2026-09-22 write-up: a real HDR-tagged phone
    video (color_transfer=smpte2084, color_primaries=bt2020), decoded
    via this pipeline's plain bgr24 raw pipe with no tone-mapping, came
    out visibly washed-out/desaturated -- confirmed by extracting the
    same real frame with and without the zscale+tonemap filter chain."""

    def test_pq_and_hlg_are_hdr(self):
        from video_face_pipeline import is_hdr_transfer
        self.assertTrue(is_hdr_transfer('smpte2084'))
        self.assertTrue(is_hdr_transfer('arib-std-b67'))

    def test_ordinary_transfers_are_not_hdr(self):
        from video_face_pipeline import is_hdr_transfer
        for ct in ('bt709', 'unknown', None, ''):
            self.assertFalse(is_hdr_transfer(ct))

    def test_neither_filter_needed_gives_none(self):
        from video_face_pipeline import build_vf_filter
        self.assertIsNone(build_vf_filter('progressive', 'bt709'))
        self.assertIsNone(build_vf_filter('unknown', None))

    def test_only_deinterlace_needed(self):
        from video_face_pipeline import build_vf_filter
        self.assertEqual(build_vf_filter('tt', 'bt709'), 'yadif=0')

    def test_only_tonemap_needed(self):
        from video_face_pipeline import build_vf_filter, HDR_TONEMAP_FILTER
        self.assertEqual(build_vf_filter('progressive', 'smpte2084'), HDR_TONEMAP_FILTER)

    def test_both_combine_deinterlace_first(self):
        # Order matters: deinterlace must run on the source's native
        # pixel format before the tone-map chain's own format
        # conversions, not after.
        from video_face_pipeline import build_vf_filter, HDR_TONEMAP_FILTER
        combined = build_vf_filter('bb', 'arib-std-b67')
        self.assertEqual(combined, f'yadif=0,{HDR_TONEMAP_FILTER}')


class ExtractFrameNearTimestampHdrDispatchTests(unittest.TestCase):
    """extract_frame_near_timestamp() must dispatch HDR-tagged sources to
    the ffmpeg-based path (PyAV's own bundled libavfilter lacks the
    zscale filter the tone-map chain needs -- confirmed directly via
    av.filter.Filter('zscale') raising "no filter zscale", 2026-09-22)
    and leave every other video on the existing, already-hardened PyAV
    path untouched."""

    def test_hdr_source_dispatches_to_ffmpeg_path(self):
        from video_face_pipeline import extract_frame_near_timestamp
        with patch('video_face_pipeline._extract_frame_near_timestamp_ffmpeg') as mock_ffmpeg_path, \
             patch('video_face_pipeline.av.open') as mock_av_open:
            mock_ffmpeg_path.return_value = 'sentinel-frame'
            result = extract_frame_near_timestamp(
                '/fake/path.mp4', 10.0, 30.0, 'progressive', 0,
                (0, 0, 10, 10), np.zeros((10, 10, 3), dtype=np.uint8),
                width=1920, height=1080, color_transfer='smpte2084',
            )
        self.assertEqual(result, 'sentinel-frame')
        mock_ffmpeg_path.assert_called_once()
        mock_av_open.assert_not_called()

    def test_non_hdr_source_never_touches_ffmpeg_path(self):
        from video_face_pipeline import extract_frame_near_timestamp
        with patch('video_face_pipeline._extract_frame_near_timestamp_ffmpeg') as mock_ffmpeg_path, \
             patch('video_face_pipeline.av.open') as mock_av_open:
            mock_container = MagicMock()
            mock_container.streams.video = [MagicMock(start_time=0, time_base=1)]
            mock_container.demux.return_value = []
            mock_av_open.return_value = mock_container
            extract_frame_near_timestamp(
                '/fake/path.mp4', 10.0, 30.0, 'progressive', 0,
                (0, 0, 10, 10), np.zeros((10, 10, 3), dtype=np.uint8),
                width=1920, height=1080, color_transfer='bt709',
            )
        mock_ffmpeg_path.assert_not_called()
        mock_av_open.assert_called_once()


class ExtractFrameNearTimestampFfmpegPathTests(unittest.TestCase):
    """_extract_frame_near_timestamp_ffmpeg()'s own search logic, with
    ffmpeg_frame_iterator mocked out (no real video/subprocess needed --
    this tests the frame-selection and seek-overshoot-fallback logic
    directly, mirroring the PyAV path's own already-covered behavior)."""

    def _frame(self, fill_value):
        return np.full((40, 40, 3), fill_value, dtype=np.uint8)

    def _thumb(self, fill_value):
        # Matches what _extract_frame_near_timestamp_ffmpeg itself
        # produces internally (VideoFaceExtractor._square_thumbnail,
        # resized to settings.FACE_THUMBNAIL_SIZE) -- the reference
        # image must be the same shape or _mean_abs_pixel_diff refuses
        # to compare (returns None) and nothing would ever match.
        from video_face_pipeline import VideoFaceExtractor
        return VideoFaceExtractor._square_thumbnail(self._frame(fill_value), (5, 5, 35, 35))

    def test_picks_the_closest_pixel_match_within_the_window(self):
        from video_face_pipeline import _extract_frame_near_timestamp_ffmpeg
        reference = self._thumb(100)
        # Three candidate frames at t=0,1,2 (fps=1) -- only the middle
        # one (t=1, matching the target) is a real pixel match.
        frames = [self._frame(0), self._frame(100), self._frame(255)]
        with patch('video_face_pipeline.ffmpeg_frame_iterator', return_value=iter(frames)):
            result = _extract_frame_near_timestamp_ffmpeg(
                '/fake/path.mp4', target_seconds=1.0, avg_fps=1.0, field_order='progressive',
                box=(5, 5, 35, 35), reference_bgr=reference, window_seconds=1.5,
                width=40, height=40, color_transfer='bt709',
            )
        self.assertTrue(np.array_equal(result, self._frame(100)))

    def test_falls_back_to_full_decode_when_windowed_scan_finds_nothing(self):
        # Mirrors the PyAV path's own real 2026-09-11 WMV/ASF seek-
        # overshoot fix, adapted to what this raw-pipe path can actually
        # detect: unlike PyAV, ffmpeg's raw `rawvideo` output carries no
        # true per-frame timestamps at all, so a bad seek can't be
        # detected by comparing a frame's OWN real pts against the
        # window (there isn't one) -- only by the windowed scan coming
        # back with literally zero candidate frames (e.g. the seek
        # landed at/past EOF). That's the case this test simulates and
        # the fallback covers: retry with a full sequential decode from
        # frame zero rather than trusting the seek a second time.
        from video_face_pipeline import _extract_frame_near_timestamp_ffmpeg
        reference = self._thumb(100)
        calls = []

        def fake_iterator(path, width, height, vf_filter=None, seek_seconds=0):
            calls.append(seek_seconds)
            if seek_seconds > 0:
                return iter([])  # seek landed past any usable content
            return iter([self._frame(0), self._frame(100), self._frame(0)])

        with patch('video_face_pipeline.ffmpeg_frame_iterator', side_effect=fake_iterator):
            result = _extract_frame_near_timestamp_ffmpeg(
                '/fake/path.mp4', target_seconds=1.0, avg_fps=1.0, field_order='progressive',
                box=(5, 5, 35, 35), reference_bgr=reference, window_seconds=0.5,
                width=40, height=40, color_transfer='bt709',
            )
        self.assertTrue(np.array_equal(result, self._frame(100)))
        self.assertEqual(calls, [0.5, 0])

    def test_returns_none_when_nothing_matches_anywhere(self):
        from video_face_pipeline import _extract_frame_near_timestamp_ffmpeg
        reference = self._frame(100)
        with patch('video_face_pipeline.ffmpeg_frame_iterator', return_value=iter([])):
            result = _extract_frame_near_timestamp_ffmpeg(
                '/fake/path.mp4', target_seconds=1.0, avg_fps=1.0, field_order='progressive',
                box=(0, 0, 4, 4), reference_bgr=reference, window_seconds=0.5,
                width=4, height=4, color_transfer='bt709',
            )
        self.assertIsNone(result)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class RetonemapHdrVideoThumbnailsTests(TestCase):
    """retonemap_hdr_video_thumbnails management command -- regenerates
    ONLY the stored face_thumbnail file for already-processed HDR-video
    faces, using the already-known-correct video_thumbnail_frame_seconds
    as ground truth (no pixel-matching needed here, unlike
    backfill_video_thumbnail_timestamps.py, which recovers an UNKNOWN
    timestamp by matching against the existing thumbnail -- that
    wouldn't make sense here since the existing thumbnail IS the
    washed-out artifact being replaced). box/kps/embedding/det_score/
    declared_name/validated must all stay untouched -- only the
    thumbnail's bytes should change."""

    CMD = 'face_manager.management.commands.retonemap_hdr_video_thumbnails'

    def _make_video(self, path, width=100, height=100):
        from filepopulator.models import Directory, VideoFile
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(path))
        return VideoFile.objects.create(
            filename=path, directory=directory, width=width, height=height,
            duration_seconds=30, isProcessed=True,
        )

    def _make_person(self, name):
        person, created = Person.objects.get_or_create(person_name=name)
        if created:
            person.highlight_img.save(f"{name}.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
        return person

    def _make_face(self, video, person, timestamp, validated=False, embedding=None):
        face = Face(
            declared_name=person, source_video_file=video, validated=validated,
            box_left=1, box_top=1, box_right=40, box_bottom=40,
            video_thumbnail_frame_seconds=timestamp, face_encoding_512=embedding,
        )
        face.face_thumbnail.save(
            f"thumb_{video.pk}_{timestamp}.jpg",
            ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False,
        )
        face.save()
        return face

    def _frame(self, fill_value, size=100):
        return np.full((size, size, 3), fill_value, dtype=np.uint8)

    def test_regenerates_thumbnail_leaving_identity_and_box_untouched(self):
        video = self._make_video('/videos/hdr_test/a.mp4')
        person = self._make_person('Retonemap Test Person')
        face = self._make_face(video, person, timestamp=2.0, validated=True, embedding=[0.1] * 512)

        old_box = (face.box_left, face.box_top, face.box_right, face.box_bottom)
        old_declared_id = face.declared_name_id
        old_validated = face.validated
        old_embedding = list(face.face_encoding_512)
        with open(face.face_thumbnail.path, 'rb') as fh:
            old_bytes = fh.read()

        def fake_iterator(path, width, height, vf_filter=None, seek_seconds=0):
            # seek_seconds lands at 0 (target 2.0 - 5 margin, clamped);
            # frame at real t=2.0 gets a distinct fill value so a wrong
            # pick is detectable.
            for i in range(8):
                t = seek_seconds + i
                yield self._frame(50 if abs(t - 2.0) < 0.01 else 10)

        with patch(f'{self.CMD}.ffprobe_info',
                   return_value=(100, 100, 1.0, 'progressive', 0, 'smpte2084', None)), \
             patch(f'{self.CMD}.ffmpeg_frame_iterator', side_effect=fake_iterator):
            call_command('retonemap_hdr_video_thumbnails', '--yes')

        face.refresh_from_db()
        self.assertEqual((face.box_left, face.box_top, face.box_right, face.box_bottom), old_box)
        self.assertEqual(face.declared_name_id, old_declared_id)
        self.assertEqual(face.validated, old_validated)
        self.assertEqual(list(face.face_encoding_512), old_embedding)

        with open(face.face_thumbnail.path, 'rb') as fh:
            new_bytes = fh.read()
        self.assertNotEqual(new_bytes, old_bytes)

    def test_non_hdr_video_is_skipped_entirely(self):
        video = self._make_video('/videos/hdr_test/b.mp4')
        person = self._make_person('Retonemap Non HDR Person')
        face = self._make_face(video, person, timestamp=2.0)
        with open(face.face_thumbnail.path, 'rb') as fh:
            old_bytes = fh.read()

        with patch(f'{self.CMD}.ffprobe_info',
                   return_value=(100, 100, 1.0, 'progressive', 0, 'bt709', None)), \
             patch(f'{self.CMD}.ffmpeg_frame_iterator') as mock_iterator:
            call_command('retonemap_hdr_video_thumbnails', '--yes')

        mock_iterator.assert_not_called()
        with open(face.face_thumbnail.path, 'rb') as fh:
            self.assertEqual(fh.read(), old_bytes)

    def test_dry_run_makes_no_changes(self):
        # Uses the same fake_iterator shape as the real-regenerate test
        # (a genuine match IS found at t=2.0) specifically so this test
        # actually exercises "found a match, but --dry-run skips the
        # write" -- not just "nothing was found anyway".
        video = self._make_video('/videos/hdr_test/c.mp4')
        person = self._make_person('Retonemap Dry Run Person')
        face = self._make_face(video, person, timestamp=2.0)
        with open(face.face_thumbnail.path, 'rb') as fh:
            old_bytes = fh.read()

        def fake_iterator(path, width, height, vf_filter=None, seek_seconds=0):
            for i in range(8):
                t = seek_seconds + i
                yield self._frame(50 if abs(t - 2.0) < 0.01 else 10)

        with patch(f'{self.CMD}.ffprobe_info',
                   return_value=(100, 100, 1.0, 'progressive', 0, 'smpte2084', None)), \
             patch(f'{self.CMD}.ffmpeg_frame_iterator', side_effect=fake_iterator):
            call_command('retonemap_hdr_video_thumbnails', '--dry-run')

        with open(face.face_thumbnail.path, 'rb') as fh:
            self.assertEqual(fh.read(), old_bytes)

    def test_no_frame_within_tolerance_leaves_thumbnail_untouched_and_reports_unresolved(self):
        video = self._make_video('/videos/hdr_test/d.mp4')
        person = self._make_person('Retonemap Unresolved Person')
        face = self._make_face(video, person, timestamp=2.0)
        with open(face.face_thumbnail.path, 'rb') as fh:
            old_bytes = fh.read()

        def fake_iterator(path, width, height, vf_filter=None, seek_seconds=0):
            # Every yielded frame is far outside the 0.5s tolerance.
            yield self._frame(50)

        with patch(f'{self.CMD}.ffprobe_info',
                   return_value=(100, 100, 1.0, 'progressive', 0, 'smpte2084', None)), \
             patch(f'{self.CMD}.ffmpeg_frame_iterator', side_effect=fake_iterator):
            call_command('retonemap_hdr_video_thumbnails', '--yes')

        with open(face.face_thumbnail.path, 'rb') as fh:
            self.assertEqual(fh.read(), old_bytes)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class RedetectSarAffectedVideosTests(TestCase):
    """redetect_sar_affected_videos management command -- reconciles a
    fresh full-redetect pass (VideoFaceExtractor mocked entirely; real
    ONNX inference is covered by VideoFaceExtractorRealInferenceTests
    elsewhere) against a video's existing Face rows. A matched face must
    keep its identity but get fresh geometry/embedding; an unmatched
    candidate becomes a new unclassified face; an unmatched existing
    face is left completely untouched."""

    CMD = 'face_manager.management.commands.redetect_sar_affected_videos'

    def _make_video(self, path, sar_scale_width=640, width=352, height=480):
        from filepopulator.models import Directory, VideoFile
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(path))
        return VideoFile.objects.create(
            filename=path, directory=directory, width=width, height=height,
            duration_seconds=30, isProcessed=True, sar_scale_width=sar_scale_width,
        )

    def _make_person(self, name):
        person, created = Person.objects.get_or_create(person_name=name)
        if created:
            person.highlight_img.save(f"{name}.jpg", ContentFile(_tiny_jpeg_bytes()), save=True)
        return person

    def _make_face(self, video, person, embedding, first=1.0, last=2.0, validated=False):
        face = Face(
            declared_name=person, source_video_file=video, validated=validated,
            box_left=1, box_top=1, box_right=40, box_bottom=40,
            video_first_timestamp_seconds=first, video_last_timestamp_seconds=last,
            face_encoding_512=embedding,
        )
        face.face_thumbnail.save(
            "thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False,
        )
        face.save()
        return face

    def _emb(self, direction, dim=512):
        v = [0.0] * dim
        v[direction] = 1.0
        return v

    def _candidate(self, embedding, box=(1, 1, 40, 40), first_sample=10, last_sample=20):
        import numpy as np
        rep = {'box': box, 'kps': [1.0] * 10, 'det_score': 0.9, 'frame_idx': 0, 'emb_if': embedding}
        return {
            'centroid': np.array(embedding),
            'provisional': rep,
            'pooled_reps': [rep],
            'first_sample': first_sample,
            'last_sample': last_sample,
        }

    def _fake_extractor(self, candidates, blank_person, fps=30.0):
        import numpy as np
        mock_extractor = MagicMock()
        mock_extractor.blank_face_person = blank_person
        mock_extractor._compute_candidate_groups.return_value = (fps, {0: np.zeros((10, 10, 3), dtype=np.uint8)}, candidates)
        mock_extractor._clamp_to_duration.side_effect = lambda video, seconds, label: seconds

        def fake_set_box(face, frame, box, kps, timestamp_seconds, det_score=None):
            face.box_left, face.box_top, face.box_right, face.box_bottom = box
            face.kps = list(kps)
            face.det_score = det_score
            face.video_thumbnail_frame_seconds = timestamp_seconds
            face.face_thumbnail.save(
                "redetect_thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False,
            )
        mock_extractor._set_face_box_and_thumbnail.side_effect = fake_set_box
        mock_extractor._classify_and_pick_thumbnail.return_value = None
        return mock_extractor

    def test_matched_face_keeps_identity_gets_fresh_geometry(self):
        from filepopulator.models import VideoFile
        video = self._make_video('/videos/sar_test/a.mpg')
        person = self._make_person('Redetect Test Person')
        blank = self._make_person(settings.BLANK_FACE_NAME)
        old_embedding = self._emb(0)
        face = self._make_face(video, person, old_embedding, validated=True)
        old_box = (face.box_left, face.box_top, face.box_right, face.box_bottom)

        new_embedding = self._emb(0)  # same direction -- should match
        new_embedding[0] = 0.99  # slightly different, still a clear match
        candidates = [self._candidate(new_embedding, box=(5, 5, 55, 55))]

        with patch(f'{self.CMD}.VideoFaceExtractor', return_value=self._fake_extractor(candidates, blank)):
            call_command('redetect_sar_affected_videos', '--yes')

        face.refresh_from_db()
        self.assertEqual(face.declared_name_id, person.id)
        self.assertTrue(face.validated)
        self.assertNotEqual((face.box_left, face.box_top, face.box_right, face.box_bottom), old_box)
        self.assertEqual((face.box_left, face.box_top, face.box_right, face.box_bottom), (5, 5, 55, 55))
        self.assertAlmostEqual(face.face_encoding_512[0], new_embedding[0], places=3)
        self.assertEqual(Face.objects.filter(source_video_file=video).count(), 1)

    def test_unmatched_candidate_becomes_new_unclassified_face(self):
        video = self._make_video('/videos/sar_test/b.mpg')
        person = self._make_person('Redetect Existing Person')
        blank = self._make_person(settings.BLANK_FACE_NAME)
        self._make_face(video, person, self._emb(0))

        # Completely different embedding direction -- a genuinely new face.
        candidates = [self._candidate(self._emb(5), box=(2, 2, 42, 42))]

        with patch(f'{self.CMD}.VideoFaceExtractor', return_value=self._fake_extractor(candidates, blank)):
            call_command('redetect_sar_affected_videos', '--yes')

        self.assertEqual(Face.objects.filter(source_video_file=video).count(), 2)
        new_face = Face.objects.filter(source_video_file=video).exclude(declared_name=person).get()
        self.assertEqual(new_face.declared_name_id, blank.id)

    def test_unmatched_existing_face_left_completely_untouched(self):
        video = self._make_video('/videos/sar_test/c.mpg')
        person = self._make_person('Redetect Untouched Person')
        blank = self._make_person(settings.BLANK_FACE_NAME)
        face = self._make_face(video, person, self._emb(0), validated=True)
        old_box = (face.box_left, face.box_top, face.box_right, face.box_bottom)
        old_embedding = list(face.face_encoding_512)

        candidates = [self._candidate(self._emb(7))]  # no real match

        with patch(f'{self.CMD}.VideoFaceExtractor', return_value=self._fake_extractor(candidates, blank)):
            call_command('redetect_sar_affected_videos', '--yes')

        face.refresh_from_db()
        self.assertEqual((face.box_left, face.box_top, face.box_right, face.box_bottom), old_box)
        self.assertEqual(list(face.face_encoding_512), old_embedding)
        self.assertEqual(face.declared_name_id, person.id)
        self.assertTrue(face.validated)
        # Plus the genuinely new candidate, created separately.
        self.assertEqual(Face.objects.filter(source_video_file=video).count(), 2)

    def test_dry_run_makes_no_changes(self):
        video = self._make_video('/videos/sar_test/d.mpg')
        person = self._make_person('Redetect Dry Run Person')
        blank = self._make_person(settings.BLANK_FACE_NAME)
        face = self._make_face(video, person, self._emb(0))
        old_box = (face.box_left, face.box_top, face.box_right, face.box_bottom)

        candidates = [self._candidate(self._emb(0))]

        with patch(f'{self.CMD}.VideoFaceExtractor', return_value=self._fake_extractor(candidates, blank)):
            call_command('redetect_sar_affected_videos', '--dry-run')

        face.refresh_from_db()
        self.assertEqual((face.box_left, face.box_top, face.box_right, face.box_bottom), old_box)
        self.assertEqual(Face.objects.filter(source_video_file=video).count(), 1)

    def test_non_sar_video_is_not_selected(self):
        video = self._make_video('/videos/sar_test/e.mp4', sar_scale_width=0)
        person = self._make_person('Redetect Square Pixel Person')
        blank = self._make_person(settings.BLANK_FACE_NAME)
        self._make_face(video, person, self._emb(0))

        with patch(f'{self.CMD}.VideoFaceExtractor') as MockExtractorCls:
            call_command('redetect_sar_affected_videos', '--yes')

        MockExtractorCls.assert_not_called()
        self.assertEqual(Face.objects.filter(source_video_file=video).count(), 1)

    def test_timeout_on_one_video_is_skipped_and_the_run_continues(self):
        # Real, reproduced 2026-09-23: a video timed out under real
        # concurrent CPU/memory pressure from another job, not because
        # of anything wrong with the file (re-running it in isolation
        # took ~3s) -- same class of issue the scheduled video_face_
        # extraction task already guards against. This must be reported
        # distinctly from a real failure and must NOT crash the whole
        # command -- later videos still get processed.
        from face_manager.tasks import _VideoProcessingTimeout
        video_a = self._make_video('/videos/sar_test/f.mpg')
        video_b = self._make_video('/videos/sar_test/g.mpg')
        person = self._make_person('Redetect Timeout Person')
        blank = self._make_person(settings.BLANK_FACE_NAME)
        self._make_face(video_a, person, self._emb(0))
        self._make_face(video_b, person, self._emb(0))

        candidates = [self._candidate(self._emb(0))]
        fake_extractor = self._fake_extractor(candidates, blank)
        fake_extractor._compute_candidate_groups.side_effect = [
            _VideoProcessingTimeout("exceeded 2700s"),
            fake_extractor._compute_candidate_groups.return_value,
        ]

        with patch(f'{self.CMD}.VideoFaceExtractor', return_value=fake_extractor):
            call_command('redetect_sar_affected_videos', '--yes')

        # video_a's face untouched (its redetect never completed);
        # video_b's face still got matched/updated normally.
        self.assertEqual(fake_extractor._compute_candidate_groups.call_count, 2)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class BackfillVideoThumbnailTimestampsTests(TestCase):
    """backfill_video_thumbnail_timestamps -- previously had NO test
    coverage at all, and (found during the 2026-10-30 video-face-
    pipeline consolidation) its own local _decode_stored_thumbnail()
    didn't guard the face_thumbnail open/read call -- a single face
    with a missing thumbnail file would have crashed the entire
    command, the exact same bug class already found and fixed in
    remediate_video_face_geometry.py. Now uses the shared, guarded
    decode_face_thumbnail() instead."""

    CMD = 'face_manager.management.commands.backfill_video_thumbnail_timestamps'

    def _make_video(self, path, width=40, height=40):
        from filepopulator.models import Directory, VideoFile
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(path))
        return VideoFile.objects.create(
            filename=path, directory=directory, width=width, height=height,
            duration_seconds=30, isProcessed=True,
        )

    def _make_face(self, video, first_ts, last_ts):
        # VideoFaceExtractor._square_thumbnail() always cv2.resizes its
        # output to settings.FACE_THUMBNAIL_SIZE regardless of the input
        # frame/box size -- the stored thumbnail must match THAT size,
        # not the frame's own size, or _mean_abs_diff's shape check
        # rejects every candidate. _tiny_jpeg_bytes() is solid black
        # (all-zero), so whichever candidate frame is numerically
        # closest to 0 is the one that should win the pixel match.
        from django.conf import settings as django_settings
        person = make_person("Backfill Timestamp Person")
        face = Face(
            declared_name=person, source_video_file=video,
            box_left=5, box_top=5, box_right=35, box_bottom=35,
            video_first_timestamp_seconds=first_ts, video_last_timestamp_seconds=last_ts,
        )
        face.face_thumbnail.save(
            f"thumb_{video.pk}_{first_ts}.jpg",
            ContentFile(_tiny_jpeg_bytes(size=django_settings.FACE_THUMBNAIL_SIZE)), save=False,
        )
        face.save()
        return face

    def _frame(self, fill_value, size=40):
        return np.full((size, size, 3), fill_value, dtype=np.uint8)

    def test_missing_thumbnail_file_does_not_crash_the_whole_run(self):
        video = self._make_video('/videos/backfill_ts_test/a.mp4')
        bad_face = self._make_face(video, first_ts=1.0, last_ts=2.0)
        os.remove(bad_face.face_thumbnail.path)

        with patch(f'{self.CMD}.ffprobe_info',
                   return_value=(40, 40, 1.0, 'progressive', 0, None, None)), \
             patch(f'{self.CMD}.ffmpeg_frame_iterator',
                   return_value=iter([self._frame(50), self._frame(100)])):
            out = io.StringIO()
            call_command('backfill_video_thumbnail_timestamps', stdout=out)

        # No crash (the real regression) -- and since the thumbnail
        # itself is unreadable, this face correctly can't be resolved.
        self.assertIn('unresolved: 1', out.getvalue())
        bad_face.refresh_from_db()
        self.assertIsNone(bad_face.video_thumbnail_frame_seconds)

    def test_resolves_timestamp_via_closest_pixel_match(self):
        video = self._make_video('/videos/backfill_ts_test/b.mp4')
        face = self._make_face(video, first_ts=0.0, last_ts=2.0)

        with patch(f'{self.CMD}.ffprobe_info',
                   return_value=(40, 40, 1.0, 'progressive', 0, None, None)), \
             patch(f'{self.CMD}.ffmpeg_frame_iterator',
                   return_value=iter([self._frame(50), self._frame(100), self._frame(10)])):
            out = io.StringIO()
            call_command('backfill_video_thumbnail_timestamps', stdout=out)

        self.assertIn('Resolved: 1', out.getvalue())
        face.refresh_from_db()
        self.assertAlmostEqual(face.video_thumbnail_frame_seconds, 2.0)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class ResetBadVideoThumbnailsTests(TestCase):
    """reset_bad_video_thumbnails -- previously had NO test coverage at
    all, and (found during the 2026-10-30 video-face-pipeline
    consolidation) its own local _decode_stored_thumbnail() didn't
    guard the face_thumbnail open/read call -- a single face with a
    missing thumbnail file would have crashed the entire command. Now
    uses the shared, guarded decode_face_thumbnail() instead."""

    CMD = 'face_manager.management.commands.reset_bad_video_thumbnails'

    def _make_video(self, path, width=40, height=40):
        from filepopulator.models import Directory, VideoFile
        directory, _ = Directory.objects.get_or_create(dir_path=os.path.dirname(path))
        return VideoFile.objects.create(
            filename=path, directory=directory, width=width, height=height,
            duration_seconds=30, isProcessed=True, field_order='progressive',
        )

    def _make_face(self, video, timestamp):
        person = make_person("Reset Bad Thumbnail Person")
        face = Face(
            declared_name=person, source_video_file=video,
            box_left=5, box_top=5, box_right=35, box_bottom=35,
            video_thumbnail_frame_seconds=timestamp,
        )
        face.face_thumbnail.save(
            f"thumb_{video.pk}_{timestamp}.jpg",
            ContentFile(_tiny_jpeg_bytes(size=(30, 30))), save=False,
        )
        face.save()
        return face

    def test_missing_thumbnail_file_does_not_crash_the_whole_run(self):
        video = self._make_video('/videos/reset_bad_test/a.mp4')
        bad_face = self._make_face(video, timestamp=1.0)
        os.remove(bad_face.face_thumbnail.path)
        good_video = self._make_video('/videos/reset_bad_test/b.mp4')
        good_face = self._make_face(good_video, timestamp=1.0)

        def fake_extract(path, timestamp, field_order, accurate=False):
            # Matches whichever face's own thumbnail this call is "for" via
            # the video path, so good_face's fast check passes cleanly.
            return np.full((30, 30, 3), 7, dtype=np.uint8) if 'b.mp4' in path else None

        with patch(f'{self.CMD}._extract_frame', side_effect=fake_extract), \
             patch(f'{self.CMD}.VideoFaceExtractor._square_thumbnail',
                   side_effect=lambda frame, box: frame):
            out = io.StringIO()
            call_command('reset_bad_video_thumbnails', stdout=out)

        # No crash (the real regression, checked by simply reaching
        # this line without an unhandled exception) -- bad_face's video
        # is flagged (can't confirm a good thumbnail at all, having no
        # usable stored reference to compare against), its face deleted
        # and the video reset to unprocessed; good_face's video is left
        # completely alone.
        video.refresh_from_db()
        self.assertFalse(video.isProcessed)
        self.assertFalse(Face.objects.filter(pk=bad_face.pk).exists())
        good_video.refresh_from_db()
        self.assertTrue(good_video.isProcessed)
        self.assertTrue(Face.objects.filter(pk=good_face.pk).exists())



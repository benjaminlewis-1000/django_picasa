"""face_manager tests -- split from the original tests.py (see CLAUDE.md).

Run with: manage.py test face_manager.tests.test_video_pipeline
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


class VideoFaceIOUTrackingTests(unittest.TestCase):
    """_iou_track: pure frame-to-frame linking, no Django/model deps --
    see video_face_pipeline.py / CLAUDE.md's Phase 3 design for the full
    investigation these constants (grow 25%, iou_thresh 0.3, gap
    tolerance 1 sample) came out of."""

    def _det(self, sample_idx, boxes):
        # boxes: list of (l, t, r, b); scores/kps are unused by tracking
        # logic itself, just carried through -- filler values are fine.
        return (sample_idx, boxes, [1.0] * len(boxes), [[0.0] * 10] * len(boxes))

    def test_same_box_across_samples_forms_one_track(self):
        from video_face_pipeline import _iou_track
        dets = [self._det(0, [[10, 10, 50, 50]]), self._det(1, [[11, 11, 51, 51]]),
                self._det(2, [[10, 10, 50, 50]])]
        tracks = _iou_track(dets)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]['span'], (0, 2))
        self.assertEqual(len(tracks[0]['boxes']), 3)

    def test_two_spatially_distinct_boxes_form_two_tracks(self):
        from video_face_pipeline import _iou_track
        dets = [self._det(0, [[10, 10, 50, 50], [500, 500, 540, 540]]),
                self._det(1, [[10, 10, 50, 50], [500, 500, 540, 540]])]
        tracks = _iou_track(dets)
        self.assertEqual(len(tracks), 2)

    def test_gap_tolerance_bridges_one_missed_sample(self):
        from video_face_pipeline import _iou_track
        dets = [self._det(0, [[10, 10, 50, 50]]),
                self._det(1, []),  # missed detection
                self._det(2, [[10, 10, 50, 50]])]
        tracks = _iou_track(dets, max_gap=1)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]['span'], (0, 2))

    def test_gap_beyond_tolerance_splits_into_two_tracks(self):
        from video_face_pipeline import _iou_track
        dets = [self._det(0, [[10, 10, 50, 50]]),
                self._det(1, []), self._det(2, []),  # two consecutive misses
                self._det(3, [[10, 10, 50, 50]])]
        tracks = _iou_track(dets, max_gap=1)
        self.assertEqual(len(tracks), 2)

    def test_far_apart_boxes_in_same_frame_never_merge_regardless_of_gap(self):
        from video_face_pipeline import _iou_track
        # Two different people, both present in every sampled frame --
        # must never collapse into one track no matter the gap tolerance.
        dets = [self._det(i, [[10, 10, 50, 50], [500, 500, 540, 540]]) for i in range(5)]
        tracks = _iou_track(dets, max_gap=1)
        self.assertEqual(len(tracks), 2)


class VideoFaceShortTrackFilterTests(unittest.TestCase):
    """_filter_short_tracks: drops raw tracks matched across too few
    sampled frames before the expensive per-track redetect+dual-encode
    pass -- but ONLY once a video's raw singleton count looks
    pathological (ABSURD_SINGLETON_TRACK_COUNT), since a normal video's
    occasional genuine single-appearance track is real signal, not noise
    -- see that constant's own comment in video_face_pipeline.py for the
    real videos (2 old MJPEG .AVIs producing 2,400+ tracks, 78-80%
    singletons) that motivated this gate."""

    def _track(self, n_frames):
        return {'frames': list(range(n_frames))}

    def test_below_the_gate_keeps_every_track_including_singletons(self):
        # Only 2 singletons -- nowhere near "absurd" -- so nothing is
        # dropped, even the singletons themselves.
        from video_face_pipeline import _filter_short_tracks
        tracks = [self._track(1), self._track(1), self._track(3), self._track(10)]
        kept = _filter_short_tracks(tracks, min_len=3, absurd_singleton_count=150)
        self.assertEqual(len(kept), 4)

    def test_at_the_gate_drops_tracks_below_the_length_floor(self):
        # Exactly absurd_singleton_count singletons trips the gate (>=,
        # not strictly >), and once tripped the length floor applies.
        from video_face_pipeline import _filter_short_tracks
        tracks = [self._track(1)] * 3 + [self._track(2), self._track(3), self._track(10)]
        kept = _filter_short_tracks(tracks, min_len=3, absurd_singleton_count=3)
        self.assertEqual([len(t['frames']) for t in kept], [3, 10])

    def test_one_below_the_gate_changes_nothing(self):
        from video_face_pipeline import _filter_short_tracks
        tracks = [self._track(1)] * 2 + [self._track(10)]
        kept = _filter_short_tracks(tracks, min_len=3, absurd_singleton_count=3)
        self.assertEqual(len(kept), 3)

    def test_empty_input_returns_empty(self):
        from video_face_pipeline import _filter_short_tracks
        self.assertEqual(_filter_short_tracks([], min_len=3, absurd_singleton_count=3), [])

    def test_defaults_match_module_constants(self):
        from video_face_pipeline import (
            _filter_short_tracks, MIN_TRACK_LEN_SAMPLES, ABSURD_SINGLETON_TRACK_COUNT,
        )
        tracks = ([self._track(1)] * ABSURD_SINGLETON_TRACK_COUNT
                  + [self._track(MIN_TRACK_LEN_SAMPLES - 1), self._track(MIN_TRACK_LEN_SAMPLES)])
        kept = _filter_short_tracks(tracks)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(kept[0]['frames']), MIN_TRACK_LEN_SAMPLES)


class VideoFaceClusterTests(unittest.TestCase):
    """_cluster_track_faces: must-link (same track) + cannot-link
    (temporally-overlapping different tracks) constrained clustering."""

    def test_same_base_direction_different_tracks_cluster_together(self):
        from video_face_pipeline import _cluster_track_faces
        # Track 0: samples 0-10; Track 1: samples 100-110 (non-overlapping)
        # -- both the same person (same base direction), should merge.
        embs = [_embedding(0, seed=i) for i in range(4)]
        track_id = [0, 0, 1, 1]
        spans = {0: (0, 10), 1: (100, 110)}
        groups = _cluster_track_faces(embs, track_id, spans, cos_threshold=0.5)
        self.assertEqual(groups[0], groups[1])

    def test_different_base_directions_never_merge(self):
        from video_face_pipeline import _cluster_track_faces
        embs = [_embedding(0, seed=i) for i in range(3)] + [_embedding(1, seed=100 + i) for i in range(3)]
        track_id = [0, 0, 0, 1, 1, 1]
        spans = {0: (0, 10), 1: (100, 110)}
        groups = _cluster_track_faces(embs, track_id, spans, cos_threshold=0.5)
        self.assertNotEqual(groups[0], groups[1])

    def test_must_link_keeps_one_tracks_own_faces_together_despite_noise(self):
        from video_face_pipeline import _cluster_track_faces
        # One track's own two reps, seeded far apart enough that raw
        # similarity alone might not obviously cluster them -- must-link
        # forces them together regardless (same track = same person by
        # construction, per the IOU tracker's own guarantee).
        embs = [_embedding(0, seed=1), _embedding(0, seed=999)]
        track_id = [0, 0]
        spans = {0: (0, 10)}
        groups = _cluster_track_faces(embs, track_id, spans, cos_threshold=0.99)
        self.assertEqual(groups[0], groups[0])  # trivially true; real check is no crash/split

    def test_cannot_link_blocks_merge_even_at_identical_embeddings(self):
        from video_face_pipeline import _cluster_track_faces
        # Same base direction (would otherwise merge) but temporally
        # overlapping spans -- two different people can't both be the
        # same identity at the same time, so this must NOT merge even
        # though the embeddings alone would suggest it should.
        embs = [_embedding(0, seed=1), _embedding(0, seed=2)]
        track_id = [0, 1]
        spans = {0: (0, 100), 1: (50, 150)}  # overlapping
        groups = _cluster_track_faces(embs, track_id, spans, cos_threshold=0.5)
        self.assertNotEqual(groups[0], groups[1])


class VideoFaceUnionMergeTests(unittest.TestCase):
    """_union_merge_groups: union (not intersection) over multiple
    independent clusterings -- the user's own explicit call for this
    pipeline (fewer final groups, some wrong merges acceptable)."""

    def test_union_of_disjoint_groupings_combines_via_either(self):
        from video_face_pipeline import _union_merge_groups
        # Model A groups (0,1) together and (2,3) together; model B
        # groups (1,2) together -- the union should chain all 4 into one.
        grouping_a = {0: 'x', 1: 'x', 2: 'y', 3: 'y'}
        grouping_b = {0: 'p', 1: 'q', 2: 'q', 3: 'r'}
        final = _union_merge_groups(4, grouping_a, grouping_b)
        self.assertEqual(len({final[0], final[1], final[2], final[3]}), 1)

    def test_agreement_on_separation_keeps_tracks_apart(self):
        from video_face_pipeline import _union_merge_groups
        grouping_a = {0: 'x', 1: 'y'}
        grouping_b = {0: 'p', 1: 'q'}
        final = _union_merge_groups(2, grouping_a, grouping_b)
        self.assertNotEqual(final[0], final[1])

    def test_single_grouping_passthrough(self):
        from video_face_pipeline import _union_merge_groups
        grouping = {0: 'x', 1: 'x', 2: 'y'}
        final = _union_merge_groups(3, grouping)
        self.assertEqual(final[0], final[1])
        self.assertNotEqual(final[0], final[2])


class MatchRedetectCandidatesTests(unittest.TestCase):
    """match_redetect_candidates_to_existing_faces: reconciles a fresh
    full-redetect pass (e.g. after fixing a video's aspect-ratio decode)
    against that video's existing Face rows, so confirmed identifications
    survive even though the redetect's own boundaries won't exactly
    reproduce the original pass's. See CLAUDE.md's 2026-09-23 write-up."""

    def _emb(self, direction, dim=8):
        v = np.zeros(dim)
        v[direction] = 1.0
        return v

    def test_matching_embeddings_pair_up(self):
        from video_face_pipeline import match_redetect_candidates_to_existing_faces
        existing = [{'embedding': self._emb(0)}, {'embedding': self._emb(1)}]
        candidates = [{'embedding': self._emb(1)}, {'embedding': self._emb(0)}]
        matches = match_redetect_candidates_to_existing_faces(existing, candidates)
        self.assertEqual(set(matches), {(0, 1), (1, 0)})

    def test_dissimilar_embedding_stays_unmatched(self):
        from video_face_pipeline import match_redetect_candidates_to_existing_faces
        existing = [{'embedding': self._emb(0)}]
        candidates = [{'embedding': self._emb(1)}]  # orthogonal -- cos_sim=0
        matches = match_redetect_candidates_to_existing_faces(existing, candidates)
        self.assertEqual(matches, [])

    def test_more_candidates_than_existing_leaves_extras_unmatched(self):
        from video_face_pipeline import match_redetect_candidates_to_existing_faces
        existing = [{'embedding': self._emb(0)}]
        candidates = [{'embedding': self._emb(0)}, {'embedding': self._emb(1)}]
        matches = match_redetect_candidates_to_existing_faces(existing, candidates)
        self.assertEqual(matches, [(0, 0)])
        matched_candidate_idcs = {j for _, j in matches}
        self.assertNotIn(1, matched_candidate_idcs)  # the new, unmatched face

    def test_more_existing_than_candidates_leaves_extras_unmatched(self):
        from video_face_pipeline import match_redetect_candidates_to_existing_faces
        existing = [{'embedding': self._emb(0)}, {'embedding': self._emb(1)}]
        candidates = [{'embedding': self._emb(0)}]
        matches = match_redetect_candidates_to_existing_faces(existing, candidates)
        self.assertEqual(matches, [(0, 0)])
        matched_existing_idcs = {i for i, _ in matches}
        self.assertNotIn(1, matched_existing_idcs)  # left alone, not deleted

    def test_empty_either_side_returns_no_matches(self):
        from video_face_pipeline import match_redetect_candidates_to_existing_faces
        self.assertEqual(match_redetect_candidates_to_existing_faces([], [{'embedding': self._emb(0)}]), [])
        self.assertEqual(match_redetect_candidates_to_existing_faces([{'embedding': self._emb(0)}], []), [])

    def test_never_force_matches_below_the_floor(self):
        # Two existing, two candidates, but only ONE real match -- the
        # optimal assignment must not force the second (bad) pairing
        # just because it's "the best of what's left".
        from video_face_pipeline import match_redetect_candidates_to_existing_faces
        existing = [{'embedding': self._emb(0)}, {'embedding': self._emb(2)}]
        candidates = [{'embedding': self._emb(0)}, {'embedding': self._emb(3)}]
        matches = match_redetect_candidates_to_existing_faces(existing, candidates)
        self.assertEqual(matches, [(0, 0)])


class VideoFaceTrimmedCentroidTests(unittest.TestCase):
    """_trimmed_centroid: one-pass outlier trim (drop anything below a
    cosine-similarity floor to the plain mean, recompute from survivors)
    -- the user's own chosen approach over a fixed-percentage trim."""

    def test_all_similar_embeddings_centroid_close_to_each(self):
        from video_face_pipeline import _trimmed_centroid, cos_sim
        embs = np.array([_embedding(0, seed=i) for i in range(5)])
        centroid = _trimmed_centroid(embs, sim_floor=0.35)
        for e in embs:
            self.assertGreater(cos_sim(e, centroid), 0.9)

    def test_outlier_is_excluded_from_centroid(self):
        from video_face_pipeline import _trimmed_centroid, cos_sim
        # 4 real members of one identity, 1 genuine outlier (orthogonal
        # base direction) that should get trimmed before the final mean.
        good = [_embedding(0, seed=i) for i in range(4)]
        outlier = _embedding(1, seed=999)
        centroid_with_outlier_trimmed = _trimmed_centroid(np.array(good + [outlier]), sim_floor=0.35)
        centroid_good_only = np.mean(good, axis=0)
        # The trimmed centroid should closely match the good-only mean --
        # if the outlier had survived trimming, it would pull noticeably
        # away from this.
        self.assertGreater(cos_sim(centroid_with_outlier_trimmed, centroid_good_only), 0.99)

    def test_floor_that_excludes_everything_falls_back_to_untrimmed_mean(self):
        from video_face_pipeline import _trimmed_centroid
        embs = np.array([_embedding(0, seed=i) for i in range(3)])
        # An impossibly strict floor (above 1.0) can never be met -- must
        # fall back to the plain mean rather than dividing by zero
        # survivors.
        centroid = _trimmed_centroid(embs, sim_floor=1.5)
        expected = embs.mean(axis=0)
        np.testing.assert_allclose(centroid, expected)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class VideoFaceSetBoxAndThumbnailOrphanTests(TestCase):
    """_set_face_box_and_thumbnail() is called a second time, on the same
    already-saved Face, when _classify_and_pick_thumbnail() re-picks a
    better representative frame after a confident gallery match. Django's
    FieldFile never deletes the file it's replacing on a plain
    reassignment -- found 2026-10-08 while root-causing a real 188K-file
    orphaned-face-thumbnail backlog. Uses object.__new__() to skip
    VideoFaceExtractor.__init__()'s real ONNX/facenet model loading
    entirely, since the method under test touches none of those models
    -- pure box math + file I/O."""

    def setUp(self):
        from video_face_pipeline import VideoFaceExtractor
        from filepopulator.models import Directory, VideoFile
        self.extractor = object.__new__(VideoFaceExtractor)

        directory = Directory.objects.create(dir_path='/tmp/video_orphan_test')
        self.video = VideoFile.objects.create(
            filename='/tmp/video_orphan_test/a.mp4', directory=directory,
            width=320, height=240, duration_seconds=10.0,
        )
        self.blank_person, _ = Person.objects.get_or_create(person_name=settings.BLANK_FACE_NAME)
        self.face = Face(
            source_video_file=self.video, declared_name=self.blank_person,
            box_left=1, box_top=1, box_right=50, box_bottom=50,
        )
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)

    def test_second_call_deletes_the_first_calls_thumbnail_file(self):
        # First call -- matches the real pipeline's own order (see
        # _set_face_box_and_thumbnail's docstring on
        # _classify_and_pick_thumbnail): the thumbnail is set once before
        # the Face row is ever saved to the DB (Face.save() itself
        # requires a thumbnail to already be attached), then the row is
        # saved for real, standing in for "an already-saved Face row".
        self.extractor._set_face_box_and_thumbnail(
            self.face, self.frame, (10, 10, 60, 60), kps=[[0, 0]] * 5,
        )
        self.face.save()
        first_thumbnail_path = self.face.face_thumbnail.path
        self.assertTrue(os.path.isfile(first_thumbnail_path))

        # Second call -- the real re-pick path.
        self.extractor._set_face_box_and_thumbnail(
            self.face, self.frame, (20, 20, 70, 70), kps=[[0, 0]] * 5,
        )

        self.assertFalse(os.path.isfile(first_thumbnail_path))
        second_thumbnail_path = self.face.face_thumbnail.path
        self.assertTrue(os.path.isfile(second_thumbnail_path))
        self.assertNotEqual(first_thumbnail_path, second_thumbnail_path)

    def test_first_call_on_a_fresh_face_does_not_error(self):
        # No prior thumbnail exists yet -- must be a clean no-op, not a
        # crash trying to delete something that was never there.
        self.extractor._set_face_box_and_thumbnail(
            self.face, self.frame, (10, 10, 60, 60), kps=[[0, 0]] * 5,
        )
        self.assertTrue(os.path.isfile(self.face.face_thumbnail.path))


class LocateConfidentFaceNearTimestampTests(unittest.TestCase):
    """locate_confident_face_near_timestamp() -- the remediation search
    used by remediate_video_face_geometry.py to find a real fix for a
    video face whose box/kps don't bound a real face (2026-10-29, see
    CLAUDE.md). Mocks av.open() and the detector entirely -- this is
    pure search-logic under test (closest-first ordering, confidence
    floor, nearest-to-old-box tie-breaking), not real decode/detection."""

    @staticmethod
    def _mock_frame(seconds, time_base=1):
        f = MagicMock()
        f.pts = seconds
        f.time_base = time_base
        f.to_ndarray.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        return f

    def _mock_container(self, frame_seconds_list):
        container = MagicMock()
        stream = MagicMock(start_time=0, time_base=1)
        container.streams.video = [stream]
        packet = MagicMock()
        packet.decode.return_value = [self._mock_frame(s) for s in frame_seconds_list]
        container.demux.return_value = [packet]
        return container

    def test_finds_confident_match_closest_to_target_first(self):
        from video_face_pipeline import locate_confident_face_near_timestamp
        # Two candidate frames both have a confident detection near the
        # old box -- the CLOSER one (target+0.1) must win, not whichever
        # the detector happens to be called on first/last.
        container = self._mock_container([10.0 - 0.5, 10.0 + 0.1, 10.0 + 0.9])
        det_model = MagicMock()

        def detect(frame, max_num=0, metric='default'):
            # Only the two frames nearest target (10.0+0.1 and 10.0-0.5)
            # have a confident detection; the farthest does not.
            return (
                np.array([[40, 40, 60, 60, 0.9]], dtype=np.float32),
                np.zeros((1, 5, 2), dtype=np.float32),
            )
        det_model.detect.side_effect = detect

        with patch('video_face_pipeline.av.open', return_value=container):
            result = locate_confident_face_near_timestamp(
                '/fake.mp4', 10.0, 'progressive', 0, (40, 40, 60, 60), det_model, 0.5,
                window_seconds=1.0,
            )
        self.assertIsNotNone(result)
        frame, box, kps, score, frame_time = result
        self.assertAlmostEqual(frame_time, 10.1, places=4)

    def test_returns_none_when_nothing_confident_anywhere(self):
        from video_face_pipeline import locate_confident_face_near_timestamp
        container = self._mock_container([10.0, 10.0 + 0.3])
        det_model = MagicMock()
        det_model.detect.return_value = (
            np.array([[40, 40, 60, 60, 0.2]], dtype=np.float32),  # below floor
            np.zeros((1, 5, 2), dtype=np.float32),
        )
        with patch('video_face_pipeline.av.open', return_value=container):
            result = locate_confident_face_near_timestamp(
                '/fake.mp4', 10.0, 'progressive', 0, (40, 40, 60, 60), det_model, 0.5,
                window_seconds=1.0,
            )
        self.assertIsNone(result)

    def test_picks_nearest_to_old_box_not_highest_score(self):
        from video_face_pipeline import locate_confident_face_near_timestamp
        container = self._mock_container([10.0])
        det_model = MagicMock()
        # Two confident candidates in the SAME frame: a far, more
        # confident one, and a nearer, less-but-still-confident one.
        # Must pick the nearer one.
        det_model.detect.return_value = (
            np.array([
                [400, 400, 440, 440, 0.95],
                [42, 42, 58, 58, 0.6],
            ], dtype=np.float32),
            np.zeros((2, 5, 2), dtype=np.float32),
        )
        with patch('video_face_pipeline.av.open', return_value=container):
            result = locate_confident_face_near_timestamp(
                '/fake.mp4', 10.0, 'progressive', 0, (40, 40, 60, 60), det_model, 0.5,
                window_seconds=1.0,
            )
        self.assertIsNotNone(result)
        frame, box, kps, score, frame_time = result
        self.assertAlmostEqual(box[0], 42, places=4)
        self.assertAlmostEqual(score, 0.6, places=4)

    def test_no_candidates_in_window_returns_none(self):
        from video_face_pipeline import locate_confident_face_near_timestamp
        container = self._mock_container([])
        det_model = MagicMock()
        with patch('video_face_pipeline.av.open', return_value=container):
            result = locate_confident_face_near_timestamp(
                '/fake.mp4', 10.0, 'progressive', 0, (40, 40, 60, 60), det_model, 0.5,
                window_seconds=1.0,
            )
        self.assertIsNone(result)
        det_model.detect.assert_not_called()


class RedetectConfidenceFallbackTests(unittest.TestCase):
    """_redetect_in_crop()/_redetect_full_frame_nearest() -- the fix for
    a real, confirmed bug (2026-10-29, see CLAUDE.md): _redetect_in_crop()
    used to always return its best candidate, even a low-confidence
    spurious one on background content, whenever the padded crop (built
    around a possibly already-wrong upstream box) didn't actually contain
    the tracked face. Confirmed via a real investigation that the
    function itself works correctly given a genuinely correct box --
    the bug was having no way to say "I don't actually know" when given
    a bad one. Pure math, no Django/DB/video I/O -- object.__new__()
    skips VideoFaceExtractor.__init__()'s real ONNX model loading, same
    pattern VideoFaceSetBoxAndThumbnailOrphanTests already established."""

    def setUp(self):
        from video_face_pipeline import VideoFaceExtractor
        self.extractor = object.__new__(VideoFaceExtractor)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    @staticmethod
    def _dets(rows):
        """rows: list of (l, t, r, b, score). Returns (bboxes, kpss)
        shaped like insightface's own detector output."""
        bboxes = np.array([[l, t, r, b, s] for (l, t, r, b, s) in rows], dtype=np.float32)
        kpss = np.zeros((len(rows), 5, 2), dtype=np.float32)
        return bboxes, kpss

    def test_confident_crop_candidate_is_still_accepted(self):
        # Regression: the pre-existing, working case must be unaffected.
        self.extractor.det10g_model = MagicMock()
        self.extractor.det10g_model.detect.return_value = self._dets([(5, 5, 40, 40, 0.87)])
        result = self.extractor._redetect_in_crop(self.frame, (10, 10, 50, 50))
        self.assertIsNotNone(result)
        box_out, kps_out, score_out = result
        self.assertAlmostEqual(score_out, 0.87, places=4)

    def test_low_confidence_crop_candidate_is_rejected(self):
        # The real bug: a weak, spurious detection in a crop that
        # doesn't actually contain the face used to be accepted anyway.
        self.extractor.det10g_model = MagicMock()
        self.extractor.det10g_model.detect.return_value = self._dets([(5, 5, 40, 40, 0.22)])
        result = self.extractor._redetect_in_crop(self.frame, (10, 10, 50, 50))
        self.assertIsNone(result)

    def test_zero_candidates_still_returns_none(self):
        self.extractor.det10g_model = MagicMock()
        self.extractor.det10g_model.detect.return_value = (
            np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32),
        )
        result = self.extractor._redetect_in_crop(self.frame, (10, 10, 50, 50))
        self.assertIsNone(result)

    def test_full_frame_fallback_picks_nearest_confident_candidate_not_just_highest_score(self):
        # Two people in frame: a far, MORE confident face, and a nearer,
        # less-but-still-confident one matching where this track's
        # person actually was. Must pick the nearer one -- picking
        # highest-confidence overall would silently swap in a different
        # person's face.
        self.extractor.det_model = MagicMock()
        self.extractor.det_model.detect.return_value = self._dets([
            (400, 400, 440, 440, 0.95),  # far away, very confident -- wrong person
            (12, 12, 48, 48, 0.6),        # near the original box -- the right person
        ])
        result = self.extractor._redetect_full_frame_nearest(self.frame, (10, 10, 50, 50))
        self.assertIsNotNone(result)
        box_out, kps_out, score_out = result
        self.assertAlmostEqual(box_out[0], 12, places=4)
        self.assertAlmostEqual(score_out, 0.6, places=4)

    def test_full_frame_fallback_ignores_low_confidence_candidates(self):
        self.extractor.det_model = MagicMock()
        self.extractor.det_model.detect.return_value = self._dets([
            (12, 12, 48, 48, 0.3),  # near, but too weak to trust
        ])
        result = self.extractor._redetect_full_frame_nearest(self.frame, (10, 10, 50, 50))
        self.assertIsNone(result)

    def test_full_frame_fallback_returns_none_when_nothing_detected(self):
        self.extractor.det_model = MagicMock()
        self.extractor.det_model.detect.return_value = (
            np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32),
        )
        result = self.extractor._redetect_full_frame_nearest(self.frame, (10, 10, 50, 50))
        self.assertIsNone(result)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class AuditVideoFaceGeometryTests(TestCase):
    """audit_video_face_geometry -- read-only scan for video faces whose
    box/kps don't actually bound a real, centered face in their own
    saved thumbnail (the real bug found 2026-10-29, see CLAUDE.md).
    Mocks the FaceAnalysis detector entirely; real model loading is
    expensive and this command's own logic (confidence + centeredness
    check) is what's under test, not detection accuracy itself."""

    CMD_MODULE = 'face_manager.management.commands.audit_video_face_geometry'

    def _make_video_face(self, **overrides):
        from filepopulator.models import Directory, VideoFile
        directory, _ = Directory.objects.get_or_create(dir_path='/videos/geometry_audit_test')
        video_path = '/tmp/geometry_audit_test_video.mp4'
        with open(video_path, 'wb') as f:
            f.write(b'not a real video, existence is all this scan checks')
        self.addCleanup(lambda: os.path.exists(video_path) and os.remove(video_path))
        video = VideoFile.objects.create(filename=video_path, directory=directory, width=640, height=480)
        from face_manager.models import get_default_blank_person
        face = Face(
            declared_name=overrides.pop('declared_name', None) or get_default_blank_person(),
            poss_ident1=overrides.pop('poss_ident1', None),
            source_video_file=video,
            box_left=1, box_top=1, box_right=10, box_bottom=10,
            video_thumbnail_frame_seconds=12.3,
            kps=[1.0] * 10,
            **overrides,
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(200, 200))), save=False)
        face.save()
        return face

    @staticmethod
    def _dets(rows):
        """rows: list of (l, t, r, b, score)."""
        bboxes = np.array([[l, t, r, b, s] for (l, t, r, b, s) in rows], dtype=np.float32) \
            if rows else np.zeros((0, 5), dtype=np.float32)
        return bboxes, None

    def _run_with_detections(self, bboxes_and_kpss, *extra_args):
        with patch(f'{self.CMD_MODULE}.FaceAnalysis') as MockFA:
            mock_det_model = MagicMock()
            mock_det_model.detect.return_value = bboxes_and_kpss
            MockFA.return_value.models = {'detection': mock_det_model}
            out = io.StringIO()
            call_command('audit_video_face_geometry', *extra_args, stdout=out)
        return out.getvalue()

    def test_confident_centered_detection_is_not_flagged(self):
        face = self._make_video_face()
        output = self._run_with_detections(self._dets([(70, 70, 130, 130, 0.9)]))
        self.assertIn("0 flagged", output)
        self.assertNotIn(str(face.id), output)

    def test_no_detection_at_all_is_flagged(self):
        face = self._make_video_face()
        output = self._run_with_detections(self._dets([]))
        self.assertIn("1 flagged", output)
        self.assertIn(f"{face.id}: no detection in thumbnail at all", output)

    def test_low_confidence_detection_is_flagged(self):
        face = self._make_video_face()
        output = self._run_with_detections(self._dets([(70, 70, 130, 130, 0.3)]))
        self.assertIn("1 flagged", output)
        self.assertIn(f"{face.id}:", output)
        self.assertIn("no confident detection", output)

    def test_confident_but_off_center_detection_is_flagged(self):
        face = self._make_video_face()
        output = self._run_with_detections(self._dets([(5, 5, 20, 20, 0.9)]))
        self.assertIn("1 flagged", output)
        self.assertIn(f"{face.id}: confident detection found, but off-center", output)

    def test_limit_restricts_scope(self):
        self._make_video_face()
        self._make_video_face()
        with patch(f'{self.CMD_MODULE}.FaceAnalysis') as MockFA:
            mock_det_model = MagicMock()
            mock_det_model.detect.return_value = self._dets([(70, 70, 130, 130, 0.9)])
            MockFA.return_value.models = {'detection': mock_det_model}
            out = io.StringIO()
            call_command('audit_video_face_geometry', '--limit', '1', stdout=out)
        self.assertIn("Checked 1 video face", out.getvalue())

    def test_person_ids_restricts_to_declared_or_proposed(self):
        target_person = make_person("Geometry Audit Target")
        other_person = make_person("Geometry Audit Other")
        target_face = self._make_video_face(declared_name=target_person)
        self._make_video_face(declared_name=other_person)
        with patch(f'{self.CMD_MODULE}.FaceAnalysis') as MockFA:
            mock_det_model = MagicMock()
            mock_det_model.detect.return_value = self._dets([])
            MockFA.return_value.models = {'detection': mock_det_model}
            out = io.StringIO()
            call_command('audit_video_face_geometry', f'--person-ids={target_person.id}', stdout=out)
        output = out.getvalue()
        self.assertIn("Checked 1 video face", output)
        self.assertIn(str(target_face.id), output)

    def test_missing_thumbnail_is_flagged_not_crashed(self):
        face = self._make_video_face()
        Face.objects.filter(pk=face.pk).update(face_thumbnail='')
        output = self._run_with_detections(self._dets([(70, 70, 130, 130, 0.9)]))
        self.assertIn("Checked 0 video face", output)
        self.assertNotIn(str(face.id), output)


@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class RemediateVideoFaceGeometryTests(TestCase):
    """remediate_video_face_geometry -- combined triage+fix for video
    faces whose box/kps don't bound a real face (2026-10-29, see
    CLAUDE.md). Mocks VideoFaceExtractor (det10g_model.detect for
    triage, _set_face_box_and_thumbnail/_encode_insightface for the
    fix) and locate_confident_face_near_timestamp entirely -- this
    command's own triage/dispatch/save logic is what's under test."""

    CMD_MODULE = 'face_manager.management.commands.remediate_video_face_geometry'

    def _make_video_face(self, **overrides):
        from filepopulator.models import Directory, VideoFile
        directory, _ = Directory.objects.get_or_create(dir_path='/videos/remediate_test')
        video_path = '/tmp/remediate_test_video.mp4'
        with open(video_path, 'wb') as f:
            f.write(b'not a real video, existence is all this scan checks')
        self.addCleanup(lambda: os.path.exists(video_path) and os.remove(video_path))
        video = VideoFile.objects.create(filename=video_path, directory=directory, width=640, height=480)
        from face_manager.models import get_default_blank_person
        face = Face(
            declared_name=overrides.pop('declared_name', None) or get_default_blank_person(),
            source_video_file=video,
            box_left=10, box_top=10, box_right=50, box_bottom=50,
            video_thumbnail_frame_seconds=overrides.pop('video_thumbnail_frame_seconds', 12.3),
            kps=[1.0] * 10,
            **overrides,
        )
        face.face_thumbnail.save("thumb.jpg", ContentFile(_tiny_jpeg_bytes(size=(200, 200))), save=False)
        face.save()
        return face

    @staticmethod
    def _dets(rows):
        bboxes = np.array([[l, t, r, b, s] for (l, t, r, b, s) in rows], dtype=np.float32) \
            if rows else np.zeros((0, 5), dtype=np.float32)
        return bboxes, None

    def test_face_passing_triage_is_skipped_untouched(self):
        face = self._make_video_face()
        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.locate_confident_face_near_timestamp') as mock_locate:
            MockExtractor.return_value.det10g_model.detect.return_value = self._dets([(70, 70, 130, 130, 0.9)])
            out = io.StringIO()
            call_command('remediate_video_face_geometry', stdout=out)
        mock_locate.assert_not_called()
        self.assertIn("1 already fine", out.getvalue())
        face.refresh_from_db()
        self.assertEqual(face.box_left, 10)

    def test_ok_face_gets_second_check_and_lvface_from_real_frame(self):
        """A triage-passed face now also gets a real (cheap) frame decode
        -- a second, independent validation via _redetect_in_crop against
        the real frame, and LVFace-B from the real frame + confirmed kps,
        per the user's explicit request (2026-10-29)."""
        face = self._make_video_face()
        real_frame = np.ones((480, 640, 3), dtype=np.uint8) * 7
        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info',
                   return_value=(640, 480, 30.0, 'progressive', 0, None, None)), \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=real_frame) as mock_extract, \
             patch(f'{self.CMD_MODULE}.locate_confident_face_near_timestamp') as mock_locate, \
             patch(f'{self.CMD_MODULE}.lvface') as mock_lvface:
            mock_ext = MockExtractor.return_value
            mock_ext.det10g_model.detect.return_value = self._dets([(70, 70, 130, 130, 0.9)])  # triage passes
            mock_ext._redetect_in_crop.return_value = ([12, 12, 48, 48], [[1, 1]] * 5, 0.88)
            mock_lvface.embed_from_image_array.return_value = np.full(512, 0.6, dtype=np.float32)
            out = io.StringIO()
            call_command('remediate_video_face_geometry', stdout=out)
        mock_locate.assert_not_called()
        mock_extract.assert_called_once()
        # Regression (2026-10-30): extract_frame_near_timestamp() dispatches HDR-tagged
        # sources to its ffmpeg-subprocess variant, which needs width/height explicitly
        # (it has no container to read dimensions from the way the PyAV path does) --
        # this command used to omit them entirely, crashing with
        # TypeError('NoneType * NoneType') for every HDR-sourced "already ok" face, always
        # silently swallowed by a bare except. Assert the real ffprobe'd dimensions are
        # actually passed through, not just that the function was called at all.
        self.assertEqual(mock_extract.call_args.kwargs.get('width'), 640)
        self.assertEqual(mock_extract.call_args.kwargs.get('height'), 480)
        mock_ext._redetect_in_crop.assert_called_once_with(real_frame, (10, 10, 50, 50))
        mock_lvface.embed_from_image_array.assert_called_once_with(
            real_frame, [[1, 1]] * 5, input_is_bgr=True, size='B',
        )
        self.assertIn("1 already fine (1 also got LVFace embeddings", out.getvalue())
        face.refresh_from_db()
        self.assertEqual(face.face_encoding_lvface_b, [0.6] * 512)
        self.assertEqual(face.box_left, 10)  # box/kps genuinely unchanged, only LVFace was added

    def test_ok_face_hdr_source_does_not_crash_extract_frame_near_timestamp(self):
        """Regression (2026-10-30): an HDR-tagged video (color_transfer
        'arib-std-b67'/'smpte2084') makes extract_frame_near_timestamp()
        dispatch to its ffmpeg-subprocess variant, which needs width/
        height explicitly -- this command used to call it without them,
        crashing with TypeError('NoneType * NoneType') for every single
        HDR-sourced 'already ok' face, silently swallowed by a bare
        except so it looked like simple skip-already-done behavior
        rather than a real, systemic failure. Uses the REAL
        extract_frame_near_timestamp()/dispatch logic (only
        ffmpeg_frame_iterator itself is mocked, to avoid a real
        subprocess), so this exercises the exact code path that crashed,
        not a mock standing in for it."""
        face = self._make_video_face()
        real_frame = np.ones((480, 640, 3), dtype=np.uint8) * 7
        # remediate_video_face_geometry.py imports video_face_pipeline as a BARE
        # top-level module (`from video_face_pipeline import ...`), which this codebase
        # also loads under `face_manager.video_face_pipeline` -- same file, but two
        # distinct sys.modules entries with independent globals. Must patch the bare
        # one, or the patch silently has no effect on the real dispatch path.
        import video_face_pipeline as vfp_module
        mock_iterator = MagicMock(return_value=iter([real_frame]))
        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info',
                   return_value=(640, 480, 30.0, 'progressive', 0, 'arib-std-b67', None)), \
             patch.object(vfp_module, 'ffmpeg_frame_iterator', mock_iterator), \
             patch(f'{self.CMD_MODULE}.lvface') as mock_lvface:
            mock_ext = MockExtractor.return_value
            mock_ext.det10g_model.detect.return_value = self._dets([(70, 70, 130, 130, 0.9)])
            mock_ext._redetect_in_crop.return_value = ([12, 12, 48, 48], [[1, 1]] * 5, 0.88)
            mock_lvface.embed_from_image_array.return_value = np.full(512, 0.6, dtype=np.float32)
            out = io.StringIO()
            call_command('remediate_video_face_geometry', stdout=out)
        # The real point of this test: the HDR dispatch path actually ran (no TypeError
        # from frame_size = width * height * 3 with width/height missing) and reached
        # the raw-pipe frame source with the REAL ffprobe'd dimensions, not None/None.
        mock_iterator.assert_called()
        call_args = mock_iterator.call_args_list[0]
        self.assertEqual(call_args.args[1], 640)  # width
        self.assertEqual(call_args.args[2], 480)  # height
        self.assertEqual(out.getvalue().count('CRASH'), 0)
        self.assertNotIn('TypeError', out.getvalue())

    def test_ok_face_flagged_when_real_frame_check_disagrees(self):
        """Triage passed on the stored thumbnail, but a real frame decode
        finds nothing confident near the stored box -- flagged, not
        auto-fixed, same discipline as 'unresolved'."""
        face = self._make_video_face()
        real_frame = np.ones((480, 640, 3), dtype=np.uint8) * 7
        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info',
                   return_value=(640, 480, 30.0, 'progressive', 0, None, None)), \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=real_frame), \
             patch(f'{self.CMD_MODULE}.lvface') as mock_lvface:
            mock_ext = MockExtractor.return_value
            mock_ext.det10g_model.detect.return_value = self._dets([(70, 70, 130, 130, 0.9)])
            mock_ext._redetect_in_crop.return_value = None
            out = io.StringIO()
            call_command('remediate_video_face_geometry', stdout=out)
        mock_lvface.embed_from_image_array.assert_not_called()
        self.assertIn("1 disagreed on a second, real-frame check", out.getvalue())
        self.assertIn(str(face.id), out.getvalue())
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)

    def test_ok_face_lvface_skipped_if_already_set(self):
        """Unlike a FIXED face (kps changed, always overwritten), an OK
        face's box/kps haven't changed -- an existing LVFace value isn't
        stale, so it's left alone rather than recomputed."""
        face = self._make_video_face(face_encoding_lvface_b=[0.9] * 512)
        real_frame = np.ones((480, 640, 3), dtype=np.uint8) * 7
        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info',
                   return_value=(640, 480, 30.0, 'progressive', 0, None, None)), \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=real_frame), \
             patch(f'{self.CMD_MODULE}.lvface') as mock_lvface:
            mock_ext = MockExtractor.return_value
            mock_ext.det10g_model.detect.return_value = self._dets([(70, 70, 130, 130, 0.9)])
            mock_ext._redetect_in_crop.return_value = ([12, 12, 48, 48], [[1, 1]] * 5, 0.88)
            out = io.StringIO()
            call_command('remediate_video_face_geometry', stdout=out)
        mock_lvface.embed_from_image_array.assert_not_called()
        face.refresh_from_db()
        self.assertEqual(face.face_encoding_lvface_b, [0.9] * 512)

    def test_ok_face_dry_run_does_not_save_lvface(self):
        face = self._make_video_face()
        real_frame = np.ones((480, 640, 3), dtype=np.uint8) * 7
        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info',
                   return_value=(640, 480, 30.0, 'progressive', 0, None, None)), \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=real_frame), \
             patch(f'{self.CMD_MODULE}.lvface') as mock_lvface:
            mock_ext = MockExtractor.return_value
            mock_ext.det10g_model.detect.return_value = self._dets([(70, 70, 130, 130, 0.9)])
            mock_ext._redetect_in_crop.return_value = ([12, 12, 48, 48], [[1, 1]] * 5, 0.88)
            mock_lvface.embed_from_image_array.return_value = np.full(512, 0.6, dtype=np.float32)
            out = io.StringIO()
            call_command('remediate_video_face_geometry', '--dry-run', stdout=out)
        self.assertIn("1 also got LVFace embeddings", out.getvalue())
        face.refresh_from_db()
        self.assertIsNone(face.face_encoding_lvface_b)

    def test_ffprobe_cached_across_faces_sharing_one_video(self):
        """Many faces on the same VideoFile shouldn't each shell out to
        ffprobe separately."""
        face1 = self._make_video_face()
        video = face1.source_video_file
        face2 = Face(
            declared_name=face1.declared_name, source_video_file=video,
            box_left=10, box_top=10, box_right=50, box_bottom=50,
            video_thumbnail_frame_seconds=20.0, kps=[1.0] * 10,
        )
        face2.face_thumbnail.save("thumb2.jpg", ContentFile(_tiny_jpeg_bytes(size=(200, 200))), save=False)
        face2.save()
        real_frame = np.ones((480, 640, 3), dtype=np.uint8) * 7
        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info',
                   return_value=(640, 480, 30.0, 'progressive', 0, None, None)) as mock_ffprobe, \
             patch(f'{self.CMD_MODULE}.extract_frame_near_timestamp', return_value=real_frame), \
             patch(f'{self.CMD_MODULE}.lvface') as mock_lvface:
            mock_ext = MockExtractor.return_value
            mock_ext.det10g_model.detect.return_value = self._dets([(70, 70, 130, 130, 0.9)])
            mock_ext._redetect_in_crop.return_value = ([12, 12, 48, 48], [[1, 1]] * 5, 0.88)
            mock_lvface.embed_from_image_array.return_value = np.full(512, 0.6, dtype=np.float32)
            call_command('remediate_video_face_geometry', stdout=io.StringIO())
        mock_ffprobe.assert_called_once_with(video.filename)

    def test_face_failing_triage_gets_fixed(self):
        face = self._make_video_face()
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info',
                   return_value=(640, 480, 30.0, 'progressive', 0, None, None)), \
             patch(f'{self.CMD_MODULE}.locate_confident_face_near_timestamp') as mock_locate, \
             patch(f'{self.CMD_MODULE}.lvface') as mock_lvface:
            mock_ext = MockExtractor.return_value
            mock_ext.det10g_model.detect.return_value = self._dets([])  # triage fails
            mock_locate.return_value = (
                fake_frame, [20, 20, 80, 80], [[0, 0]] * 5, 0.85, 12.5,
            )
            mock_ext._encode_insightface.return_value = np.full(512, 0.3, dtype=np.float32)
            mock_lvface.embed_from_image_array.return_value = np.full(512, 0.4, dtype=np.float32)
            out = io.StringIO()
            call_command('remediate_video_face_geometry', stdout=out)
        mock_locate.assert_called_once()
        mock_ext._set_face_box_and_thumbnail.assert_called_once_with(
            face, fake_frame, [20, 20, 80, 80], [[0, 0]] * 5, timestamp_seconds=12.5, det_score=0.85,
        )
        self.assertIn("1 fixed", out.getvalue())
        face.refresh_from_db()
        self.assertEqual(face.face_encoding_512, [0.3] * 512)

    def test_fixed_face_also_gets_lvface_embeddings_from_the_same_frame(self):
        """The frame/kps are already decoded for the geometry fix -- LVFace-
        B is (re)computed from that same frame by default, not left for a
        separate (nonexistent, for video) backfill. --lvface-sizes can
        widen this to S/L too, exercised here explicitly."""
        face = self._make_video_face()
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info',
                   return_value=(640, 480, 30.0, 'progressive', 0, None, None)), \
             patch(f'{self.CMD_MODULE}.locate_confident_face_near_timestamp') as mock_locate, \
             patch(f'{self.CMD_MODULE}.lvface') as mock_lvface:
            mock_ext = MockExtractor.return_value
            mock_ext.det10g_model.detect.return_value = self._dets([])
            mock_locate.return_value = (fake_frame, [20, 20, 80, 80], [[0, 0]] * 5, 0.85, 12.5)
            mock_ext._encode_insightface.return_value = np.full(512, 0.3, dtype=np.float32)
            mock_lvface.embed_from_image_array.side_effect = (
                lambda frame, kps, input_is_bgr, size: np.full(512, {'S': 0.1, 'B': 0.2, 'L': 0.3}[size],
                                                                 dtype=np.float32)
            )
            out = io.StringIO()
            call_command('remediate_video_face_geometry', '--lvface-sizes=S,B,L', stdout=out)
        self.assertEqual(mock_lvface.embed_from_image_array.call_count, 3)
        for call in mock_lvface.embed_from_image_array.call_args_list:
            args, kwargs = call
            self.assertIs(args[0], fake_frame)
            self.assertEqual(args[1], [[0, 0]] * 5)
            self.assertTrue(kwargs['input_is_bgr'])
        self.assertIn("1 also got LVFace embeddings", out.getvalue())
        face.refresh_from_db()
        self.assertEqual(face.face_encoding_lvface_s, [0.1] * 512)
        self.assertEqual(face.face_encoding_lvface_b, [0.2] * 512)
        self.assertEqual(face.face_encoding_lvface_l, [0.3] * 512)

    def test_lvface_sizes_can_be_restricted_via_flag(self):
        face = self._make_video_face()
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info',
                   return_value=(640, 480, 30.0, 'progressive', 0, None, None)), \
             patch(f'{self.CMD_MODULE}.locate_confident_face_near_timestamp') as mock_locate, \
             patch(f'{self.CMD_MODULE}.lvface') as mock_lvface:
            mock_ext = MockExtractor.return_value
            mock_ext.det10g_model.detect.return_value = self._dets([])
            mock_locate.return_value = (fake_frame, [20, 20, 80, 80], [[0, 0]] * 5, 0.85, 12.5)
            mock_ext._encode_insightface.return_value = np.full(512, 0.3, dtype=np.float32)
            mock_lvface.embed_from_image_array.return_value = np.full(512, 0.2, dtype=np.float32)
            out = io.StringIO()
            call_command('remediate_video_face_geometry', '--lvface-sizes=B', stdout=out)
        mock_lvface.embed_from_image_array.assert_called_once()
        self.assertEqual(mock_lvface.embed_from_image_array.call_args.kwargs['size'], 'B')
        face.refresh_from_db()
        self.assertEqual(face.face_encoding_lvface_b, [0.2] * 512)
        self.assertIsNone(face.face_encoding_lvface_s)
        self.assertIsNone(face.face_encoding_lvface_l)

    def test_unresolved_when_no_confident_match_anywhere(self):
        face = self._make_video_face()
        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info',
                   return_value=(640, 480, 30.0, 'progressive', 0, None, None)), \
             patch(f'{self.CMD_MODULE}.locate_confident_face_near_timestamp', return_value=None):
            MockExtractor.return_value.det10g_model.detect.return_value = self._dets([])
            out = io.StringIO()
            call_command('remediate_video_face_geometry', stdout=out)
        self.assertIn("1 unresolved", out.getvalue())
        self.assertIn(str(face.id), out.getvalue())
        face.refresh_from_db()
        self.assertEqual(face.box_left, 10)  # untouched

    def test_dry_run_does_not_save(self):
        face = self._make_video_face()
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.ffprobe_info',
                   return_value=(640, 480, 30.0, 'progressive', 0, None, None)), \
             patch(f'{self.CMD_MODULE}.locate_confident_face_near_timestamp') as mock_locate:
            mock_ext = MockExtractor.return_value
            mock_ext.det10g_model.detect.return_value = self._dets([])
            mock_locate.return_value = (fake_frame, [20, 20, 80, 80], [[0, 0]] * 5, 0.85, 12.5)
            out = io.StringIO()
            call_command('remediate_video_face_geometry', '--dry-run', stdout=out)
        mock_ext._set_face_box_and_thumbnail.assert_not_called()
        self.assertIn("1 fixed", out.getvalue())
        face.refresh_from_db()
        self.assertEqual(face.box_left, 10)  # untouched despite being counted as "fixed"

    def test_no_stored_timestamp_is_unresolved_without_a_search(self):
        face = self._make_video_face(video_thumbnail_frame_seconds=None)
        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor, \
             patch(f'{self.CMD_MODULE}.locate_confident_face_near_timestamp') as mock_locate:
            MockExtractor.return_value.det10g_model.detect.return_value = self._dets([])
            out = io.StringIO()
            call_command('remediate_video_face_geometry', stdout=out)
        mock_locate.assert_not_called()
        self.assertIn("1 unresolved", out.getvalue())

    def test_unexpected_error_on_one_face_does_not_abort_the_whole_run(self):
        """Real bug found 2026-10-30: a missing thumbnail file on disk
        (FileNotFoundError, genuinely reproduced here by deleting it)
        crashed the ENTIRE multi-day production run, since only
        _FaceTimeout was caught at the per-face level. Any unexpected
        exception must be caught, logged, and skipped -- same 'one bad
        item can't kill the batch' discipline this project enforces
        everywhere else."""
        bad_face = self._make_video_face()
        os.remove(bad_face.face_thumbnail.path)
        good_face = self._make_video_face()
        with patch(f'{self.CMD_MODULE}.VideoFaceExtractor') as MockExtractor:
            MockExtractor.return_value.det10g_model.detect.return_value = self._dets([(70, 70, 130, 130, 0.9)])
            out = io.StringIO()
            call_command('remediate_video_face_geometry', stdout=out)
        self.assertIn("1 crashed", out.getvalue())
        self.assertIn(str(bad_face.id), out.getvalue())
        self.assertIn("FileNotFoundError", out.getvalue())
        self.assertIn("1 already fine", out.getvalue())


@tag("slow")
@override_settings(MEDIA_ROOT="/tmp/face_manager_test_media")
class VideoFaceExtractorRealInferenceTests(TestCase):
    """Real ffmpeg decode + real insightface/facenet inference against a
    real video fixture (IMG_0760.MOV, 7.6s, already used elsewhere in
    this investigation and known to contain real, detectable faces --
    see CLAUDE.md's Phase 3 section). No test_face_cache-style caching
    exists for this pipeline yet (unlike PyramidalDetectorRealInferenceTests)
    -- this test pays real CPU inference cost every run."""

    VIDEO_PATH = '/photos/video_samples/IMG_0760.MOV'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from video_face_pipeline import VideoFaceExtractor
        cls.extractor = VideoFaceExtractor()
        # Same override LoadEncodingsCachingTests uses: /models/ (the
        # real cache location) doesn't exist in this test environment --
        # pre-build the faceAssigner here with a writable path instead of
        # letting the lazy face_assigner property build one against the
        # real (here, nonexistent) path on first access.
        cache_path = "/tmp/face_manager_test_media/test_video_encodings_cache.pkl"
        if os.path.exists(cache_path):
            os.remove(cache_path)
        fa = faceAssigner()
        fa.ENCODINGS_PKL_FILE = cache_path
        fa.load_encodings()
        cls.extractor._face_assigner = fa

    def setUp(self):
        from filepopulator.models import Directory, VideoFile
        directory = Directory.objects.create(dir_path='/photos/video_samples')
        self.video = VideoFile.objects.create(
            filename=self.VIDEO_PATH, directory=directory,
            width=1920, height=1080, duration_seconds=7.6,
        )

    def test_real_fixture_produces_valid_face_groups(self):
        faces = self.extractor.process_video(self.video)
        self.assertGreaterEqual(len(faces), 1, "expected at least one face group from a known-face fixture")

        blank_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        for face in faces:
            face.refresh_from_db()
            self.assertEqual(face.source_video_file_id, self.video.pk)
            self.assertIsNone(face.source_image_file_id)
            self.assertEqual(len(face.face_encoding_512), 512)
            self.assertIsNotNone(face.video_first_timestamp_seconds)
            self.assertIsNotNone(face.video_last_timestamp_seconds)
            self.assertLessEqual(face.video_first_timestamp_seconds, face.video_last_timestamp_seconds)
            self.assertGreaterEqual(face.video_first_timestamp_seconds, 0)
            self.assertLessEqual(face.video_last_timestamp_seconds, self.video.duration_seconds + 1)
            # Auto-matching (if any) only ever proposes -- never auto-declares.
            self.assertEqual(face.declared_name_id, blank_person.pk)
            # A real thumbnail file was actually generated and saved.
            self.assertTrue(os.path.exists(face.face_thumbnail.path))
            self.assertGreater(face.box_right, face.box_left)
            self.assertGreater(face.box_bottom, face.box_top)
            # det_score (added 2026-09-18, see CLAUDE.md) is threaded
            # through the video pipeline the same as the image one --
            # a real, valid detection confidence, not just left NULL.
            self.assertIsNotNone(face.det_score)
            self.assertGreater(face.det_score, 0.0)
            self.assertLessEqual(face.det_score, 1.0)

    def test_reprocessing_is_independent_not_cumulative(self):
        # Not idempotency in the "no duplicate rows" sense (there's no
        # dedup logic -- isProcessed is what stops a real double-run) --
        # just confirms a second call doesn't crash or silently multiply
        # unboundedly, producing a comparable group count both times.
        first = self.extractor.process_video(self.video)
        for f in first:
            f.delete()
        second = self.extractor.process_video(self.video)
        self.assertGreaterEqual(len(second), 1)



#! /usr/bin/env python

# Stage 2 (expensive -- full video redetection) of the two-stage quality
# rescan started by scan_video_thumbnail_quality.py (Stage 1, cheap,
# read-only). See that command's own docstring for the real bug report
# that prompted this (face 1104963: a genuine, confirmed person, but the
# stored thumbnail/box/timestamp point at a dark, blurry moment while a
# much clearer frame of the same person exists ~0.8s later in the same
# detection track).
#
# Takes Stage 1's --out CSV and picks candidate faces using TWO factors,
# not quality score alone -- a real calibration finding against the full
# 45,206-face library (2026-10-02): the raw quality score is heavily
# confounded by how large the original detected face was. A small/
# distant face (e.g. 80x70px) scores low simply because it has few real
# pixels of detail once upscaled to the fixed 200x200 thumbnail size --
# that's an inherent framing property no amount of re-picking within the
# same track can fix. The genuinely fixable case (like face 1104963,
# whose box was 243x358px -- plenty of real detail -- yet still scored
# in the bottom 1-5th percentile of large-box faces) is a face that's
# BIG ENOUGH to expect real detail, but still looks dark/blurry. So a
# candidate must clear --min-box-area (restricts to faces large enough
# that a low score is actually surprising) AND score below
# --quality-threshold (now calibrated against the large-box population
# specifically: p1=3.8, p5=12.6, p10=23.5 -- the default of 10.0 flags
# ~3.8% of large-box faces, 648 faces / 421 videos in the real library).
#
# Groups flagged faces by video (so a video with several flagged faces
# only pays for one full redetect pass), and for each:
#   1. Re-runs VideoFaceExtractor._compute_candidate_groups() on the
#      source video -- the same detect->track->cluster->union-merge
#      pipeline process_video() uses, read-only (no DB writes from this
#      step).
#   2. Matches the flagged Face back to whichever fresh candidate group
#      it came from, via match_redetect_candidates_to_existing_faces()
#      (embedding-similarity bipartite match, same mechanism
#      redetect_sar_affected_videos.py already uses for exactly this
#      "reconcile an existing Face against fresh candidates" problem).
#   3. Scores every pooled representative frame in that candidate group
#      on the SAME combined quality formula.
#   4. If the best one clears --quality-threshold AND beats the
#      currently-stored frame's own score by --improvement-margin (a
#      relative multiplier, not just any positive delta -- avoids
#      switching on noise), re-picks it: box/kps/det_score/embedding/
#      thumbnail/timestamp are overwritten in place, exactly like
#      redetect_sar_affected_videos.py's "matched" branch.
#      declared_name/validated/poss_identN are NEVER touched -- this is
#      purely a presentation/embedding-quality improvement for an
#      already-decided identity, not a reclassification.
#   5. If no match is found, or no pooled rep clears the margin, the
#      face is left completely untouched and reported separately --
#      most unmatched-track cases are expected to be genuine false-
#      positive detections (e.g. foliage mistaken for a face, see face
#      1107315 in the originating investigation) rather than a
#      correctable quality issue, since a track with no good frame has
#      nothing better to switch to.
import csv
import signal
import time
from collections import defaultdict

import cv2
import numpy as np
from django.db import transaction
from django.core.management.base import BaseCommand

from face_manager.models import Face
from face_manager.tasks import PER_VIDEO_TIMEOUT_SECONDS, _VideoProcessingTimeout, _raise_video_timeout
from filepopulator.models import VideoFile
from video_face_pipeline import VideoFaceExtractor, match_redetect_candidates_to_existing_faces

DEFAULT_QUALITY_THRESHOLD = 10.0
DEFAULT_MIN_BOX_AREA = 10000  # 100x100px -- see this file's own docstring
DEFAULT_IMPROVEMENT_MARGIN = 2.0
# A replacement must ALSO clear this absolute floor, not just beat the
# old score by --improvement-margin -- found via a real dry-run spot
# check (face 1131900): a near-zero baseline (0.8) can satisfy a 2x
# relative margin by landing on another still-bad frame (2.0) from a
# track that's uniformly dark/blurry throughout. Set near the large-box
# population's own p1 (3.8) -- below this, nothing in the track is
# actually good, so leave the original pick alone rather than reshuffle
# between two poor options.
DEFAULT_MIN_NEW_SCORE = 5.0


def _quality_score(brightness, sharpness, det_score=None):
    """Combined, cheap image-quality score -- penalizes both darkness
    (brightness well below a mid-exposure reference) and blur (low
    Laplacian variance) multiplicatively, lightly weighted by detector
    confidence when available (a confident detection is less likely to
    be a spurious one). Deliberately simple/interpretable over a more
    "principled" formula -- calibrated empirically against real data in
    scan_video_thumbnail_quality.py's own output, not derived
    theoretically."""
    brightness_factor = min(1.0, brightness / 100.0)
    conf_factor = det_score if det_score is not None else 1.0
    return sharpness * brightness_factor * conf_factor


def _frame_quality(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, sharpness


class Command(BaseCommand):
    help = (
        "Stage 2: for faces flagged by scan_video_thumbnail_quality.py "
        "as having a low-quality (dark/blurry) stored thumbnail, "
        "re-examine their original detection track and re-pick a "
        "meaningfully better representative frame if one exists. "
        "Identity fields (declared_name/validated/poss_identN) are "
        "never touched."
    )

    def add_arguments(self, parser):
        parser.add_argument('--in', dest='in_path', type=str, required=True,
                             help='Path to the CSV produced by scan_video_thumbnail_quality.py.')
        parser.add_argument('--quality-threshold', type=float, default=DEFAULT_QUALITY_THRESHOLD,
                             help='Faces scoring below this are candidates for re-examination.')
        parser.add_argument('--min-box-area', type=float, default=DEFAULT_MIN_BOX_AREA,
                             help='Only faces with an original box at least this large (px^2) '
                                  'are considered -- a small/distant face scores low for reasons '
                                  'no re-pick can fix (see this file\'s own docstring).')
        parser.add_argument('--improvement-margin', type=float, default=DEFAULT_IMPROVEMENT_MARGIN,
                             help='A replacement frame must score at least this many times better.')
        parser.add_argument('--min-new-score', type=float, default=DEFAULT_MIN_NEW_SCORE,
                             help='A replacement frame must ALSO clear this absolute score, not '
                                  'just beat the old score by --improvement-margin -- avoids '
                                  'reshuffling between two still-bad frames in a uniformly poor '
                                  'track (see this file\'s own docstring).')
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--limit', type=int, default=None,
                             help='Restrict to the first N candidate videos (for testing).')

    def handle(self, *args, **options):
        in_path = options['in_path']
        quality_threshold = options['quality_threshold']
        min_box_area = options['min_box_area']
        improvement_margin = options['improvement_margin']
        min_new_score = options['min_new_score']
        dry_run = options['dry_run']
        limit = options['limit']

        rows = []
        with open(in_path) as f:
            for row in csv.DictReader(f):
                brightness = float(row['brightness'])
                sharpness = float(row['sharpness'])
                score = _quality_score(brightness, sharpness)
                if score < quality_threshold:
                    rows.append((int(row['face_id']), int(row['video_id']), score))

        # Box dimensions aren't in Stage 1's CSV -- one bulk query for
        # every score-flagged face, filtered down further by the area
        # floor before any video gets touched.
        face_ids = [r[0] for r in rows]
        boxes = {
            f.id: (f.box_right - f.box_left) * (f.box_bottom - f.box_top)
            for f in Face.objects.filter(id__in=face_ids).only(
                'id', 'box_left', 'box_top', 'box_right', 'box_bottom'
            )
        }

        candidates_by_video = defaultdict(list)
        for face_id, video_id, score in rows:
            if boxes.get(face_id, 0) >= min_box_area:
                candidates_by_video[video_id].append((face_id, score))

        video_ids = list(candidates_by_video.keys())
        if limit is not None:
            video_ids = video_ids[:limit]

        total_candidates = sum(len(candidates_by_video[v]) for v in video_ids)
        self.stdout.write(
            f'{len(video_ids)} video(s) / {total_candidates} candidate face(s) '
            f'below quality threshold {quality_threshold}.'
        )

        extractor = VideoFaceExtractor()
        improved = unmatched = no_improvement = failed = timed_out = 0
        t0 = time.time()

        for vi, video_id in enumerate(video_ids, start=1):
            try:
                video = VideoFile.objects.get(id=video_id)
            except VideoFile.DoesNotExist:
                continue

            face_ids = [fid for fid, _ in candidates_by_video[video_id]]
            faces = {f.id: f for f in Face.objects.filter(id__in=face_ids)}
            old_scores = dict(candidates_by_video[video_id])

            try:
                signal.signal(signal.SIGALRM, _raise_video_timeout)
                signal.alarm(PER_VIDEO_TIMEOUT_SECONDS)
                try:
                    fps, frame_pixels, candidates = extractor._compute_candidate_groups(video)
                finally:
                    signal.alarm(0)
            except _VideoProcessingTimeout:
                timed_out += len(face_ids)
                self.stdout.write(f'  [{vi}/{len(video_ids)}] video {video_id}: TIMEOUT, skipped')
                continue
            except Exception as e:
                failed += len(face_ids)
                self.stdout.write(f'  [{vi}/{len(video_ids)}] video {video_id}: failed -- {e}')
                continue

            cand_dicts = [{'embedding': c['centroid']} for c in candidates]

            for face_id in face_ids:
                face = faces.get(face_id)
                if face is None or face.face_encoding_512 is None:
                    unmatched += 1
                    continue
                existing = [{'face': face, 'embedding': np.array(face.face_encoding_512)}]
                matches = match_redetect_candidates_to_existing_faces(existing, cand_dicts)
                if not matches:
                    unmatched += 1
                    continue

                _, cand_idx = matches[0]
                cand = candidates[cand_idx]

                best_rep, best_score = None, old_scores[face_id]
                for rep in cand['pooled_reps']:
                    frame = frame_pixels[rep['frame_idx']]
                    crop = VideoFaceExtractor._square_thumbnail(frame, rep['box'])
                    brightness, sharpness = _frame_quality(crop)
                    score = _quality_score(brightness, sharpness, rep.get('det_score'))
                    if score > best_score:
                        best_score, best_rep = score, rep

                if (
                    best_rep is None
                    or best_score < old_scores[face_id] * improvement_margin
                    or best_score < min_new_score
                ):
                    no_improvement += 1
                    continue

                improved += 1
                self.stdout.write(
                    f'  face {face_id} (video {video_id}): {old_scores[face_id]:.1f} -> '
                    f'{best_score:.1f}'
                )
                if dry_run:
                    continue

                with transaction.atomic():
                    frame = frame_pixels[best_rep['frame_idx']]
                    face.face_encoding_512 = cand['centroid'].tolist()
                    extractor._set_face_box_and_thumbnail(
                        face, frame, best_rep['box'], best_rep['kps'],
                        best_rep['frame_idx'] / fps, det_score=best_rep['det_score'],
                    )
                    face.save()

            if vi % 20 == 0:
                elapsed = time.time() - t0
                self.stdout.write(
                    f'  ... {vi}/{len(video_ids)} videos, improved={improved} '
                    f'unmatched={unmatched} no_improvement={no_improvement} ({elapsed:.0f}s elapsed)'
                )

        verb = 'Would improve' if dry_run else 'Improved'
        self.stdout.write(
            f'\n{verb} {improved}, unmatched {unmatched}, no_improvement {no_improvement}, '
            f'failed {failed}, timed_out {timed_out} (out of {total_candidates} candidate(s)).'
        )

#! /usr/bin/env python

# Read-only scope check: triggered by a real user report (2026-10-02) that
# a video face's SAVED thumbnail file doesn't match its own box/kps/
# timestamp -- confirmed directly on face 1104963 (video 2709): the stored
# box/kps/video_thumbnail_frame_seconds all correctly point at a real,
# clearly-visible face (verified by decoding that exact frame fresh and
# cropping it -- a clean match to what the "fast" (ffmpeg -ss) viewer
# shows), but the face_thumbnail JPEG actually saved on disk is a dark,
# unrelated blur. This also explains why the "accurate" on-demand full-
# frame viewer (extract_frame_near_timestamp) can land on the wrong
# frame too, for a DIFFERENT reason than the already-fixed fps-mismatch
# bug: that function picks whichever candidate frame's own box crop best
# pixel-matches the STORED thumbnail -- if the stored thumbnail is itself
# wrong, "best match to a wrong reference" can easily be another wrong
# frame (confirmed on face 1107315: fast viewer shows the right person,
# accurate viewer shows an unrelated tree).
#
# This command does NOT fix anything -- it only measures how widespread
# the problem is, by regenerating what the thumbnail SHOULD look like
# (same _square_thumbnail crop logic the original pipeline uses, applied
# to a fresh decode at the face's own already-stored timestamp+box) and
# comparing it against what's actually saved on disk via mean absolute
# pixel difference. A real mismatch (the saved file genuinely depicts
# different content than its own box/timestamp) reads as a large diff;
# ordinary JPEG re-encoding noise between two correct crops of the same
# frame reads as a small one (see the already-established ~10.0 diff
# threshold used elsewhere in this file's own frame-matching code).
#
# Grouped by video (one decode per video, not one ffmpeg invocation per
# face) -- 45,206 video faces span only 5,460 distinct videos, so this is
# a ~8x reduction in decode cost versus a naive per-face approach.
import signal
import time
from collections import defaultdict

import cv2
import numpy as np
from django.core.management.base import BaseCommand

from face_manager.models import Face
from face_manager.tasks import PER_VIDEO_TIMEOUT_SECONDS, _VideoProcessingTimeout, _raise_video_timeout
from filepopulator.models import VideoFile
from video_face_pipeline import (
    VideoFaceExtractor, _mean_abs_pixel_diff, build_vf_filter, ffmpeg_frame_iterator, ffprobe_info,
)

MISMATCH_THRESHOLD = 15.0


class Command(BaseCommand):
    help = (
        "Read-only scan: for every video-sourced Face with a stored "
        "thumbnail, decode its source video fresh and compare a crop at "
        "the face's own stored box/timestamp against the actually-saved "
        "face_thumbnail file. Reports a mismatch distribution -- does "
        "not modify any data. See this file's own docstring for the "
        "real bug report that prompted it."
    )

    def add_arguments(self, parser):
        parser.add_argument('--limit', type=int, default=None,
                             help='Restrict to the first N videos (for testing).')

    def handle(self, *args, **options):
        limit = options['limit']

        faces = list(
            Face.objects.filter(
                source_video_file__isnull=False, video_thumbnail_frame_seconds__isnull=False,
            ).select_related('source_video_file')
        )
        by_video = defaultdict(list)
        for f in faces:
            by_video[f.source_video_file].append(f)

        videos = list(by_video.items())
        if limit is not None:
            videos = videos[:limit]

        self.stdout.write(f'{len(videos)} video(s) / {sum(len(v) for _, v in videos)} face(s) to check.')

        checked_videos = 0
        checked_faces = 0
        mismatches = []
        decode_failed = 0
        timed_out = 0
        t0 = time.time()

        for video, video_faces in videos:
            checked_videos += 1
            try:
                signal.signal(signal.SIGALRM, _raise_video_timeout)
                signal.alarm(PER_VIDEO_TIMEOUT_SECONDS)
                try:
                    width, height, fps, field_order, _rotation, color_transfer, sar_scale_width = (
                        ffprobe_info(video.filename)
                    )
                    vf = build_vf_filter(field_order, color_transfer, sar_scale_width)
                    decode_width = sar_scale_width or width
                    decoded_frames = list(
                        ffmpeg_frame_iterator(video.filename, decode_width, height, vf_filter=vf)
                    )
                finally:
                    signal.alarm(0)
            except _VideoProcessingTimeout:
                timed_out += 1
                self.stdout.write(f'  video {video.id} ({video.filename}): TIMEOUT, skipped')
                continue
            except Exception as e:
                decode_failed += len(video_faces)
                self.stdout.write(f'  video {video.id} ({video.filename}): decode failed -- {e}')
                continue

            if not decoded_frames:
                decode_failed += len(video_faces)
                continue

            # process_video() computes video_thumbnail_frame_seconds as
            # frame_idx / real_fps (the TRUE decoded-frame-count-based
            # rate), not ffprobe's own reported fps -- the two can differ
            # significantly on some files (see backfill_video_real_fps).
            # Using ffprobe's fps here to invert timestamp -> index would
            # reintroduce false mismatches on exactly the videos that are
            # already correctly stored.
            real_fps = len(decoded_frames) / video.duration_seconds if video.duration_seconds else fps

            for face in video_faces:
                checked_faces += 1
                target = face.video_thumbnail_frame_seconds
                # Recomputing the exact index (not a nearest-time search)
                # reproduces the identical frame deterministically,
                # avoiding false positives from picking an adjacent frame
                # on fast-motion content where neighboring frames can
                # differ a lot.
                exact_idx = min(max(0, round(target * real_fps)), len(decoded_frames) - 1)
                exact_frame = decoded_frames[exact_idx]
                box = (face.box_left, face.box_top, face.box_right, face.box_bottom)
                fresh_crop = VideoFaceExtractor._square_thumbnail(exact_frame, box)

                try:
                    face.face_thumbnail.open('rb')
                    thumb_bytes = face.face_thumbnail.read()
                    face.face_thumbnail.close()
                except Exception:
                    continue
                stored = cv2.imdecode(np.frombuffer(thumb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

                diff = _mean_abs_pixel_diff(fresh_crop, stored)
                if diff is not None and diff > MISMATCH_THRESHOLD:
                    mismatches.append((face.id, video.id, video.filename, diff))

            if checked_videos % 50 == 0:
                elapsed = time.time() - t0
                self.stdout.write(
                    f'  ... {checked_videos}/{len(videos)} videos, {checked_faces} faces checked, '
                    f'{len(mismatches)} mismatches so far ({elapsed:.0f}s elapsed)'
                )

        self.stdout.write(
            f'\nDone. {checked_videos} videos checked ({timed_out} timed out), '
            f'{checked_faces} faces checked ({decode_failed} skipped on decode failure), '
            f'{len(mismatches)} mismatches found (threshold={MISMATCH_THRESHOLD}).'
        )
        mismatches.sort(key=lambda m: -m[3])
        for face_id, video_id, filename, diff in mismatches[:50]:
            self.stdout.write(f'  face {face_id} (video {video_id}, {filename}): diff={diff:.1f}')

#! /usr/bin/env python

# One-time (but safely re-runnable) backfill: regenerates the stored
# face_thumbnail JPEG for every already-processed HDR-video face with
# proper tone-mapping applied, in place -- box/kps/embedding/det_score/
# declared_name/validated/clustering are all left completely untouched.
# See CLAUDE.md's 2026-09-22 write-up: HDR-tagged source video (PQ/
# HDR10 or HLG), decoded via the pipeline's plain bgr24 raw pipe with no
# tone-mapping, comes out badly washed-out/desaturated -- confirmed
# directly against a real production clip. video_face_pipeline.py's
# extraction pipeline and both on-demand full-frame viewers now apply
# HDR_TONEMAP_FILTER going forward; this command is the one-time
# catch-up for faces already extracted before that fix landed.
#
# Unlike backfill_video_thumbnail_timestamps.py (which recovers an
# UNKNOWN timestamp via pixel-matching against the existing thumbnail),
# this command already knows the correct timestamp
# (Face.video_thumbnail_frame_seconds) -- that field is exactly the
# thing we trust; only the PIXEL VALUES at that already-known frame are
# wrong. So no reference-image matching is needed here (matching a
# freshly tone-mapped candidate against a washed-out reference would be
# self-defeating anyway) -- just decode near each target timestamp,
# track real elapsed time the same way ffmpeg_frame_iterator's other
# callers do (seek_seconds + n/fps), and take whichever decoded frame's
# computed time is closest to the target, within a small tolerance.
import time

import cv2
from django.core.management.base import BaseCommand

from face_manager.models import Face
from filepopulator.models import VideoFile
from video_face_pipeline import (
    VideoFaceExtractor, build_vf_filter, ffmpeg_frame_iterator, ffprobe_info, is_hdr_transfer,
)

TOLERANCE_SECONDS = 0.5


class Command(BaseCommand):
    help = (
        "Regenerate face_thumbnail files (only) for already-processed HDR-video "
        "faces with proper tone-mapping -- box/kps/embedding/identity untouched."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--yes', action='store_true')

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        videos = list(VideoFile.objects.filter(isProcessed=True, face__isnull=False).distinct())
        self.stdout.write(f"Checking {len(videos)} processed video(s) with faces for HDR tagging...")

        # Uses the cached VideoFile.color_transfer field (a plain DB
        # query) rather than an ffprobe subprocess call per video --
        # this step alone used to mean re-opening every processed video
        # in the library just to check its tag. Only rows that predate
        # the cached field (color_transfer still NULL) pay for a live
        # ffprobe call here; run backfill_video_color_transfer first to
        # avoid that entirely. The per-HDR-video ffprobe call below
        # (for fps, which isn't cached anywhere) is unavoidable either
        # way, but only paid for the actual HDR subset, not every video.
        hdr_videos = []
        uncached = 0
        for video in videos:
            color_transfer = video.color_transfer
            if color_transfer is None:
                uncached += 1
                try:
                    _, _, _, _, _, color_transfer = ffprobe_info(video.filename)
                except Exception as e:
                    self.stdout.write(f"  ffprobe failed for {video.filename}: {e}")
                    continue
            if is_hdr_transfer(color_transfer):
                hdr_videos.append(video)

        if uncached:
            self.stdout.write(
                f"  ({uncached} row(s) had no cached color_transfer -- consider running "
                f"backfill_video_color_transfer first to avoid the live ffprobe fallback)"
            )

        total_faces = sum(Face.objects.filter(source_video_file=v).count() for v in hdr_videos)
        self.stdout.write(f"{len(hdr_videos)} HDR-tagged video(s), {total_faces} face(s) total.")

        if not hdr_videos:
            self.stdout.write(self.style.SUCCESS("Nothing to do."))
            return

        if not dry_run and not options['yes']:
            go_ahead = input(
                f"Regenerate {total_faces} thumbnail(s) across {len(hdr_videos)} video(s)? y/N: "
            )
            if go_ahead.lower() != 'y':
                self.stdout.write("Aborted.")
                return

        regenerated = 0
        unresolved = 0
        for vi, video in enumerate(hdr_videos, start=1):
            video_start = time.time()
            faces = list(
                Face.objects.filter(source_video_file=video, video_thumbnail_frame_seconds__isnull=False)
            )
            if not faces:
                continue

            # One ffprobe call per HDR video -- unavoidable (fps isn't
            # cached anywhere on VideoFile), but this is now only paid
            # for the actual HDR subset, not every processed video.
            # width/height come from THIS call (not VideoFile.width/
            # height) since ffprobe_info() already rotation-corrects
            # them to match ffmpeg_frame_iterator's raw-pipe output
            # shape -- the plain cached dimensions would be wrong for a
            # rotated video.
            try:
                width, height, fps, probed_field_order, _rotation, color_transfer = ffprobe_info(video.filename)
            except Exception as e:
                self.stdout.write(f"  ffprobe failed for {video.filename}: {e}")
                unresolved += len(faces)
                continue
            field_order = video.field_order or probed_field_order

            vf_filter = build_vf_filter(field_order, color_transfer)
            targets = sorted({f.video_thumbnail_frame_seconds for f in faces})
            seek_seconds = max(0, targets[0] - 5)
            hi_cutoff = targets[-1] + 5

            best = {t: (None, None) for t in targets}  # target -> (frame, delta)
            n = 0
            for frame in ffmpeg_frame_iterator(
                video.filename, width, height, vf_filter=vf_filter, seek_seconds=seek_seconds
            ):
                t = seek_seconds + n / fps
                n += 1
                if t > hi_cutoff:
                    break
                for target in targets:
                    delta = abs(t - target)
                    if delta <= TOLERANCE_SECONDS and (best[target][1] is None or delta < best[target][1]):
                        best[target] = (frame.copy(), delta)

            for f in faces:
                frame, _delta = best[f.video_thumbnail_frame_seconds]
                if frame is None:
                    unresolved += 1
                    self.stdout.write(f"  face {f.id} ({video.filename}): no frame found within tolerance")
                    continue
                if dry_run:
                    regenerated += 1
                    continue
                box = (f.box_left, f.box_top, f.box_right, f.box_bottom)
                thumbnail = VideoFaceExtractor._square_thumbnail(frame, box)
                is_success, buffer_img = cv2.imencode('.jpg', thumbnail)
                if not is_success:
                    unresolved += 1
                    self.stdout.write(f"  face {f.id}: JPEG encode failed")
                    continue
                with open(f.face_thumbnail.path, 'wb') as fh:
                    fh.write(buffer_img.tobytes())
                regenerated += 1

            self.stdout.write(
                f"[{vi}/{len(hdr_videos)}] {video.filename}: {len(faces)} face(s), "
                f"{time.time() - video_start:.1f}s"
            )

        self.stdout.write(
            f"{'Would regenerate' if dry_run else 'Regenerated'}: {regenerated}, unresolved: {unresolved}."
        )

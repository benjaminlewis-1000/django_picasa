import cv2
import numpy as np

from django.core.management.base import BaseCommand

from face_manager.models import Face
from video_face_pipeline import VideoFaceExtractor, ffmpeg_frame_iterator, ffprobe_info, sample_stride


def _decode_stored_thumbnail(face):
    face.face_thumbnail.open('rb')
    try:
        raw = face.face_thumbnail.read()
    finally:
        face.face_thumbnail.close()
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _mean_abs_diff(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


class Command(BaseCommand):
    help = (
        "One-time backfill for Face.video_thumbnail_frame_seconds on video-"
        "sourced rows created before that field existed. Cheaper than a full "
        "pipeline re-run: for each affected face, re-decodes only the sampled "
        "frames within its own [video_first_timestamp_seconds, "
        "video_last_timestamp_seconds] window, reproduces _square_thumbnail() "
        "at the face's already-stored box for each candidate, and picks "
        "whichever candidate's pixels most closely match the already-saved "
        "face_thumbnail JPEG (mean-abs-pixel-diff, tolerating JPEG lossy "
        "compression noise but not a genuinely different frame). No re-"
        "detection/re-classification -- box/embedding/identity are untouched, "
        "this only recovers which frame the existing thumbnail came from."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument(
            '--match-threshold', type=float, default=10.0,
            help='Max mean-abs-pixel-diff (0-255 scale) to accept a match.',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        threshold = options['match_threshold']

        faces = list(
            Face.objects.filter(
                source_video_file__isnull=False,
                video_thumbnail_frame_seconds__isnull=True,
            ).select_related('source_video_file')
        )

        by_video = {}
        for f in faces:
            by_video.setdefault(f.source_video_file_id, []).append(f)

        self.stdout.write(f'{len(faces)} faces across {len(by_video)} videos missing a thumbnail timestamp.')

        resolved = 0
        unresolved = 0
        for video_id, video_faces in by_video.items():
            video = video_faces[0].source_video_file
            spans = [
                (f.video_first_timestamp_seconds, f.video_last_timestamp_seconds)
                for f in video_faces
                if f.video_first_timestamp_seconds is not None and f.video_last_timestamp_seconds is not None
            ]
            if not spans:
                unresolved += len(video_faces)
                continue

            try:
                width, height, fps, field_order = ffprobe_info(video.filename)
            except Exception as e:
                self.stdout.write(f'  ffprobe failed for {video.filename}: {e}')
                unresolved += len(video_faces)
                continue

            vf_filter = 'yadif=0' if field_order not in ('progressive', 'unknown') else None
            stride = sample_stride(fps)
            lo = min(s[0] for s in spans)
            hi = max(s[1] for s in spans)
            lo_frame = max(0, int(lo * fps) - stride)
            hi_frame = int(hi * fps) + stride

            frames = {}
            idx = 0
            for frame in ffmpeg_frame_iterator(video.filename, width, height, vf_filter=vf_filter):
                if idx > hi_frame:
                    break
                if idx >= lo_frame and (idx - lo_frame) % stride == 0:
                    frames[idx] = frame.copy()
                idx += 1

            if not frames:
                self.stdout.write(f'  no frames decoded for {video.filename} (span {lo}-{hi}s)')
                unresolved += len(video_faces)
                continue

            for f in video_faces:
                stored = _decode_stored_thumbnail(f)
                box = (f.box_left, f.box_top, f.box_right, f.box_bottom)
                best_idx, best_diff = None, None
                for fidx, frame in frames.items():
                    candidate = VideoFaceExtractor._square_thumbnail(frame, box)
                    diff = _mean_abs_diff(candidate, stored)
                    if diff is not None and (best_diff is None or diff < best_diff):
                        best_idx, best_diff = fidx, diff

                if best_idx is not None and best_diff <= threshold:
                    ts = best_idx / fps
                    resolved += 1
                    if not dry_run:
                        f.video_thumbnail_frame_seconds = ts
                        f.save(update_fields=['video_thumbnail_frame_seconds'])
                else:
                    unresolved += 1
                    self.stdout.write(
                        f'  face {f.id} ({video.filename}): no match within threshold '
                        f'(best_diff={best_diff})'
                    )

        self.stdout.write(
            f"{'Would resolve' if dry_run else 'Resolved'}: {resolved}, unresolved: {unresolved}."
        )

import cv2
import numpy as np
import subprocess

from django.core.management.base import BaseCommand

from face_manager.models import Face
from video_face_pipeline import VideoFaceExtractor, ffprobe_info


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


def _extract_frame_at(path, timestamp, width, height, vf_filter=None):
    frame_size = width * height * 3
    cmd = ['ffmpeg', '-v', 'error', '-i', path, '-ss', str(timestamp)]
    if vf_filter:
        cmd += ['-vf', vf_filter]
    cmd += ['-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60)
    if len(result.stdout) < frame_size:
        return None
    return np.frombuffer(result.stdout[:frame_size], dtype=np.uint8).reshape(height, width, 3)


class Command(BaseCommand):
    help = (
        "Re-validates Face.video_thumbnail_frame_seconds values already "
        "written by backfill_video_thumbnail_timestamps, resetting any "
        "whose stored timestamp doesn't actually pixel-match its saved "
        "face_thumbnail (above --match-threshold) back to NULL. One-off "
        "cleanup needed after that command briefly defaulted to accepting "
        "the closest available candidate unconditionally, which could "
        "write a confidently-wrong timestamp for a face whose tracked "
        "[first,last] span has a real discontinuity (left frame, "
        "reappeared later) -- the closest 'match' in that gap can still "
        "be a poor one. Re-run backfill_video_thumbnail_timestamps "
        "afterward (with its real threshold restored) to properly "
        "re-resolve anything reset here."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--filenames-file', required=True,
            help='Text file, one video filename per line, to scope validation to.',
        )
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--match-threshold', type=float, default=10.0)

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        threshold = options['match_threshold']
        with open(options['filenames_file']) as fh:
            filenames = [line.strip() for line in fh if line.strip()]

        faces = Face.objects.filter(
            source_video_file__filename__in=filenames,
            video_thumbnail_frame_seconds__isnull=False,
        ).select_related('source_video_file')

        by_video = {}
        for f in faces:
            by_video.setdefault(f.source_video_file_id, []).append(f)

        self.stdout.write(f'{faces.count()} faces across {len(by_video)} videos to validate.')

        checked = 0
        reset = 0
        for video_id, video_faces in by_video.items():
            video = video_faces[0].source_video_file
            try:
                width, height, fps, field_order = ffprobe_info(video.filename)
            except Exception as e:
                self.stdout.write(f'  ffprobe failed for {video.filename}: {e}')
                continue
            vf_filter = 'yadif=0' if field_order not in ('progressive', 'unknown') else None

            for f in video_faces:
                checked += 1
                frame = _extract_frame_at(
                    video.filename, f.video_thumbnail_frame_seconds, width, height, vf_filter
                )
                if frame is None:
                    reset += 1
                    self.stdout.write(f'  face {f.id} ({video.filename}): re-extract failed -- resetting to NULL')
                    if not dry_run:
                        f.video_thumbnail_frame_seconds = None
                        f.save(update_fields=['video_thumbnail_frame_seconds'])
                    continue

                box = (f.box_left, f.box_top, f.box_right, f.box_bottom)
                candidate = VideoFaceExtractor._square_thumbnail(frame, box)
                stored = _decode_stored_thumbnail(f)
                diff = _mean_abs_diff(candidate, stored)

                if diff is None or diff > threshold:
                    reset += 1
                    self.stdout.write(
                        f'  face {f.id} ({video.filename}): diff={diff} -- resetting to NULL'
                    )
                    if not dry_run:
                        f.video_thumbnail_frame_seconds = None
                        f.save(update_fields=['video_thumbnail_frame_seconds'])

        self.stdout.write(
            f"Checked {checked}, {'would reset' if dry_run else 'reset'} {reset}."
        )

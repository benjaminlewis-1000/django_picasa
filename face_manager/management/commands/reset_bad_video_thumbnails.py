import cv2
import numpy as np
import subprocess

from django.core.management.base import BaseCommand

from face_manager.models import Face
from filepopulator.models import VideoFile
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


def _extract_frame_fast(path, timestamp, field_order):
    """Same fast-seek approach as api/views.py's _extract_video_face_frame
    -- -ss placed before -i, approximate but not scaling with video
    length. Returns a raw BGR frame array, or None on failure."""
    cmd = ['ffmpeg', '-v', 'error', '-ss', str(timestamp), '-i', path]
    if field_order not in ('progressive', 'unknown'):
        cmd += ['-vf', 'yadif=0']
    cmd += ['-frames:v', '1', '-f', 'image2', '-q:v', '2', 'pipe:1']
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    arr = np.frombuffer(result.stdout, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


class Command(BaseCommand):
    help = (
        "Full sweep: re-checks every video-sourced Face's stored "
        "video_thumbnail_frame_seconds against its saved face_thumbnail "
        "(re-extracts the frame at that exact timestamp and pixel-diffs "
        "against the stored thumbnail). Faces that don't hold up -- most "
        "likely a bad backfill_video_thumbnail_timestamps guess on a "
        "narrow-span face, where the comparison window had too few "
        "candidates and a wrong frame's background happened to still "
        "score well against a mostly-background crop -- have their WHOLE "
        "VIDEO's faces deleted and isProcessed reset to False, so the "
        "real pipeline reprocesses it from scratch (accurate detections "
        "and timestamps, not a reconstruction). Only the flagged video's "
        "faces are touched, not the whole library."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--match-threshold', type=float, default=10.0)

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        threshold = options['match_threshold']

        faces = Face.objects.filter(
            source_video_file__isnull=False,
            video_thumbnail_frame_seconds__isnull=False,
        ).select_related('source_video_file')

        total = faces.count()
        self.stdout.write(f'{total} video-sourced faces to check.')

        bad_video_ids = set()
        checked = 0
        flagged = 0
        for f in faces.iterator():
            checked += 1
            video = f.source_video_file
            field_order = video.field_order
            if field_order is None:
                try:
                    _, _, _, field_order = ffprobe_info(video.filename)
                except Exception:
                    field_order = 'unknown'

            frame = _extract_frame_fast(video.filename, f.video_thumbnail_frame_seconds, field_order)
            if frame is None:
                flagged += 1
                bad_video_ids.add(video.id)
                self.stdout.write(f'  face {f.id} ({video.filename}): re-extract failed')
                continue

            box = (f.box_left, f.box_top, f.box_right, f.box_bottom)
            candidate = VideoFaceExtractor._square_thumbnail(frame, box)
            stored = _decode_stored_thumbnail(f)
            diff = _mean_abs_diff(candidate, stored)

            if diff is None or diff > threshold:
                flagged += 1
                bad_video_ids.add(video.id)
                self.stdout.write(f'  face {f.id} ({video.filename}): diff={diff}')

            if checked % 200 == 0:
                self.stdout.write(f'  ... {checked}/{total} checked, {flagged} flagged so far')

        self.stdout.write(
            f'Checked {checked} faces. {flagged} flagged, across {len(bad_video_ids)} video(s).'
        )

        if not bad_video_ids:
            return

        if dry_run:
            self.stdout.write(self.style.WARNING(f'Dry run -- would reset {len(bad_video_ids)} video(s).'))
            return

        n_deleted = 0
        for f in Face.objects.filter(source_video_file_id__in=bad_video_ids):
            f.delete()
            n_deleted += 1
        n_reset = VideoFile.objects.filter(id__in=bad_video_ids).update(isProcessed=False)
        self.stdout.write(
            self.style.SUCCESS(f'Deleted {n_deleted} face(s), reset {n_reset} video(s) to unprocessed.')
        )

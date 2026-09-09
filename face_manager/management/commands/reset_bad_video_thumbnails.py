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


def _extract_frame(path, timestamp, field_order, accurate=False):
    """accurate=False: -ss placed before -i, fast/approximate (can land a
    few frames off the true target -- fine for a "roughly this moment"
    viewer, NOT precise enough on its own to decide whether to delete
    real data). accurate=True: -ss placed after -i, frame-exact but
    decodes the whole video from the start to reach the target -- only
    used as a slower confirmation check on faces the fast pass already
    flagged, not on all of them, to keep the sweep's overall cost down.
    Returns a raw BGR frame array, or None on failure."""
    if accurate:
        cmd = ['ffmpeg', '-v', 'error', '-i', path, '-ss', str(timestamp)]
    else:
        cmd = ['ffmpeg', '-v', 'error', '-ss', str(timestamp), '-i', path]
    if field_order not in ('progressive', 'unknown'):
        cmd += ['-vf', 'yadif=0']
    cmd += ['-frames:v', '1', '-f', 'image2', '-q:v', '2', 'pipe:1']
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=300 if accurate else 30,
        )
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
                    _, _, _, field_order, _rotation = ffprobe_info(video.filename)
                except Exception:
                    field_order = 'unknown'

            box = (f.box_left, f.box_top, f.box_right, f.box_bottom)
            stored = _decode_stored_thumbnail(f)

            frame = _extract_frame(video.filename, f.video_thumbnail_frame_seconds, field_order)
            diff = None
            if frame is not None:
                candidate = VideoFaceExtractor._square_thumbnail(frame, box)
                diff = _mean_abs_diff(candidate, stored)

            if diff is not None and diff <= threshold:
                continue  # fast check already confirms a good match

            # Fast check failed or was inconclusive -- confirm with a
            # slower, frame-exact re-check before trusting it, since a
            # fast approximate seek landing a few frames off the true
            # target can make even genuinely-correct data look wrong.
            frame_accurate = _extract_frame(
                video.filename, f.video_thumbnail_frame_seconds, field_order, accurate=True
            )
            diff_accurate = None
            if frame_accurate is not None:
                candidate_accurate = VideoFaceExtractor._square_thumbnail(frame_accurate, box)
                diff_accurate = _mean_abs_diff(candidate_accurate, stored)

            if diff_accurate is None or diff_accurate > threshold:
                flagged += 1
                bad_video_ids.add(video.id)
                self.stdout.write(
                    f'  face {f.id} ({video.filename}): fast_diff={diff} accurate_diff={diff_accurate}'
                )

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

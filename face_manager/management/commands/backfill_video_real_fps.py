from django.core.management.base import BaseCommand

from face_manager.models import Face
from filepopulator.models import VideoFile
from video_face_pipeline import VideoFaceExtractor, ffmpeg_frame_iterator, ffprobe_info


def _count_real_frames(video_path, width, height):
    """Decode-only frame count, no detection/embedding -- much cheaper
    than the full pipeline, but still a real full decode (no shortcut
    exists to know a video's true frame count without actually reading
    every frame)."""
    count = 0
    for _ in ffmpeg_frame_iterator(video_path, width, height):
        count += 1
    return count


class Command(BaseCommand):
    help = (
        "One-time sweep over already-processed videos: for each, decodes "
        "the file to count its REAL frame total (no detection/embedding, "
        "just a count) and derives real_fps = real_frame_count / "
        "duration_seconds. Compares against ffprobe's own fps (whichever "
        "of r_frame_rate/avg_frame_rate ffprobe_info picked) -- if they "
        "differ by more than --fps-threshold, every one of that video's "
        "Face rows had its video_thumbnail_frame_seconds/"
        "video_first_timestamp_seconds/video_last_timestamp_seconds "
        "computed with the wrong fps at processing time (confirmed "
        "2026-09-09 on a real file where ffprobe's own frame-count "
        "metadata undercounted the true decoded frame count by exactly "
        "4x -- a genuine per-file metadata quirk, not a 'wrong ffprobe "
        "field selected' bug). Rescales those three fields by "
        "(ffprobe_fps / real_fps) -- pure arithmetic on the SAME "
        "underlying raw frame index, no redetection needed, since boxes/"
        "kps/embeddings/thumbnails are computed directly from decoded "
        "pixel content and are unaffected by this bug -- then clamps to "
        "duration_seconds as a final safety net. Videos where the two "
        "fps values already agree are left completely untouched (the "
        "common case): only real_frame_count needs a full decode either "
        "way, so this is a real, non-trivial per-video cost, but is a "
        "one-time sweep -- anything processed going forward already gets "
        "the correct value at write time (video_face_pipeline.py's own "
        "real_fps computation)."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--fps-threshold', type=float, default=0.05,
                             help='Relative difference (e.g. 0.05 = 5%%) above which a video is considered affected.')
        parser.add_argument('--video-id', type=int, default=None,
                             help='Restrict to a single VideoFile id (for testing/spot-checking).')

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        threshold = options['fps_threshold']
        only_id = options['video_id']

        videos = VideoFile.objects.filter(
            isProcessed=True, duration_seconds__isnull=False,
        ).filter(face__isnull=False).distinct()
        if only_id is not None:
            videos = videos.filter(id=only_id)

        total = videos.count()
        self.stdout.write(f'{total} already-processed video(s) with faces to check.')

        checked = 0
        affected = 0
        faces_rescaled = 0
        for video in videos.iterator():
            checked += 1
            try:
                width, height, ffprobe_fps, _field_order, _rotation = ffprobe_info(video.filename)
                real_frame_count = _count_real_frames(video.filename, width, height)
            except Exception as e:
                self.stdout.write(f'  video {video.id} ({video.filename}): failed to check -- {e}')
                continue

            duration = video.duration_seconds
            if not duration or real_frame_count == 0:
                continue
            real_fps = real_frame_count / duration

            if abs(real_fps - ffprobe_fps) / ffprobe_fps <= threshold:
                if checked % 20 == 0:
                    self.stdout.write(f'  ... {checked}/{total} checked, {affected} affected so far')
                continue

            affected += 1
            ratio = ffprobe_fps / real_fps
            self.stdout.write(
                f'  video {video.id} ({video.filename}): ffprobe_fps={ffprobe_fps:.3f} '
                f'real_fps={real_fps:.3f} ({real_frame_count} real frames / {duration:.2f}s) '
                f'-- rescaling by {ratio:.4f}'
            )

            if dry_run:
                continue

            for face in Face.objects.filter(source_video_file=video):
                changed = False
                for field, label in [
                    ('video_thumbnail_frame_seconds', 'video_thumbnail_frame_seconds'),
                    ('video_first_timestamp_seconds', 'video_first_timestamp_seconds'),
                    ('video_last_timestamp_seconds', 'video_last_timestamp_seconds'),
                ]:
                    value = getattr(face, field)
                    if value is None:
                        continue
                    corrected = VideoFaceExtractor._clamp_to_duration(video, value * ratio, label)
                    setattr(face, field, corrected)
                    changed = True
                if changed:
                    face.save(update_fields=[
                        'video_thumbnail_frame_seconds',
                        'video_first_timestamp_seconds',
                        'video_last_timestamp_seconds',
                    ])
                    faces_rescaled += 1

        self.stdout.write(
            f'Checked {checked} video(s). {affected} affected.'
            + ('' if dry_run else f' {faces_rescaled} face(s) rescaled.')
        )
        if dry_run and affected:
            self.stdout.write(self.style.WARNING('Dry run -- no changes written.'))
        elif faces_rescaled:
            self.stdout.write(self.style.SUCCESS(f'Rescaled {faces_rescaled} face(s) across {affected} video(s).'))

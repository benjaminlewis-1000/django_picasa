from django.core.management.base import BaseCommand

from face_manager.models import Face
from video_face_pipeline import (
    VideoFaceExtractor, decode_face_thumbnail, extract_frame_near_timestamp, ffprobe_info,
)

IOU_MATCH_THRESHOLD = 0.3


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Command(BaseCommand):
    help = (
        "One-time backfill for Face.det_score on video-sourced faces that "
        "predate the video pipeline capturing it (~90% of the historical "
        "video population as of 2026-10-13). For each face, re-extracts "
        "the exact frame its thumbnail/box originally came from "
        "(extract_frame_near_timestamp, matched against the already-"
        "stored thumbnail -- the same mechanism the on-demand accurate "
        "frame viewer uses, so this doesn't depend on getting any single "
        "seek/index computation exactly right), redetects in a padded "
        "crop around the stored box, and verifies the result actually "
        "matches (IOU >= 0.3) before trusting its det_score. Real video "
        "decode cost per face -- much slower than the image-side "
        "backfill_det_score, which only redetects on a static photo -- "
        "so this is a separate, standalone command."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--limit', type=int, default=None)
        parser.add_argument(
            '--ignore-only', action='store_true',
            help='Only process faces currently declared .ignore/.realignore -- lets a '
                 'reclassification pass follow much sooner than waiting on the full '
                 'backlog (which also includes faces already correctly declared to a '
                 'real person, and ones still genuinely unassigned).',
        )
        parser.add_argument(
            '--pad-pct', type=float, default=None,
            help='Override _redetect_in_crop\'s default crop padding (REDETECT_PAD_PCT, '
                 '0.6) -- a larger crop gives a real face near the stored box\'s edge '
                 'another chance to be found, for a retry pass on faces the default '
                 'padding already missed.',
        )
        parser.add_argument(
            '--det-thresh', type=float, default=None,
            help='Override the detector\'s own det_thresh (default 0.5 -- note this is '
                 'a hard cutoff baked into the model itself: a genuine face below it is '
                 'discarded before this command ever sees it, not filtered by anything '
                 'here). Lowering it for a retry pass recovers real, lower-confidence '
                 'faces missed the first time -- they still only pass '
                 'VIDEO_DET_SCORE_FLOOR (0.5) if their real score clears it, so this '
                 'cannot inflate a score, only reveal one that was being silently '
                 'discarded.',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        limit = options['limit']
        pad_pct = options['pad_pct']
        det_thresh = options['det_thresh']

        qs = Face.objects.filter(
            source_video_file__isnull=False, det_score__isnull=True,
            video_thumbnail_frame_seconds__isnull=False,
        ).select_related('source_video_file').order_by('source_video_file_id')
        if options['ignore_only']:
            qs = qs.filter(declared_name__person_name__in=['.ignore', '.realignore'])
        if limit:
            qs = qs[:limit]

        self.stdout.write("Loading VideoFaceExtractor (real model load)...")
        extractor = VideoFaceExtractor()
        if det_thresh is not None:
            self.stdout.write(f"Overriding det_thresh: {extractor.det10g_model.det_thresh} -> {det_thresh}")
            extractor.det10g_model.det_thresh = det_thresh

        total = matched = no_frame = no_detect = iou_mismatch = 0
        ffprobe_cache = {}

        for face in qs.iterator(chunk_size=200):
            total += 1
            video = face.source_video_file

            try:
                if video.id not in ffprobe_cache:
                    ffprobe_cache[video.id] = ffprobe_info(video.filename)
                width, height, avg_fps, probed_field_order, rotation, color_transfer, sar_scale_width = (
                    ffprobe_cache[video.id]
                )
            except Exception:
                no_frame += 1
                continue

            field_order = video.field_order or probed_field_order
            box = (face.box_left, face.box_top, face.box_right, face.box_bottom)

            reference_bgr = decode_face_thumbnail(face)
            if reference_bgr is None:
                no_frame += 1
                continue

            try:
                frame = extract_frame_near_timestamp(
                    video.filename, face.video_thumbnail_frame_seconds, avg_fps,
                    field_order, rotation, box, reference_bgr,
                    width=(sar_scale_width or width), height=height,
                    color_transfer=color_transfer, sar_scale_width=sar_scale_width,
                )
            except Exception:
                frame = None
            if frame is None:
                no_frame += 1
                continue

            if pad_pct is not None:
                result = extractor._redetect_in_crop(frame, box, pad_pct=pad_pct)
            else:
                result = extractor._redetect_in_crop(frame, box)
            if result is None:
                no_detect += 1
                continue
            redetected_box, _kps, det_score = result
            if _iou(box, redetected_box) < IOU_MATCH_THRESHOLD:
                iou_mismatch += 1
                continue

            matched += 1
            if not dry_run:
                Face.objects.filter(pk=face.pk).update(det_score=det_score)

            if total % 500 == 0:
                self.stdout.write(
                    f"[{total}] matched={matched} no_frame={no_frame} "
                    f"no_detect={no_detect} iou_mismatch={iou_mismatch}"
                )

        self.stdout.write(
            f"Done. total={total} matched={matched} "
            f"({'would set' if dry_run else 'set'} det_score) "
            f"no_frame={no_frame} no_detect={no_detect} "
            f"iou_mismatch={iou_mismatch} dry_run={dry_run}."
        )

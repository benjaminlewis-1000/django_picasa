import signal
import time
from collections import defaultdict

import torch
import torchvision.ops.boxes as bops
from django.conf import settings
from django.core.management.base import BaseCommand

import common
from face_extract_encode import FaceExtractor
from face_manager.models import Face, Person

# Per-image hard timeout -- a real production hang was found running this
# command at scale: one image drove CPU to 1400%+ with ZERO progress for
# minutes (a real query, not a slow-but-progressing one -- confirmed by
# watching the actual det_score count in the DB stop moving entirely
# while the process kept burning CPU). No specific root cause was chased
# (the image itself wasn't identified before this guard was added), but
# this is the exact same class of risk face_manager/tasks.py's own
# PER_VIDEO_TIMEOUT_SECONDS/_VideoProcessingTimeout guards against for
# video processing -- a single pathological input (huge resolution,
# some ONNX/NMS edge case) can otherwise hang an entire long-running
# batch job indefinitely with nothing to show for it. A single-pass
# detection call on a normal photo takes well under a second in
# practice; 30s is a generous margin, not a tight budget.
PER_IMAGE_TIMEOUT_SECONDS = 30


class _ImageTimeout(Exception):
    pass


def _raise_image_timeout(signum, frame):
    raise _ImageTimeout(f"Image processing exceeded {PER_IMAGE_TIMEOUT_SECONDS}s")


class Command(BaseCommand):
    help = (
        "One-time backfill of Face.det_score (insightface's own detection "
        "confidence, 0-1) for the unlabeled poss_ident1=.ignore population "
        "-- this field never existed before 2026-09-18, so every existing "
        "face has it NULL. Investigated first against a real sample before "
        "committing to this: validated real faces average ~0.85, confirmed-"
        "wrong .ignore candidates ~0.58, with a real, mostly non-overlapping "
        "gap between them -- see CLAUDE.md. Re-detects each face's source "
        "image with the same PyramidalDetector the live pipeline uses, "
        "matches results back to each face's stored box by IOU, and stores "
        "the matched detection's own det_score. Also backfills Face.kps "
        "(the 5-point landmarks) from the same matched detection -- free, "
        "since the redetection already happened for det_score; lets a "
        "future reencode_missing_faces() run replay the exact alignment "
        "for these faces instead of falling back to its own crop-based "
        "redetect path. Groups faces by source image first, so a photo "
        "with several .ignore faces only pays for one decode+detect pass, "
        "not one per face. A face whose box no longer matches any "
        "redetection is left NULL (both fields) and counted separately -- "
        "not retried automatically, since this is a one-time sweep, not a "
        "scheduled task."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--limit', type=int, default=None,
                             help='Restrict to the first N source images (for testing).')
        parser.add_argument('--single-pass', action='store_true',
                             help=(
                                 "Skip the PyramidalDetector's 3x3 tiled sub-detection pass "
                                 "(cut_list=[1,3] -> [1]) -- roughly 10x fewer detection calls "
                                 "per image, at the cost of missing some small/edge faces the "
                                 "tiled pass would have caught (those land in 'unmatched' "
                                 "instead of getting a score). Measured real throughput on the "
                                 "full cut_list: ~0.10 img/s, a multi-day run for the full "
                                 "~73k-face population -- --single-pass is the practical "
                                 "default for a first sweep; faces left unmatched can be "
                                 "revisited with a second, non---single-pass run later (the "
                                 "same query naturally targets only what's still NULL)."
                             ))

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        limit = options['limit']
        single_pass = options['single_pass']

        ignore_person = Person.objects.get(person_name=settings.SOFT_IGNORE_NAME)
        faces = Face.objects.filter(
            poss_ident1=ignore_person, det_score__isnull=True, source_image_file__isnull=False,
        ).select_related('source_image_file')

        by_image = defaultdict(list)
        for face in faces:
            by_image[face.source_image_file].append(face)

        image_list = list(by_image.items())
        if limit is not None:
            image_list = image_list[:limit]

        total_faces = sum(len(fs) for _, fs in image_list)
        self.stdout.write(
            f"{len(image_list)} distinct source images, {total_faces} faces to backfill "
            f"{'(dry run)' if dry_run else ''}"
        )

        extractor = FaceExtractor()
        if single_pass:
            extractor.app.cut_list = [1]
        matched = unmatched = decode_failed = timed_out = 0
        t0 = time.time()

        for idx, (img_obj, image_faces) in enumerate(image_list):
            try:
                signal.signal(signal.SIGALRM, _raise_image_timeout)
                signal.alarm(PER_IMAGE_TIMEOUT_SECONDS)
                try:
                    img_numpy = common.open_img_oriented(img_obj.filename, as_numpy=True)
                    if img_numpy is None:
                        decode_failed += len(image_faces)
                        continue
                    detections = extractor.app.get(img_numpy)
                finally:
                    signal.alarm(0)
            except _ImageTimeout:
                self.stdout.write(f"  TIMEOUT on {img_obj.filename} -- skipping")
                timed_out += len(image_faces)
                continue
            except Exception:
                decode_failed += len(image_faces)
                continue

            det_boxes = torch.tensor(
                [d.bbox.tolist() for d in detections], dtype=torch.float32
            ) if detections else torch.zeros((0, 4))

            for face in image_faces:
                stored_box = torch.tensor(
                    [[face.box_left, face.box_top, face.box_right, face.box_bottom]],
                    dtype=torch.float32,
                )
                if len(detections) == 0:
                    unmatched += 1
                    continue
                ious = bops.box_iou(stored_box, det_boxes)[0]
                best_idx = int(torch.argmax(ious))
                if float(ious[best_idx]) < 0.3:
                    unmatched += 1
                    continue
                matched += 1
                if not dry_run:
                    best_det = detections[best_idx]
                    face.det_score = float(best_det.det_score)
                    face.kps = FaceExtractor._flatten_kps(best_det['kps'])
                    # update_fields here is also what keeps Face.save()
                    # from opening (and never closing) the thumbnail file
                    # on every one of these thousands of calls -- see its
                    # own comment. A real crash (OSError: Too many open
                    # files, ~600/38489 images) before that fix existed.
                    face.save(update_fields=['det_score', 'kps'])

            if (idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                remaining = len(image_list) - (idx + 1)
                eta_min = remaining / rate / 60 if rate else float('inf')
                self.stdout.write(
                    f"  ...{idx+1}/{len(image_list)} images ({rate:.2f} img/s, "
                    f"ETA {eta_min:.0f}min) -- matched={matched} unmatched={unmatched} "
                    f"decode_failed={decode_failed} timed_out={timed_out}"
                )

        elapsed = time.time() - t0
        self.stdout.write(
            f"DONE: {len(image_list)} images, {elapsed:.0f}s. "
            f"matched={matched} unmatched={unmatched} decode_failed={decode_failed} "
            f"timed_out={timed_out}"
        )

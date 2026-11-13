#! /usr/bin/env python

# Fixes the real, confirmed bug found 2026-11-12 (see CLAUDE.md, "face
# 875939 thumbnail/box investigation"): a legacy (kps IS NULL --
# detected before the Aug-2026 kps migration) population of Face rows
# has box coordinates stored in a STALE coordinate space relative to
# the source image's CURRENT width/height. The source image was
# apparently re-ingested/replaced at a different (usually higher)
# resolution at some point without the pre-existing faces' boxes being
# rescaled. Confirmed via direct visual inspection (face 875939 and
# others, across many unrelated images/years -- not a single bulk
# event): a crop at the CURRENT stored box coordinates shows background
# only; the same box scaled by the image's own implied ratio lands
# exactly on the real face, matching the already-correct stored
# thumbnail. Full population scan: 12,861 such faces exist, 7,920
# (61.6%) show a >5% scale mismatch. The MODERN (kps-having) population
# is confirmed essentially unaffected (65 of 766,508, ~0.0085%) -- this
# command's target population is deliberately scoped to kps IS NULL.
#
# Fix is deliberately narrow, per the user's explicit call (no reason
# to touch anything downstream of the box, and recomputing encodings
# isn't needed since kps is never touched): ONLY box_left/top/right/
# bottom, relative_size (recomputed as a consistency check, not new
# information), and face_thumbnail (regenerated from the real image at
# the corrected box) are ever written. kps, face_encoding_512(_hv),
# declared_name, validated, poss_identN, det_score are never touched.
#
# Safety, built in response to the user's explicit hesitance this
# session given a recent kps-backfill incident that reportedly went
# wrong: every candidate's rescaled box is cropped from the REAL
# current image and pixel-diffed against its own EXISTING (already-
# correct) stored thumbnail before anything is written -- a face that
# doesn't match closely is skipped and reported, never guessed at or
# force-written. Faces on the same image are also cross-checked against
# each other's own implied scale ratio first: the investigation found
# the ratio is consistent across every face on one image (it varies
# image-to-image, not face-to-face) -- if an image's own faces disagree
# with each other, something about that image is unusual, and the
# whole image is skipped/flagged rather than guessing which face is
# right.

import os
import time
from collections import defaultdict

import cv2
import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand

import common
from face_manager.models import Face
from video_face_pipeline import decode_face_thumbnail

# Below this, a face's own implied ratio is close enough to 1.0 that no
# rescale is needed at all -- matches the >5% mismatch threshold used
# throughout the originating investigation.
RATIO_NOOP_TOLERANCE = 0.05

# How far two faces' own implied ratios on the SAME image may disagree
# from each other before this command refuses to guess and skips the
# whole image instead.
RATIO_AGREEMENT_TOLERANCE = 0.05

# How far the recomputed relative_size (from the new, rescaled box) may
# differ from the ORIGINAL stored relative_size before the rescale is
# treated as internally inconsistent and refused. These should be very
# close by construction (the ratio is derived FROM this relationship),
# so a real mismatch here means something unexpected is going on.
RELATIVE_SIZE_CONSISTENCY_TOLERANCE = 0.10

# Mean absolute per-pixel BGR difference (0-255 scale) between the
# rescaled-box crop and the face's existing stored thumbnail, at or
# under which the rescale is trusted enough to write. Validate against
# real examples via --dry-run before trusting this default at scale.
DEFAULT_DIFF_THRESHOLD = 15.0

# Matches FaceExtractor.thumbnail_extension_mult -- duplicated here
# rather than instantiating a whole FaceExtractor (which loads the full
# detector/recognition model set) just to read one plain attribute this
# command has no other use for.
THUMBNAIL_EXTENSION_MULT = 2


def _square_crop_and_resize(img_numpy, box):
    """Rebuilds the exact crop/pad/resize FaceExtractor.get_square_face_img()
    produces, from a plain (left, top, right, bottom) box tuple instead
    of requiring a real insightface Face wrapper -- this command has no
    detection to run, only an existing box to re-crop. Returns None if
    the box is degenerate once clamped to the frame.

    Deliberately keeps get_square_face_img()'s own BGR2RGB conversion
    immediately before resize, even though it's a real mislabeling (the
    source array from common.open_img_oriented() is already RGB, so
    this call actually swaps it back to BGR) -- every existing stored
    thumbnail in this codebase was generated with that same mislabeling
    baked in, so reproducing it here is what makes the pixel-diff
    safety check below an apples-to-apples comparison, not a new bug.
    """
    img_h, img_w = img_numpy.shape[:2]
    bb_l, bb_t, bb_r, bb_b = (int(round(v)) for v in box)
    bb_l = max(0, bb_l)
    bb_t = max(0, bb_t)
    bb_r = min(bb_r, img_w)
    bb_b = min(bb_b, img_h)
    if bb_r <= bb_l or bb_b <= bb_t:
        return None

    face_h = bb_b - bb_t
    face_w = bb_r - bb_l
    face_center_vert = (bb_b - bb_t) // 2 + bb_t
    face_center_horiz = (bb_r - bb_l) // 2 + bb_l

    vert_margin = min(face_center_vert, img_h - face_center_vert)
    horiz_margin = min(face_center_horiz, img_w - face_center_horiz)
    if vert_margin <= 0 or horiz_margin <= 0:
        return None

    detection_max_dim = max(face_h, face_w)
    max_allowable_margin = min(vert_margin, horiz_margin)
    ideal_thumbnail_margin = detection_max_dim * THUMBNAIL_EXTENSION_MULT // 2
    actual_margin = min(ideal_thumbnail_margin, max_allowable_margin)
    actual_margin = max(actual_margin, detection_max_dim // 2)

    chip_l = face_center_horiz - actual_margin
    chip_r = face_center_horiz + actual_margin
    chip_t = face_center_vert - actual_margin
    chip_b = face_center_vert + actual_margin

    left_pad = top_pad = right_pad = bot_pad = 0
    if chip_l < 0:
        left_pad, chip_l = -chip_l, 0
    if chip_r > img_w:
        right_pad, chip_r = chip_r - img_w, img_w
    if chip_t < 0:
        top_pad, chip_t = -chip_t, 0
    if chip_b > img_h:
        bot_pad, chip_b = chip_b - img_h, img_h

    chip_h = (chip_b - chip_t) + top_pad + bot_pad
    chip_w = (chip_r - chip_l) + left_pad + right_pad
    crop = img_numpy[int(chip_t):int(chip_b), int(chip_l):int(chip_r)]
    crop = np.pad(
        crop,
        ((int(top_pad), int(bot_pad)), (int(left_pad), int(right_pad)), (0, 0)),
        'constant',
    )
    if crop.shape[0] != chip_h or crop.shape[1] != chip_w or crop.shape[0] != crop.shape[1] or crop.shape[0] == 0:
        return None

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, settings.FACE_THUMBNAIL_SIZE)


def _mean_abs_pixel_diff(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def _load_image(img_obj):
    try:
        img_numpy = common.open_img_oriented(
            img_obj.filename, as_numpy=True,
            extra_rotation_degrees=img_obj.confirmed_rotation_override_degrees,
        )
    except Exception:
        return None
    if img_numpy is None or len(getattr(img_numpy, 'shape', ())) != 3:
        return None
    return img_numpy


class Command(BaseCommand):
    help = (
        "Rescales legacy (kps IS NULL) Face box coordinates to match their "
        "source image's CURRENT width/height, where the two have drifted "
        "apart (see this command's own module docstring). Every rescale is "
        "visually verified against the face's existing, already-correct "
        "stored thumbnail before anything is written; anything that doesn't "
        "match is skipped and reported, never guessed at."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Report what would happen without writing anything.",
        )
        parser.add_argument(
            '--diff-threshold', type=float, default=DEFAULT_DIFF_THRESHOLD,
            help=(
                "Max mean-abs-pixel-diff (0-255 scale) between the rescaled "
                "crop and the existing stored thumbnail to trust the rescale "
                f"(default {DEFAULT_DIFF_THRESHOLD})."
            ),
        )
        parser.add_argument(
            '--limit', type=int, default=None,
            help="Stop after this many candidate images (for a quick sample).",
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        diff_threshold = options['diff_threshold']
        limit = options['limit']

        faces_qs = Face.objects.filter(
            source_image_file__isnull=False,
            kps__isnull=True,
            relative_size__isnull=False,
            relative_size__gt=0,
        ).select_related('source_image_file').order_by('source_image_file_id', 'id')

        by_image = defaultdict(list)
        for face in faces_qs.iterator():
            by_image[face.source_image_file_id].append(face)

        images_total = len(by_image)
        self.stdout.write(
            f"{images_total} images with candidate legacy (kps IS NULL) faces "
            f"({dry_run and 'DRY RUN' or 'REAL RUN'}, diff-threshold={diff_threshold})"
        )

        counts = defaultdict(int)
        start = time.time()
        processed_images = 0

        for img_id, faces in by_image.items():
            if limit is not None and processed_images >= limit:
                break
            processed_images += 1

            result = self._process_image(faces, diff_threshold, dry_run)
            for key, value in result.items():
                counts[key] += value

            if processed_images % 200 == 0:
                elapsed = time.time() - start
                self.stdout.write(
                    f"[{processed_images}/{images_total}] {dict(counts)} ({elapsed:.0f}s)"
                )

        elapsed = time.time() - start
        self.stdout.write(f"DONE in {elapsed:.0f}s. {dict(counts)}")

    def _process_image(self, faces, diff_threshold, dry_run):
        counts = defaultdict(int)
        img_obj = faces[0].source_image_file
        img_w, img_h = img_obj.width, img_obj.height
        if not img_w or not img_h:
            counts['bad_image_dims'] += len(faces)
            return counts
        current_min_dim = min(img_w, img_h)

        per_face_ratio = {}
        for face in faces:
            box_w = face.box_right - face.box_left
            box_h = face.box_bottom - face.box_top
            stored_min_dim_box = min(box_w, box_h)
            if stored_min_dim_box <= 0 or not face.relative_size:
                counts['degenerate_box'] += 1
                continue
            old_min_dim = stored_min_dim_box / face.relative_size
            if old_min_dim <= 0:
                counts['degenerate_box'] += 1
                continue
            per_face_ratio[face.id] = current_min_dim / old_min_dim

        if not per_face_ratio:
            return counts

        ratios = list(per_face_ratio.values())
        ratio_min, ratio_max = min(ratios), max(ratios)

        if ratio_max <= 1 + RATIO_NOOP_TOLERANCE and ratio_min >= 1 - RATIO_NOOP_TOLERANCE:
            counts['already_ok'] += len(per_face_ratio)
            return counts

        if (ratio_max - ratio_min) > RATIO_AGREEMENT_TOLERANCE * max(1.0, ratio_min):
            counts['inconsistent_image'] += len(per_face_ratio)
            return counts

        ratio = sum(ratios) / len(ratios)

        img_numpy = _load_image(img_obj)
        if img_numpy is None:
            counts['image_decode_failed'] += len(per_face_ratio)
            return counts

        for face in faces:
            if face.id not in per_face_ratio:
                continue
            outcome = self._process_face(face, ratio, img_numpy, img_w, img_h, diff_threshold, dry_run)
            counts[outcome] += 1
        return counts

    def _process_face(self, face, ratio, img_numpy, img_w, img_h, diff_threshold, dry_run):
        new_left = max(0, min(face.box_left * ratio, img_w))
        new_top = max(0, min(face.box_top * ratio, img_h))
        new_right = max(0, min(face.box_right * ratio, img_w))
        new_bottom = max(0, min(face.box_bottom * ratio, img_h))

        new_left, new_top = int(round(new_left)), int(round(new_top))
        new_right, new_bottom = int(round(new_right)), int(round(new_bottom))

        if new_right <= new_left or new_bottom <= new_top:
            return 'degenerate_rescaled_box'

        new_box_w = new_right - new_left
        new_box_h = new_bottom - new_top
        current_min_dim = min(img_w, img_h)
        new_relative_size = min(new_box_w, new_box_h) / current_min_dim if current_min_dim > 0 else 0

        if face.relative_size and new_relative_size:
            rel_diff = abs(new_relative_size - face.relative_size) / face.relative_size
            if rel_diff > RELATIVE_SIZE_CONSISTENCY_TOLERANCE:
                return 'relative_size_mismatch'

        new_crop = _square_crop_and_resize(img_numpy, (new_left, new_top, new_right, new_bottom))
        if new_crop is None:
            return 'crop_failed'

        old_thumb = decode_face_thumbnail(face)
        diff = _mean_abs_pixel_diff(new_crop, old_thumb)
        if diff is None:
            return 'no_existing_thumbnail_to_verify_against'
        if diff > diff_threshold:
            return 'visual_mismatch'

        if dry_run:
            return 'would_fix'

        is_success, buffer_img = cv2.imencode('.jpg', new_crop)
        if not is_success:
            return 'encode_failed'
        new_thumb_bytes = buffer_img.tobytes()

        old_thumbnail_path = None
        try:
            old_thumbnail_path = face.face_thumbnail.path
        except ValueError:
            pass

        face.box_left = new_left
        face.box_top = new_top
        face.box_right = new_right
        face.box_bottom = new_bottom
        face.relative_size = new_relative_size

        thumb_filename = f'rescale_{face.source_image_file_id}_{face.id}.jpg'
        face.face_thumbnail.save(thumb_filename, ContentFile(new_thumb_bytes), save=False)
        face.save()

        if old_thumbnail_path:
            try:
                os.remove(old_thumbnail_path)
            except FileNotFoundError:
                pass

        return 'fixed'

import os
from collections import defaultdict
from io import BytesIO

import cv2
import insightface
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand

import common
from face_manager.face_extract_encode import FaceExtractor
from face_manager.models import Face


class _ThumbnailCropper:
    """Minimal stand-in for FaceExtractor -- get_square_face_img() and
    hash_numpy_array() only read self.thumbnail_extension_mult, so this
    avoids loading the full insightface model stack for a backfill that
    needs none of it (we're re-cropping from an already-stored box, not
    detecting or re-encoding anything)."""
    thumbnail_extension_mult = 2


class Command(BaseCommand):
    help = (
        "Regenerate face_thumbnail files for faces whose thumbnail file "
        "is missing on disk (the field still has a name, but the file "
        "itself is gone -- e.g. a restored-from-backup database "
        "referencing thumbnails that have since been regenerated with a "
        "different random-suffixed filename, or deleted outright). "
        "Reuses the already-stored box (no re-detection) to re-crop the "
        "exact same thumbnail from the source image -- classification "
        "fields (declared_name, poss_identN, validated, etc.) are never "
        "touched."
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        cropper = _ThumbnailCropper()

        self.stdout.write('Scanning for faces with a missing thumbnail file...')
        missing_by_image = defaultdict(list)
        total_checked = 0
        candidates = (
            Face.objects
            .exclude(face_thumbnail='')
            .filter(source_image_file__isnull=False)
            .select_related('source_image_file')
        )
        for f in candidates.iterator(chunk_size=2000):
            total_checked += 1
            try:
                exists = os.path.isfile(f.face_thumbnail.path)
            except Exception:
                exists = False
            if not exists:
                missing_by_image[f.source_image_file_id].append(f)

        total_missing = sum(len(v) for v in missing_by_image.values())
        self.stdout.write(
            f'Checked {total_checked} faces; {total_missing} missing thumbnails '
            f'across {len(missing_by_image)} images.'
        )

        regenerated = 0
        decode_failed = 0
        crop_failed = 0
        for i, (img_id, faces) in enumerate(missing_by_image.items()):
            img_obj = faces[0].source_image_file
            try:
                img_numpy = common.open_img_oriented(img_obj.filename, as_numpy=True)
                if img_numpy is None:
                    raise OSError('decode returned None')
            except Exception as e:
                decode_failed += len(faces)
                self.stdout.write(f'  decode failed for {img_obj.filename}: {e}')
                continue

            for f in faces:
                try:
                    fake_detected = insightface.app.common.Face(
                        bbox=[f.box_left, f.box_top, f.box_right, f.box_bottom]
                    )
                    thumb = FaceExtractor.get_square_face_img(cropper, fake_detected, img_obj, img_numpy)
                except Exception as e:
                    crop_failed += 1
                    self.stdout.write(f'  crop failed for face {f.id} ({img_obj.filename}): {e}')
                    continue

                if dry_run:
                    regenerated += 1
                    continue

                is_success, buffer_img = cv2.imencode('.jpg', thumb)
                temp_thumb = BytesIO(buffer_img)
                temp_thumb.seek(0)
                thumb_hash = FaceExtractor.hash_numpy_array(cropper, thumb)
                thumb_filename = f'{img_obj.pixel_hash}_{img_obj.file_hash}_facehash_{thumb_hash[:10]}.jpg'
                f.face_thumbnail.save(thumb_filename, ContentFile(temp_thumb.read()))
                temp_thumb.close()
                regenerated += 1

            if (i + 1) % 500 == 0:
                self.stdout.write(
                    f'[{i+1}/{len(missing_by_image)} images] regenerated={regenerated} '
                    f'decode_failed={decode_failed} crop_failed={crop_failed}'
                )

        self.stdout.write(
            f'Done. Regenerated: {regenerated}, decode_failed: {decode_failed}, '
            f'crop_failed: {crop_failed}, dry_run={dry_run}.'
        )

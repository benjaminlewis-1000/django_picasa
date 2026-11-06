"""DB/model-touching orchestration for APPLYING a confirmed rotation
correction -- the counterpart to face_manager.rotation_audit (which
only ever detects). See CLAUDE.md's 2026-11-03/11-05/11-06 orientation-
detection write-up and face_manager.rotation_correction_geometry's own
docstring for the design.

The rotation itself is NEVER baked into the original file, and the full
pixel-level correction is applied only VIRTUALLY, at decode time, via
common.open_img_oriented's extra_rotation_degrees param (see
ImageFile.confirmed_rotation_override_degrees) -- the same chokepoint
every consumer (serving, face extraction, ingestion) already goes
through for EXIF orientation. That decode-time override is what makes
the image show correctly EVERYWHERE the instant a human confirms, with
no waiting on this module at all.

What THIS module does is the one remaining piece of real, necessary
work: a one-time transform of each EXISTING Face's box/kps into the
new (post-rotation) coordinate space, and regenerating their
thumbnails/ImageFile.thumbnail_small to match -- pure geometry and
array ops, no model inference, no re-encoding (empirically confirmed
unnecessary: a global rigid rotation + the matching kps transform
reproduces the identical insightface/LVFace-B embedding, since
face_align.norm_crop's kps-based alignment is itself rotation-
invariant -- see the 2026-11-06 session write-up). Once that transform
has landed and isProcessed is cleared, the STANDARD, already-scheduled
face_extraction pipeline naturally reconciles this image on its next
pass: it now decodes through the SAME rotation-aware chokepoint, so its
own already-tested Hungarian-matching logic correctly matches (not
duplicates) our already-transformed existing faces and adds any
genuinely missed one as a new row -- no bespoke reconciliation/
redetection code needed here at all.
"""
from io import BytesIO

import cv2
import insightface
import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from django.db import transaction

import common
from face_manager.rotation_correction_geometry import (
    snap_to_canonical_rotation,
    transform_box,
    transform_kps,
)
from filepopulator.models import ImageFile


def _save_face_thumbnail(face, image, extractor, rotated_numpy, new_box):
    # get_square_face_img()/hash_numpy_array() are pure array-slice/
    # resize/hash operations -- no ONNX model call, despite living on
    # FaceExtractor. get_square_face_img() casts bbox to int internally,
    # so this doesn't need to pre-cast.
    insight_face_for_thumb = insightface.app.common.Face(bbox=np.array(new_box, dtype=np.int64))
    thumb = extractor.get_square_face_img(insight_face_for_thumb, image, rotated_numpy)
    _, buffer_img = cv2.imencode(".jpg", thumb)
    temp_thumb = BytesIO(buffer_img)
    temp_thumb.seek(0)
    thumb_hash = extractor.hash_numpy_array(thumb)
    thumb_filename = f'{image.pixel_hash}_{image.file_hash}_facehash_{thumb_hash[:10]}_rotated.jpg'
    face.face_thumbnail.save(thumb_filename, ContentFile(temp_thumb.read()), save=False)
    temp_thumb.close()


def apply_rotation_correction(image, extractor):
    """Transforms every existing Face's box/kps into image.
    proposed_rotation_degrees's coordinate space (snapped to the nearest
    canonical 90-degree correction) and regenerates their thumbnails and
    ImageFile.thumbnail_small to match -- no re-encoding, no missed-face
    redetection (see this module's own docstring for why neither is
    needed). Clears image.isProcessed so the standard, already-scheduled
    face_extraction pipeline reconciles this image on its next pass.

    Caller is responsible for the rotation_scan_state/
    rotation_correction_applied bookkeeping -- this function only does
    the geometry/thumbnail side effects, and does NOT save() `image`
    itself (the caller decides which fields to persist and when, e.g.
    via update_fields).

    Returns {'updated_face_ids': [...]}.

    All DB writes happen inside one transaction -- a failure partway
    through (e.g. a thumbnail write failing for the Nth of several
    faces) rolls back every Face change for this image rather than
    leaving it half-corrected. A half-corrected image would otherwise be
    genuinely unsafe to simply retry: a face already transformed once
    would get the SAME transform applied a SECOND time on retry (no way
    to tell "already corrected" from "not yet corrected" by inspecting a
    Face row alone), silently corrupting its box/kps instead of fixing
    them. Thumbnail FILE writes are not rolled back by this -- a stray
    orphaned file from a failed attempt is harmless, same as this
    project already tolerates elsewhere (content-addressed filenames).
    """
    # Decoded explicitly WITHOUT the confirmed override, to get the OLD
    # (pre-correction) dimensions the existing Face rows' stored box/kps
    # are still in -- transform_box()/transform_kps() need these to map
    # into the new coordinate space. extra_rotation_degrees=None here is
    # deliberate, not an oversight: this is the one place in the whole
    # pipeline that needs the PRE-correction geometry.
    old_pil = common.open_img_oriented(image.filename, as_numpy=False, extra_rotation_degrees=None)
    if old_pil is None:
        raise RuntimeError(f"Could not decode {image.filename} for rotation correction.")
    old_w, old_h = old_pil.size
    snapped = snap_to_canonical_rotation(image.proposed_rotation_degrees)

    # The FINAL pixels, via the same rotation-aware chokepoint every
    # other consumer uses (EXIF + the confirmed override, composed in
    # that order -- see common.open_img_oriented.apply_additional_
    # rotation's own docstring for why that's correct, not a double
    # application). Decoded as a PIL Image (not numpy) since
    # ImageFile._generate_thumbnail() below requires one; the numpy copy
    # face-thumbnail cropping needs is derived from it, not decoded a
    # second time.
    rotated_pil = common.open_img_oriented(
        image.filename, as_numpy=False,
        extra_rotation_degrees=image.confirmed_rotation_override_degrees,
    )
    if rotated_pil is None:
        raise RuntimeError(f"Could not decode {image.filename} for rotation correction.")
    rotated_numpy = np.array(rotated_pil)

    with transaction.atomic():
        updated_face_ids = []
        for face in image.face_set.all():
            new_box = transform_box(
                face.box_left, face.box_top, face.box_right, face.box_bottom, old_w, old_h, snapped,
            )
            new_kps = transform_kps(list(face.kps), old_w, old_h, snapped) if face.kps else None

            face.box_left, face.box_top, face.box_right, face.box_bottom = new_box
            if new_kps is not None:
                face.kps = new_kps

            _save_face_thumbnail(face, image, extractor, rotated_numpy, new_box)

            face.save()
            updated_face_ids.append(face.id)

        image.image = rotated_pil
        image.width, image.height = rotated_pil.size
        image._generate_thumbnail()
        image.isProcessed = False

    return {'updated_face_ids': updated_face_ids}


def run_rotation_correction(image_ids=None):
    """DB orchestration for apply_rotation_correction() -- the counterpart
    to face_manager.rotation_audit's run_rotation_audit(). Shared by
    correct_image_rotation (management command, manual trigger) and
    face_manager.tasks.correct_image_rotation_task (the scheduled Celery
    task), matching rotation_audit.py's own
    task-is-a-thin-wrapper-around-one-shared-function shape.

    Processes every ImageFile with rotation_scan_state=CONFIRMED and
    rotation_correction_applied=False (optionally narrowed to
    `image_ids`). No dry-run mode here (unlike run_rotation_audit) since
    there's nothing to preview beyond what KeyedImageView's
    rotation_preview type already shows a reviewer before they confirm.

    Returns {'corrected': int, 'failed': int, 'failures': [(id, str), ...]}.
    """
    # Imported lazily, not because it's heavy here (get_square_face_img/
    # hash_numpy_array are the only methods this module actually calls,
    # neither touches a model) but to avoid a real import-time cost this
    # function doesn't need: FaceExtractor.__init__() unconditionally
    # loads real ONNX models regardless of which methods end up called.
    from face_extract_encode import FaceExtractor

    ScanState = ImageFile.RotationScanState
    queryset = ImageFile.objects.filter(
        rotation_scan_state=ScanState.CONFIRMED, rotation_correction_applied=False,
    )
    if image_ids is not None:
        queryset = queryset.filter(id__in=image_ids)
    images = list(queryset)

    if not images:
        return {'corrected': 0, 'failed': 0, 'failures': []}

    extractor = FaceExtractor()
    corrected, failures = 0, []
    for image in images:
        # Per-image, not a single shared lock -- this function is called
        # from BOTH an immediate on-demand task (dispatched right after a
        # human confirms, for fast turnaround) and a periodic scheduled
        # sweep (a safety net for whatever the on-demand dispatch missed,
        # e.g. a lost broker message). A single shared lock would force
        # the fast path to wait out the whole sweep if they ever overlap,
        # defeating its own purpose; a per-image lock only ever excludes
        # two callers from double-applying the SAME row at once, which is
        # the actual risk (apply_rotation_correction() is not idempotent
        # against being run twice on an unflagged-in-between row).
        with common.advisory_lock(f'face_manager.correct_image_rotation.{image.id}') as acquired:
            if not acquired:
                continue
            # Re-check under the lock -- another caller may have already
            # corrected this exact row between this function's initial
            # query and acquiring the lock.
            image.refresh_from_db(fields=['rotation_correction_applied'])
            if image.rotation_correction_applied:
                continue

            try:
                apply_rotation_correction(image, extractor)
            except Exception as e:
                failures.append((image.id, str(e)))
                msg = f"ImageFile {image.id} ({image.filename}): rotation correction failed -- {e}"
                settings.LOGGER.error(msg)
                common.log_error('face_manager.rotation_correction', msg)
                continue

            image.rotation_correction_applied = True
            image.save(update_fields=[
                'thumbnail_small', 'width', 'height', 'isProcessed', 'rotation_correction_applied',
            ])
            corrected += 1

    return {'corrected': corrected, 'failed': len(failures), 'failures': failures}

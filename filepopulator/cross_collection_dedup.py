# Resolves CrossCollectionImageDuplicate candidates: pairs of ImageFile
# rows, in two different PhotoCollections, whose phash Hamming distance
# is exactly 0 (SimilarImagePair.hamming_distance=0) -- the "same real
# photo independently ingested into two collections" case. See
# CrossCollectionImageDuplicate's own docstring (filepopulator/models.py)
# and CLAUDE.md (2026-10-21) for the full investigation this design is
# based on: real production sample sizes, false-positive rate, and why a
# second perceptual hash (dhash) was tried and rejected (its distance is
# dominated by resizing/recompression noise, not genuine same-shot-vs-
# different-shot signal -- confirmed against both known-true-duplicate
# and known-burst-sequence pairs showing the same distribution).
#
# Classification (classify_pair): exactly reproduces the logic validated
# against the full real cross-collection population (4,727 pairs,
# 2026-10-21): 40.9% exact metadata match, 58.0% one side missing camera
# info (both safe to auto-merge), 1.0% same camera/model but different
# dateTaken (a real burst-shot signature, or occasionally a corrupted
# timestamp on an otherwise-identical photo), 0.02% a genuine camera-make
# conflict (one confirmed real example: two near-blank/white images that
# coincidentally hashed identically). The latter two are never auto-
# merged -- see resolve_pair().
#
# Neither ImageFile row is ever deleted -- both are real, distinct files.
# Face reconciliation (resolve_pair) rescales the secondary image's face
# boxes into the primary's coordinate space (pairs can have different
# resolutions -- confirmed on real data: 37.5% of exact-phash pairs do)
# and matches via optimal bipartite assignment on IOU, the same
# "optimal assignment, then filter by a real threshold" pattern this
# project already uses for the analogous video-duplicate case
# (merge_duplicate_videofiles.py) and for image face reconciliation
# during detection (face_extract_encode.py).
import os

import numpy as np
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from scipy.optimize import linear_sum_assignment

IOU_THRESH = 0.3


def _iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _rescale_box(box, scale_x, scale_y):
    return (box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y)


def match_cross_collection_faces(primary_faces, secondary_faces, scale_x, scale_y, iou_thresh=IOU_THRESH):
    """primary_faces/secondary_faces: list of dicts with a 'box'
    (left, top, right, bottom) key. scale_x/scale_y rescale secondary's
    boxes into primary's coordinate space before comparing (the two
    images in a pair are not always the same resolution).

    Returns a list of (primary_index, secondary_index) pairs considered
    the same real face. Any index absent from every pair is unmatched
    and must be treated by the caller as a distinct face. Never force-
    matched: linear_sum_assignment finds the single best overall
    pairing, but any pair whose IOU falls under iou_thresh is rejected
    afterward."""
    n, m = len(primary_faces), len(secondary_faces)
    if n == 0 or m == 0:
        return []

    cost = np.ones((n, m))
    for i, p in enumerate(primary_faces):
        for j, s in enumerate(secondary_faces):
            s_box_scaled = _rescale_box(s['box'], scale_x, scale_y)
            cost[i, j] = 1 - _iou(p['box'], s_box_scaled)

    rows, cols = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(rows, cols):
        if (1 - cost[i, j]) >= iou_thresh:
            matches.append((i, j))
    return matches


def _has_real_label(person_name):
    return person_name not in settings.IGNORED_NAMES


def _survivor_key(face_info):
    """Lower sorts first (wins). Same preference this project already
    uses for the equivalent same-image (dedupe_overlapping_faces.py) and
    duplicate-video (merge_duplicate_videofiles.py) cases: validated >
    has a real label > has kps > has det_score > lowest id."""
    return (
        0 if face_info['validated'] else 1,
        0 if _has_real_label(face_info['person_name']) else 1,
        0 if face_info['has_kps'] else 1,
        0 if face_info['has_det_score'] else 1,
        face_info['id'],
    )


def classify_pair(image_a, image_b):
    """Which CrossCollectionImageDuplicate.REASON_* bucket this pair
    falls into. Deliberately NOT branching on resolution/file-size at
    all -- only dateTaken and camera make/model, exactly reproducing the
    logic validated against real production data (see this module's own
    docstring)."""
    from filepopulator.models import CrossCollectionImageDuplicate

    a_make = (image_a.camera_make or '').strip()
    b_make = (image_b.camera_make or '').strip()
    a_model = (image_a.camera_model or '').strip()
    b_model = (image_b.camera_model or '').strip()

    if image_a.dateTaken == image_b.dateTaken and a_make == b_make and a_model == b_model:
        return CrossCollectionImageDuplicate.REASON_EXACT_METADATA

    if not a_make or not b_make:
        return CrossCollectionImageDuplicate.REASON_STRIPPED_EXIF

    if a_make.lower() == b_make.lower() and a_model.lower() == b_model.lower():
        return CrossCollectionImageDuplicate.REASON_BURST_SIGNATURE

    return CrossCollectionImageDuplicate.REASON_CAMERA_CONFLICT


def _file_size_or_fallback(image):
    try:
        return os.path.getsize(image.filename)
    except OSError:
        return -1


def _pick_primary(image_a, image_b):
    """Larger file (by real on-disk size, not resolution -- can be
    identical, e.g. the real motivating example: same 4000x3000 photo,
    535KB vs 2.67MB) wins. A missing file sorts last (-1), so the other
    side wins automatically rather than crashing."""
    if _file_size_or_fallback(image_a) >= _file_size_or_fallback(image_b):
        return image_a, image_b
    return image_b, image_a


def _load_face_infos(image):
    from face_manager.models import Face

    infos = []
    for f in Face.objects.filter(source_image_file=image).select_related('declared_name'):
        infos.append({
            'id': f.pk,
            'box': (f.box_left, f.box_top, f.box_right, f.box_bottom),
            'validated': f.validated,
            'person_name': f.declared_name.person_name if f.declared_name_id else settings.BLANK_FACE_NAME,
            'has_kps': f.kps is not None,
            'has_det_score': f.det_score is not None,
        })
    return infos


def resolve_pair(pair, dry_run=False):
    """pair: a SimilarImagePair with hamming_distance==0, image_a/image_b
    confirmed (by the caller) to be in different PhotoCollections.

    Classifies the pair; if it falls into a safe-to-auto-merge bucket
    (EXACT_METADATA/STRIPPED_EXIF), reconciles their Face rows (IOU-
    matched, rescaled for resolution) and marks the non-primary image so
    it's never independently re-extracted. Otherwise (BURST_SIGNATURE/
    CAMERA_CONFLICT) makes no Face/ImageFile changes at all -- just
    records a NEEDS_REVIEW row for a human to decide later.

    The SAME code path computes the plan whether dry_run is True or
    False -- only the actual writes are skipped -- so dry-run reporting
    is guaranteed to match what a real run would do.

    Returns a stats dict: {'reason', 'status', 'primary_id',
    'faces_matched', 'faces_transferred', 'faces_deleted'}."""
    from face_manager.models import Face
    from filepopulator.models import CrossCollectionImageDuplicate, ImageFile

    image_a, image_b = pair.image_a, pair.image_b
    lo_id, hi_id = sorted((image_a.id, image_b.id))
    reason = classify_pair(image_a, image_b)

    if reason in (CrossCollectionImageDuplicate.REASON_BURST_SIGNATURE, CrossCollectionImageDuplicate.REASON_CAMERA_CONFLICT):
        if not dry_run:
            CrossCollectionImageDuplicate.objects.update_or_create(
                image_a_id=lo_id, image_b_id=hi_id,
                defaults={
                    'status': CrossCollectionImageDuplicate.STATUS_NEEDS_REVIEW,
                    'reason': reason, 'decided_at': timezone.now(),
                },
            )
        return {
            'reason': reason, 'status': CrossCollectionImageDuplicate.STATUS_NEEDS_REVIEW,
            'primary_id': None, 'faces_matched': 0, 'faces_transferred': 0, 'faces_deleted': 0,
        }

    primary, secondary = _pick_primary(image_a, image_b)

    scale_x = (primary.width / secondary.width) if (primary.width and secondary.width) else 1.0
    scale_y = (primary.height / secondary.height) if (primary.height and secondary.height) else 1.0

    primary_faces = _load_face_infos(primary)
    secondary_faces = _load_face_infos(secondary)
    matches = match_cross_collection_faces(primary_faces, secondary_faces, scale_x, scale_y)
    matched_secondary_idx = {j for _, j in matches}

    primary_delete_ids = []
    secondary_delete_ids = []
    secondary_transfer_ids = []

    for i, j in matches:
        p_info, s_info = primary_faces[i], secondary_faces[j]
        if _survivor_key(s_info) < _survivor_key(p_info):
            # The secondary copy's version of this same real face is the
            # better one (e.g. it's the copy a human actually labeled) --
            # keep it, drop primary's.
            primary_delete_ids.append(p_info['id'])
            secondary_transfer_ids.append(s_info['id'])
        else:
            secondary_delete_ids.append(s_info['id'])

    for j, s_info in enumerate(secondary_faces):
        if j not in matched_secondary_idx:
            # Only the secondary's extraction run found this face at all
            # -- never discard it.
            secondary_transfer_ids.append(s_info['id'])

    if not dry_run:
        with transaction.atomic():
            for face in Face.objects.filter(pk__in=primary_delete_ids):
                face.delete()
            for face in Face.objects.filter(pk__in=secondary_delete_ids):
                face.delete()
            if secondary_transfer_ids:
                Face.objects.filter(pk__in=secondary_transfer_ids).update(source_image_file=primary)

            # Keep the primary queryable as "visible from" the secondary's
            # collection too, same cross-linking convention
            # create_image_file()'s duplicate-discovery branches already
            # use.
            for collection in secondary.collections.all():
                primary.collections.add(collection)

            # Never independently re-extracted again -- all of its real
            # faces now live on primary (or already did).
            ImageFile.objects.filter(pk=secondary.pk).update(isProcessed=True)

            CrossCollectionImageDuplicate.objects.update_or_create(
                image_a_id=lo_id, image_b_id=hi_id,
                defaults={
                    'status': CrossCollectionImageDuplicate.STATUS_AUTO_MERGED,
                    'reason': reason, 'primary_image': primary,
                    'faces_matched': len(matches),
                    'faces_transferred': len(secondary_transfer_ids),
                    'faces_deleted': len(primary_delete_ids) + len(secondary_delete_ids),
                    'decided_at': timezone.now(),
                },
            )

    return {
        'reason': reason, 'status': CrossCollectionImageDuplicate.STATUS_AUTO_MERGED,
        'primary_id': primary.id, 'faces_matched': len(matches),
        'faces_transferred': len(secondary_transfer_ids),
        'faces_deleted': len(primary_delete_ids) + len(secondary_delete_ids),
    }


def find_unresolved_cross_collection_pairs():
    """Every exact-phash (hamming_distance=0) SimilarImagePair, in
    different collections, not yet covered by a CrossCollectionImageDuplicate
    row -- the candidate set both the one-off management command and the
    recurring Celery task work from."""
    from filepopulator.models import CrossCollectionImageDuplicate, SimilarImagePair

    already_decided = set(
        CrossCollectionImageDuplicate.objects.values_list('image_a_id', 'image_b_id')
    )

    candidates = SimilarImagePair.objects.filter(hamming_distance=0).select_related(
        'image_a__directory__collection', 'image_b__directory__collection'
    )
    for pair in candidates.iterator(chunk_size=2000):
        if (pair.image_a_id, pair.image_b_id) in already_decided:
            continue
        a_collection = pair.image_a.directory.collection_id
        b_collection = pair.image_b.directory.collection_id
        if a_collection is None or b_collection is None or a_collection == b_collection:
            continue
        yield pair

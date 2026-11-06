"""DB-touching orchestration for the rotation-review scan -- the
counterpart to face_manager.rotation_geometry's pure math. Shared by
audit_image_rotation (management command, manual trigger) and
face_manager.tasks.audit_image_rotation_task (the scheduled Celery
task) so there's exactly one place this logic lives, matching
FaceExtractor.reencode_missing_faces()'s own
task-is-a-thin-wrapper-around-one-shared-function shape elsewhere in
this codebase.
"""
from django.db.models import Prefetch, Q

from face_manager.models import Face
from face_manager.rotation_geometry import (
    ROTATION_SCAN_LOGIC_VERSION,
    UPRIGHT_TOLERANCE_DEG,
    aggregate_orientation,
    circular_distance,
    eye_angle_degrees,
    should_overwrite_rotation_proposal,
)
from filepopulator.models import ImageFile

# A real, confirmed production incident (2026-11-06): the unbatched version
# of this function loaded every eligible ImageFile PLUS every one of its
# Face rows IN FULL (face_encoding_512/lvface_b included, ~2KB+ per face,
# via the default prefetch_related('face_set')) all at once -- against the
# real ~271k-image backlog this measured 25GB+ RSS and still climbing, on a
# host already low on free memory, before it was killed. evaluate_image()
# only ever reads Face.kps, so the prefetch now pulls just id/kps per face,
# and the whole scan is capped to a bounded chunk per call rather than the
# entire backlog at once -- the scheduled hourly task naturally grinds
# through a large backlog over many runs instead of one unbounded pass.
DEFAULT_MAX_IMAGES_PER_RUN = 2000


def _candidate_images_queryset():
    ScanState = ImageFile.RotationScanState
    return ImageFile.objects.filter(
        Q(rotation_scan_state=ScanState.NOT_SCANNED) | Q(rotation_scan_state=ScanState.FLAGGED),
        rotation_scanned_at_version__lt=ROTATION_SCAN_LOGIC_VERSION,
    ).order_by('id')


def evaluate_image(image, faces_with_kps):
    """Pure-ish evaluation of one image's faces (DB-free): returns a dict
    describing what, if anything, should change -- {'examined': bool,
    'estimate': float|None, 'source': str|None, 'significant': bool} --
    or None if there's nothing usable to examine (no faces with kps).
    Separated from the write logic below so it's directly testable
    without a DB.
    """
    angles = [eye_angle_degrees(face.kps) for face in faces_with_kps]
    if not angles:
        return None

    if len(angles) >= 2:
        result = aggregate_orientation(angles)
        if result is None:
            # Real disagreement among 2+ faces -- genuinely ambiguous,
            # not a signal either way. Still "examined" so the version
            # bump applies (no point re-deriving the same ambiguity
            # again under an unchanged logic version).
            return {'examined': True, 'estimate': None, 'source': None, 'significant': False}
        estimate, _inlier_count = result
        source = ImageFile.RotationSignalSource.LANDMARK_MULTI_FACE
    else:
        estimate = angles[0]
        source = ImageFile.RotationSignalSource.LANDMARK_SINGLE_FACE

    significant = circular_distance(estimate, 0) > UPRIGHT_TOLERANCE_DEG
    return {'examined': True, 'estimate': estimate, 'source': source, 'significant': significant}


def run_rotation_audit(write=False, max_images=DEFAULT_MAX_IMAGES_PER_RUN):
    """Scans up to `max_images` eligible ImageFiles (see
    _candidate_images_queryset, ordered by id for stable chunking across
    repeated calls) and either reports (write=False, the default) or
    persists (write=True) the resulting rotation_scan_state/
    proposed_rotation_degrees/rotation_signal_source/
    rotation_scanned_at_version changes.

    Deliberately NOT the whole backlog in one call -- see
    DEFAULT_MAX_IMAGES_PER_RUN's own comment. `max_images=None` removes
    the cap entirely (only for a deliberate, supervised one-off run with
    its own external memory monitoring -- never from the scheduled task).

    Returns a dict: {'examined': int, 'counts': {...}, 'updated': int}.
    """
    ScanState = ImageFile.RotationScanState

    queryset = _candidate_images_queryset().prefetch_related(
        Prefetch('face_set', queryset=Face.objects.only('id', 'kps', 'source_image_file')),
    )
    if max_images is not None:
        queryset = queryset[:max_images]
    candidates = list(queryset)
    counts = {
        'no_usable_faces': 0,
        'ambiguous_disagreement': 0,
        'confirmed_upright': 0,
        'flagged_new_or_updated': 0,
        'skipped_lower_priority': 0,
    }

    to_update = []
    for image in candidates:
        faces_with_kps = [f for f in image.face_set.all() if f.kps]
        outcome = evaluate_image(image, faces_with_kps)
        if outcome is None:
            counts['no_usable_faces'] += 1
            continue

        if outcome['estimate'] is None:
            counts['ambiguous_disagreement'] += 1
            image.rotation_scanned_at_version = ROTATION_SCAN_LOGIC_VERSION
            to_update.append(image)
            continue

        allowed = should_overwrite_rotation_proposal(image.rotation_signal_source, outcome['source'])
        if not allowed:
            counts['skipped_lower_priority'] += 1
            image.rotation_scanned_at_version = ROTATION_SCAN_LOGIC_VERSION
            to_update.append(image)
            continue

        if outcome['significant']:
            counts['flagged_new_or_updated'] += 1
            image.rotation_scan_state = ScanState.FLAGGED
            image.proposed_rotation_degrees = outcome['estimate']
            image.rotation_signal_source = outcome['source']
        else:
            counts['confirmed_upright'] += 1
            image.rotation_scan_state = ScanState.NOT_SCANNED
            image.proposed_rotation_degrees = None
            image.rotation_signal_source = None
        image.rotation_scanned_at_version = ROTATION_SCAN_LOGIC_VERSION
        to_update.append(image)

    if write:
        ImageFile.objects.bulk_update(
            to_update,
            ['rotation_scan_state', 'proposed_rotation_degrees',
             'rotation_signal_source', 'rotation_scanned_at_version'],
            batch_size=1000,
        )

    return {'examined': len(candidates), 'counts': counts, 'updated': len(to_update)}

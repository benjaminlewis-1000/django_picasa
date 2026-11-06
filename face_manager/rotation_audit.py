"""DB-touching orchestration for the rotation-review scan -- the
counterpart to face_manager.rotation_geometry's pure math. Shared by
audit_image_rotation (management command, manual trigger) and
face_manager.tasks.audit_image_rotation_task (the scheduled Celery
task) so there's exactly one place this logic lives, matching
FaceExtractor.reencode_missing_faces()'s own
task-is-a-thin-wrapper-around-one-shared-function shape elsewhere in
this codebase.
"""
import hashlib
import inspect
import sys
import time

from django.db.models import Count, Prefetch, Q

from face_manager import rotation_geometry
from face_manager.models import Face
from face_manager.rotation_geometry import (
    UPRIGHT_TOLERANCE_DEG,
    aggregate_orientation,
    circular_distance,
    eye_angle_degrees,
    should_overwrite_rotation_proposal,
)
from filepopulator.models import ImageFile


def _rotation_scan_logic_hash():
    """sha256(source of THIS module + rotation_geometry)[:16] -- any
    meaningful change to either module's evaluation math or gating
    conditions changes this automatically, so nobody has to remember to
    hand-bump a version constant (this module used to require exactly
    that; forgetting would silently keep stale verdicts around rather
    than erroring loudly). Mirrors face_manager.test_face_cache's own
    sha256(source)-keyed cache-invalidation pattern -- same tradeoff
    accepted: a comment-only edit also triggers a harmless re-scan, which
    costs nothing since this is cheap pure-geometry math on already-
    stored kps, no image decode or model inference.
    """
    src = inspect.getsource(sys.modules[__name__]) + inspect.getsource(rotation_geometry)
    return hashlib.sha256(src.encode('utf-8')).hexdigest()[:16]

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

# The per-batch math itself is cheap (pure geometry on already-stored
# kps, no image decode or model inference) -- the real constraint this
# whole module exists to respect is MEMORY per batch (see above), not
# wall-clock time. So rather than stopping after exactly one batch and
# waiting for the next hourly tick, run_rotation_audit_until_done() below
# keeps looping fresh, independently-bounded batches until either nothing
# eligible remains or this time budget is spent -- a large backlog can
# realistically be ground through in one scheduled run. Comfortably under
# the hourly schedule interval (3600s) so the task reliably finishes (and
# releases its advisory lock) before the next tick, rather than running
# right up against it.
DEFAULT_MAX_RUNTIME_SECONDS = 3000


def _candidate_images_queryset():
    """Real production breakdown checked 2026-11-06 (270,470 NOT_SCANNED
    images): only 52 were isProcessed=False (not yet through face
    detection at all); 33,762 had faces but at least one still missing
    kps (the in-progress kps backfill, not a permanent state); 62,354
    genuinely have zero faces at all; the remaining 174,302 were fully
    ready to evaluate right now. isProcessed/kps-completeness are
    required here (not just version-gated) so the first two groups are
    never pulled into a batch at all -- they cost nothing and are
    automatically reconsidered once face detection/kps backfill actually
    finishes on them, with no separate "rescan later" bookkeeping needed,
    since they were never marked scanned in the first place. An image
    with genuinely zero faces still passes (Count(filter=...) over zero
    rows is 0) and reaches evaluate_image(), which correctly reports
    'no_usable_faces' for it once, same as before.
    """
    ScanState = ImageFile.RotationScanState
    return (
        ImageFile.objects.filter(
            Q(rotation_scan_state=ScanState.NOT_SCANNED) | Q(rotation_scan_state=ScanState.FLAGGED),
            isProcessed=True,
        )
        .exclude(rotation_scan_logic_hash=_rotation_scan_logic_hash())
        .annotate(_faces_missing_kps=Count('face', filter=Q(face__kps__isnull=True)))
        .filter(_faces_missing_kps=0)
        .order_by('id')
    )


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
    rotation_scan_logic_hash changes.

    Deliberately NOT the whole backlog in one call -- see
    DEFAULT_MAX_IMAGES_PER_RUN's own comment. `max_images=None` removes
    the cap entirely (only for a deliberate, supervised one-off run with
    its own external memory monitoring -- never from the scheduled task).

    Returns a dict: {'examined': int, 'counts': {...}, 'updated': int}.
    """
    ScanState = ImageFile.RotationScanState
    current_hash = _rotation_scan_logic_hash()

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
            image.rotation_scan_logic_hash = current_hash
            to_update.append(image)
            continue

        allowed = should_overwrite_rotation_proposal(image.rotation_signal_source, outcome['source'])
        if not allowed:
            counts['skipped_lower_priority'] += 1
            image.rotation_scan_logic_hash = current_hash
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
        image.rotation_scan_logic_hash = current_hash
        to_update.append(image)

    if write:
        ImageFile.objects.bulk_update(
            to_update,
            ['rotation_scan_state', 'proposed_rotation_degrees',
             'rotation_signal_source', 'rotation_scan_logic_hash'],
            batch_size=1000,
        )

    return {'examined': len(candidates), 'counts': counts, 'updated': len(to_update)}


def run_rotation_audit_until_done(batch_size=DEFAULT_MAX_IMAGES_PER_RUN,
                                   max_runtime_seconds=DEFAULT_MAX_RUNTIME_SECONDS):
    """Repeatedly calls run_rotation_audit(write=True, max_images=
    batch_size) -- never holding more than one batch's worth of data in
    memory at once -- until either nothing eligible remains or
    max_runtime_seconds is exceeded, whichever comes first. Used by the
    scheduled task so a large backlog is ground through in one run
    rather than trickling `batch_size` images per hourly tick.

    write=True ONLY -- looping only makes forward progress because each
    written batch stamps rotation_scan_logic_hash, which is what makes
    _candidate_images_queryset() return DIFFERENT rows on the next
    iteration. A dry-run loop would just re-examine the identical first
    batch forever (nothing persists to advance the query), so there's no
    write=False variant of this function -- use run_rotation_audit()
    directly (a single batch) for a dry-run preview instead.

    Returns {'examined': int, 'updated': int, 'counts': {...}, 'batches': int,
    'hit_time_limit': bool}.
    """
    start = time.monotonic()
    totals = {'examined': 0, 'updated': 0, 'counts': {}, 'batches': 0, 'hit_time_limit': False}
    while True:
        result = run_rotation_audit(write=True, max_images=batch_size)
        totals['examined'] += result['examined']
        totals['updated'] += result['updated']
        totals['batches'] += 1
        for key, value in result['counts'].items():
            totals['counts'][key] = totals['counts'].get(key, 0) + value

        if result['examined'] == 0:
            break
        if time.monotonic() - start >= max_runtime_seconds:
            totals['hit_time_limit'] = True
            break
    return totals

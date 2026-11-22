"""DB-touching orchestration for the rotation-review scan's SECOND
signal source -- the ML orientation classifier (ImageFile.
RotationSignalSource.ML_CLASSIFIER). Mirrors face_manager.rotation_
audit's shape exactly (one shared function the management command and
the scheduled Celery task both call), but classifies via the trained
model (face_manager.rotation_classifier_model) instead of pure
face-landmark geometry -- so, unlike the landmark path, this one needs
no detected faces/kps at all and can examine every NOT_SCANNED image,
not just the ones with at least one face.

Built 2026-11-21 per the user's explicit request to have this
infrastructure ready even though it isn't turned on yet: nothing here
is wired into CELERY_BEAT_SCHEDULE, so it never runs on its own --
only a manual run_rotation_ml_audit management-command invocation (or
a direct call to the functions below) ever examines/writes anything.

Deliberately uses its own, separately-tracked staleness hash
(ImageFile.rotation_ml_scan_hash), NOT rotation_audit's
rotation_scan_logic_hash -- see that field's own docstring in
filepopulator/models.py. The landmark path's hash is keyed on SOURCE
CODE alone (cheap pure geometry, safe to re-run the whole backlog on
any edit); this one is keyed on source code PLUS the trained
checkpoint's own weight bytes, since a retrain-and-re-export with
unchanged code must still re-qualify every row, and re-running real
model inference on an unrelated code edit elsewhere would be
genuinely expensive.

Once FLAGGED (by this signal OR the landmark one), a row is excluded
from both audits' candidate querysets -- only a human confirm/dismiss
ever moves it again (see rotation_audit._candidate_images_queryset()'s
own 2026-11-21 note on why the earlier source-priority overwrite
table was removed).
"""
import hashlib
import inspect
import os
import sys
import time

from django.conf import settings

from face_manager import rotation_classifier_model
from face_manager.rotation_classifier_model import (
    INPUT_SIZE,
    MAX_ASPECT_RATIO,
    MAX_IMAGE_PIXELS,
    decode_upright,
    letterbox_to_square,
    to_model_input,
    undo_rotation_degrees,
)
from filepopulator.models import ImageFile

# Deliberately the ONNX export, not train_rotation_classifier.py's own
# checkpoint.pt -- that file is overwritten with the LATEST weights on
# every CHECKPOINT_EVERY_STEPS tick, not just on a new best, so it drifts
# away from the best val_acc actually reached as training continues past
# that point. export_onnx() in the training command, by contrast, only
# ever fires when a NEW best val_acc is found, so this file always holds
# the best model seen so far -- exactly the weights this live-inference
# path should use. Also matches this project's existing all-ONNX CPU-
# inference convention elsewhere (insightface's det_500m/det_10g, LVFace).
ONNX_MODEL_PATH = '/models/rotation_classifier/rotation_classifier.onnx'

# Empirically calibrated 2026-11-22 against this checkpoint's best-exported
# ONNX weights, using REAL human-adjudicated rotation-review rows (960
# CONFIRMED + 2374 DISMISSED_NOT_ROTATED) as ground truth -- a harder,
# more adversarial population than a random held-out split, since every
# DISMISSED row was something another signal already thought looked
# rotated and a human said no.
#
# Gate is on the AGGREGATE "not confidently upright" mass (1 - P(upright)),
# not the old per-class-argmax confidence -- see the dated CLAUDE.md
# write-up ("Is one threshold for orientation the right way to go about
# this?"). The old rule required the network to commit confidently to ONE
# specific rotated class; a model split ~evenly between two rotated
# classes (e.g. 35%/35%/20% upright) never cleared that gate even though
# it was clearly, confidently NOT upright. At this threshold (0.80): ~10.6%
# false-flag rate on genuinely-upright images, ~83.5% recall on
# genuinely-rotated ones (full sweep in the dated write-up) -- same
# false-flag cost as the old 0.75 per-class rule, meaningfully higher
# recall. Deliberately on the loose/high-recall end of that curve, not the
# conservative end -- per explicit user direction: confirmed/dismissed
# rows are a one-time review, never re-surfaced, so a wrong flag just
# costs one quick human dismissal rather than a standing cost, and
# catching more real rotations is worth a higher false-flag rate here.
CONFIDENCE_FLOOR = 0.80

# Same batching/runtime-budget reasoning as rotation_audit.py's own
# constants -- see that module's comments. A full classifier forward
# pass is far more expensive per image than the landmark path's pure
# geometry, so these are deliberately smaller.
DEFAULT_MAX_IMAGES_PER_RUN = 500
DEFAULT_MAX_RUNTIME_SECONDS = 3000


def _rotation_ml_scan_hash():
    """sha256(source of this module + rotation_classifier_model +
    the exported ONNX model's own raw bytes)[:16] -- see this module's own
    docstring for why the model's weight bytes are folded in here,
    unlike rotation_audit's source-only hash. A missing model file
    hashes to the empty-bytes case deterministically (not an error) --
    callers that actually need a loaded model handle FileNotFoundError
    separately, at load time.
    """
    src = inspect.getsource(sys.modules[__name__]) + inspect.getsource(rotation_classifier_model)
    try:
        with open(ONNX_MODEL_PATH, 'rb') as f:
            model_bytes = f.read()
    except FileNotFoundError:
        model_bytes = b''
    return hashlib.sha256(src.encode('utf-8') + model_bytes).hexdigest()[:16]


class _OnnxLogitsModel:
    """Minimal callable wrapper around an onnxruntime InferenceSession so
    classify_image()'s torch-tensor-in/torch-tensor-out interface --
    shared with the plain-callable stub models this module's own tests
    use in place of a real model -- works unchanged regardless of which
    backend _load_model() returns."""

    def __init__(self, session):
        self._session = session
        self._input_name = session.get_inputs()[0].name

    def __call__(self, tensor):
        import torch

        logits = self._session.run(None, {self._input_name: tensor.numpy()})[0]
        return torch.from_numpy(logits)

    def eval(self):
        return self


def _load_model():
    """Loads the best-exported ONNX model fresh -- raises
    FileNotFoundError if it doesn't exist (callers decide how to handle
    that; see run_rotation_ml_audit_until_done()). Deliberately not
    cached at module scope: this is infra built ahead of being
    scheduled, not yet exercised at a frequency where reload cost
    matters -- revisit if/when this is actually wired into a live,
    repeatedly-invoked task.
    """
    import onnxruntime as ort

    if not os.path.exists(ONNX_MODEL_PATH):
        raise FileNotFoundError(ONNX_MODEL_PATH)
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    return _OnnxLogitsModel(session)


def _candidate_images_queryset():
    """Every NOT_SCANNED image not yet examined under the current ML
    hash -- no face/kps requirement at all (the whole point of this
    signal is covering the landmark path's permanent blind spot:
    faceless content, and any face whose kps will never be recoverable).
    isProcessed=True is still required, matching the landmark path's own
    reasoning, so a row mid-ingestion is naturally reconsidered once
    processing actually settles rather than needing separate "retry
    later" bookkeeping.

    FLAGGED is deliberately excluded, same as rotation_audit's own
    queryset -- see this module's docstring.
    """
    ScanState = ImageFile.RotationScanState
    return (
        ImageFile.objects.filter(rotation_scan_state=ScanState.NOT_SCANNED, isProcessed=True)
        .exclude(rotation_ml_scan_hash=_rotation_ml_scan_hash())
        .order_by('id')
    )


def classify_image(arr, model):
    """Pure inference on an already-decoded HxWx3 uint8 RGB array --
    model is passed in (not loaded here) so this is directly testable
    with a cheap stub instead of the real checkpoint. Returns the full
    4-class softmax probability vector (numpy array, class 0 = upright),
    not just the top class -- the aggregate flagging rule below needs
    P(upright) specifically, not only whichever single class wins."""
    import torch

    square = letterbox_to_square(arr, size=INPUT_SIZE)
    tensor = torch.from_numpy(to_model_input(square)).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    return probs.numpy()


def evaluate_image_ml(filename, width, height, model):
    """DB-free evaluation of one image: decode + classify (or skip/fail
    first). Returns a dict: {'outcome': 'size_skip'|'decode_failure'|
    'classified', 'degrees': float|None, 'significant': bool}.

    Flags on the AGGREGATE "not confidently upright" mass
    (1 - P(upright)), not on any single class's own confidence -- see
    CONFIDENCE_FLOOR's docstring above. The proposed DIRECTION is still
    whichever rotated class (1..3) has the most probability mass, but
    that choice no longer gates whether the image is flagged at all.

    'size_skip'/'decode_failure' are terminal for THIS code/model
    version (same width/height will always skip; a genuinely unreadable
    file won't become readable on retry under the same hash) -- the
    caller still stamps rotation_ml_scan_hash for these so they aren't
    re-attempted forever, same reasoning as rotation_audit's own
    no_usable_faces handling.
    """
    if width and height:
        if width * height > MAX_IMAGE_PIXELS or max(width, height) / min(width, height) > MAX_ASPECT_RATIO:
            return {'outcome': 'size_skip', 'degrees': None, 'significant': False}

    arr = decode_upright(filename, None)
    if arr is None:
        return {'outcome': 'decode_failure', 'degrees': None, 'significant': False}

    probs = classify_image(arr, model)
    predicted_rotated_class = 1 + int(probs[1:].argmax())
    degrees = undo_rotation_degrees(predicted_rotated_class)
    not_upright_mass = 1.0 - float(probs[0])
    significant = not_upright_mass >= CONFIDENCE_FLOOR
    return {'outcome': 'classified', 'degrees': degrees, 'significant': significant}


def run_rotation_ml_audit(write=False, max_images=DEFAULT_MAX_IMAGES_PER_RUN, model=None):
    """Scans up to `max_images` eligible ImageFiles and either reports
    (write=False, the default) or persists (write=True) the resulting
    rotation_scan_state/proposed_rotation_degrees/rotation_signal_source/
    rotation_ml_scan_hash changes.

    `model=None` loads the real checkpoint fresh (raises
    FileNotFoundError if it doesn't exist yet -- e.g. training hasn't
    produced one, or this is a test environment with no /models mount;
    callers that want to loop many batches under one loaded model, see
    run_rotation_ml_audit_until_done(), should load once and pass it in).

    Returns a dict: {'examined': int, 'counts': {...}, 'updated': int}.
    """
    ScanState = ImageFile.RotationScanState
    if model is None:
        model = _load_model()
    current_hash = _rotation_ml_scan_hash()

    queryset = _candidate_images_queryset().only(
        'id', 'filename', 'width', 'height', 'rotation_scan_state',
        'proposed_rotation_degrees', 'rotation_signal_source', 'rotation_ml_scan_hash',
    )
    if max_images is not None:
        queryset = queryset[:max_images]
    candidates = list(queryset)
    counts = {
        'size_skip': 0,
        'decode_failure': 0,
        'confirmed_upright': 0,
        'flagged_new': 0,
    }

    to_update = []
    for image in candidates:
        outcome = evaluate_image_ml(image.filename, image.width, image.height, model)
        if outcome['outcome'] in ('size_skip', 'decode_failure'):
            counts[outcome['outcome']] += 1
            image.rotation_ml_scan_hash = current_hash
            to_update.append(image)
            continue

        if outcome['significant']:
            counts['flagged_new'] += 1
            image.rotation_scan_state = ScanState.FLAGGED
            image.proposed_rotation_degrees = outcome['degrees']
            image.rotation_signal_source = ImageFile.RotationSignalSource.ML_CLASSIFIER
        else:
            counts['confirmed_upright'] += 1
        image.rotation_ml_scan_hash = current_hash
        to_update.append(image)

    if write:
        ImageFile.objects.bulk_update(
            to_update,
            ['rotation_scan_state', 'proposed_rotation_degrees',
             'rotation_signal_source', 'rotation_ml_scan_hash'],
            batch_size=1000,
        )

    return {'examined': len(candidates), 'counts': counts, 'updated': len(to_update)}


def run_rotation_ml_audit_until_done(batch_size=DEFAULT_MAX_IMAGES_PER_RUN,
                                      max_runtime_seconds=DEFAULT_MAX_RUNTIME_SECONDS):
    """Loads the checkpoint once, then repeatedly calls
    run_rotation_ml_audit(write=True, ...) with that one loaded model
    until either nothing eligible remains or max_runtime_seconds is
    exceeded -- mirrors rotation_audit.run_rotation_audit_until_done()'s
    shape. If no checkpoint exists yet, returns immediately with
    skipped_no_checkpoint=True rather than raising -- this module is
    infra built ahead of training necessarily having produced a usable
    checkpoint in every environment (e.g. a fresh/CI database) that
    might import it.

    Returns {'examined': int, 'updated': int, 'counts': {...}, 'batches': int,
    'hit_time_limit': bool, 'skipped_no_checkpoint': bool}.
    """
    try:
        model = _load_model()
    except FileNotFoundError:
        settings.LOGGER.debug("run_rotation_ml_audit_until_done: no checkpoint found, skipping.")
        return {
            'examined': 0, 'updated': 0, 'counts': {}, 'batches': 0,
            'hit_time_limit': False, 'skipped_no_checkpoint': True,
        }

    start = time.monotonic()
    totals = {
        'examined': 0, 'updated': 0, 'counts': {}, 'batches': 0,
        'hit_time_limit': False, 'skipped_no_checkpoint': False,
    }
    while True:
        result = run_rotation_ml_audit(write=True, max_images=batch_size, model=model)
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

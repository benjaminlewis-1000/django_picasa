"""LVFace (ViT-based face recognition, bytedance-research/LVFace,
Glint360K-trained) -- a second, independently-trained embedding model used
as a veto gate alongside insightface's own classification (see
face_manager.models.Person.use_lvface_verification's docstring for the
full investigation behind this).

Same "second independently-trained embedding model used alongside the
primary one" shape as facenet-pytorch's role in the video face pipeline
(face_manager/video_face_pipeline.py) -- but unlike facenet's embeddings
(computed transiently, never stored), LVFace's ARE persisted on every
Face row (see Face.face_encoding_lvface_b's own comment for why), since
this one needs to support both live extraction-time encoding AND cheap
reuse during classification without re-running inference.

Three published sizes (S/B/L, all Glint360K-trained) are supported --
added 2026-10-26 alongside the kps/box mismatch investigation (see
CLAUDE.md) to properly re-test which size is actually worth using,
after the original B-only validation turned out to rest on a flawed
sample. Model files live on the persistent /models bind mount (same
convention as insightface's buffalo_l/buffalo_s packs and the
assign_faces.py encoding cache) -- not baked into the Docker image,
fetched once onto that mount the same way facenet's torch checkpoint
already is.
"""
import cv2
import numpy as np
from insightface.utils import face_align

# Same /models bind mount convention as everywhere else in this codebase
# (assign_faces.py's ENCODINGS_PKL_FILE, insightface's own buffalo_l/
# buffalo_s packs) -- not a Django setting, since none of those sibling
# paths are either.
LVFACE_MODEL_PATHS = {
    'S': '/models/LVFace-S_Glint360K.onnx',
    'B': '/models/LVFace-B_Glint360K.onnx',
    'L': '/models/LVFace-L_Glint360K.onnx',
}
# Backward-compat alias -- several call sites/tests reference this
# directly for the (still-default) B model.
LVFACE_MODEL_PATH = LVFACE_MODEL_PATHS['B']

_sessions = {}


def _get_session(size='B'):
    """Lazily builds the ONNXRuntime session for the given size -- not at
    import time, so importing this module (e.g. for a test that mocks
    everything) never requires any of the real model files to be present
    on disk. Each size gets its own independently-cached session."""
    if size not in LVFACE_MODEL_PATHS:
        raise ValueError(f"Unknown LVFace size {size!r}, expected one of {sorted(LVFACE_MODEL_PATHS)}")
    if size not in _sessions:
        import onnxruntime as ort
        _sessions[size] = ort.InferenceSession(LVFACE_MODEL_PATHS[size], providers=['CPUExecutionProvider'])
    return _sessions[size]


def reset_session(size=None):
    """Drops the cached session(s) so the next embed_from_image_array()/
    embed_from_path() call rebuilds a fresh one. Same reasoning as
    FaceExtractor._build_detector()'s own periodic-rebuild comment --
    ONNXRuntime's CPU memory arena grows to the largest allocation ever
    seen by a session and never shrinks back down for that session's
    lifetime, so a long-running loop that never rebuilds this session
    (e.g. reencode_missing_faces()'s combined image loop, found
    2026-10-25 to grow into double-digit GB RSS over a multi-hour run)
    needs a way to periodically start fresh. size=None (the default)
    resets every cached session; pass a specific size ('S'/'B'/'L') to
    reset only that one. A no-op if no session has been built yet."""
    if size is None:
        _sessions.clear()
    else:
        _sessions.pop(size, None)


def embed_from_image_array(img, kps, input_is_bgr=True, size='B'):
    """Computes an LVFace embedding for one face, given the already-
    decoded source image and that face's 5-point kps (flattened
    [x1,y1,...,x5,y5] or an already-reshaped (5, 2) array) in the
    image's own pixel coordinate space -- the exact same kps already
    stored on Face for insightface's own re-encoding path (see
    reencode_missing_faces()), so no new detection is needed.

    input_is_bgr: True for a cv2.imread()-style array (this module's own
    embed_from_path() below); False for a PIL-sourced RGB array, e.g.
    common.open_img_oriented(..., as_numpy=True)'s output, which is what
    FaceExtractor.reencode_missing_faces() already has on hand -- getting
    this wrong silently feeds the model color-swapped input with no
    error, so callers must get it right rather than this function
    guessing.

    size: 'S', 'B' (default, matches the historical single-size
    behavior), or 'L'.

    Returns a unit-normalized float32 (512,) array, or None if the
    embedding comes back degenerate (zero norm -- shouldn't happen in
    practice, same defensive check used throughout this codebase's other
    embedding code).
    """
    kps_arr = np.asarray(kps, dtype=np.float32).reshape(5, 2)
    aligned = face_align.norm_crop(img, landmark=kps_arr, image_size=112)
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB) if input_is_bgr else aligned
    chw = rgb.transpose(2, 0, 1).astype(np.float32)
    normed_input = (chw / 255.0 - 0.5) / 0.5
    batch = normed_input[np.newaxis, ...]

    session = _get_session(size)
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: batch})[0][0]

    norm = np.linalg.norm(out)
    if norm == 0:
        return None
    return (out / norm).astype(np.float32)


def embed_from_path(image_path, kps, size='B'):
    """Convenience wrapper: decodes the image from disk first. Returns
    None (rather than raising) if the file can't be decoded, matching
    this codebase's established tolerance for an occasionally-
    unreadable file during a bulk backfill -- the caller decides whether
    that's worth logging/retrying."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return embed_from_image_array(img, kps, size=size)

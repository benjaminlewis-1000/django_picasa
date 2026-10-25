"""LVFace-B (ViT-based face recognition, bytedance-research/LVFace,
Glint360K-trained) -- a second, independently-trained embedding model used
as a veto gate alongside insightface's own classification (see
face_manager.models.Person.use_lvface_verification's docstring for the
full investigation behind this).

Same "second independently-trained embedding model used alongside the
primary one" shape as facenet-pytorch's role in the video face pipeline
(face_manager/video_face_pipeline.py) -- but unlike facenet's embeddings
(computed transiently, never stored), LVFace-B's ARE persisted on every
Face row (see Face.face_encoding_lvface_b's own comment for why), since
this one needs to support both live extraction-time encoding AND cheap
reuse during classification without re-running inference.

Model file lives on the persistent /models bind mount (same convention
as insightface's buffalo_l/buffalo_s packs and the assign_faces.py
encoding cache) at LVFACE_MODEL_PATH -- not baked into the Docker image
(455MB), fetched once onto that mount the same way facenet's torch
checkpoint already is.
"""
import cv2
import numpy as np
from insightface.utils import face_align

# Same /models bind mount convention as everywhere else in this codebase
# (assign_faces.py's ENCODINGS_PKL_FILE, insightface's own buffalo_l/
# buffalo_s packs) -- not a Django setting, since none of those sibling
# paths are either.
LVFACE_MODEL_PATH = '/models/LVFace-B_Glint360K.onnx'

_session = None


def _get_session():
    """Lazily builds the ONNXRuntime session -- not at import time, so
    importing this module (e.g. for a test that mocks everything) never
    requires the real 455MB model file to be present on disk."""
    global _session
    if _session is None:
        import onnxruntime as ort
        _session = ort.InferenceSession(LVFACE_MODEL_PATH, providers=['CPUExecutionProvider'])
    return _session


def reset_session():
    """Drops the cached session so the next embed_from_image_array()/
    embed_from_path() call rebuilds a fresh one. Same reasoning as
    FaceExtractor._build_detector()'s own periodic-rebuild comment --
    ONNXRuntime's CPU memory arena grows to the largest allocation ever
    seen by a session and never shrinks back down for that session's
    lifetime, so a long-running loop that never rebuilds this session
    (e.g. reencode_missing_faces()'s combined image loop, found
    2026-10-25 to grow into double-digit GB RSS over a multi-hour run)
    needs a way to periodically start fresh. A no-op if no session has
    been built yet."""
    global _session
    _session = None


def embed_from_image_array(img, kps, input_is_bgr=True):
    """Computes an LVFace-B embedding for one face, given the already-
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

    session = _get_session()
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: batch})[0][0]

    norm = np.linalg.norm(out)
    if norm == 0:
        return None
    return (out / norm).astype(np.float32)


def embed_from_path(image_path, kps):
    """Convenience wrapper: decodes the image from disk first. Returns
    None (rather than raising) if the file can't be decoded, matching
    this codebase's established tolerance for an occasionally-
    unreadable file during a bulk backfill -- the caller decides whether
    that's worth logging/retrying."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return embed_from_image_array(img, kps)

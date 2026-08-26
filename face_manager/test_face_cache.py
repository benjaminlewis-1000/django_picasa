"""Test-only cache for expensive real insightface inference.

Not imported by any app code -- this exists purely so the face_manager
test suite doesn't re-run CPU face detection/encoding on every test run
against images that haven't changed, using a detector that hasn't changed.

Cache key = sha256(image bytes) + sha256(pyramidal_detector.py source).
If either the source image or the pyramidal detector's code changes, the
key changes and the cache is transparently recomputed -- no manual
invalidation needed.
"""
import hashlib
import inspect
import os
import pickle

import cv2

from face_manager import pyramidal_detector

# Backed by a bind-mounted host directory so it survives container
# restarts/rebuilds, not the ephemeral per-test MEDIA_ROOT.
CACHE_DIR = os.environ.get("FACE_TEST_CACHE_DIR", "/media/_test_face_cache")


def _detector_source_hash() -> str:
    src = inspect.getsource(pyramidal_detector)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


def _image_hash(image_path: str) -> str:
    with open(image_path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


_PLAIN_FIELDS = ("bbox", "det_score", "embedding", "age", "gender", "kps")


def _to_plain_dict(face) -> dict:
    """insightface Face objects don't reliably round-trip through pickle
    (they carry numpy views tied to the detector's session). Only the
    plain numeric fields tests actually need are cached."""
    return {k: face[k] for k in _PLAIN_FIELDS if k in face}


def cached_detect(detector: "pyramidal_detector.PyramidalDetector", image_path: str) -> list:
    """Run detector.get() on image_path, or return the cached result (as
    plain dicts, not insightface Face objects) if this exact (image bytes,
    detector source) pair has been seen before."""
    key = f"{_image_hash(image_path)}_{_detector_source_hash()}.pkl"
    cache_path = os.path.join(CACHE_DIR, key)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    np_image = cv2.imread(image_path)
    result = [_to_plain_dict(f) for f in detector.get(np_image)]

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(result, fh)
    return result

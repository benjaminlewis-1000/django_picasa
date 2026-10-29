"""Test-only cache for expensive real ffprobe/exiftool subprocess calls.

Not imported by any app code -- this exists purely so the filepopulator
test suite doesn't re-run real ffprobe/exiftool against the same small
set of fixture files over and over (VideoIngestionTests alone has dozens
of test methods that each call add_videos_from_root_dir()/
create_video_file(), which internally calls _run_ffprobe()/_run_exiftool()
-- the same handful of real/synthetic fixture files, repeated across
every one of those tests).

Mirrors face_manager/test_face_cache.py's exact pattern: cache key =
sha256(file bytes) + sha256(video_scripts.py source). If either the
source file or video_scripts.py's own code changes (including
EXIFTOOL_FIELDS or the ffprobe/exiftool command lines themselves), the
key changes and the cache is transparently recomputed -- no manual
invalidation needed.

Two things this does NOT share with test_face_cache.py, found via a
real test failure, not assumed:
- A nonexistent file (one real test deliberately passes a path that was
  never created, to exercise the "file doesn't exist" failure path) has
  no stable content to hash -- _file_hash() would raise FileNotFoundError
  before the real function ever got a chance to handle that case the way
  it always has. Falls back to calling the real function directly,
  uncached, whenever the file can't be read for hashing.
- Unlike deterministic ONNX inference, a subprocess call can fail
  TRANSIENTLY (e.g. starved of CPU by something else running on the same
  host) in a way that has nothing to do with the file's actual content.
  Caching that failure would make it permanent -- every later test run
  would keep replaying a transient hiccup forever instead of the real,
  correct result a retry would have gotten. Only a genuinely successful
  result (ffprobe: not None; exiftool: a non-empty dict) is ever written
  to the cache; a failure/empty result is always recomputed fresh next
  time, exactly matching the original (uncached) behavior for that case.
"""
import hashlib
import inspect
import json
import os

from filepopulator import video_scripts

# Captured at import time, BEFORE anything patches video_scripts._run_ffprobe/
# _run_exiftool to point at the cached_* wrappers below -- calling through
# the module attribute instead (video_scripts._run_ffprobe(...)) would
# resolve to whatever's CURRENTLY installed there once patched, i.e. the
# wrapper calling itself, infinite recursion.
_original_run_ffprobe = video_scripts._run_ffprobe
_original_run_exiftool = video_scripts._run_exiftool

# Backed by a bind-mounted host directory so it survives container
# restarts/rebuilds, not the ephemeral per-test MEDIA_ROOT.
CACHE_DIR = os.environ.get("VIDEO_TEST_CACHE_DIR", "/media/_test_video_cache")


def _video_scripts_source_hash() -> str:
    src = inspect.getsource(video_scripts)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


def _file_hash(file_path: str) -> str | None:
    try:
        with open(file_path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except OSError:
        return None


def _cache_path(file_path: str, suffix: str) -> str | None:
    file_hash = _file_hash(file_path)
    if file_hash is None:
        return None
    key = f"{file_hash}_{_video_scripts_source_hash()}_{suffix}.json"
    return os.path.join(CACHE_DIR, key)


def _write_cache(cache_path: str, result) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w") as fh:
        json.dump({"value": result}, fh)


def cached_run_ffprobe(file_path: str):
    cache_path = _cache_path(file_path, "ffprobe")
    if cache_path is None:
        return _original_run_ffprobe(file_path)

    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            return json.load(fh)["value"]

    result = _original_run_ffprobe(file_path)
    if result is not None:  # never cache a failure -- see module docstring
        _write_cache(cache_path, result)
    return result


def cached_run_exiftool(file_path: str):
    cache_path = _cache_path(file_path, "exiftool")
    if cache_path is None:
        return _original_run_exiftool(file_path)

    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            return json.load(fh)["value"]

    result = _original_run_exiftool(file_path)
    if result:  # never cache a failure/empty result -- see module docstring
        _write_cache(cache_path, result)
    return result

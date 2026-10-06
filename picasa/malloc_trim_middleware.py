# Real incident, 2026-10-05: found while investigating gunicorn worker
# memory growth triggered by real mobile-app traffic (the face-tagging
# workflow's face_array/face_source endpoints -- api/views.py's
# KeyedImageView -- repeatedly decode/crop real photos and video frames to
# serve face thumbnails). This is the same class of native-buffer
# fragmentation already root-caused and fixed in the batch face-extraction
# pipeline (face_manager/face_extract_encode.py's _trim_malloc_arena()) --
# PIL/cv2/onnxruntime native allocations that glibc's malloc arena holds
# onto rather than returning to the OS, invisible to Python's own garbage
# collector since it's not a Python-level leak.
#
# Rather than patching each image-decoding view individually (there's more
# than just face_array/face_source -- full_big/full_medium/full_small,
# slideshow, and the video-frame extraction paths all decode real media
# too), this is deliberately a response-level middleware: malloc_trim(0) is
# cheap enough to call after every single request regardless of what that
# request actually did, so this covers any current or future view that
# decodes an image/video frame without needing to track down and patch
# each one by hand.
import ctypes

try:
    _libc = ctypes.CDLL('libc.so.6')
except OSError:
    _libc = None


class MallocTrimMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if _libc is not None:
            try:
                _libc.malloc_trim(0)
            except Exception:
                pass
        return response

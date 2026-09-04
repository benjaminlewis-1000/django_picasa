import pillow_heif

# Registers a PIL plugin so PIL.Image.open() handles .heic/.heif
# transparently, the same as any other format -- must happen before any
# Image.open() call anywhere in the app, so it lives at import time here
# rather than deferred into filepopulator's HEIC-specific code.
pillow_heif.register_heif_opener()

from .open_img_oriented import open_img_oriented, apply_exif_orientation
from .equalize import clahe_equalize_bgr
from .advisory_lock import advisory_lock

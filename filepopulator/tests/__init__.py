"""filepopulator tests package -- shared fixtures."""
import numpy as np


def _tiny_jpeg_bytes(size=(30, 30)):
    import cv2
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    from io import BytesIO
    return BytesIO(buf).read()

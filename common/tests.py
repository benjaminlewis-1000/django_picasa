import os

from django.conf import settings
from django.test import TestCase

from common.open_img_oriented import open_img_oriented


class OpenImgOrientedTests(TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            open_img_oriented("/tmp/does_not_exist_at_all.jpg", as_numpy=True)

    def test_reads_normal_jpeg_as_numpy(self):
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/naming/good/1.JPG"
        arr = open_img_oriented(path, as_numpy=True)
        self.assertEqual(len(arr.shape), 3)

    def test_reads_normal_jpeg_as_pil_image(self):
        path = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/naming/good/1.JPG"
        img = open_img_oriented(path, as_numpy=False)
        self.assertEqual(img.mode, "RGB")

    def test_corrupted_image_raises_rather_than_returning_none(self):
        # KNOWN GAP (documented, not fixed here): open_img_oriented()'s
        # try/except only wraps the initial PIL.Image.open() call, which
        # succeeds even for a truncated/broken JPEG (PIL parses the header
        # lazily and doesn't decode pixels yet). The actual decode error
        # only surfaces later, at `np.array(image)` (as_numpy=True) or a
        # caller's own pixel access -- outside this function's guard -- so
        # despite this function *looking* like it handles bad images
        # gracefully (returns None on failure), corrupted files actually
        # propagate an unguarded OSError. This is the root of the
        # `find_and_encode_faces()`/`_generate_md5_hash()` retry-forever
        # bugs documented in face_manager/filepopulator tests -- this test
        # pins down where in the call stack it actually originates.
        path = "/photos/corrupted/20220827_130217.jpg"
        with self.assertRaises(OSError):
            open_img_oriented(path, as_numpy=True)

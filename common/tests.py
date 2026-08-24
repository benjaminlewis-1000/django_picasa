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

    def test_corrupted_image_returns_none(self):
        # Regression test for a fixed bug: open_img_oriented()'s try/except
        # used to only wrap the initial PIL.Image.open() call, which
        # succeeds even for a truncated/broken JPEG (PIL parses the header
        # lazily and doesn't decode pixels yet). The actual decode error
        # only surfaced later, at `np.array(image)` (as_numpy=True) --
        # outside the old guard -- so despite *looking* like it handled bad
        # images gracefully, corrupted files actually raised an unguarded
        # OSError. Now that call is wrapped too, so this genuinely returns
        # None on failure as documented.
        path = "/photos/corrupted/20220827_130217.jpg"
        self.assertIsNone(open_img_oriented(path, as_numpy=True))

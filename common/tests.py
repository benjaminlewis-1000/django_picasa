import os

import numpy as np
from django.conf import settings
from django.test import TestCase
from PIL import Image

from common.open_img_oriented import open_img_oriented, apply_exif_orientation


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


def _diagonal_marker_image(width=200, height=150, color=(120, 60, 200)):
    # Same construction as ci_fixtures/generate_fixtures.py's
    # _solid_jpeg_bytes(): a solid color block with a white diagonal
    # stripe breaking symmetry, so a transform's effect on geometry is
    # directly checkable by locating the white pixels afterward.
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = color
    for i in range(min(width, height)):
        img[i, i] = (255, 255, 255)
    return Image.fromarray(img)


class ApplyExifOrientationTests(TestCase):
    """Regression test for a fixed bug: this module used to only handle
    EXIF orientations 3, 6, 8 (via rotate()), silently doing nothing for
    2, 4, 5, 7 -- while filepopulator/models.py had its own separate,
    complete 8-value implementation (via transpose()). apply_exif_
    orientation() is now that single shared implementation, used by both.

    Expected (width, height) and first/last white-pixel (y, x) coordinates
    below were captured directly from this function's own output on a
    known 200x150 source image, not re-derived by hand -- they pin down
    today's verified-correct behavior (matching filepopulator's original,
    already-relied-upon transform) as a regression baseline."""

    def setUp(self):
        self.image = _diagonal_marker_image()

    def _white_pixel_range(self, out_image):
        arr = np.array(out_image)
        coords = np.argwhere((arr == 255).all(axis=2))
        return coords[0].tolist(), coords[-1].tolist()

    def test_orientation_0_is_a_no_op_same_as_1(self):
        # orientation 0 isn't a standard EXIF value, but is observed in
        # ~1090 images in the live library -- explicitly treated the same
        # as 1 (no rotation), not left to fall through unhandled.
        out0 = apply_exif_orientation(self.image, 0)
        out1 = apply_exif_orientation(self.image, 1)
        self.assertEqual(out0.size, self.image.size)
        self.assertEqual(np.array(out0).tolist(), np.array(out1).tolist())
        self.assertEqual(self._white_pixel_range(out0), ([0, 0], [149, 149]))

    def test_orientation_2_mirrors_left_right_without_resizing(self):
        # Previously unhandled entirely (silent no-op) -- now a real flip.
        out = apply_exif_orientation(self.image, 2)
        self.assertEqual(out.size, self.image.size)
        self.assertEqual(self._white_pixel_range(out), ([0, 199], [149, 50]))

    def test_orientation_3_rotates_180_without_resizing(self):
        out = apply_exif_orientation(self.image, 3)
        self.assertEqual(out.size, self.image.size)
        self.assertEqual(self._white_pixel_range(out), ([0, 50], [149, 199]))

    def test_orientation_4_mirrors_top_bottom_without_resizing(self):
        # Previously unhandled entirely (silent no-op) -- now a real flip.
        out = apply_exif_orientation(self.image, 4)
        self.assertEqual(out.size, self.image.size)
        self.assertEqual(self._white_pixel_range(out), ([0, 149], [149, 0]))

    def test_orientation_5_flips_and_rotates_swapping_dimensions(self):
        # Previously unhandled entirely (silent no-op).
        out = apply_exif_orientation(self.image, 5)
        self.assertEqual(out.size, (self.image.size[1], self.image.size[0]))
        self.assertEqual(self._white_pixel_range(out), ([0, 0], [149, 149]))

    def test_orientation_6_rotates_swapping_dimensions(self):
        out = apply_exif_orientation(self.image, 6)
        self.assertEqual(out.size, (self.image.size[1], self.image.size[0]))
        self.assertEqual(self._white_pixel_range(out), ([0, 149], [149, 0]))

    def test_orientation_7_flips_and_rotates_swapping_dimensions(self):
        # Previously unhandled entirely (silent no-op).
        out = apply_exif_orientation(self.image, 7)
        self.assertEqual(out.size, (self.image.size[1], self.image.size[0]))
        self.assertEqual(self._white_pixel_range(out), ([50, 0], [199, 149]))

    def test_orientation_8_rotates_swapping_dimensions(self):
        out = apply_exif_orientation(self.image, 8)
        self.assertEqual(out.size, (self.image.size[1], self.image.size[0]))
        self.assertEqual(self._white_pixel_range(out), ([50, 149], [199, 0]))

    def test_unrecognized_orientation_value_is_left_untouched(self):
        out = apply_exif_orientation(self.image, 99)
        self.assertEqual(np.array(out).tolist(), np.array(self.image).tolist())


class OpenImgOrientedRealOrientationFixturesTests(TestCase):
    """End-to-end check that open_img_oriented() actually reads the EXIF
    Orientation tag and applies apply_exif_orientation() with it, using
    the real (JPEG-compressed) per-orientation fixtures. Only asserts on
    output dimensions here (whether the transform swapped width/height),
    not exact pixel positions -- a single-pixel-wide diagonal stripe
    against a solid background doesn't reliably survive JPEG's lossy
    compression, so pixel-perfect geometry is covered instead by
    ApplyExifOrientationTests against an uncompressed array."""

    ORIENTATION_DIR = f"{settings.FILEPOPULATOR_VAL_DIRECTORY}/orientation"

    def test_dimensions_swap_only_for_the_four_90_degree_orientations(self):
        for orientation in range(1, 9):
            path = f"{self.ORIENTATION_DIR}/2_{orientation}.jpg"
            with self.subTest(orientation=orientation):
                img = open_img_oriented(path, as_numpy=False)
                if orientation in (5, 6, 7, 8):
                    self.assertGreater(img.size[1], img.size[0])
                else:
                    self.assertGreater(img.size[0], img.size[1])

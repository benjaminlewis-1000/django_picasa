"""face_manager.rotation_classifier_model -- the pure numpy/PIL
preprocessing helpers for the planned ML orientation-classifier signal
(see CLAUDE.md's "rotation-review ML classifier" open TODO). build_model()
itself (loads a real torchvision backbone + pretrained weights, needs
network access the first time) is exercised separately, tagged slow.
"""
import unittest

import numpy as np
from django.test import TestCase, tag

from face_manager.rotation_classifier_model import (
    INPUT_SIZE, NUM_CLASSES, apply_synthetic_rotation, letterbox_to_square,
    to_model_input, undo_rotation_degrees,
)


def _asymmetric_image(h=60, w=100):
    """A real (not accidentally-symmetric) HxWx3 uint8 array -- distinct
    markers in each corner so a rotation's actual effect is directly
    checkable, not just its shape."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[0:5, 0:5] = (255, 0, 0)      # top-left: red
    img[0:5, -5:] = (0, 255, 0)      # top-right: green
    img[-5:, 0:5] = (0, 0, 255)      # bottom-left: blue
    img[-5:, -5:] = (255, 255, 0)    # bottom-right: yellow
    return img


class ApplySyntheticRotationTests(unittest.TestCase):
    def test_k0_is_unchanged(self):
        img = _asymmetric_image()
        np.testing.assert_array_equal(apply_synthetic_rotation(img, 0), img)

    def test_k1_rotates_counterclockwise_90(self):
        img = _asymmetric_image()
        rotated = apply_synthetic_rotation(img, 1)
        # np.rot90(k=1) is CCW: the original top-right (green) corner
        # lands at top-left after a 90-degree CCW turn.
        self.assertEqual(rotated.shape, (img.shape[1], img.shape[0], 3))
        np.testing.assert_array_equal(rotated[0, 0], (0, 255, 0))

    def test_k2_rotates_180(self):
        img = _asymmetric_image()
        rotated = apply_synthetic_rotation(img, 2)
        # top-left (red) lands at bottom-right under a full 180.
        np.testing.assert_array_equal(rotated[-1, -1], (255, 0, 0))

    def test_four_rotations_return_to_start(self):
        img = _asymmetric_image()
        result = img
        for _ in range(4):
            result = apply_synthetic_rotation(result, 1)
        np.testing.assert_array_equal(result, img)

    def test_output_is_contiguous(self):
        # np.rot90 returns a VIEW with non-trivial strides -- a caller
        # (e.g. PIL.Image.fromarray, or a torch.from_numpy) can behave
        # incorrectly or raise on a non-contiguous array, so this must
        # actually copy, not just rely on rot90's own return value.
        img = _asymmetric_image()
        rotated = apply_synthetic_rotation(img, 1)
        self.assertTrue(rotated.flags['C_CONTIGUOUS'])


class LetterboxToSquareTests(unittest.TestCase):
    def test_output_is_square_and_requested_size(self):
        img = _asymmetric_image(h=60, w=100)
        out = letterbox_to_square(img, size=224)
        self.assertEqual(out.shape, (224, 224, 3))

    def test_landscape_image_gets_horizontal_bands_not_stretched(self):
        # A wide (landscape) source should be padded top/bottom, not
        # stretched vertically to fill the square -- check the corner
        # markers land close to where a correct aspect-preserving resize
        # would put them, not smeared across the whole height.
        img = _asymmetric_image(h=60, w=100)  # aspect ratio 100:60
        out = letterbox_to_square(img, size=224)
        # top rows should be pure pad color (no content yet)
        self.assertTrue((out[0] == 128).all())
        # some row near vertical center should contain non-pad content
        center_row = out[112]
        self.assertFalse((center_row == 128).all())

    def test_square_image_needs_no_padding(self):
        img = _asymmetric_image(h=80, w=80)
        out = letterbox_to_square(img, size=224)
        # a square source resized to a square target should have content
        # reaching every edge (no pad bands at all)
        self.assertFalse((out[0] == 128).all())
        self.assertFalse((out[:, 0] == 128).all())

    def test_does_not_mutate_input(self):
        img = _asymmetric_image()
        original = img.copy()
        letterbox_to_square(img, size=224)
        np.testing.assert_array_equal(img, original)


class ToModelInputTests(unittest.TestCase):
    def test_shape_is_chw(self):
        img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        out = to_model_input(img)
        self.assertEqual(out.shape, (3, INPUT_SIZE, INPUT_SIZE))

    def test_dtype_is_float32(self):
        img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        out = to_model_input(img)
        self.assertEqual(out.dtype, np.float32)

    def test_mid_gray_input_is_not_all_zero(self):
        # A genuinely uniform mid-value input should still produce a
        # real (non-degenerate) normalized value, not something that
        # silently collapses to 0 everywhere (which would make the
        # letterbox pad color invisible to the model).
        img = np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8)
        out = to_model_input(img)
        self.assertFalse(np.allclose(out, 0.0))


class UndoRotationDegreesTests(unittest.TestCase):
    def test_every_class_has_a_defined_mapping(self):
        for k in range(NUM_CLASSES):
            self.assertIsInstance(undo_rotation_degrees(k), float)

    def test_class_0_is_already_upright(self):
        self.assertEqual(undo_rotation_degrees(0), 0.0)

    def test_mapping_is_consistent_with_apply_synthetic_rotation_convention(self):
        # apply_synthetic_rotation(img, 1) rotates the image 90 degrees
        # CCW (np.rot90's own convention) to SIMULATE "this is what a
        # 90-degree-CW-rotated photo looks like" -- undoing that requires
        # rotating back CW, i.e. +90 under rotation_geometry's CCW-
        # positive atan2 convention... concretely: class 1 must map to
        # a nonzero correction, and classes 1 and 3 must be opposite in
        # sign (one undoes a CW rotation, the other a CCW one).
        self.assertEqual(undo_rotation_degrees(1), -undo_rotation_degrees(3))
        self.assertNotEqual(undo_rotation_degrees(1), 0.0)
        self.assertEqual(abs(undo_rotation_degrees(2)), 180.0)


@tag("slow")
class BuildModelRealWeightsTests(TestCase):
    """Loads the real torchvision backbone + pretrained ImageNet
    weights (a real network fetch the first time, cached afterward) --
    tagged slow since this is meaningfully heavier than this module's
    other, pure-numpy tests."""

    def test_build_model_produces_correct_output_shape(self):
        import torch

        from face_manager.rotation_classifier_model import INPUT_SIZE as SIZE
        from face_manager.rotation_classifier_model import build_model
        model = build_model(pretrained=True)
        model.eval()
        x = torch.zeros(2, 3, SIZE, SIZE)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(tuple(out.shape), (2, NUM_CLASSES))

    def test_build_model_without_pretrained_weights_still_works(self):
        import torch

        from face_manager.rotation_classifier_model import INPUT_SIZE as SIZE
        from face_manager.rotation_classifier_model import build_model
        model = build_model(pretrained=False)
        model.eval()
        x = torch.zeros(1, 3, SIZE, SIZE)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(tuple(out.shape), (1, NUM_CLASSES))

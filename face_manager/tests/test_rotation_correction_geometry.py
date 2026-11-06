"""Tests for face_manager.rotation_correction_geometry -- pure geometry,
no DB/model/PIL-decode dependency. The point-transform formulas are
locked against values empirically measured from real PIL.Image.
transpose() output (see that module's own docstring for how).
"""
import math
import unittest

from face_manager.rotation_correction_geometry import (
    snap_to_canonical_rotation,
    transform_box,
    transform_kps,
    transform_point,
    transformed_image_size,
)
from face_manager.rotation_geometry import eye_angle_degrees


class SnapToCanonicalRotationTests(unittest.TestCase):
    def test_exact_values_snap_to_themselves(self):
        self.assertEqual(snap_to_canonical_rotation(90.0), 90.0)
        self.assertEqual(snap_to_canonical_rotation(-90.0), -90.0)
        self.assertEqual(snap_to_canonical_rotation(180.0), 180.0)

    def test_near_values_snap_to_nearest(self):
        self.assertEqual(snap_to_canonical_rotation(85.0), 90.0)
        self.assertEqual(snap_to_canonical_rotation(-88.0), -90.0)
        self.assertEqual(snap_to_canonical_rotation(170.0), 180.0)
        self.assertEqual(snap_to_canonical_rotation(-170.0), 180.0)


class TransformedImageSizeTests(unittest.TestCase):
    def test_90_and_negative_90_swap_dimensions(self):
        self.assertEqual(transformed_image_size(800, 600, 90.0), (600, 800))
        self.assertEqual(transformed_image_size(800, 600, -90.0), (600, 800))

    def test_180_keeps_dimensions(self):
        self.assertEqual(transformed_image_size(800, 600, 180.0), (800, 600))


class TransformPointTests(unittest.TestCase):
    """Locked against values empirically measured from real PIL
    Image.transpose() output (W=10, H=6) -- see the module docstring.
    """
    W, H = 10, 6

    def test_rotate_90_matches_pil(self):
        self.assertEqual(transform_point(2, 1, self.W, self.H, 90.0), (1, 7))
        self.assertEqual(transform_point(7, 4, self.W, self.H, 90.0), (4, 2))
        self.assertEqual(transform_point(0, 0, self.W, self.H, 90.0), (0, 9))
        self.assertEqual(transform_point(9, 5, self.W, self.H, 90.0), (5, 0))

    def test_rotate_180_matches_pil(self):
        self.assertEqual(transform_point(2, 1, self.W, self.H, 180.0), (7, 4))
        self.assertEqual(transform_point(0, 0, self.W, self.H, 180.0), (9, 5))

    def test_rotate_negative_90_matches_pil(self):
        self.assertEqual(transform_point(2, 1, self.W, self.H, -90.0), (4, 2))
        self.assertEqual(transform_point(7, 4, self.W, self.H, -90.0), (1, 7))
        self.assertEqual(transform_point(0, 0, self.W, self.H, -90.0), (5, 0))
        self.assertEqual(transform_point(9, 5, self.W, self.H, -90.0), (0, 9))

    def test_invalid_degrees_raises(self):
        with self.assertRaises(ValueError):
            transform_point(0, 0, self.W, self.H, 45.0)


class TransformBoxTests(unittest.TestCase):
    def test_box_maps_to_axis_aligned_box(self):
        # A box from (2,1) to (7,4) in a 10x6 image, rotated 90.
        box = transform_box(2, 1, 7, 4, 10, 6, 90.0)
        # Corners (2,1)->(1,7) and (7,4)->(4,2); box is their min/max.
        self.assertEqual(box, (1, 2, 4, 7))


class TransformKpsAndEyeAngleRoundTripTests(unittest.TestCase):
    """The real correctness property: transforming a face's kps by the
    canonical rotation that was itself derived from that face's own
    measured eye angle should bring the corrected angle to ~0.
    """

    def _kps_for_eye_angle(self, angle_deg, cx=100.0, cy=100.0, eye_sep=50.0):
        half = eye_sep / 2.0
        rad = math.radians(angle_deg)
        dx = half * math.cos(rad)
        dy = half * math.sin(rad)
        left_eye = (cx - dx, cy - dy)
        right_eye = (cx + dx, cy + dy)
        return [left_eye[0], left_eye[1], right_eye[0], right_eye[1],
                cx, cy + 20, cx - 15, cy + 35, cx + 15, cy + 35]

    def _assert_corrects_to_upright(self, measured_angle, snapped_degrees):
        kps = self._kps_for_eye_angle(measured_angle)
        corrected = transform_kps(kps, 200, 200, snapped_degrees)
        corrected_angle = eye_angle_degrees(corrected)
        self.assertAlmostEqual(
            min(abs(corrected_angle), 360 - abs(corrected_angle)), 0, delta=1,
            msg=f"measured={measured_angle} snapped={snapped_degrees} -> corrected={corrected_angle}",
        )

    def test_90_degree_measured_corrects_with_rotate_90(self):
        self._assert_corrects_to_upright(90.0, 90.0)

    def test_negative_90_degree_measured_corrects_with_rotate_negative_90(self):
        self._assert_corrects_to_upright(-90.0, -90.0)

    def test_180_degree_measured_corrects_with_rotate_180(self):
        self._assert_corrects_to_upright(180.0, 180.0)

    def test_wrong_direction_does_not_correct(self):
        # Using the OPPOSITE canonical rotation from what a +90 signal
        # calls for should NOT land near upright -- confirms the test
        # above isn't vacuously true.
        kps = self._kps_for_eye_angle(90.0)
        corrected = transform_kps(kps, 200, 200, -90.0)
        corrected_angle = eye_angle_degrees(corrected)
        self.assertGreater(min(abs(corrected_angle), 360 - abs(corrected_angle)), 45)


if __name__ == '__main__':
    unittest.main()

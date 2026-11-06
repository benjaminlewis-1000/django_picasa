"""Tests for face_manager.rotation_geometry -- pure geometry, no DB/model
dependency. See CLAUDE.md's 2026-11-03 orientation-detection-signal write-up
for the design/validation this is built on.
"""
import math
import unittest

from face_manager.rotation_geometry import (
    aggregate_orientation,
    circular_distance,
    circular_mean,
    circular_median,
    eye_angle_degrees,
    should_overwrite_rotation_proposal,
)


def _kps_for_eye_angle(angle_deg, eye_separation=50.0):
    """Build a synthetic 5-point kps array whose eye-to-eye vector sits at
    the given angle. Nose/mouth points are placeholders (unused by
    eye_angle_degrees) but included to match the real 5-point shape.
    """
    cx, cy = 100.0, 100.0
    half = eye_separation / 2.0
    rad = math.radians(angle_deg)
    dx = half * math.cos(rad)
    dy = half * math.sin(rad)
    left_eye = (cx - dx, cy - dy)
    right_eye = (cx + dx, cy + dy)
    nose = (cx, cy + 20)
    mouth_l = (cx - 15, cy + 35)
    mouth_r = (cx + 15, cy + 35)
    return [left_eye, right_eye, nose, mouth_l, mouth_r]


class EyeAngleDegreesTests(unittest.TestCase):
    def test_upright_face_is_near_zero(self):
        kps = _kps_for_eye_angle(0.0)
        self.assertAlmostEqual(eye_angle_degrees(kps), 0.0, places=5)

    def test_flat_list_kps_also_works(self):
        pts = _kps_for_eye_angle(0.0)
        flat = [coord for point in pts for coord in point]
        self.assertAlmostEqual(eye_angle_degrees(flat), 0.0, places=5)

    def test_rotated_90_detected(self):
        kps = _kps_for_eye_angle(90.0)
        self.assertAlmostEqual(eye_angle_degrees(kps), 90.0, places=5)

    def test_rotated_negative_90_detected(self):
        kps = _kps_for_eye_angle(-90.0)
        self.assertAlmostEqual(eye_angle_degrees(kps), -90.0, places=5)

    def test_rotated_180_detected(self):
        kps = _kps_for_eye_angle(180.0)
        # atan2's range means 180 and -180 are the same angle; accept either.
        angle = eye_angle_degrees(kps)
        self.assertAlmostEqual(abs(angle), 180.0, places=5)


class CircularDistanceTests(unittest.TestCase):
    def test_simple_case_no_wraparound(self):
        self.assertAlmostEqual(circular_distance(10, 30), 20)

    def test_wraparound_gap_is_small(self):
        # 2 and 358 are only 4 degrees apart circularly, not 356.
        self.assertAlmostEqual(circular_distance(2, 358), 4)

    def test_negative_and_positive_near_zero(self):
        self.assertAlmostEqual(circular_distance(-2, 2), 4)

    def test_opposite_angles_give_180(self):
        self.assertAlmostEqual(circular_distance(0, 180), 180)

    def test_distance_is_symmetric(self):
        self.assertAlmostEqual(circular_distance(10, 350), circular_distance(350, 10))

    def test_distance_never_exceeds_180(self):
        self.assertLessEqual(circular_distance(0, 179.9), 180)
        self.assertLessEqual(circular_distance(1, 358), 180)


class CircularMeanTests(unittest.TestCase):
    def test_simple_cluster_no_wraparound(self):
        self.assertAlmostEqual(circular_mean([10, 20, 30]), 20, places=5)

    def test_straddles_zero_degree_wraparound(self):
        # Naive arithmetic mean of [355, 5] is 180 -- the worst possible
        # answer. The correct circular mean is 0.
        self.assertAlmostEqual(circular_mean([355, 5]), 0, places=5)

    def test_straddles_zero_three_points(self):
        self.assertAlmostEqual(circular_mean([358, 0, 2]), 0, places=5)


class CircularMedianTests(unittest.TestCase):
    def test_simple_cluster_no_wraparound(self):
        self.assertAlmostEqual(circular_median([10, 20, 30]), 20, places=5)

    def test_single_value(self):
        self.assertAlmostEqual(circular_median([42]), 42, places=5)

    def test_straddles_zero_degree_wraparound(self):
        # Naive numeric median of [358, 0, 2] would be 0 by luck here, but
        # [359, 1] averaged naively gives 180 -- wrong. Confirm the real
        # wraparound case comes back near 0, not 180.
        self.assertAlmostEqual(circular_median([359, 1, 0]), 0, places=4)

    def test_even_count_straddling_wraparound(self):
        result = circular_median([355, 5])
        self.assertAlmostEqual(circular_distance(result, 0), 0, places=4)


class AggregateOrientationTests(unittest.TestCase):
    def test_zero_or_one_face_returns_none(self):
        self.assertIsNone(aggregate_orientation([]))
        self.assertIsNone(aggregate_orientation([42]))

    def test_two_faces_agreeing_within_threshold(self):
        result = aggregate_orientation([88, 92])
        self.assertIsNotNone(result)
        estimate, inlier_count = result
        self.assertAlmostEqual(estimate, 90, delta=1)
        self.assertEqual(inlier_count, 2)

    def test_two_faces_disagreeing_falls_back(self):
        self.assertIsNone(aggregate_orientation([0, 90]))

    def test_two_faces_agree_across_wraparound(self):
        # 358 and 2 are only 4 degrees apart circularly.
        result = aggregate_orientation([358, 2])
        self.assertIsNotNone(result)
        estimate, _ = result
        self.assertAlmostEqual(circular_distance(estimate, 0), 0, delta=1)

    def test_three_faces_all_agree(self):
        result = aggregate_orientation([88, 90, 93])
        self.assertIsNotNone(result)
        estimate, inlier_count = result
        self.assertAlmostEqual(estimate, 90, delta=2)
        self.assertEqual(inlier_count, 3)

    def test_three_faces_one_outlier_is_dropped(self):
        # Two faces cluster near 90; one is a wildly different stray
        # detection (e.g. a bad landmark fit) near -30. The outlier should
        # be dropped and the verdict should come from the two agreeing faces.
        result = aggregate_orientation([88, 92, -30])
        self.assertIsNotNone(result)
        estimate, inlier_count = result
        self.assertAlmostEqual(estimate, 90, delta=2)
        self.assertEqual(inlier_count, 2)

    def test_three_faces_no_agreement_falls_back(self):
        self.assertIsNone(aggregate_orientation([0, 90, 180]))

    def test_outlier_detection_across_wraparound(self):
        # Two faces cluster near 358/2 (effectively 0), one outlier near 150.
        result = aggregate_orientation([358, 2, 150])
        self.assertIsNotNone(result)
        estimate, inlier_count = result
        self.assertAlmostEqual(circular_distance(estimate, 0), 0, delta=2)
        self.assertEqual(inlier_count, 2)

    def test_custom_agreement_threshold(self):
        # 0 and 15 disagree under the default 20-degree threshold's sibling
        # test above would pass, but confirm a tighter custom threshold
        # correctly rejects a 15-degree gap.
        self.assertIsNone(aggregate_orientation([0, 15], agreement_threshold_deg=10))
        result = aggregate_orientation([0, 15], agreement_threshold_deg=20)
        self.assertIsNotNone(result)


class FourOrientationDeterministicDetectionTests(unittest.TestCase):
    """Per the agreed design: rotate kps through all four orientations and
    confirm each is detected deterministically, using >=2 agreeing faces
    (the real aggregation policy), not a single face in isolation.
    """

    def _angles_for_two_faces(self, base_angle, jitter=3.0):
        return [base_angle - jitter, base_angle + jitter]

    def test_upright_0_degrees(self):
        angles = [eye_angle_degrees(_kps_for_eye_angle(a))
                  for a in self._angles_for_two_faces(0.0)]
        result = aggregate_orientation(angles)
        self.assertIsNotNone(result)
        estimate, _ = result
        self.assertAlmostEqual(circular_distance(estimate, 0), 0, delta=1)

    def test_rotated_90_degrees(self):
        angles = [eye_angle_degrees(_kps_for_eye_angle(a))
                  for a in self._angles_for_two_faces(90.0)]
        result = aggregate_orientation(angles)
        self.assertIsNotNone(result)
        estimate, _ = result
        self.assertAlmostEqual(circular_distance(estimate, 90), 0, delta=1)

    def test_rotated_negative_90_degrees(self):
        angles = [eye_angle_degrees(_kps_for_eye_angle(a))
                  for a in self._angles_for_two_faces(-90.0)]
        result = aggregate_orientation(angles)
        self.assertIsNotNone(result)
        estimate, _ = result
        self.assertAlmostEqual(circular_distance(estimate, -90), 0, delta=1)

    def test_rotated_180_degrees(self):
        angles = [eye_angle_degrees(_kps_for_eye_angle(a))
                  for a in self._angles_for_two_faces(180.0)]
        result = aggregate_orientation(angles)
        self.assertIsNotNone(result)
        estimate, _ = result
        self.assertAlmostEqual(circular_distance(estimate, 180), 0, delta=1)

    def test_all_four_orientations_are_mutually_distinguishable(self):
        estimates = {}
        for label, base in [('upright', 0.0), ('cw90', 90.0), ('ccw90', -90.0), ('upside_down', 180.0)]:
            angles = [eye_angle_degrees(_kps_for_eye_angle(a))
                      for a in self._angles_for_two_faces(base)]
            result = aggregate_orientation(angles)
            self.assertIsNotNone(result, f"expected a verdict for {label}")
            estimates[label] = result[0]
        # Every pair of distinct orientations' estimates should be clearly
        # separated (far more than the aggregation's own 20-degree tolerance).
        labels = list(estimates.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                dist = circular_distance(estimates[labels[i]], estimates[labels[j]])
                self.assertGreater(
                    dist, 45,
                    f"{labels[i]} ({estimates[labels[i]]}) and {labels[j]} "
                    f"({estimates[labels[j]]}) are not clearly distinguishable",
                )


class ShouldOverwriteRotationProposalTests(unittest.TestCase):
    def test_never_scanned_always_allows_write(self):
        self.assertTrue(should_overwrite_rotation_proposal(None, 'landmark_single_face'))
        self.assertTrue(should_overwrite_rotation_proposal(None, 'ml_classifier'))

    def test_higher_priority_source_overwrites_lower(self):
        self.assertTrue(should_overwrite_rotation_proposal('landmark_single_face', 'landmark_multi_face'))
        self.assertTrue(should_overwrite_rotation_proposal('landmark_multi_face', 'ml_classifier'))
        self.assertTrue(should_overwrite_rotation_proposal('landmark_single_face', 'ml_classifier'))

    def test_lower_priority_source_does_not_overwrite_higher(self):
        self.assertFalse(should_overwrite_rotation_proposal('ml_classifier', 'landmark_multi_face'))
        self.assertFalse(should_overwrite_rotation_proposal('ml_classifier', 'landmark_single_face'))
        self.assertFalse(should_overwrite_rotation_proposal('landmark_multi_face', 'landmark_single_face'))

    def test_same_source_rescan_overwrites(self):
        # A re-scan from the same source (e.g. after a logic-version bump)
        # is allowed to refresh its own prior proposal.
        self.assertTrue(should_overwrite_rotation_proposal('landmark_multi_face', 'landmark_multi_face'))
        self.assertTrue(should_overwrite_rotation_proposal('ml_classifier', 'ml_classifier'))


if __name__ == '__main__':
    unittest.main()

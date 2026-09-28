"""Independent connectivity oracle plus synthetic geometric regression tests."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage as ndi

from cavity_ranking import (boundary_seed, escape_clearance, directional_enclosure,
                            rank_cavity_candidates, cavity_above_escape)
from center_point_debug import synthetic_skull


class CavityRankingTests(unittest.TestCase):
    def test_escape_matches_threshold_connectivity_oracle(self):
        rng = np.random.default_rng(13)
        distance = rng.integers(0, 8, size=(7, 8, 9)).astype(float) / 2
        for connectivity in (1, 2, 3):
            expected = np.zeros_like(distance)
            for threshold in np.unique(distance):
                mask = distance >= threshold
                reached = ndi.binary_propagation(boundary_seed(mask), mask=mask,
                         structure=ndi.generate_binary_structure(3, connectivity))
                expected[reached] = threshold
            actual = escape_clearance(distance, connectivity)
            np.testing.assert_array_equal(actual, expected)
            self.assertTrue(np.all(actual <= distance))

    def test_widest_of_two_exits_and_sealed_region(self):
        distance = np.zeros((9, 9, 9))
        distance[3:6, 3:6, 3:6] = 7
        distance[:4, 4, 4] = 2
        distance[5:, 4, 4] = 4
        self.assertEqual(escape_clearance(distance)[4, 4, 4], 4)
        distance[5:, 4, 4] = 0
        self.assertEqual(escape_clearance(distance)[4, 4, 4], 2)
        distance[:4, 4, 4] = 0
        self.assertEqual(escape_clearance(distance)[4, 4, 4], 0)

    def test_cavity_outranks_larger_exterior_peak(self):
        # An exterior pocket open widely to the right, plus a smaller chamber
        # open through a 3x3 tunnel to the left. All cavities use the same EDT.
        bone = np.ones((55, 49, 49), dtype=bool)
        bone[8:23, 17:32, 17:32] = False
        bone[:9, 23:26, 23:26] = False
        bone[29:, 7:42, 7:42] = False
        result = rank_cavity_candidates(bone, sigma=0, min_distance=3,
                                       max_grid_size=None, n_rays=0)
        self.assertLess(result.points[0, 0], 23)
        self.assertTrue(np.any(result.peak > result.peak[0]))
        self.assertGreater(result.bottleneck[0], 0)
        self.assertTrue(cavity_above_escape(result)[tuple(result.points[0])])
        region = cavity_above_escape(result)
        self.assertFalse(boundary_seed(region).any())

    def test_shell_ray_scores_and_unchanged_input(self):
        bone = synthetic_skull(45)
        before = bone.copy()
        scores = directional_enclosure(bone, np.array([[22, 22, 22], [1, 1, 1]]), 96)
        self.assertGreater(scores[0], 0.9)
        self.assertLess(scores[1], 0.4)
        result = rank_cavity_candidates(bone, sigma=1, min_distance=3, max_grid_size=None)
        self.assertLess(np.linalg.norm(result.points[0] - 22), 2)
        np.testing.assert_array_equal(bone, before)
        self.assertGreater(result.peak[0], result.escape[0])
        np.testing.assert_allclose(result.score, result.bottleneck * (0.5 + 0.5 * result.enclosure))

    def test_coarse_coordinates_and_true_original_radius(self):
        bone = synthetic_skull(57)
        with self.assertWarns(UserWarning):
            result = rank_cavity_candidates(bone, sigma=0, min_distance=3, max_grid_size=20, n_rays=0)
        self.assertEqual(result.stride, 3)
        self.assertLessEqual(max(result.analysis_distance.shape), 20)
        np.testing.assert_array_equal(result.points // 3, result.grid_points)
        expected = ndi.distance_transform_edt(~bone)[tuple(result.points.T)]
        np.testing.assert_array_equal(result.peak, expected)
        self.assertFalse(bone[tuple(result.points[0])])
        self.assertLess(np.linalg.norm(result.points[0] - 28), 3)

    def test_axis_permutation(self):
        bone = np.ones((35, 39, 43), dtype=bool)
        bone[7:26, 8:29, 10:33] = False
        bone[:8, 17:20, 20:23] = False
        result = rank_cavity_candidates(bone, sigma=0, min_distance=3, max_grid_size=None, n_rays=0)
        other = rank_cavity_candidates(bone.transpose(2, 0, 1), sigma=0, min_distance=3,
                                      max_grid_size=None, n_rays=0)
        self.assertAlmostEqual(result.score[0], other.score[0])
        np.testing.assert_array_equal(result.escape_field.transpose(2, 0, 1), other.escape_field)

    def test_compatibility_tuple_and_empty_inputs(self):
        from find_center_point import get_candidate_points
        bone = synthetic_skull(33)
        points, radii = get_candidate_points(bone, sigma=0, min_distance=2, n_rays=0)
        self.assertEqual(points.shape[1], 3)
        np.testing.assert_array_equal(radii, ndi.distance_transform_edt(~bone)[tuple(points.T)])
        for empty in (np.zeros((9, 9, 9), bool), np.ones((9, 9, 9), bool)):
            with self.assertRaises(ValueError):
                rank_cavity_candidates(empty)
        with self.assertRaises(ValueError):
            rank_cavity_candidates(bone, max_candidates=0)

    def test_pipeline_tuple_and_no_hull_in_default_path(self):
        from find_center_point import find_center_points
        bone = synthetic_skull(33)
        with patch("find_center_point.preprocessing", return_value=(bone, 50)), \
             patch("find_center_point.get_alpha_shape", side_effect=AssertionError("Hull invoked")):
            points, radii, cloud, result = find_center_points(
                "unused.nii", sigma=0, min_distance=2, n_rays=0, return_scores=True)
        np.testing.assert_array_equal(points, result.points)
        np.testing.assert_array_equal(radii, result.peak)
        self.assertEqual(cloud.shape[1], 3)

    def test_cut_open_chamber_and_boundary_voxels(self):
        field = np.zeros((11, 11, 11))
        field[:, 3:8, 3:8] = 5
        result = escape_clearance(field)
        np.testing.assert_array_equal(result, field)
        np.testing.assert_array_equal(boundary_seed(result), boundary_seed(field))

    def test_coarse_escape_is_lower_bound_on_refined_points(self):
        bone = synthetic_skull(41)
        with self.assertWarns(UserWarning):
            result = rank_cavity_candidates(bone, sigma=0, min_distance=2,
                                           max_grid_size=15, n_rays=0)
        exact = escape_clearance(ndi.distance_transform_edt(~bone))
        self.assertTrue(np.all(result.escape <= exact[tuple(result.points.T)]))


if __name__ == "__main__":
    unittest.main()

"""Numerical stage checks and simple 2D controls, using only the Agg backend."""

import unittest
from unittest.mock import patch

import numpy as np

from hole_detection import compute_watershed_stages, detect_openings


def chamber():
    bone = np.ones((11, 13, 15), dtype=bool)
    bone[3:8, 3:10, 3:12] = False
    bone[:4, 6, 7] = False  # Narrow passage to the i=0 image face.
    bone[1, 1, 1] = False  # Unseeded isolated air pocket.
    return bone


class WatershedStageTests(unittest.TestCase):
    def test_stages_expose_physical_distance_markers_and_both_seeded_regions(self):
        bone = chamber()
        original = bone.copy()
        bone.flags.writeable = False
        stages = compute_watershed_stages(bone, (5, 6, 7), (2, 3, 4))
        self.assertEqual(stages.distance[5, 6, 7], 6)  # Three steps along i to bone, each size 2.
        self.assertEqual(stages.distance[1, 1, 1], 2)
        self.assertTrue(np.all(stages.distance[bone] == 0))
        self.assertTrue(np.all(stages.distance[~bone] > 0))
        self.assertEqual(stages.markers[5, 6, 7], 1)
        self.assertEqual(stages.markers[0, 6, 7], 2)
        self.assertEqual(stages.regions[5, 6, 7], 1)
        self.assertEqual(stages.regions[0, 6, 7], 2)
        self.assertEqual(stages.regions[1, 1, 1], 0)
        self.assertTrue(np.all(stages.regions[bone] == 0))
        np.testing.assert_array_equal(bone, original)
        np.testing.assert_array_equal(stages.regions, detect_openings(bone, (5, 6, 7), (2, 3, 4)).regions)

    def test_markers_cover_all_six_empty_faces_and_round_center_consistently(self):
        bone = np.zeros((15, 17, 19), dtype=bool)
        bone[3:12, 3:14, 3:16] = True
        bone[5:10, 5:12, 5:14] = False
        stages = compute_watershed_stages(bone, (7.2, 8.1, 8.8))
        self.assertEqual(stages.center_ijk, (7, 8, 9))
        self.assertEqual(np.count_nonzero(stages.markers == 1), 1)
        for axis in range(3):
            for end in (0, -1):
                self.assertTrue(np.all(np.take(stages.markers, end, axis=axis) == 2))


class SliceInspectionTests(unittest.TestCase):
    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from test_watershed_steps import plot_watershed_steps

        self.bone = chamber()
        self.stages = compute_watershed_stages(self.bone, (5, 6, 7), (2, 3, 4))
        for values in (self.bone, self.stages.distance, self.stages.markers, self.stages.regions):
            values.flags.writeable = False
        redraw = patch("matplotlib.backends.backend_agg.FigureCanvasAgg.draw_idle")
        redraw.start()
        self.addCleanup(redraw.stop)
        self.figure, self.controls = plot_watershed_steps(self.bone, self.stages)
        self.addCleanup(plt.close, self.figure)

    def test_axis_and_slice_controls_show_matching_planes_and_physical_extents(self):
        controls = self.controls
        np.testing.assert_array_equal(controls["images"][1].get_array(), self.stages.distance[:, :, 7].T)
        controls["axis"].set_active(0)
        self.assertEqual(controls["slice"].valmax, 10)
        controls["slice"].set_val(3)
        np.testing.assert_array_equal(controls["images"][0].get_array(), self.bone[3].T)
        np.testing.assert_array_equal(controls["images"][1].get_array(), self.stages.distance[3].T)
        np.testing.assert_allclose(controls["images"][1].get_extent(), [-1.5, 37.5, -2, 58])
        controls["axis"].set_active(2)
        self.assertEqual(controls["slice"].val, 7)
        controls["axis"].set_active(0)
        self.assertEqual(controls["slice"].val, 3)
        controls["seed"]._observers.process("clicked", None)
        self.assertEqual(controls["slice"].val, 5)

    def test_label_colors_distinguish_bone_cavity_exterior_and_unseeded_air(self):
        controls = self.controls
        marker_view, region_view = [controls["images"][i].get_array() for i in (2, 3)]
        self.assertEqual(marker_view[6, 5], 2)  # Cavity marker, transposed (j,i).
        self.assertEqual(region_view[6, 5], 2)  # Cavity region.
        self.assertEqual(marker_view[6, 0], 3)  # Exterior marker.
        self.assertEqual(region_view[6, 0], 3)  # Exterior region.
        self.assertEqual(region_view[0, 5], 1)  # Bone is distinguishable from unseeded air.
        controls["slice"].set_val(1)
        self.assertEqual(controls["images"][3].get_array()[1, 1], 0)

    def test_contrast_changes_display_without_changing_distance_values(self):
        before = self.stages.distance.copy()
        self.controls["contrast"].set_val(3)
        self.assertEqual(self.controls["images"][1].get_clim(), (0, 3))
        self.controls["slice"].set_val(6)
        self.assertEqual(self.controls["images"][1].get_clim(), (0, 3))
        np.testing.assert_array_equal(self.stages.distance, before)


if __name__ == "__main__":
    unittest.main()

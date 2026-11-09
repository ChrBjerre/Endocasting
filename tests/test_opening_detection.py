"""Synthetic geometric and viewer checks for the two-marker watershed method."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage as ndi

from hole_detection import detect_openings


def box_skull(openings=1):
    """Known closed chamber, optionally with 5x5 and 7x7 square tunnels."""
    bone = np.zeros((43, 43, 43), dtype=bool)
    bone[8:35, 8:35, 8:35] = True
    bone[12:31, 12:31, 12:31] = False
    if openings:
        bone[21:, 19:24, 19:24] = False
    if openings == 2:
        bone[18:25, :22, 18:25] = False
    return bone


class HoleDetectionTests(unittest.TestCase):
    def test_one_opening_is_in_tunnel_and_input_is_unchanged(self):
        bone = box_skull().astype(np.uint8)
        original = bone.copy()
        bone.flags.writeable = False
        result = detect_openings(bone, (21, 21, 21))
        self.assertEqual(len(result.openings), 1)
        opening = result.openings[0]
        self.assertTrue(30 <= opening.centroid_ijk[0] <= 35)
        np.testing.assert_allclose(opening.centroid_ijk[1:], (21, 21))
        self.assertEqual(opening.area, 25)
        self.assertEqual(opening.voxel_count, 25)
        np.testing.assert_array_equal(bone, original)
        self.assertTrue(np.all(result.regions[bone != 0] == 0))
        self.assertTrue(np.all(result.regions[result.opening_labels > 0] == 1))
        # Every labeled opening voxel really touches exterior across a face.
        touching = ndi.binary_dilation(result.regions == 2, ndi.generate_binary_structure(3, 1))
        self.assertTrue(np.all(touching[result.opening_labels > 0]))

    def test_two_separate_openings_are_not_merged(self):
        result = detect_openings(box_skull(2), (21, 21, 21))
        self.assertEqual(len(result.openings), 2)
        self.assertEqual(sorted(o.area for o in result.openings), [25, 49])
        np.testing.assert_array_equal(np.unique(result.opening_labels), [0, 1, 2])

    def test_sealed_cavity_and_unseeded_air_are_not_openings(self):
        bone = box_skull(0)
        bone[9, 9, 9] = False  # Isolated air voxel in the bone wall.
        result = detect_openings(bone, (21, 21, 21))
        self.assertEqual(result.openings, [])
        self.assertFalse(result.opening_labels.any())
        self.assertEqual(result.regions[21, 21, 21], 1)
        self.assertEqual(result.regions[0, 0, 0], 2)
        self.assertEqual(result.regions[9, 9, 9], 0)

    def test_anisotropic_spacing_changes_area_and_coordinates(self):
        result = detect_openings(box_skull(), (21, 21, 21), spacing=(2, 3, 4))
        opening = result.openings[0]
        # A 5x5 opening normal to axis i has face area spacing_j*spacing_k.
        self.assertEqual(opening.area, 25 * 3 * 4)
        self.assertAlmostEqual(opening.equivalent_diameter, 2 * np.sqrt(300 / np.pi))
        np.testing.assert_allclose(opening.centroid_physical,
                                   np.asarray(opening.centroid_ijk) * [2, 3, 4])

    def test_edt_receives_physical_spacing(self):
        actual = ndi.distance_transform_edt
        with patch("hole_detection.ndi.distance_transform_edt", wraps=actual) as edt:
            detect_openings(box_skull(), (21, 21, 21), (2, 3, 4))
        np.testing.assert_array_equal(edt.call_args.kwargs["sampling"], [2, 3, 4])

    def test_uniform_spacing_scale_preserves_labels_and_scales_sizes(self):
        first = detect_openings(box_skull(2), (21, 21, 21))
        scaled = detect_openings(box_skull(2), (21, 21, 21), spacing=(3, 3, 3))
        np.testing.assert_array_equal(first.opening_labels, scaled.opening_labels)
        for a, b in zip(first.openings, scaled.openings):
            self.assertAlmostEqual(b.area, 9 * a.area)
            self.assertAlmostEqual(b.equivalent_diameter, 3 * a.equivalent_diameter)

    def test_all_six_volume_faces_seed_the_exterior(self):
        # Rotate/reflect the tunnel onto each image face.
        for axis in range(3):
            for flipped in (False, True):
                with self.subTest(axis=axis, flipped=flipped):
                    bone = np.swapaxes(box_skull(), 0, axis)
                    if flipped:
                        bone = np.flip(bone, axis=axis)
                    result = detect_openings(bone, (21, 21, 21))
                    self.assertEqual(len(result.openings), 1)
                    self.assertEqual(result.openings[0].area, 25)
                    for boundary_axis in range(3):
                        for end in (0, -1):
                            self.assertTrue(np.all(np.take(result.regions, end, axis=boundary_axis) == 2))

    def test_invalid_inputs_raise_clear_errors(self):
        bone = box_skull()
        for center in ((0, 21, 21), (-1, 21, 21), (43, 21, 21), (8, 8, 8), (np.nan, 21, 21)):
            with self.subTest(center=center), self.assertRaises(ValueError):
                detect_openings(bone, center)
        for spacing in ((0, 1, 1), (1, -1, 1), (1, np.inf, 1), (1, 1)):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                detect_openings(bone, (21, 21, 21), spacing)
        invalid = bone.astype(float)
        invalid[1, 1, 1] = 0.5
        with self.assertRaisesRegex(ValueError, "only 0"):
            detect_openings(invalid, (21, 21, 21))
        with self.assertRaisesRegex(ValueError, "3D binary"):
            detect_openings(np.zeros((5, 5)), (2, 2, 2))
        with self.assertRaisesRegex(ValueError, "some bone"):
            detect_openings(np.zeros((5, 5, 5)), (2, 2, 2))
        no_exterior = np.ones((5, 5, 5), dtype=bool)
        no_exterior[2, 2, 2] = False
        with self.assertRaisesRegex(ValueError, "boundary voxels"):
            detect_openings(no_exterior, (2, 2, 2))


class HoleViewerTests(unittest.TestCase):
    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.figures_before = set(plt.get_fignums())
        redraw = patch("matplotlib.backends.backend_agg.FigureCanvasAgg.draw_idle")
        redraw.start()
        self.addCleanup(redraw.stop)

    def tearDown(self):
        import matplotlib.pyplot as plt
        for number in set(plt.get_fignums()) - self.figures_before:
            plt.close(number)

    def make_view(self, **kwargs):
        from hole_detection_viewer import plot_openings
        kwargs.setdefault("mesh_renderer", "matplotlib")
        bone = box_skull(2)
        result = detect_openings(bone, (21, 21, 21))
        fig, widgets = plot_openings(bone, result, **kwargs)
        return bone, result, fig, widgets

    def test_display_pooling_keeps_thin_bone_and_respects_point_budget(self):
        from hole_detection_viewer import _surface_points

        volume = np.zeros((13, 15, 17), dtype=bool)
        volume[1, 2, 3] = True  # Would disappear with volume[::4, ::4, ::4].
        volume[-1, -1, -1] = True
        original = volume.copy()
        points = _surface_points(volume, (2, 3, 4), step=4, max_points=100)
        self.assertEqual(len(points), 2)
        self.assertTrue(np.all(points >= 0))
        self.assertTrue(np.all(points <= (np.asarray(volume.shape) - 1) * [2, 3, 4]))
        capped = _surface_points(volume, (2, 3, 4), step=4, max_points=1)
        self.assertEqual(len(capped), 1)
        np.testing.assert_array_equal(volume, original)

    def test_all_3d_point_clouds_are_bounded(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from hole_detection_viewer import plot_openings

        bone = box_skull(2)
        result = detect_openings(bone, (21, 21, 21))
        original_labels = result.opening_labels.copy()
        with patch("matplotlib.backends.backend_agg.FigureCanvasAgg.draw_idle"):
            figure, widgets = plot_openings(bone, result, max_points=12)
            try:
                widgets["open_slices"](show=False)
                for opening in result.openings:
                    widgets["select_opening"](opening.id)
                    counts = widgets["render_counts"]
                    for name in ("skull", "cavity", "selected_opening"):
                        self.assertLessEqual(counts[name], 12)
                    self.assertEqual(counts["opening_markers"], 0)
                # Skull, cavity, centroids, selected patch, seed: no triangle meshes.
                self.assertEqual(len(figure.axes[0].collections), 5)
                self.assertTrue(all(type(artist).__name__ != "Poly3DCollection"
                                    for artist in figure.axes[0].collections))
                self.assertTrue(all(not widgets[axis].drawon for axis in "ijk"))
            finally:
                plt.close(figure)
        np.testing.assert_array_equal(result.opening_labels, original_labels)

    def test_opening_selection_recenters_all_slice_controls(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from hole_detection_viewer import plot_openings

        bone = box_skull(2)
        result = detect_openings(bone, (21, 21, 21))
        # Exercise callbacks without expensive, repeated 3D canvas rendering.
        with patch("matplotlib.backends.backend_agg.FigureCanvasAgg.draw_idle"):
            figure, widgets = plot_openings(bone, result, surface_step=3)
            try:
                self.assertNotIn("slice_figure", widgets)
                widgets["open_slices"](show=False)
                for opening in result.openings:
                    widgets["select_opening"](opening.id)
                    actual = [widgets[axis].val for axis in "ijk"]
                    np.testing.assert_array_equal(actual, np.floor(np.asarray(opening.centroid_ijk) + 0.5))
                widgets["seed"]._observers.process("clicked", None)
                np.testing.assert_array_equal([widgets[axis].val for axis in "ijk"], result.center_ijk)
                widgets["opacity"].set_val(0.7)
                widgets["options"].set_active(1)
            finally:
                plt.close(figure)

    def test_sealed_cavity_viewer_has_working_slice_controls(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from hole_detection_viewer import plot_openings

        bone = box_skull(0)
        with patch("matplotlib.backends.backend_agg.FigureCanvasAgg.draw_idle"):
            figure, widgets = plot_openings(bone, detect_openings(bone, (21, 21, 21)), surface_step=3)
            try:
                self.assertNotIn("opening", widgets)
                self.assertEqual(widgets["state"]["retained_ids"], [])
                widgets["open_slices"](show=False)
                widgets["i"].set_val(10)
            finally:
                plt.close(figure)

    def test_arrows_navigate_largest_first_and_wrap(self):
        from matplotlib.backend_bases import KeyEvent
        _, result, figure, widgets = self.make_view()
        ordered = [o.id for o in sorted(result.openings, key=lambda opening: -opening.area)]
        state = widgets["state"]
        self.assertEqual(state["selected_id"], ordered[0])
        self.assertEqual(widgets["render_counts"]["opening_markers"], 0)
        widgets["next"]._observers.process("clicked", None)
        self.assertEqual(state["selected_id"], ordered[1])
        event = KeyEvent("key_press_event", figure.canvas, key="right")
        figure.canvas.callbacks.process("key_press_event", event)
        self.assertEqual(state["selected_id"], ordered[0])
        widgets["previous"]._observers.process("clicked", None)
        self.assertEqual(state["selected_id"], ordered[-1])

    def test_cutoff_excludes_small_candidates_inclusively_and_handles_empty_list(self):
        bone, result, _, widgets = self.make_view()
        original_labels, original_regions, original_bone = result.opening_labels.copy(), result.regions.copy(), bone.copy()
        small, large = sorted(result.openings, key=lambda opening: opening.area)
        state = widgets["state"]
        widgets["select_opening"](small.id)
        widgets["threshold"].set_val(small.area)
        self.assertEqual(len(state["retained_ids"]), 2)
        widgets["threshold"].set_val((small.area + large.area) / 2)
        self.assertEqual(state["retained_ids"], [large.id])
        self.assertEqual(state["selected_id"], large.id)
        with self.assertRaisesRegex(ValueError, "excluded"):
            widgets["select_opening"](small.id)
        widgets["navigate"](1)
        self.assertEqual(state["selected_id"], large.id)
        histogram_bars = widgets["histogram"].patches
        self.assertEqual(sum(bar.get_height() for bar in histogram_bars[len(histogram_bars)//2:]), 1)
        widgets["threshold"].set_val(large.area * 1.02)
        self.assertEqual(state["retained_ids"], [])
        self.assertIsNone(state["selected_id"])
        self.assertEqual(widgets["render_counts"]["selected_opening"], 0)
        self.assertFalse(widgets["next"].active)
        widgets["navigate"](-1)
        widgets["threshold"].set_val(0)
        self.assertEqual(state["selected_id"], large.id)
        self.assertTrue(widgets["next"].active)
        np.testing.assert_array_equal(result.opening_labels, original_labels)
        np.testing.assert_array_equal(result.regions, original_regions)
        np.testing.assert_array_equal(bone, original_bone)

    def test_area_diameter_switch_preserves_cutoff_and_retained_candidates(self):
        _, _, _, widgets = self.make_view()
        widgets["threshold"].set_val(30)
        retained = widgets["state"]["retained_ids"].copy()
        widgets["metric"].set_active(1)
        self.assertEqual(widgets["state"]["metric"], "Diameter")
        self.assertAlmostEqual(widgets["threshold"].val, 2 * np.sqrt(30 / np.pi))
        self.assertEqual(widgets["state"]["retained_ids"], retained)
        widgets["metric"].set_active(0)
        self.assertAlmostEqual(widgets["threshold"].val, 30)
        self.assertEqual(widgets["state"]["retained_ids"], retained)

    def test_histogram_click_sets_cutoff_and_other_points_obey_filter(self):
        from matplotlib.backend_bases import MouseEvent
        _, _, figure, widgets = self.make_view()
        widgets["options"].set_active(0)
        self.assertEqual(widgets["render_counts"]["opening_markers"], 1)
        x, y = widgets["histogram"].transData.transform((30, 0.3))
        event = MouseEvent("button_press_event", figure.canvas, x, y, button=1)
        figure.canvas.callbacks.process("button_press_event", event)
        self.assertAlmostEqual(widgets["threshold"].val, 30, delta=0.2)
        self.assertEqual(len(widgets["state"]["retained_ids"]), 1)
        self.assertEqual(widgets["render_counts"]["opening_markers"], 0)

    def test_focus_zoom_and_separate_slice_redraws_remain_bounded(self):
        _, result, figure, widgets = self.make_view(max_points=50)
        whole_width = np.ptp(figure.axes[0].get_xlim())
        widgets["focus"]._observers.process("clicked", None)
        self.assertTrue(widgets["state"]["focus"])
        self.assertLess(np.ptp(figure.axes[0].get_xlim()), whole_width)
        self.assertLessEqual(widgets["render_counts"]["skull"], 50)
        widgets["navigate"](1)
        self.assertLessEqual(widgets["render_counts"]["selected_opening"], 50)
        widgets["whole"]._observers.process("clicked", None)
        self.assertAlmostEqual(np.ptp(figure.axes[0].get_xlim()), whole_width)
        slices = widgets["open_slices"](show=False)
        self.assertIsNot(slices, figure)
        with patch.object(figure.canvas, "draw_idle") as main_redraw:
            widgets["i"].set_val(10)
        main_redraw.assert_not_called()
        self.assertEqual(len(result.openings), 2)

    def test_mesh_mode_uses_original_voxels_and_hides_the_opening_overlay(self):
        from matplotlib.backend_bases import MouseEvent
        from skull_mesh import local_bone_mesh
        _, result, figure, widgets = self.make_view()
        original_labels = result.opening_labels.copy()
        self.assertEqual(widgets["render_counts"]["mesh_faces"], 0)
        with patch("skull_mesh.local_bone_mesh", wraps=local_bone_mesh) as build:
            widgets["options"].set_active(2)
            self.assertTrue(widgets["state"]["focus"])
            self.assertFalse(widgets["options"].get_status()[3])
            self.assertEqual(widgets["opacity"].val, 1)
            self.assertGreater(widgets["render_counts"]["mesh_faces"], 0)
            self.assertLessEqual(widgets["render_counts"]["mesh_faces"], 60000)
            self.assertEqual(build.call_count, 1)
            # Filtering the same candidate reuses its already extracted mesh.
            widgets["threshold"].set_val(1)
            self.assertEqual(build.call_count, 1)
            widgets["navigate"](1)
            self.assertEqual(build.call_count, 2)
        visible_meshes = [artist for artist in figure.axes[0].collections
                          if type(artist).__name__ == "Poly3DCollection" and artist.get_visible()]
        self.assertEqual(len(visible_meshes), 1)
        # Mesh mode must remain a surface even while dragging to rotate.
        x, y = figure.axes[0].transAxes.transform((0.5, 0.5))
        press = MouseEvent("button_press_event", figure.canvas, x, y, button=1)
        figure.canvas.callbacks.process("button_press_event", press)
        self.assertEqual(widgets["state"]["render_mode"], "mesh")
        self.assertTrue(visible_meshes[0].get_visible())
        release = MouseEvent("button_release_event", figure.canvas, x, y, button=1)
        figure.canvas.callbacks.process("button_release_event", release)
        self.assertTrue(visible_meshes[0].get_visible())
        widgets["options"].set_active(3)
        self.assertTrue(widgets["options"].get_status()[3])
        widgets["whole"]._observers.process("clicked", None)
        self.assertFalse(widgets["options"].get_status()[2])
        self.assertFalse(widgets["state"]["focus"])
        self.assertEqual(widgets["render_counts"]["mesh_faces"], 0)
        self.assertAlmostEqual(widgets["opacity"].val, 0.23)
        np.testing.assert_array_equal(result.opening_labels, original_labels)

    def test_mesh_budget_failure_never_displays_a_point_cloud_as_mesh(self):
        _, _, figure, widgets = self.make_view(mesh_max_faces=1)
        widgets["options"].set_active(2)
        self.assertEqual(widgets["state"]["render_mode"], "mesh unavailable")
        self.assertEqual(widgets["render_counts"]["mesh_faces"], 0)
        # The skull point preview is hidden instead of masquerading as a mesh.
        self.assertFalse(figure.axes[0].collections[0].get_visible())
        self.assertTrue(any("Mesh unavailable" in text.get_text() for text in figure.texts))


if __name__ == "__main__":
    unittest.main()

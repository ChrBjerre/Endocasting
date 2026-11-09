"""Process-bridge checks; native OpenGL tests require an explicit opt-in."""

import importlib.util
import os
import json
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("vtkmodules") and os.environ.get("ENDOCASTING_RUN_GL_TESTS") == "1",
                     "Native graphics tests are opt-in; no renderer windows are opened by the default suite")
class GpuMeshViewerTests(unittest.TestCase):
    def test_more_than_60000_triangles_rotate_without_reuploading_geometry(self):
        from vtkmodules.vtkFiltersSources import vtkSphereSource
        from vtkmodules.util.numpy_support import vtk_to_numpy
        from gpu_mesh_viewer import MeshScene
        from skull_mesh import BoneMesh

        sphere = vtkSphereSource()
        sphere.SetThetaResolution(248)
        sphere.SetPhiResolution(124)
        sphere.Update()
        surface = sphere.GetOutput()
        vertices = vtk_to_numpy(surface.GetPoints().GetData()).copy()
        faces = vtk_to_numpy(surface.GetPolys().GetData()).reshape(-1, 4)[:, 1:].copy()
        mesh = BoneMesh(vertices, faces, np.stack((vertices.min(axis=0), vertices.max(axis=0))))
        self.assertGreater(len(faces), 60000)
        scene = MeshScene(offscreen=True)
        self.addCleanup(scene.close)
        scene.set_mesh(mesh, key=1, title="GPU mesh test")
        scene.render()
        original_input = scene.mapper.GetInput()
        self.assertEqual(original_input.GetNumberOfPolys(), len(faces))
        self.assertTrue(scene.window.SupportsOpenGL())
        image = scene.snapshot()
        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[2], 3)
        self.assertGreater(float(image.std()), 1)
        scene.renderer.GetActiveCamera().Azimuth(25)
        position = scene.renderer.GetActiveCamera().GetPosition()
        scene.render()
        scene.set_mesh(mesh, key=1, title="Same geometry, new label")
        self.assertIs(scene.mapper.GetInput(), original_input)
        self.assertEqual(scene.renderer.GetActiveCamera().GetPosition(), position)
        scene.set_opacity(0.7)
        self.assertAlmostEqual(scene.actor.GetProperty().GetOpacity(), 0.7)

    def test_arrow_events_delegate_to_candidate_navigation(self):
        from gpu_mesh_viewer import MeshScene
        directions = []
        scene = MeshScene(offscreen=True, on_navigate=directions.append)
        self.addCleanup(scene.close)
        for key in ("Right", "Left"):
            scene.interactor.SetKeySym(key)
            scene.interactor.InvokeEvent("KeyPressEvent")
        self.assertEqual(directions, [1, -1])

    def test_histogram_navigation_and_overlay_sync_with_gpu_mesh(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from test_hole_detection import synthetic_skull
        from hole_detection import detect_openings
        from hole_detection_viewer import plot_openings

        bone, center = synthetic_skull()
        result = detect_openings(bone, center)
        with patch("matplotlib.backends.backend_agg.FigureCanvasAgg.draw_idle"):
            figure, controls = plot_openings(bone, result, mesh_renderer="vtk", mesh_offscreen=True)
            self.addCleanup(plt.close, figure)
            self.addCleanup(controls["close"])
            controls["options"].set_active(2)
            panel = controls["gpu_panel"]
            self.assertEqual(controls["state"]["render_mode"], "mesh")
            self.assertFalse(any(type(item).__name__ == "Poly3DCollection"
                                 for item in figure.axes[0].collections))
            self.assertEqual(panel.scene.points["opening"], [])
            controls["options"].set_active(3)
            self.assertGreater(len(panel.scene.points["opening"]), 0)
            initial_id = controls["state"]["selected_id"]
            controls["navigate"](1)
            self.assertNotEqual(controls["state"]["selected_id"], initial_id)
            controls["threshold"].set_val(60)
            self.assertEqual(controls["state"]["retained_ids"], [initial_id])
            self.assertEqual(controls["state"]["selected_id"], initial_id)
            controls["whole"]._observers.process("clicked", None)
            self.assertFalse(panel.visible)
            self.assertEqual(controls["state"]["render_mode"], "points")


class MeshBridgeTests(unittest.TestCase):
    """Validate process isolation and UI links without starting a renderer."""
    def setUp(self):
        from gpu_mesh_viewer import RemoteMeshScene
        from skull_mesh import BoneMesh

        patcher = patch("subprocess.Popen")
        self.start_process = patcher.start()
        self.addCleanup(patcher.stop)
        self.process = self.start_process.return_value
        self.process.poll.return_value = None
        self.process.wait.return_value = 0
        self.navigation = []
        self.scene = RemoteMeshScene(on_navigate=self.navigation.append)
        self.addCleanup(self.scene.close)
        self.mesh = BoneMesh(np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]]),
                             np.array([[0, 1, 2]]), np.array([[0, 0, 0], [1, 1, 1]]))

    def test_mesh_upload_and_opacity_changes_do_not_share_native_graphics(self):
        self.scene.set_mesh(self.mesh, key=1, title="Opening 1")
        self.scene.set_points("opening", [[0.1, 0.2, 0.3]])
        self.scene.render()
        self.start_process.assert_called_once()
        command = self.start_process.call_args.args[0]
        self.assertIn("--scene-dir", command)
        state = json.loads((self.scene.directory / "state.json").read_text())
        with np.load(self.scene.directory / state["geometry"], allow_pickle=False) as data:
            np.testing.assert_array_equal(data["faces"], self.mesh.faces)
            np.testing.assert_array_equal(data["vertices"], self.mesh.vertices)
        self.scene.set_opacity(0.4)
        self.scene.set_mesh(self.mesh, key=1, title="Still opening 1")
        self.scene.render()
        self.assertEqual(self.scene.generation, 1)
        self.assertEqual(json.loads((self.scene.directory / "state.json").read_text())["opacity"], 0.4)
        self.start_process.assert_called_once()
        self.scene.close()
        self.process.wait.assert_called()
        self.process.terminate.assert_not_called()

    def test_remote_frame_and_arrow_requests_update_parent_without_windows(self):
        self.scene.set_mesh(self.mesh, key=1, title="Opening 1")
        self.scene.render()
        image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
        np.save(self.scene.directory / "image.npy", image, allow_pickle=False)
        (self.scene.directory / "navigation.json").write_text(json.dumps(dict(steps=2)))
        frames = []
        self.scene.on_frame = lambda: frames.append(True)
        self.scene.process_events()
        np.testing.assert_array_equal(self.scene.snapshot(), image)
        self.assertEqual(frames, [True])
        self.assertEqual(self.navigation, [2])
        (self.scene.directory / "navigation.json").write_text(json.dumps(dict(steps=1)))
        self.scene.process_events()
        self.assertEqual(self.navigation, [2, -1])
        self.assertEqual(frames, [True])

    def test_renderer_protocol_loads_geometry_and_writes_snapshot_without_graphics(self):
        from gpu_mesh_viewer import run_renderer
        self.scene.set_mesh(self.mesh, key=1, title="Opening 1")
        self.scene.render()
        image = np.full((4, 5, 3), 120, dtype=np.uint8)
        with patch("gpu_mesh_viewer.MeshScene") as native:
            native.return_value.clouds = {"opening": None}
            native.return_value.snapshot.return_value = image
            run_renderer(self.scene.directory, offscreen=True)
            arguments = native.return_value.set_mesh.call_args
            np.testing.assert_array_equal(arguments.args[0].faces, self.mesh.faces)
            self.assertEqual(arguments.kwargs["title"], "Opening 1")
            native.return_value.render.assert_called_once()
            native.return_value.close.assert_called_once()
        self.scene.process_events()
        np.testing.assert_array_equal(self.scene.snapshot(), image)


if __name__ == "__main__":
    unittest.main()

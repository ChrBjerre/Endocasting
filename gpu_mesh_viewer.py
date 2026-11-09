"""VTK/OpenGL mesh rendering in a process separate from Matplotlib/Qt.

Rotation changes a VTK camera without rebuilding geometry or redrawing plots.
Atomic files link meshes, navigation and snapshots across the process boundary.
Importing this module never constructs native graphics objects.
"""

import numpy as np


class MeshScene:
    def __init__(self, *, render_window=None, offscreen=False, on_navigate=None):
        # Import registration modules before constructing OpenGL/window objects.
        import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
        import vtkmodules.vtkRenderingFreeType  # noqa: F401
        import vtkmodules.vtkRenderingUI  # noqa: F401
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
        from vtkmodules.vtkRenderingCore import (vtkActor, vtkPolyDataMapper, vtkRenderer,
                                                 vtkRenderWindow, vtkRenderWindowInteractor,
                                                 vtkTextActor)

        self.window = render_window or vtkRenderWindow()
        self.offscreen = offscreen
        self.window.SetOffScreenRendering(int(offscreen))
        self.window.SetSize(1100, 800)
        self.window.SetMultiSamples(0)
        self.window.SetWindowName("Endocasting | OpenGL bone mesh")
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.96, 0.97, 0.98)
        self.window.AddRenderer(self.renderer)
        self.mapper = vtkPolyDataMapper()
        self.mapper.ScalarVisibilityOff()
        self.actor = vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetColor(0.73, 0.78, 0.83)
        self.actor.GetProperty().SetAmbient(0.25)
        self.actor.GetProperty().SetDiffuse(0.75)
        self.renderer.AddActor(self.actor)
        self.clouds = {}
        for name, color, size in (("opening", (0.91, 0.55, 0.13), 5),
                                   ("other", (0.24, 0.62, 0.74), 5),
                                   ("cavity", (0.24, 0.62, 0.74), 2),
                                   ("seed", (0.20, 0.71, 0.42), 8)):
            actor = vtkActor()
            actor.SetMapper(vtkPolyDataMapper())
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetPointSize(size)
            self.renderer.AddActor(actor)
            self.clouds[name] = actor
        self.label = vtkTextActor()
        self.label.SetPosition(16, 16)
        self.label.GetTextProperty().SetColor(0.10, 0.15, 0.23)
        self.label.GetTextProperty().SetFontSize(16)
        self.renderer.AddActor2D(self.label)
        self.interactor = self.window.GetInteractor() or vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)
        self.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        self.interactor.AddObserver("KeyPressEvent", self._on_key)
        self.interactor.AddObserver("ExitEvent", self._on_exit)
        self.on_navigate = on_navigate
        self.closed = False
        self.started = False
        self.mesh_key = None
        self.finalized = False

    def _on_key(self, interactor, _):
        key = interactor.GetKeySym()
        if key in ("Left", "Right") and self.on_navigate is not None:
            self.on_navigate(-1 if key == "Left" else 1)

    def _on_exit(self, *_):
        self.closed = True

    def set_mesh(self, mesh, *, key, title):
        """Upload geometry only when the candidate/crop changes; keep camera pose."""
        from skull_mesh import mesh_to_polydata

        if key != self.mesh_key:
            if self.mesh_key is None:
                center = mesh.bounds.mean(axis=0)
                extent = max(float(np.ptp(mesh.bounds, axis=0).max()), 1)
                camera = self.renderer.GetActiveCamera()
                camera.SetFocalPoint(*center)
                camera.SetPosition(*(center + extent * np.array([1.3, -1.8, 1.0])))
                camera.SetViewUp(0, 0, 1)
            self.mapper.SetInputData(mesh_to_polydata(mesh))
            self.mesh_key = key
            self.renderer.ResetCamera(*mesh.bounds.T.ravel())
            self.renderer.ResetCameraClippingRange()
        self.label.SetInput(f"{title}\n{len(mesh.vertices):,} vertices | {len(mesh.faces):,} triangles\n"
                            "Drag: rotate | wheel: zoom | arrows: previous/next | crop sides are cuts")
        self.actor.SetVisibility(True)

    def set_points(self, name, points):
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        vertices = vtkPoints()
        vertices.SetData(numpy_to_vtk(points, deep=True))
        cells = vtkCellArray()
        indices = np.column_stack((np.ones(len(points)), np.arange(len(points)))).astype(np.int64)
        cells.SetCells(len(points), numpy_to_vtkIdTypeArray(indices.ravel(), deep=True))
        cloud = vtkPolyData()
        cloud.SetPoints(vertices)
        cloud.SetVerts(cells)
        self.clouds[name].GetMapper().SetInputData(cloud)
        self.clouds[name].SetVisibility(bool(len(points)))

    def set_opacity(self, value):
        self.actor.GetProperty().SetOpacity(value)

    def render(self):
        if self.closed:
            return
        self.window.Render()
        if not self.started and not self.offscreen:
            self.interactor.Initialize()
            self.interactor.Enable()
        self.started = True

    def snapshot(self):
        from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
        from vtkmodules.util.numpy_support import vtk_to_numpy

        capture = vtkWindowToImageFilter()
        capture.SetInput(self.window)
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.ShouldRerenderOff()
        capture.Update()
        output = capture.GetOutput()
        width, height, _ = output.GetDimensions()
        return vtk_to_numpy(output.GetPointData().GetScalars()).reshape(height, width, 3)[::-1].copy()

    def process_events(self):
        if self.started and not self.closed and not self.offscreen:
            self.interactor.ProcessEvents()
            if self.interactor.GetDone():
                self.closed = True

    def close(self):
        if self.finalized:
            return
        if self.started:
            self.interactor.Disable()
        self.window.Finalize()
        self.closed = True
        self.finalized = True


def _write_json(path, value):
    import json
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


class RemoteMeshScene:
    """Parent-side controller: no VTK render windows or Qt graphics objects."""
    def __init__(self, *, offscreen=False, on_navigate=None):
        import tempfile
        from pathlib import Path

        self._temporary = tempfile.TemporaryDirectory(prefix="endocasting-mesh-")
        self.directory = Path(self._temporary.name)
        self.offscreen, self.on_navigate = offscreen, on_navigate
        self.closed, self.started = False, False
        self.mesh_key, self.generation = None, 0
        self.geometry_file = None
        self.process, self.log = None, None
        self.points = {}
        self.title, self.opacity, self.visible = "Bone mesh", 1.0, True
        self.image = None
        self.image_stamp, self.steps = None, 0
        self.on_frame = None

    def set_mesh(self, mesh, *, key, title):
        if key != self.mesh_key:
            temporary = self.directory / "geometry.tmp"
            with temporary.open("wb") as stream:
                np.savez(stream, vertices=mesh.vertices, faces=mesh.faces, bounds=mesh.bounds)
            self.mesh_key, self.generation = key, self.generation + 1
            self.geometry_file = self.directory / f"geometry-{self.generation}.npz"
            temporary.replace(self.geometry_file)
        self.title = title

    def set_points(self, name, points):
        self.points[name] = np.asarray(points).reshape(-1, 3).tolist()

    def set_opacity(self, value):
        self.opacity = float(value)

    def render(self):
        import subprocess
        import sys
        from pathlib import Path

        _write_json(self.directory / "state.json", dict(generation=self.generation,
                    geometry=self.geometry_file.name, title=self.title,
                    opacity=self.opacity, points=self.points, visible=self.visible))
        if self.offscreen and self.process is not None and self.process.poll() is not None:
            self.process.wait()
            self.process = None
            self.log.close()
            self.image, self.image_stamp = None, None
            (self.directory / "image.npy").unlink(missing_ok=True)
        if self.process is None:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--scene-dir", str(self.directory)]
            if self.offscreen:
                command.append("--offscreen")
            self.log = (self.directory / "renderer.log").open("w", encoding="utf-8")
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=self.log,
                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.started = True

    def snapshot(self):
        import time
        # Only initial startup waits. Subsequent frames arrive asynchronously.
        deadline = time.monotonic() + 10
        while self.image is None:
            self.process_events()
            if self.closed:
                error = (self.directory / "renderer.log").read_text(encoding="utf-8")[-2000:]
                raise RuntimeError(f"GPU mesh process exited before drawing a frame. {error}")
            if time.monotonic() >= deadline:
                raise RuntimeError("GPU mesh process did not produce its first frame within 10 seconds")
            if self.image is None:
                time.sleep(0.02)
        return self.image

    def process_events(self):
        import json

        path = self.directory / "image.npy"
        if path.exists():
            stamp = path.stat().st_mtime_ns
            if stamp != self.image_stamp:
                self.image = np.load(path, allow_pickle=False)
                self.image_stamp = stamp
                if self.on_frame is not None:
                    self.on_frame()
        request = self.directory / "navigation.json"
        if request.exists():
            steps = json.loads(request.read_text(encoding="utf-8"))["steps"]
            if steps != self.steps:
                delta, self.steps = steps - self.steps, steps
                if self.on_navigate is not None:
                    self.on_navigate(delta)
        if self.process is not None and self.process.poll() is not None:
            if not (self.offscreen and self.image is not None):
                self.closed = True

    def close(self):
        import subprocess
        if self.closed and self.process is None:
            return
        (self.directory / "close").touch()
        if self.process is not None:
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()  # Only the renderer child created by this controller.
                self.process.wait(timeout=2)
            self.process = None
        if self.log is not None:
            self.log.close()
        self.closed = True
        self._temporary.cleanup()


class MeshPanel:
    """Linked GPU window, isolated from the inspector's GUI toolkit and lifetime."""
    region = (0.01, 0.22, 0.68, 0.63)

    def __init__(self, figure, *, offscreen=False, on_navigate=None):
        self.figure = figure
        self.embedded, self.visible, self.closed = False, False, False
        self.scene = RemoteMeshScene(offscreen=offscreen, on_navigate=on_navigate)
        self.timer = figure.canvas.new_timer(interval=50)
        self.timer.add_callback(self.scene.process_events)

    def show(self):
        self.visible = self.scene.visible = True
        self.scene.render()
        self.timer.start()

    def hide(self):
        self.visible = self.scene.visible = False
        if self.scene.started and not self.scene.closed:
            self.scene.render()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.timer.stop()
        self.scene.close()


def run_renderer(directory, *, offscreen=False):
    """Child process: one native VTK window and its own standard event loop."""
    import json
    from skull_mesh import BoneMesh

    scene = MeshScene(offscreen=offscreen)
    generation, stamp, steps = -1, None, [0]

    def save_frame(*_):
        temporary = directory / "image.tmp"
        with temporary.open("wb") as stream:
            np.save(stream, scene.snapshot(), allow_pickle=False)
        temporary.replace(directory / "image.npy")

    def request_navigation(direction):
        steps[0] += direction
        _write_json(directory / "navigation.json", dict(steps=steps[0]))

    scene.on_navigate = request_navigation

    def refresh(*_):
        nonlocal generation, stamp
        if (directory / "close").exists():
            scene.interactor.TerminateApp()
            return
        path = directory / "state.json"
        current_stamp = path.stat().st_mtime_ns
        if current_stamp == stamp:
            return
        state = json.loads(path.read_text(encoding="utf-8"))
        if generation != state["generation"]:
            with np.load(directory / state["geometry"], allow_pickle=False) as data:
                mesh = BoneMesh(data["vertices"], data["faces"], data["bounds"])
            scene.set_mesh(mesh, key=state["generation"], title=state["title"])
            generation = state["generation"]
        else:
            scene.label.SetInput(state["title"] + "\nDrag to rotate | wheel to zoom | arrows: previous/next")
        for name in scene.clouds:
            scene.set_points(name, state["points"].get(name, []))
        scene.set_opacity(state["opacity"])
        scene.window.SetShowWindow(bool(state["visible"]) and not offscreen)
        scene.render()
        stamp = current_stamp
        save_frame()

    try:
        refresh()
        if not offscreen:
            scene.interactor.AddObserver("TimerEvent", refresh)
            scene.interactor.AddObserver("EndInteractionEvent", save_frame)
            scene.interactor.CreateRepeatingTimer(50)
            scene.interactor.Start()
    finally:
        scene.close()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--offscreen", action="store_true")
    arguments = parser.parse_args()
    run_renderer(arguments.scene_dir, offscreen=arguments.offscreen)

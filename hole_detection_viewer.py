"""A large 3D opening inspector, size histogram and optional linked slice window."""

import numpy as np


def _surface_points(volume, spacing, *, step, max_points, value=None, origin=(0, 0, 0)):
    """Sample occupied block surfaces for drawing, without changing analysis data."""
    from scipy import ndimage as ndi

    occupied = volume if value is None else volume == value
    if step > 1:
        for axis, length in enumerate(volume.shape):
            occupied = np.logical_or.reduceat(occupied, np.arange(0, length, step), axis=axis)
    surface = occupied & ~ndi.binary_erosion(occupied)
    indices = np.flatnonzero(surface)
    if len(indices) > max_points:
        indices = indices[np.linspace(0, len(indices) - 1, max_points, dtype=int)]
    starts = np.column_stack(np.unravel_index(indices, surface.shape)) * step
    ends = np.minimum(starts + step, volume.shape)
    centers = (starts + ends - 1) / 2 + np.asarray(origin)
    return centers * np.asarray(spacing)


def plot_openings(bone, result, *, unit="units", surface_step=None, max_points=8000,
                  mesh_max_faces=60000, mesh_renderer="vtk", mesh_offscreen=None):
    """Return (main figure, controls), retaining controls while the UI is open.

    Previous/next and keyboard arrows navigate retained candidates largest first.
    Histogram cutoff filters inspection only; source arrays and measurements are
    untouched. The separate slice window is created lazily via 'Slices'.
    All axes use spacing-scaled array coordinates, not anatomical orientation.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.ticker import MaxNLocator
    from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
    from scipy import ndimage as ndi

    if surface_step is None:
        surface_step = max(1, int(np.ceil(max(bone.shape) / 96)))
    if not isinstance(surface_step, (int, np.integer)) or surface_step < 1:
        raise ValueError("surface_step must be a positive integer")
    if not isinstance(max_points, (int, np.integer)) or max_points < 1:
        raise ValueError("max_points must be a positive integer")
    if not isinstance(mesh_max_faces, (int, np.integer)) or mesh_max_faces < 1:
        raise ValueError("mesh_max_faces must be a positive integer")
    if mesh_renderer not in ("vtk", "matplotlib"):
        raise ValueError("mesh_renderer must be vtk or matplotlib")
    if mesh_offscreen is None:
        mesh_offscreen = plt.get_backend().lower() == "agg"

    spacing = np.asarray(result.spacing)
    openings = {opening.id: opening for opening in result.openings}
    ordered = sorted(openings, key=lambda key: (-openings[key].area, key))
    boxes = ndi.find_objects(result.opening_labels)
    state = dict(metric="Area", threshold=0.0, retained_ids=ordered.copy(),
                 selected_id=ordered[0] if ordered else None,
                 mode="opening" if ordered else "empty", focus=False, render_mode="points")
    widgets = {"state": state}
    counts = dict(skull=0, cavity=0, opening_markers=0, selected_opening=0)
    counts["mesh_faces"] = 0
    counts["mesh_vertices"] = 0
    widgets["render_counts"] = counts
    orange, blue, ink, background = "#e88c20", "#3e9fbc", "#17243a", "#f5f7fa"

    fig = plt.figure(figsize=(17, 10), facecolor=background)
    ax3d = fig.add_axes((0.005, 0.19, 0.68, 0.75), projection="3d", facecolor=background)
    ax3d.set_axis_off()
    ax3d.set_proj_type("ortho")
    ax3d.view_init(elev=22, azim=-55)
    fig.suptitle("Endocranium opening inspector", fontsize=21, fontweight="bold", y=0.98, color=ink)
    fig.text(0.035, 0.935, "Inspect largest first. Use the arrows to move; drag the skull to rotate.",
             fontsize=11, color="#44546a")
    selection_title = fig.text(0.035, 0.865, "", fontsize=13, color=ink)
    view_status = fig.text(0.035, 0.90, "View: point preview", fontsize=11, color=ink)
    mesh_note = fig.text(0.035, 0.20, "", fontsize=9, color=ink)
    mesh_cache = dict(key=None, artist=None, mesh=None)
    gpu = [None]
    gpu_image = [None]
    display_clouds = {name: np.empty((0, 3)) for name in ("opening", "other", "cavity", "seed")}
    summary = fig.text(0.73, 0.43, "", va="top", fontsize=11, color=ink, linespacing=1.65)
    status = fig.text(0.73, 0.945, "", fontsize=11, color=ink)
    fig.text(0.035, 0.014,
             "Orange: selected opening. Grey: skull. Cutoff filters inspection only; size alone does not confirm a real opening.",
             fontsize=9, color="#44546a")

    def cloud(color, size, alpha=1):
        return ax3d.scatter([], [], [], color=color, s=size, alpha=alpha,
                             linewidths=0, depthshade=False)

    skull = cloud("#8193ab", 2.5, 0.23)
    cavity = cloud(blue, 3, 0.2)
    others = cloud(blue, 18, 0.65)
    selected_patch = cloud(orange, 15)
    seed_marker = cloud("#32b66a", 45)
    highlight, = ax3d.plot([], [], [], "o", ms=11, mfc="none", mec=orange, mew=2)
    cavity.set_visible(False)
    others.set_visible(False)

    def set_points(artist, points):
        artist.set_offsets(points[:, :2])
        artist.set_3d_properties(points[:, 2], zdir="z")

    whole_skull = _surface_points(bone, spacing, step=surface_step, max_points=max_points)
    whole_cavity = [None]
    if len(whole_skull):
        padding = spacing * surface_step
        whole_bounds = np.stack((whole_skull.min(axis=0) - padding,
                                 whole_skull.max(axis=0) + padding))
    else:
        whole_bounds = np.stack((-0.5 * spacing, (np.asarray(bone.shape) - 0.5) * spacing))

    def apply_bounds(bounds):
        for axis, setter in enumerate((ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim)):
            setter(bounds[0, axis], bounds[1, axis])
        ax3d.set_box_aspect(np.maximum(bounds[1] - bounds[0], spacing), zoom=1.15)

    def active_point():
        if state["selected_id"] is not None:
            return np.asarray(openings[state["selected_id"]].centroid_ijk)
        return np.asarray(result.center_ijk)

    def get_gpu():
        from gpu_mesh_viewer import MeshPanel

        if gpu[0] is None or gpu[0].scene.closed:
            if gpu[0] is not None:
                gpu[0].close()
            gpu[0] = MeshPanel(fig, offscreen=mesh_offscreen, on_navigate=lambda direction: navigate(direction))
            gpu[0].scene.on_frame = lambda: capture_gpu_view(redraw=True)
            widgets["gpu_panel"] = gpu[0]
        if gpu_image[0] is None:
            snapshot_axes = fig.add_axes(MeshPanel.region)
            snapshot_axes.set_axis_off()
            gpu_image[0] = snapshot_axes.imshow(np.zeros((2, 2, 3)), aspect="equal")
        return gpu[0]

    def capture_gpu_view(*, redraw=False):
        if gpu[0] is None or gpu_image[0] is None or gpu[0].scene.closed:
            return
        snapshot = gpu[0].scene.snapshot()
        gpu_image[0].set_data(snapshot)
        gpu_image[0].set_extent((0, snapshot.shape[1], snapshot.shape[0], 0))
        if redraw:
            fig.canvas.draw_idle()

    def sync_gpu():
        if mesh_renderer == "vtk" and state["render_mode"] == "mesh" and gpu[0] is not None:
            for name, points in display_clouds.items():
                gpu[0].scene.set_points(name, points)
            gpu[0].scene.set_opacity(opacity.val)
            try:
                gpu[0].show()
                # The image supports PNG export; live rotation is handled by the child.
                capture_gpu_view()
            except (OSError, RuntimeError) as error:
                gpu[0].close()
                state["render_mode"] = "mesh unavailable"
                view_status.set_text("Mesh unavailable: the GPU renderer process could not start")
                mesh_note.set_text("Renderer details are in the terminal. The inspection controls remain available.")
                print(f"GPU mesh viewer failed: {error}", flush=True)
                if gpu_image[0] is not None:
                    gpu_image[0].axes.set_visible(False)
                ax3d.set_visible(True)
                return
            gpu_image[0].axes.set_visible(True)
            ax3d.set_visible(False)
        else:
            if gpu[0] is not None:
                gpu[0].hide()
            if gpu_image[0] is not None:
                gpu_image[0].axes.set_visible(False)
            ax3d.set_visible(True)

    def update_context():
        """Focus uses fresh local samples, so nearby bone has useful detail."""
        point = active_point()
        mesh_mode = options.get_status()[2] and state["mode"] != "empty"
        if mesh_cache["artist"] is not None:
            mesh_cache["artist"].set_visible(False)
        skull.set_visible(not mesh_mode)
        state["render_mode"] = "mesh unavailable" if mesh_mode else "points"
        view_status.set_text("View: mesh requested" if mesh_mode else "View: point preview")
        mesh_note.set_text("")
        counts["mesh_faces"] = 0
        counts["mesh_vertices"] = 0
        if state["focus"] and state["mode"] != "empty":
            if state["selected_id"] is not None:
                box = boxes[state["selected_id"] - 1]
                dimensions = np.array([s.stop - s.start for s in box]) * spacing
            else:
                dimensions = np.zeros(3)
            half_width = max(float(dimensions.max()) * 1.3,
                             float((whole_bounds[1] - whole_bounds[0]).max()) * 0.10,
                             float(spacing.max()) * 4)
            lower = np.maximum(0, np.floor(point - half_width / spacing)).astype(int)
            upper = np.minimum(bone.shape, np.ceil(point + half_width / spacing + 1)).astype(int)
            roi = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
            step = max(1, int(np.ceil(max(upper - lower) / 96)))
            skull_points = _surface_points(bone[roi], spacing, step=step,
                                            max_points=max_points, origin=lower)
            cavity_points = (_surface_points(result.regions[roi], spacing, step=step,
                              max_points=min(4000, max_points), value=1, origin=lower)
                             if cavity.get_visible() else np.empty((0, 3)))
            bounds = np.stack(((lower - 0.5) * spacing, (upper - 0.5) * spacing))
            if mesh_mode:
                from skull_mesh import local_bone_mesh, simplify_bone_mesh
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                from matplotlib.colors import LightSource

                minimum = max(float(dimensions.max()) / 2 + 2 * spacing.max(), 4 * spacing.max())
                key = (tuple(point), half_width, minimum)
                if mesh_cache["key"] != key:
                    if mesh_cache["artist"] is not None:
                        mesh_cache["artist"].remove()
                    mesh = local_bone_mesh(bone, point, spacing, half_width=half_width,
                                           min_half_width=minimum, max_faces=mesh_max_faces)
                    reduction_error = None
                    if not mesh.within_budget:
                        try:
                            mesh = simplify_bone_mesh(mesh, max_faces=mesh_max_faces)
                        except ImportError:
                            reduction_error = "VTK is needed to simplify this mesh. Install vtk or raise --mesh-max-faces."
                    mesh_cache.update(key=key, mesh=mesh, artist=None, error=reduction_error)
                    if mesh_renderer == "matplotlib" and mesh.within_budget and len(mesh.faces):
                        artist = Poly3DCollection(mesh.vertices[mesh.faces],
                            facecolors="#b9c4cf", linewidths=0, shade=True,
                            lightsource=LightSource(azdeg=315, altdeg=35), alpha=opacity.val)
                        ax3d.add_collection3d(artist)
                        mesh_cache["artist"] = artist
                mesh = mesh_cache["mesh"]
                if mesh.within_budget and len(mesh.faces):
                    if mesh_renderer == "vtk":
                        try:
                            panel = get_gpu()
                            panel.scene.set_mesh(mesh, key=key,
                                title=f"Opening #{state['selected_id']} | {len(state['retained_ids'])} retained")
                        except ImportError:
                            view_status.set_text("Mesh unavailable: VTK renderer is not installed")
                            mesh_note.set_text("Install vtk, or explicitly select --mesh-renderer matplotlib (software).")
                            set_points(skull, skull_points)
                            sync_gpu()
                            return
                    else:
                        mesh_cache["artist"].set_visible(True)
                    skull.set_visible(False)
                    counts["mesh_faces"] = len(mesh.faces)
                    counts["mesh_vertices"] = len(mesh.vertices)
                    state["render_mode"] = "mesh"
                    reduced = f" (from {mesh.source_faces:,}; topology-preserving preview)" if mesh.source_faces else ""
                    view_status.set_text(f"View: bone mesh ({'OpenGL' if mesh_renderer == 'vtk' else 'software'}) | "
                                         f"{len(mesh.vertices):,} vertices | {len(mesh.faces):,} triangles{reduced}")
                    bounds = mesh.bounds
                    # The budget may have narrowed the crop. Keep the temporary
                    # point preview inside those same bounds when rotating.
                    mesh_lower = np.rint(bounds[0] / spacing).astype(int)
                    mesh_upper = np.rint(bounds[1] / spacing).astype(int) + 1
                    mesh_roi = tuple(slice(int(a), int(b)) for a, b in zip(mesh_lower, mesh_upper))
                    mesh_step = max(1, int(np.ceil(max(mesh_upper - mesh_lower) / 96)))
                    skull_points = _surface_points(bone[mesh_roi], spacing, step=mesh_step,
                                                    max_points=max_points, origin=mesh_lower)
                    if cavity.get_visible():
                        cavity_points = _surface_points(result.regions[mesh_roi], spacing, step=mesh_step,
                            max_points=min(4000, max_points), value=1, origin=mesh_lower)
                    mesh_note.set_text("Bone surface; crop sides are artificial cuts. Opening overlay is optional."
                        + (" Rotate in the linked GPU window." if mesh_renderer == "vtk"
                           and not panel.embedded and not mesh_offscreen else ""))
                elif not mesh.within_budget:
                    view_status.set_text(f"Mesh unavailable: {len(mesh.faces):,} triangles exceeds {mesh_max_faces:,} limit")
                    mesh_note.set_text(mesh_cache.get("error") or
                        "Topology-preserving reduction could not meet the limit. Increase --mesh-max-faces or inspect another candidate.")
                else:
                    view_status.set_text("Mesh unavailable: no bone surface in this crop")
                    mesh_note.set_text("No bone surface in this crop. Use Slices or Whole skull to locate bone.")
        else:
            skull_points, bounds = whole_skull, whole_bounds
            if cavity.get_visible() and whole_cavity[0] is None:
                whole_cavity[0] = _surface_points(result.regions, spacing, step=surface_step,
                                                  max_points=min(4000, max_points), value=1)
            cavity_points = (whole_cavity[0] if cavity.get_visible() else np.empty((0, 3)))
        set_points(skull, skull_points)
        set_points(cavity, cavity_points)
        counts.update(skull=len(skull_points), cavity=len(cavity_points))
        display_clouds["cavity"] = cavity_points if cavity.get_visible() else np.empty((0, 3))
        apply_bounds(bounds)
        sync_gpu()

    histogram = fig.add_axes((0.74, 0.68, 0.23, 0.23), facecolor="white")
    metric = RadioButtons(fig.add_axes((0.74, 0.575, 0.23, 0.06), facecolor=background),
                          ("Area", "Diameter"), active=0)
    metric.ax.set_frame_on(False)
    cutoff = Slider(fig.add_axes((0.79, 0.505, 0.16, 0.025)), "Minimum", 0, 1,
                     valinit=0, color=orange, valfmt="%.4g")
    cutoff.drawon = False
    cutoff_units = fig.text(0.74, 0.548, "", fontsize=10, color=ink)
    fig.text(0.74, 0.465, "Click the histogram to place the cutoff.", fontsize=9, color="#44546a")
    widgets.update(metric=metric, threshold=cutoff)

    def values_for_metric():
        attribute = "area" if state["metric"] == "Area" else "equivalent_diameter"
        return np.asarray([getattr(openings[key], attribute) for key in ordered])

    def metric_unit():
        return f"{unit}\u00b2" if state["metric"] == "Area" else unit

    def draw_histogram():
        values = values_for_metric()
        histogram.clear()
        if len(values):
            positive = values[values > 0]
            use_log = len(positive) > 0 and positive.max() / positive.min() > 50
            if use_log:
                bins = np.geomspace(positive.min() * 0.8, positive.max() * 1.1, 25)
                histogram.set_xscale("log")
            else:
                bins = np.linspace(0, max(float(values.max()) * 1.05, 1e-9), 21)
            histogram.hist(values, bins=bins, color="#d6dee8", label="All found")
            retained = set(state["retained_ids"])
            keep = np.asarray([key in retained for key in ordered])
            histogram.hist(values[keep], bins=bins,
                            color=blue, alpha=0.85, label="Retained")
            if state["threshold"] > 0:
                histogram.axvline(state["threshold"], color="#c53069", lw=1.6, label="Cutoff")
            if state["selected_id"] is not None:
                opening = openings[state["selected_id"]]
                selected_value = opening.area if state["metric"] == "Area" else opening.equivalent_diameter
                histogram.axvline(selected_value, color=orange, lw=2)
            histogram.set_xlim(bins[0], max(bins[-1], state["threshold"]))
            histogram.legend(fontsize=8, frameon=False)
        else:
            histogram.text(0.5, 0.5, "No detected openings", ha="center", transform=histogram.transAxes)
        histogram.set(title="Opening size distribution", xlabel=f"{state['metric']} [{metric_unit()}]",
                       ylabel="Count")
        histogram.tick_params(labelsize=9)
        histogram.yaxis.set_major_locator(MaxNLocator(integer=True))
        histogram.spines[["top", "right"]].set_visible(False)
        cutoff_units.set_text(f"Minimum {state['metric'].lower()} [{metric_unit()}]")

    slice_images, slice_axes, slice_crosses = [], [], []
    slice_controls = []

    def update_slices(_=None):
        if "slice_figure" not in widgets:
            return
        point = active_point()
        for axis, (ax, image, cross) in enumerate(zip(slice_axes, slice_images, slice_crosses)):
            index = int(slice_controls[axis].val)
            plane = [slice(None)] * 3
            plane[axis] = index
            plane = tuple(plane)
            regions = result.regions[plane]
            codes = np.zeros(regions.shape, dtype=np.uint8)
            codes[bone[plane] != 0] = 1
            codes[regions == 1], codes[regions == 2] = 2, 3
            if state["selected_id"] is not None:
                codes[result.opening_labels[plane] == state["selected_id"]] = 4
            image.set_data(codes.T)
            remaining = [d for d in range(3) if d != axis]
            position = point[remaining] * spacing[remaining]
            cross.set_data([position[0]], [position[1]])
            cross.set_color("#32b66a" if state["mode"] == "seed" else orange)
            cross.set_visible(state["mode"] != "empty" and abs(index - point[axis]) <= 0.5)
            ax.set_title(f"{'ijk'[axis]} = {index} | {index * spacing[axis]:.4g} {unit}", fontsize=11)
        widgets["slice_figure"].suptitle(
            f"Full-resolution slices | opening #{state['selected_id']}" if state["selected_id"] is not None
            else "Full-resolution slices | cavity seed" if state["mode"] == "seed"
            else "Full-resolution slices | no candidate retained", fontsize=14)
        widgets["slice_figure"].canvas.draw_idle()

    def center_slices():
        point = active_point()
        for axis, control in enumerate(slice_controls):
            previous = control.eventson
            control.eventson = False
            control.set_val(int(np.floor(point[axis] + 0.5)))
            control.eventson = previous
        update_slices()

    def open_slices(_=None, *, show=True):
        slice_fig = widgets.get("slice_figure")
        if slice_fig is None or not plt.fignum_exists(slice_fig.number):
            slice_images.clear()
            slice_axes.clear()
            slice_crosses.clear()
            slice_controls.clear()
            slice_fig = plt.figure(figsize=(14, 6), facecolor=background)
            widgets["slice_figure"] = slice_fig
            cmap = ListedColormap(["#0b1220", "#dbe4ed", "#284a63", "#152033", orange])
            norm = BoundaryNorm(np.arange(6) - 0.5, cmap.N)
            grid = slice_fig.add_gridspec(1, 3, left=0.06, right=0.97, bottom=0.23, top=0.86, wspace=0.28)
            point = active_point()
            for axis in range(3):
                ax = slice_fig.add_subplot(grid[0, axis])
                remaining = [d for d in range(3) if d != axis]
                extent = [-0.5 * spacing[remaining[0]], (bone.shape[remaining[0]] - 0.5) * spacing[remaining[0]],
                          -0.5 * spacing[remaining[1]], (bone.shape[remaining[1]] - 0.5) * spacing[remaining[1]]]
                image = ax.imshow(np.zeros(tuple(bone.shape[d] for d in remaining)[::-1]),
                                  origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
                cross, = ax.plot([], [], "+", ms=14, mew=2)
                ax.set(xlabel=f"{'ijk'[remaining[0]]} [{unit}]", ylabel=f"{'ijk'[remaining[1]]} [{unit}]")
                slice_axes.append(ax)
                slice_images.append(image)
                slice_crosses.append(cross)
                control = Slider(slice_fig.add_axes((0.10 + axis * 0.315, 0.11, 0.22, 0.03)),
                                 f"{'ijk'[axis]}", 0, bone.shape[axis] - 1,
                                 valinit=int(np.floor(point[axis] + 0.5)), valstep=1, valfmt="%d", color=blue)
                control.drawon = False
                control.on_changed(update_slices)
                slice_controls.append(control)
                widgets['ijk'[axis]] = control
            slice_fig.text(0.06, 0.025, "Bone: light grey. Cavity: blue. Selected opening: orange. "
                            "Coordinates are voxel axes scaled by spacing.", fontsize=10, color=ink)
        center_slices()
        if show:
            slice_fig.show()
        return slice_fig

    widgets["open_slices"] = open_slices
    options = CheckButtons(fig.add_axes((0.73, 0.125, 0.25, 0.14), facecolor=background),
                           ("Other retained points", "Cavity surface", "Local bone mesh", "Opening overlay"),
                           (False, False, False, True))
    options.ax.set_frame_on(False)
    widgets["options"] = options

    def update_selection(*, refresh_context=False):
        key = state["selected_id"]
        point = active_point()
        if key is not None:
            opening = openings[key]
            box = boxes[key - 1]
            patch = result.opening_labels[box]
            patch_points = _surface_points(patch, spacing,
                step=max(1, int(np.ceil(max(patch.shape) / 96))), max_points=min(1000, max_points),
                value=key, origin=[s.start for s in box])
            rank = state["retained_ids"].index(key) + 1
            selection_title.set_text(f"Opening #{key}  |  {rank} of {len(state['retained_ids'])} retained")
            summary.set_text(f"Opening #{key}\nArea: {opening.area:.5g} {unit}\u00b2\n"
                             f"Eq. diameter: {opening.equivalent_diameter:.5g} {unit}\n"
                             f"Voxel: {tuple(round(float(v), 2) for v in point)}\n"
                             "Order: largest to smallest")
        else:
            patch_points = np.empty((0, 3))
            selection_title.set_text("Cavity seed" if state["mode"] == "seed" else "No openings above cutoff")
            summary.set_text(f"Seed voxel: {result.center_ijk}\nUse Slices to check its placement."
                             if state["mode"] == "seed" else "No openings retained.\nLower the minimum size to inspect more.")
        set_points(selected_patch, patch_points)
        selected_patch.set_visible(options.get_status()[3])
        display_clouds["opening"] = patch_points if options.get_status()[3] else np.empty((0, 3))
        counts["selected_opening"] = len(patch_points)
        position = point * spacing
        highlight.set_data_3d([position[0]], [position[1]], [position[2]])
        highlight.set_visible(key is not None and options.get_status()[3])
        seed_points = np.asarray([position]) if state["mode"] == "seed" else np.empty((0, 3))
        set_points(seed_marker, seed_points)
        display_clouds["seed"] = seed_points
        other_keys = [candidate for candidate in state["retained_ids"] if candidate != key]
        other_points = (np.asarray([openings[candidate].centroid_physical for candidate in other_keys])
                        if options.get_status()[0] and other_keys else np.empty((0, 3)))
        set_points(others, other_points)
        display_clouds["other"] = other_points
        counts["opening_markers"] = len(other_points)
        status.set_text(f"{len(state['retained_ids'])} retained / {len(ordered)} found")
        enabled = bool(state["retained_ids"])
        widgets["previous"].set_active(enabled)
        widgets["next"].set_active(enabled)
        if refresh_context or state["focus"]:
            update_context()
        center_slices()
        draw_histogram()
        fig.canvas.draw_idle()

    def select_opening(key):
        if key not in state["retained_ids"]:
            raise ValueError("Opening is excluded by the current size cutoff")
        state.update(selected_id=key, mode="opening")
        update_selection()

    def navigate(direction):
        retained = state["retained_ids"]
        if not retained:
            return
        key = state["selected_id"]
        index = retained.index(key) if key in retained else (-1 if direction > 0 else 0)
        select_opening(retained[(index + direction) % len(retained)])

    def set_threshold(value):
        state["threshold"] = float(value)
        values = values_for_metric()
        state["retained_ids"] = [key for key, size in zip(ordered, values) if size >= value]
        if state["selected_id"] not in state["retained_ids"]:
            state["selected_id"] = next(iter(state["retained_ids"]), None)
            state["mode"] = "opening" if state["selected_id"] is not None else "empty"
        update_selection(refresh_context=state["mode"] == "empty")

    def set_metric(label):
        if state["metric"] != label:
            previous_cutoff = state["threshold"]
            state["metric"] = label
            state["threshold"] = (2 * np.sqrt(previous_cutoff / np.pi) if label == "Diameter"
                                    else np.pi * (previous_cutoff / 2)**2)
        values = values_for_metric()
        maximum = max(float(values.max()) * 1.05, state["threshold"] * 1.05, 1e-9) if len(values) else 1
        cutoff.valmax = maximum
        cutoff.ax.set_xlim(0, maximum)
        previous_events = cutoff.eventson
        cutoff.eventson = False
        cutoff.set_val(state["threshold"])
        cutoff.eventson = previous_events
        # Equivalent area/diameter cutoffs retain the exact same candidates.
        update_selection()

    def show_seed(_):
        state.update(selected_id=None, mode="seed")
        update_selection()

    def focus_opening(_):
        if state["mode"] == "empty":
            return
        state["focus"] = True
        update_context()
        fig.canvas.draw_idle()

    def whole_view(_):
        if options.get_status()[2]:
            options.set_active(2)
        state["focus"] = False
        update_context()
        fig.canvas.draw_idle()

    def make_button(key, label, x, y, width, callback):
        button = Button(fig.add_axes((x, y, width, 0.038)), label, color="white", hovercolor="#e2e9f0")
        button.on_clicked(callback)
        widgets[key] = button

    make_button("previous", "< Previous", 0.12, 0.145, 0.14, lambda _: navigate(-1))
    make_button("next", "Next >", 0.28, 0.145, 0.14, lambda _: navigate(1))
    make_button("focus", "Focus opening", 0.08, 0.085, 0.12, focus_opening)
    make_button("whole", "Whole skull", 0.21, 0.085, 0.12, whole_view)
    make_button("slices", "Slices...", 0.34, 0.085, 0.12, open_slices)
    make_button("seed", "Inspect seed", 0.47, 0.085, 0.12, show_seed)
    opacity = Slider(fig.add_axes((0.14, 0.04, 0.40, 0.018)), "Skull opacity", 0, 1,
                     valinit=0.23, color="#8193ab")
    opacity.drawon = False
    widgets.update(opacity=opacity, select_opening=select_opening, navigate=navigate)

    def set_opacity(value):
        skull.set_alpha(value)
        if mesh_cache["artist"] is not None:
            mesh_cache["artist"].set_alpha(value)
        sync_gpu()
        fig.canvas.draw_idle()

    def set_options(label):
        if label == "Local bone mesh":
            enable_mesh = options.get_status()[2]
            if enable_mesh:
                state["focus"] = True
                state["mesh_previous_opacity"] = opacity.val
                state["mesh_previous_overlay"] = options.get_status()[3]
                overlay = False
                alpha = 1.0
            else:
                overlay = state.get("mesh_previous_overlay", True)
                alpha = state.get("mesh_previous_opacity", 0.23)
            previous_events = options.eventson
            options.eventson = False
            if options.get_status()[3] != overlay:
                options.set_active(3)
            options.eventson = previous_events
            opacity.set_val(alpha)
        show_others, show_cavity, _, _ = options.get_status()
        others.set_visible(show_others)
        cavity.set_visible(show_cavity)
        update_selection(refresh_context=True)

    def on_key(event):
        if event.key in ("left", "right"):
            navigate(-1 if event.key == "left" else 1)

    def on_click(event):
        if event.inaxes == histogram and event.xdata is not None:
            cutoff.set_val(float(np.clip(event.xdata, 0, cutoff.valmax)))

    def on_scroll(event):
        if event.inaxes != ax3d:
            return
        bounds = np.asarray([ax3d.get_xlim(), ax3d.get_ylim(), ax3d.get_zlim()]).T
        center = bounds.mean(axis=0)
        half_width = (bounds[1] - bounds[0]) * (0.43 if event.button == "up" else 0.58)
        apply_bounds(np.stack((center - half_width, center + half_width)))
        fig.canvas.draw_idle()

    def close_gpu(*_):
        if gpu[0] is not None:
            gpu[0].close()

    widgets["close"] = close_gpu

    opacity.on_changed(set_opacity)
    options.on_clicked(set_options)
    cutoff.on_changed(set_threshold)
    metric.on_clicked(set_metric)
    widgets["histogram"] = histogram
    widgets["connections"] = [fig.canvas.mpl_connect("key_press_event", on_key),
                               fig.canvas.mpl_connect("button_press_event", on_click),
                               fig.canvas.mpl_connect("close_event", close_gpu),
                               fig.canvas.mpl_connect("scroll_event", on_scroll)]
    update_context()
    set_metric("Area")
    return fig, widgets

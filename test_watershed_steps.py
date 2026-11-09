"""Simple 2D slice inspection of the binary mask, EDT, markers and watershed.

No mesh viewer or OpenGL is used. Omit the input path to read script.py's path
setting without executing the pipeline; --synthetic provides a small example.
"""

import argparse
from pathlib import Path
import time

import numpy as np

from hole_detection import compute_watershed_stages


def plot_watershed_steps(bone, stages, *, unit="units"):
    """Four linked 2D panels; widgets only read slices of the computed arrays."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.widgets import Button, RadioButtons, Slider

    shape, spacing = bone.shape, np.asarray(stages.spacing)
    seed = np.asarray(stages.center_ijk)
    state = dict(axis=2, indices=seed.copy())
    fig, axes = plt.subplots(1, 4, figsize=(17, 6.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.055, right=0.91, bottom=0.34, top=0.83, wspace=0.30)
    title = fig.suptitle("", fontsize=15)
    headings = ("1. Cleaned bone\nWhite = bone", "2. Distance transform\nDistance to nearest bone",
                "3. Seed markers\nGreen = cavity; orange = boundary", "4. Watershed regions\nBlue = cavity; orange = exterior")
    marker_colors = ListedColormap(["#090c12", "#dbe4ed", "#38d96b", "#ed9c35"])
    region_colors = ListedColormap(["#090c12", "#dbe4ed", "#399bc6", "#ed9c35"])
    norm = BoundaryNorm(np.arange(5) - 0.5, 4)
    maximum = float(stages.distance.max())
    # Initially emphasize distances inside the cavity, not distant exterior air.
    display_max = min(maximum, max(2 * float(stages.distance[tuple(seed)]), float(spacing.min())))
    images, crosses = [], []
    for index, (ax, heading) in enumerate(zip(axes, headings)):
        if index == 0:
            image = ax.imshow(np.zeros((2, 2)), cmap="gray", vmin=0, vmax=1, origin="lower")
        elif index == 1:
            image = ax.imshow(np.zeros((2, 2)), cmap="magma", vmin=0, vmax=display_max, origin="lower")
        else:
            image = ax.imshow(np.zeros((2, 2)), cmap=marker_colors if index == 2 else region_colors,
                              norm=norm, origin="lower")
        image.set_interpolation("nearest")
        images.append(image)
        crosses.append(ax.plot([], [], "+", color="#38d96b", ms=13, mew=1.5)[0])
        ax.set_title(heading, fontsize=10)
    colorbar = fig.colorbar(images[1], cax=fig.add_axes((0.94, 0.37, 0.009, 0.43)))
    colorbar.set_label(f"Distance [{unit}]", fontsize=10)

    fig.text(0.065, 0.275, "Slice axis", fontsize=10)
    axis_choice = RadioButtons(fig.add_axes((0.065, 0.075, 0.16, 0.18)), ("i (axis 0)", "j (axis 1)", "k (axis 2)"), active=2)
    slice_slider = Slider(fig.add_axes((0.37, 0.19, 0.43, 0.035)), "Slice", 0, shape[2] - 1,
                          valinit=int(seed[2]), valstep=1, valfmt="%d")
    contrast = Slider(fig.add_axes((0.37, 0.10, 0.43, 0.035)), "DT display max",
                      min(float(spacing.min()) * 0.1, maximum * 0.01), maximum,
                      valinit=display_max, valfmt="%.4g")
    center_button = Button(fig.add_axes((0.85, 0.185, 0.10, 0.045)), "Seed slice")
    slice_slider.drawon = contrast.drawon = False
    fig.text(0.27, 0.035,
             "Green + marks the seed only in its plane. Black in the result = unseeded air; white = bone.\n"
             "The distance is shown positive; watershed uses its negative. All coordinates follow voxel axes (i, j, k).",
             fontsize=9)

    def update_slice(_=None, *, reset_limits=False):
        axis, index = state["axis"], int(slice_slider.val)
        state["indices"][axis] = index
        plane = [slice(None)] * 3
        plane[axis] = index
        plane = tuple(plane)
        bone_slice = bone[plane]
        markers, regions = stages.markers[plane], stages.regions[plane]
        marker_view = np.zeros(bone_slice.shape, dtype=np.uint8)
        region_view = np.zeros(bone_slice.shape, dtype=np.uint8)
        marker_view[bone_slice != 0] = region_view[bone_slice != 0] = 1
        marker_view[markers == 1], marker_view[markers == 2] = 2, 3
        region_view[regions == 1], region_view[regions == 2] = 2, 3
        remaining = [d for d in range(3) if d != axis]
        extent = [-0.5 * spacing[remaining[0]], (shape[remaining[0]] - 0.5) * spacing[remaining[0]],
                  -0.5 * spacing[remaining[1]], (shape[remaining[1]] - 0.5) * spacing[remaining[1]]]
        for ax, image, cross, values in zip(axes, images, crosses,
                                           (bone_slice, stages.distance[plane], marker_view, region_view)):
            image.set_data(values.T)
            image.set_extent(extent)
            ax.set(xlabel=f"{'ijk'[remaining[0]]} [{unit}]", ylabel=f"{'ijk'[remaining[1]]} [{unit}]")
            position = seed[remaining] * spacing[remaining]
            cross.set_data([position[0]], [position[1]])
            cross.set_visible(index == seed[axis])
        if reset_limits:
            axes[0].set_xlim(extent[0], extent[1])
            axes[0].set_ylim(extent[2], extent[3])
        title.set_text(f"Watershed steps | {'ijk'[axis]} slice {index} / {shape[axis] - 1} "
                       f"({index * spacing[axis]:.4g} {unit}) | seed {stages.center_ijk}")
        fig.canvas.draw_idle()

    def change_axis(label):
        state["axis"] = "ijk".index(label[0])
        axis = state["axis"]
        previous = slice_slider.eventson
        slice_slider.eventson = False
        slice_slider.valmax = shape[axis] - 1
        slice_slider.ax.set_xlim(0, shape[axis] - 1)
        slice_slider.set_val(int(state["indices"][axis]))
        slice_slider.eventson = previous
        update_slice(reset_limits=True)

    def change_contrast(value):
        images[1].set_clim(0, float(value))
        fig.canvas.draw_idle()

    slice_slider.on_changed(update_slice)
    axis_choice.on_clicked(change_axis)
    contrast.on_changed(change_contrast)
    center_button.on_clicked(lambda _: slice_slider.set_val(int(seed[state["axis"]])))
    update_slice(reset_limits=True)
    return fig, dict(axis=axis_choice, slice=slice_slider, contrast=contrast, seed=center_button,
                     state=state, images=images)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nifti", nargs="?", help="Input scan; defaults to the path in script.py")
    parser.add_argument("--center", nargs=3, type=float, metavar=("I", "J", "K"),
                        help="Approximate cavity seed in original array coordinates; otherwise use automatic ranking")
    parser.add_argument("--synthetic", action="store_true", help="Use a small skull with two known openings")
    parser.add_argument("--no-show", action="store_true", help="Save a 2D PNG without opening any windows")
    parser.add_argument("--output", type=Path, help="Save a PNG; --no-show defaults to results/watershed-steps.png")
    args = parser.parse_args(argv)
    if args.synthetic and args.nifti:
        parser.error("Choose a NIfTI input or --synthetic, not both")
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    from test_hole_detection import configured_input, synthetic_skull

    if args.synthetic:
        bone, seed = synthetic_skull()
        spacing, unit = (1, 1, 1), "units"
    else:
        try:
            import nibabel as nib
        except ModuleNotFoundError:
            parser.error("NIfTI input requires nibabel in your pipeline environment; --synthetic works without it")
        path = args.nifti if args.nifti else configured_input()
        image = nib.load(path)
        if len(image.shape) != 3:
            parser.error("Expected a 3D NIfTI scan")
        spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
        if not np.isfinite(spacing).all() or np.any(spacing <= 0):
            parser.error("The NIfTI affine must specify finite positive voxel sizes")
        directions = image.affine[:3, :3] / spacing
        if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-5):
            parser.error("Sheared voxel axes are unsupported; use an orthogonal volume")
        unit = {"mm": "mm", "meter": "m", "micron": "um"}.get(image.header.get_xyzt_units()[0], "units")
        from preprocessing import preprocessing
        print("Cleaning scan: Otsu threshold and largest bone component...", flush=True)
        bone, _ = preprocessing(path)
        seed = args.center
        if seed is None:
            from cavity_ranking import rank_cavity_candidates
            print("Selecting a center with the existing ranking method...", flush=True)
            candidates = rank_cavity_candidates(bone)
            seed = candidates.points[0].copy()
            del candidates
    if args.center is not None:
        seed = args.center
    print("Computing the full-resolution distance transform, markers and watershed...", flush=True)
    start = time.perf_counter()
    stages = compute_watershed_stages(bone, seed, spacing)
    # Labels only contain 0,1,2; compact them to keep inspection memory lower.
    stages.markers = stages.markers.astype(np.uint8)
    stages.regions = stages.regions.astype(np.uint8)
    print(f"Completed in {time.perf_counter() - start:.2f} s; seed = {stages.center_ijk}.", flush=True)
    print("Blue = cavity; orange = exterior. The slice controls do not rerun watershed.", flush=True)
    fig, controls = plot_watershed_steps(bone, stages, unit=unit)
    if args.output or args.no_show:
        output = args.output or Path("results/watershed-steps.png")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=130)
        print(f"Saved {output.resolve()}", flush=True)
    import matplotlib.pyplot as plt
    if not args.no_show:
        plt.show()
    plt.close(fig)
    return stages


if __name__ == "__main__":
    main()

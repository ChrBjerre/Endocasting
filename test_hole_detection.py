"""Inspect watershed openings after the pipeline's cleanup, without expansion.

Run with a NIfTI path, or omit it to use the path configured in script.py.
Use --synthetic for a small two-opening example; --no-show saves diagnostics
without opening a window. See --help for center and display options.
"""

import argparse
import ast
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from hole_detection import detect_openings


def configured_input():
    """Read only the literal input setting; importing script.py would RUN it."""
    source = Path(__file__).with_name("script.py")
    for node in ast.parse(source.read_bytes()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "path" for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("No input path configured in script.py; pass a NIfTI path explicitly")


def synthetic_skull():
    """Spherical shell with two separate square tunnels and a known seed."""
    i, j, k = np.ogrid[:81, :81, :81]
    radius2 = (i - 40)**2 + (j - 40)**2 + (k - 40)**2
    bone = (radius2 >= 22**2) & (radius2 <= 28**2)
    bone[40:, 36:45, 36:45] = False   # +i exit, 9 x 9 voxels.
    bone[37:44, :41, 37:44] = False   # -j exit, 7 x 7 voxels.
    return bone, (40, 40, 40)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nifti", nargs="?", help="Input scan; defaults to path in script.py (not executed)")
    parser.add_argument("--center", nargs=3, type=float, metavar=("I", "J", "K"),
                        help="Empty endocranium voxel in original array coordinates; otherwise use automatic ranking")
    parser.add_argument("--synthetic", action="store_true", help="Use a small skull with two known openings")
    parser.add_argument("--output", type=Path, default=Path("results/hole-detection"))
    parser.add_argument("--surface-step", type=int,
                        help="3D display block stride only; default bounds the preview grid to ~96 per axis")
    parser.add_argument("--max-points", type=int, default=8000,
                        help="Maximum skull points in the 3D preview (default: 8000); lower for slower computers")
    parser.add_argument("--mesh", action="store_true", help="Start in local bone mesh mode")
    parser.add_argument("--mesh-max-faces", type=int, default=60000,
                        help="Local mesh triangle target (default: 60000); raise to keep more detail")
    parser.add_argument("--mesh-renderer", choices=("vtk", "matplotlib"), default="vtk",
                        help="Mesh renderer: vtk/OpenGL (default) or the slower matplotlib software renderer")
    parser.add_argument("--no-show", action="store_true", help="Save reports/PNG without opening the interactive viewer")
    parser.add_argument("--save-volumes", action="store_true", help="Also save region/opening NIfTIs (requires nibabel)")
    args = parser.parse_args(argv)
    if args.synthetic and args.nifti:
        parser.error("Choose a NIfTI input or --synthetic, not both")
    if args.surface_step is not None and args.surface_step < 1:
        parser.error("--surface-step must be positive")
    if args.max_points < 1:
        parser.error("--max-points must be positive")
    if args.mesh_max_faces < 1:
        parser.error("--mesh-max-faces must be positive")
    if not args.synthetic or args.save_volumes:
        try:
            import nibabel as nib
        except ModuleNotFoundError:
            parser.error("NIfTI input/export requires nibabel in your Python environment "
                         "(python -m pip install nibabel). The --synthetic demo works without it.")
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from hole_detection_viewer import plot_openings

    if args.synthetic:
        bone, default_center = synthetic_skull()
        affine, header, spacing, unit = np.eye(4), None, (1.0, 1.0, 1.0), "mm"
        center = args.center if args.center is not None else default_center
        source, threshold, center_method = "synthetic two-opening shell", None, "known synthetic center"
    else:
        path = args.nifti if args.nifti else configured_input()
        image = nib.load(path)
        if len(image.shape) != 3:
            parser.error("Expected a 3D NIfTI scan")
        affine, header = image.affine, image.header.copy()
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        if not np.isfinite(spacing).all() or np.any(spacing <= 0):
            parser.error("NIfTI affine must specify finite positive voxel sizes")
        directions = affine[:3, :3] / spacing
        if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-5):
            parser.error("Sheared NIfTI axes are unsupported; provide a volume with orthogonal voxel axes")
        unit = {"meter": "m", "mm": "mm", "micron": "µm"}.get(header.get_xyzt_units()[0], "units")
        from preprocessing import preprocessing
        print("Cleaning scan with the existing Otsu threshold + largest bone component...", flush=True)
        bone, threshold = preprocessing(path)
        source = str(Path(path).resolve())
        center, center_method = args.center, "supplied center"
        if center is None:
            from cavity_ranking import rank_cavity_candidates
            print("Selecting the highest-ranked center with the existing pipeline method...", flush=True)
            candidates = rank_cavity_candidates(bone)
            center = candidates.points[0].copy()
            del candidates  # Release ranking fields before the full-resolution watershed.
            center_method = "automatic bottleneck/enclosure ranking"
    if args.center is not None:
        center_method = "supplied center"

    print(f"Shape: {bone.shape}; spacing: {tuple(float(v) for v in spacing)} {unit}; "
          f"center: {tuple(float(v) for v in center)}", flush=True)
    print("Running distance-transform watershed on empty space...", flush=True)
    start = time.perf_counter()
    result = detect_openings(bone, center, spacing)
    seconds = time.perf_counter() - start
    print(f"Detected {len(result.openings)} candidate openings in {seconds:.2f} s.", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)

    def world_point(point):
        return (affine[:3, :3] @ np.asarray(point) + affine[:3, 3]).tolist()

    rows = []
    for opening in result.openings:
        row = asdict(opening)
        row["centroid_world"] = world_point(opening.centroid_ijk)
        rows.append(row)
        print(f"  #{opening.id}: voxel {tuple(round(v, 2) for v in opening.centroid_ijk)}, "
              f"area ~{opening.area:.4g} {unit}², diameter ~{opening.equivalent_diameter:.4g} {unit}")
    report = dict(
        source=source, shape=list(bone.shape), threshold=threshold, spacing=result.spacing, units=unit,
        center_ijk=result.center_ijk, center_world=world_point(result.center_ijk),
        center_method=center_method, watershed_seconds=seconds, openings=rows,
        method="Watershed of negative empty-space EDT; cavity seed and empty boundary markers",
        connectivity=dict(watershed=6, interface_contacts=6, opening_components=26),
        region_labels={"0": "bone or unseeded air", "1": "endocranium seed region", "2": "exterior"},
        size_estimate="Sum of shared cavity/exterior voxel-face areas; diameter = 2*sqrt(area/pi)",
        coordinates="ijk: array indices; physical: ijk*spacing from voxel origin; world: NIfTI affine applied to ijk",
    )
    (args.output / "openings.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output / "openings.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["id", "i", "j", "k", "world_x", "world_y", "world_z", "voxel_count", "area", "equivalent_diameter", "units"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for opening, row in zip(result.openings, rows):
            writer.writerow(dict(id=opening.id, **dict(zip(("i", "j", "k"), opening.centroid_ijk)),
                                 **dict(zip(("world_x", "world_y", "world_z"), row["centroid_world"])),
                                 voxel_count=opening.voxel_count, area=opening.area,
                                 equivalent_diameter=opening.equivalent_diameter, units=unit))
    if args.save_volumes:
        for filename, data in (("regions.nii.gz", result.regions), ("openings.nii.gz", result.opening_labels)):
            label_image = nib.Nifti1Image(data.astype(np.int32, copy=False), affine, header=header)
            label_image.set_data_dtype(np.int32)
            label_image.header.set_intent("label")
            label_image.header["cal_min"], label_image.header["cal_max"] = 0, int(data.max())
            if args.synthetic:
                label_image.header.set_xyzt_units("mm")
            nib.save(label_image, args.output / filename)

    print("Building the interactive 3D and slice views...", flush=True)
    fig, widgets = plot_openings(bone, result, unit=unit, surface_step=args.surface_step,
                                 max_points=args.max_points, mesh_max_faces=args.mesh_max_faces,
                                 mesh_renderer=args.mesh_renderer, mesh_offscreen=args.no_show)
    if args.mesh:
        widgets["options"].set_active(2)
    print(f"3D view mode: {widgets['state']['render_mode']}", flush=True)
    print(f"3D display point counts: {widgets['render_counts']}", flush=True)
    fig.savefig(args.output / "overview.png", dpi=140, facecolor=fig.get_facecolor())
    print(f"Saved reports and overview.png to {args.output.resolve()}", flush=True)
    if not args.no_show:
        plt.show()
    if "slice_figure" in widgets:
        plt.close(widgets["slice_figure"])
    widgets["close"]()
    plt.close(fig)
    return result


if __name__ == "__main__":
    main()

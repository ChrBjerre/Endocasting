"""Visual diagnostics/CLI for cavity ranking; not imported by batch ranking."""

import argparse
import csv
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from cavity_ranking import cavity_above_escape, rank_cavity_candidates


def score_rows(result):
    for rank, point in enumerate(result.points):
        yield dict(rank=rank + 1, i=int(point[0]), j=int(point[1]), k=int(point[2]),
                   D_peak=float(result.peak[rank]), D_escape=float(result.escape[rank]),
                   bottleneck=float(result.bottleneck[rank]),
                   enclosure=None if np.isnan(result.enclosure[rank]) else float(result.enclosure[rank]),
                   score=float(result.score[rank]))


def plot_diagnostics(bone, result, candidate_index=0):
    """Orthogonal slices through a selected rank; all candidates are projected.

    The cavity contour uses D > D_escape: it is the component just disconnected
    from outside, not a final endocranium segmentation or an explicit escape path.
    """
    import matplotlib.pyplot as plt

    if not 0 <= candidate_index < len(result.points):
        raise ValueError("candidate rank is outside the candidate list")
    point = result.points[candidate_index]
    grid_point = result.grid_points[candidate_index]
    cavity = cavity_above_escape(result, candidate_index)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    for column, axis in enumerate(range(3)):
        remaining = [d for d in range(3) if d != axis]
        for row, (volume, title, cmap) in enumerate((
                (bone, "Original bone", "gray"),
                (result.analysis_distance, "Clearance D (coarse minimum if stride > 1)", "magma"),
                (result.escape_field, "Escape clearance; cyan = cavity above escape", "viridis"))):
            ax = axes[row, column]
            stride = 1 if row == 0 else result.stride
            location = point[axis] if row == 0 else grid_point[axis]
            image = np.take(volume, location, axis=axis).T
            extent = [-0.5, volume.shape[remaining[0]] * stride - 0.5,
                      -0.5, volume.shape[remaining[1]] * stride - 0.5]
            artist = ax.imshow(image, origin="lower", extent=extent, cmap=cmap,
                               interpolation="nearest")
            if row:
                fig.colorbar(artist, ax=ax, shrink=0.65, label="original voxels")
            if row == 2:
                section = np.take(cavity, location, axis=axis).T
                if section.any() and not section.all():
                    centers_x = np.arange(section.shape[1]) * stride + (stride - 1) / 2
                    centers_y = np.arange(section.shape[0]) * stride + (stride - 1) / 2
                    ax.contour(centers_x, centers_y, section, levels=[0.5], colors="cyan")
            for rank, candidate in enumerate(result.points):
                x, y = candidate[remaining]
                ax.plot(x, y, "o", mfc="none", mec="orange", ms=5)
                ax.annotate(str(rank + 1), (x, y), color="orange", fontsize=7)
            x, y = point[remaining]
            ax.plot(x, y, "+", color="lime", markersize=14, markeredgewidth=2)
            ax.set(xlim=(-0.5, bone.shape[remaining[0]] - 0.5),
                   ylim=(-0.5, bone.shape[remaining[1]] - 0.5),
                   xlabel=f"{'ijk'[remaining[0]]} voxel", ylabel=f"{'ijk'[remaining[1]]} voxel",
                   title=f"{title}\n{'ijk'[axis]} = {point[axis]}")
    fig.suptitle(f"Rank {candidate_index + 1} at {tuple(map(int, point))}; stride={result.stride}\n"
                 "Orange: ALL candidates projected onto each slice (not necessarily in plane). "
                 "Green: inspected candidate; rank 1 is recommended.")
    rows = list(score_rows(result))
    table_fig, table_ax = plt.subplots(figsize=(13, max(3, 0.25 * len(rows) + 1.5)))
    table_ax.axis("off")
    text = [[r["rank"], f"({r['i']}, {r['j']}, {r['k']})", f"{r['D_peak']:.3f}",
             f"{r['D_escape']:.3f}", f"{r['bottleneck']:.3f}",
             "off" if r["enclosure"] is None else f"{r['enclosure']:.3f}", f"{r['score']:.3f}"] for r in rows]
    table_ax.table(cellText=text, colLabels=["Rank", "(i,j,k)", "D peak", "D escape",
                                           "Difference", "Ray hits", "Score"], loc="center")
    table_ax.set_title("Candidate ranking: all distances in original voxel units; coarse escape is approximate")
    table_fig.tight_layout()
    return fig, table_fig


def synthetic_skull(size=65, opening=2):
    """Shell with a narrow tunnel and abundant surrounding external air."""
    xyz = np.ogrid[tuple(slice(0, size) for _ in range(3))]
    center = size // 2
    radius = sum((x - center) ** 2 for x in xyz)
    bone = (radius >= (size * 0.23) ** 2) & (radius <= (size * 0.29) ** 2)
    if opening:
        bone[center:, center-opening:center+opening+1, center-opening:center+opening+1] = False
    return bone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nifti", nargs="?", help="Prefer the PCA-resampled NIfTI used by the pipeline")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("center-debug"))
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--candidate-rank", type=int, default=1)
    parser.add_argument("--max-grid-size", type=int, default=160, help="0 means exact full-resolution escape analysis")
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--min-distance", type=float, default=10)
    parser.add_argument("--sigma", type=float, default=3)
    parser.add_argument("--rays", type=int, default=96, help="0 disables directional enclosure")
    parser.add_argument("--connectivity", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--enclosure-weight", type=float, default=0.5)
    args = parser.parse_args()
    if bool(args.nifti) == args.synthetic:
        parser.error("Provide one NIfTI path or --synthetic")
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    # Import plotting/preprocessing only after selecting the headless backend.
    from preprocessing import preprocessing
    import matplotlib.pyplot as plt

    if args.synthetic:
        bone, affine = synthetic_skull(), np.eye(4)
    else:
        image = nib.load(args.nifti)
        if len(image.shape) != 3:
            parser.error("Expected a 3D NIfTI")
        affine = image.affine
        bone, _ = preprocessing(args.nifti)
    settings = dict(sigma=args.sigma, min_distance=args.min_distance, max_candidates=args.max_candidates,
                    max_grid_size=None if args.max_grid_size == 0 else args.max_grid_size,
                    n_rays=args.rays, connectivity=args.connectivity, enclosure_weight=args.enclosure_weight)
    start = time.perf_counter()
    result = rank_cavity_candidates(bone, **settings)
    seconds = time.perf_counter() - start
    figures = plot_diagnostics(bone, result, args.candidate_rank - 1)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = list(score_rows(result))
    with (args.output / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(recommended_voxel_ijk=result.points[0].tolist(),
                  recommended_world=nib.affines.apply_affine(affine, result.points[0]).tolist(),
                  inspected_rank=args.candidate_rank, distance_units="original voxels",
                  analysis_stride=result.stride, approximate=result.stride > 1,
                  ranking_seconds=seconds, settings=settings, candidates=rows)
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, figure in zip(("slices", "scores"), figures):
        figure.savefig(args.output / f"{name}.png", dpi=130)
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))
    if not args.no_show:
        plt.show()
    for figure in figures:
        plt.close(figure)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Estimate an interior endocranium seed from a skull NIfTI, without learning.

Install: python -m pip install numpy scipy nibabel matplotlib
Run:     python find_endocranium_center.py

Edit SETTINGS below. All figures are saved; no display or GPU is required.
This is a geometric candidate finder, NOT an anatomical classifier. Check the
final overlays, particularly for large sinuses, damaged skulls, or scan holders.
The selected cavity is eroded by dilation: it is NOT an endocast segmentation.

Outputs (in a new folder beside the scan unless OUTPUT_DIR is set):
  01: intensity histogram and cavity persistence curves
  02: coarse CT and thresholded bone
  03: each dilation step, exterior flood fill, and enclosed cavities
  04: selected candidate versus up to three alternatives
  05: distance transform used to choose the point
  06: final point overlaid on three original-resolution slices
  result.json: coordinates, settings, and diagnostic notes
  candidates.csv: ranked candidates; IDs can be used with FORCE_CANDIDATE_ID
  selected_cavity_COARSE.nii.gz: eroded candidate mask with spatial metadata

All voxel indices use nibabel's original array order (i, j, k), zero-based.
NIfTI world coordinates are computed with the input affine. Do not interpret
array indices as anatomical x/y/z without consulting that affine.
"""

# ============================== SETTINGS ==============================
NIFTI_PATH = r"C:\Users\chris\Mit drev\DTU\0. Afsluttede kurser\Bachelorprojekt\DOG-ROOTS\Canislupus_CZM_712_downsampled_x025.nii"
OUTPUT_DIR = None                 # None: timestamped folder beside the scan
MAX_COARSE_DIM = 256              # Use 384/512 if small structures disappear
BONE_THRESHOLD = None            # None: Otsu; otherwise an intensity threshold
DILATION_RADII_VOX = tuple(range(17))  # 0..16 times smallest coarse voxel spacing
MIN_CAVITY_VOXELS = 64
STABLE_STEPS = 3                  # Prefer cavities present at >=3 tested radii
KEEP_LARGEST_BONE = False         # False preserves separate skull bones
FORCE_CANDIDATE_ID = None         # Optionally choose another ID from candidates.csv
SAVE_EVERY_DILATION = True        # One explanatory PNG for every tested radius
FIGURE_DPI = 140
# =====================================================================

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")             # Works on HPC / SSH without a display
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

CONNECTIVITY = ndi.generate_binary_structure(3, 1)  # 6-connected free space
ORANGE = "#d28b45"
BLUE = "#447fa6"
GREEN = "#39917e"


def load_coarse(image, max_dim):
    """Stream axial blocks; never materialize the full high-resolution volume.

    Max pooling preserves thin bright bone better than strided sampling. It
    also thickens it by up to roughly one coarse cell and retains bright noise.
    Otsu is estimated separately on a sample of ORIGINAL intensities.
    Reading a .nii.gz may dominate runtime. Sequential reads reuse its handle.
    """
    shape = np.array(image.shape, dtype=int)
    factor = max(1, int(np.ceil(shape.max() / max_dim)))
    coarse_shape = (shape + factor - 1) // factor
    coarse = np.empty(coarse_shape, dtype=np.float32)
    xs, ys = np.arange(0, shape[0], factor), np.arange(0, shape[1], factor)
    samples = []
    for k, start in enumerate(range(0, shape[2], factor)):
        block = np.array(image.dataobj[:, :, start:start + factor],
                         dtype=np.float32, copy=True)
        samples.append(block[::factor, ::factor, 0].ravel().copy())
        np.nan_to_num(block, copy=False, nan=-np.inf,
                      posinf=-np.inf, neginf=-np.inf)
        pooled = np.maximum.reduceat(block, xs, axis=0)
        pooled = np.maximum.reduceat(pooled, ys, axis=1)
        coarse[:, :, k] = pooled.max(axis=2)
        if k % max(1, coarse_shape[2] // 10) == 0:
            print(f"  Reading/downsampling: {k + 1}/{coarse_shape[2]} blocks", flush=True)
    samples = np.concatenate(samples)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError("The input has no finite sampled intensities.")
    coarse[~np.isfinite(coarse)] = samples.min()
    # Voxel centers are block centers; last partial blocks extend nominally past
    # the scan. The final original index is clipped and explicitly checked.
    transform = np.eye(4)
    transform[:3, :3] *= factor
    transform[:3, 3] = (factor - 1) / 2
    return coarse, samples, image.affine @ transform, factor


def otsu_threshold(values):
    low, high = np.percentile(values, [0.1, 99.9])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        raise ValueError("Constant sampled intensities: thresholding is impossible.")
    hist, edges = np.histogram(np.clip(values, low, high), bins=512, range=(low, high))
    centers = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(hist, dtype=float)
    moment = np.cumsum(hist * centers)
    denominator = weight[:-1] * (weight[-1] - weight[:-1])
    score = np.divide((moment[-1] * weight[:-1] - moment[:-1] * weight[-1]) ** 2,
                      denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return float(centers[np.argmax(score)])


def cavities_at_radius(distance_to_bone, radius):
    dilated = distance_to_bone <= radius
    # binary_fill_holes invades free space from ALL six faces of the volume.
    filled = ndi.binary_fill_holes(dilated, structure=CONNECTIVITY)
    enclosed = filled & ~dilated
    exterior = ~filled
    labels, _ = ndi.label(enclosed, structure=CONNECTIVITY)
    sizes = np.bincount(labels.ravel())
    return dilated, exterior, labels, sizes


def track_candidates(distance, radii, minimum, stable_steps):
    """Follow nested cavities as bone dilates; splits retain the largest child.

    Rank by peak voxel count * min(lifetime / stable_steps, 1). This deliberately
    simple heuristic rewards large persistent cavities; it does not establish
    anatomical identity. Representatives use the first (least eroded) mask.
    """
    tracks, history = {}, []
    previous_labels = np.zeros(distance.shape, dtype=np.int32)
    previous_map = {}
    next_id = 1
    for step, radius in enumerate(radii):
        _, _, labels, sizes = cavities_at_radius(distance, radius)
        good = np.flatnonzero(sizes >= minimum)
        good = good[good != 0]
        # Monotonic dilation means each new enclosed component has one parent
        # or was just disconnected from the exterior (previous label zero).
        parent_ids = ndi.maximum(previous_labels, labels=labels, index=good)
        parents = dict(zip(good.tolist(), np.asarray(parent_ids).tolist()))
        current_map, used = {}, set()
        for label in sorted(good, key=lambda value: int(sizes[value]), reverse=True):
            parent = int(parents[int(label)])
            track_id = previous_map.get(parent)
            if track_id is None or track_id in used:
                track_id = next_id
                next_id += 1
                tracks[track_id] = dict(id=track_id, first_step=step, last_step=step,
                    first_radius=float(radius), first_label=int(label),
                    peak_voxels=int(sizes[label]), observations=[])
            used.add(track_id)
            current_map[int(label)] = track_id
            tracks[track_id]["last_step"] = step
            tracks[track_id]["observations"].append([step, int(sizes[label])])
        history.append(dict(step=step, radius=float(radius), count=len(good),
                            largest_voxels=int(max(sizes[good], default=0))))
        print(f"  Radius {radius:.4g}: {len(good)} candidate cavities", flush=True)
        previous_labels, previous_map = labels, current_map
    for track in tracks.values():
        track["steps_present"] = len(track["observations"])
        track["score"] = track["peak_voxels"] * min(track["steps_present"] / stable_steps, 1)
    return sorted(tracks.values(), key=lambda t: (-t["score"], t["first_step"])), history


def slice2d(array, axis, index):
    return np.take(array, int(index), axis=axis).T


def draw_views(axes, array, index, spacing, title, window=None, masks=(),
               point=None, cmap="gray", axis_codes=None):
    for axis, ax in enumerate(axes):
        others = [i for i in range(3) if i != axis]
        extent = (-spacing[others[0]] / 2,
                  (array.shape[others[0]] - .5) * spacing[others[0]],
                  -spacing[others[1]] / 2,
                  (array.shape[others[1]] - .5) * spacing[others[1]])
        kw = {} if window is None else dict(vmin=window[0], vmax=window[1])
        ax.imshow(slice2d(array, axis, index[axis]), origin="lower", extent=extent,
                  cmap=cmap, interpolation="nearest", **kw)
        for mask, color, alpha in masks:
            plane = np.ma.masked_where(~slice2d(mask, axis, index[axis]),
                                       np.ones_like(slice2d(mask, axis, index[axis])))
            ax.imshow(plane, origin="lower", extent=extent,
                      cmap=ListedColormap([color]), vmin=0, vmax=1,
                      interpolation="nearest", alpha=alpha)
        if point is not None:
            ax.scatter(point[others[0]] * spacing[others[0]],
                       point[others[1]] * spacing[others[1]],
                       marker="+", s=210, linewidths=2, color="#e33243")
        ax.set_title(f"{title}\naxis {axis} = {index[axis]}", fontsize=10)
        suffix_x = f" ({axis_codes[others[0]]})" if axis_codes else ""
        suffix_y = f" ({axis_codes[others[1]]})" if axis_codes else ""
        ax.set_xlabel(f"axis {others[0]}{suffix_x} [mm]")
        ax.set_ylabel(f"axis {others[1]}{suffix_y} [mm]")
        ax.set_aspect("equal")


def save_figure(fig, path):
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    start_time = time.perf_counter()
    path = Path(NIFTI_PATH).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Set NIFTI_PATH at the top of the script. Not found: {path}")
    if MAX_COARSE_DIM < 16 or STABLE_STEPS < 1 or MIN_CAVITY_VOXELS < 1:
        raise ValueError("Invalid coarse size, stability, or cavity-size setting.")
    stem = path.name.removesuffix(".gz").removesuffix(".nii")
    out = (Path(OUTPUT_DIR).expanduser() if OUTPUT_DIR else path.parent /
           f"{stem}_center_{datetime.now():%Y%m%d_%H%M%S_%f}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out.resolve()}", flush=True)
    image = nib.load(str(path), mmap=True, keep_file_open=True)
    if len(image.shape) != 3 or min(image.shape) < 2:
        raise ValueError(f"Expected a 3D scalar volume; got {image.shape}.")
    native_spacing = nib.affines.voxel_sizes(image.affine)
    directions = image.affine[:3, :3] / native_spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-3):
        raise ValueError("Sheared affine: resample to an orthogonal grid first.")
    unit = image.header.get_xyzt_units()[0]
    scale_to_mm = {"mm": 1., "meter": 1000., "micron": .001}.get(unit, 1.)
    warnings = []
    if unit not in ("mm", "meter", "micron"):
        warnings.append("NIfTI spatial units unknown: assumed millimetres.")
    coarse, samples, coarse_affine, factor = load_coarse(image, MAX_COARSE_DIM)
    spacing = native_spacing * factor * scale_to_mm
    codes = nib.aff2axcodes(image.affine)
    threshold = otsu_threshold(samples) if BONE_THRESHOLD is None else float(BONE_THRESHOLD)
    bone = coarse > threshold
    if not np.any(bone) or np.all(bone):
        raise ValueError("Threshold produced empty/all-bone mask; adjust BONE_THRESHOLD.")
    if KEEP_LARGEST_BONE:
        bone_labels, _ = ndi.label(bone, structure=ndi.generate_binary_structure(3, 3))
        bone_sizes = np.bincount(bone_labels.ravel())
        bone_sizes[0] = 0
        bone = bone_labels == np.argmax(bone_sizes)
        del bone_labels
    radii_vox = np.unique(np.asarray(DILATION_RADII_VOX, dtype=float))
    if radii_vox.size == 0 or np.any(~np.isfinite(radii_vox)) or np.any(radii_vox < 0):
        raise ValueError("Dilation radii must be a nonempty sequence of finite nonnegative values.")
    radii = radii_vox * spacing.min()
    print(f"Coarse shape: {coarse.shape}; block factor: {factor}; threshold: {threshold:.5g}")
    print("Computing physical distance to bone...", flush=True)
    distance = ndi.distance_transform_edt(~bone, sampling=spacing)
    tracks, history = track_candidates(distance, radii, MIN_CAVITY_VOXELS, STABLE_STEPS)
    fields = ["id", "score", "peak_voxels", "steps_present", "first_step", "last_step", "first_radius"]
    with (out / "candidates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(tracks)
    # Save diagnostic plots even when there is no viable candidate.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    lo, hi = np.percentile(samples, [0.1, 99.9])
    axes[0].hist(samples, bins=160, range=(lo, hi), color=BLUE, log=True)
    axes[0].axvline(threshold, color=ORANGE, label=f"threshold = {threshold:.4g}")
    axes[0].set(xlabel="Original sampled intensity", ylabel="Voxel count (log)", title="Bone threshold")
    axes[0].legend()
    for track in tracks[:8]:
        obs = np.array(track["observations"])
        axes[1].plot(radii[obs[:, 0]], obs[:, 1] * np.prod(spacing), "o-", label=f"candidate {track['id']}")
    axes[1].set(xlabel="Dilation radius [mm]", ylabel="Remaining cavity volume [mm³]",
                title="Largest persistent candidates (up to 8)")
    if tracks:
        axes[1].legend(fontsize=8)
    save_figure(fig, out / "01_threshold_and_cavity_tracks.png")
    selected = None
    if tracks:
        selected = tracks[0] if FORCE_CANDIDATE_ID is None else next(
            (t for t in tracks if t["id"] == FORCE_CANDIDATE_ID), None)
        if selected is None:
            raise ValueError("FORCE_CANDIDATE_ID not found; consult candidates.csv.")
        _, _, labels, _ = cavities_at_radius(distance, selected["first_radius"])
        cavity = labels == selected["first_label"]
        interior_distance = ndi.distance_transform_edt(cavity, sampling=spacing)
        point = np.array(np.unravel_index(np.argmax(interior_distance), cavity.shape))
        if selected["steps_present"] < STABLE_STEPS:
            warnings.append("Selected cavity does not meet the requested persistence; inspect carefully.")
        if len(tracks) > 1 and tracks[1]["score"] >= .8 * tracks[0]["score"]:
            warnings.append("Two candidates have similar scores; inspect both before accepting.")
    else:
        # Keep a useful set of cross-sections on failure.
        point = np.rint(ndi.center_of_mass(bone)).astype(int)
    window = tuple(np.percentile(samples, [.5, 99.7]))
    if window[1] <= window[0]:
        window = (float(samples.min()), float(samples.max()))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    draw_views(axes[0], coarse, point, spacing, "Coarse CT (maximum per block)", window, axis_codes=codes)
    draw_views(axes[1], coarse, point, spacing, "Thresholded bone: orange", window,
               masks=[(bone, ORANGE, .65)], axis_codes=codes)
    fig.suptitle(f"Input and bone mask | factor {factor} | slices through {'chosen point' if selected else 'bone center'}")
    save_figure(fig, out / "02_input_and_bone.png")
    # Render through the SAME point at every radius so changes are comparable.
    steps = range(len(radii)) if SAVE_EVERY_DILATION else ([selected["first_step"]] if selected else [0, len(radii)-1])
    print("Rendering dilation and exterior-flood-fill steps...", flush=True)
    for step in steps:
        dilated, exterior, labels, sizes = cavities_at_radius(distance, radii[step])
        visible = (sizes >= MIN_CAVITY_VOXELS)
        visible[0] = False
        enclosed = visible[labels]
        fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
        draw_views(axes[0], coarse, point, spacing, "Bone after dilation: orange", window,
                   masks=[(dilated, ORANGE, .65)])
        draw_views(axes[1], coarse, point, spacing, "Exterior reached from borders: blue", window,
                   masks=[(exterior, BLUE, .6)])
        draw_views(axes[2], coarse, point, spacing, "Enclosed cavities: green", window,
                   masks=[(enclosed, GREEN, .8)], point=point if selected else None)
        fig.suptitle(f"Dilation step {step:02d} | radius {radii[step]:.3g} mm ({radii_vox[step]:g} coarse voxels)\n"
                     "All rows use fixed cross-sections; red cross = final selected point")
        save_figure(fig, out / f"03_dilation_{step:02d}.png")
    if selected is None:
        (out / "result.json").write_text(json.dumps(dict(status="no_candidate", warnings=warnings,
            suggestion="Check threshold, skull wall continuity, dilation range and minimum cavity size."), indent=2))
        print("No enclosed candidate found. Diagnostics saved; no center was invented.")
        return
    # Show alternative candidates at their own centers, including the selected one.
    top = [selected] + [t for t in tracks if t["id"] != selected["id"]][:3]
    fig, axes = plt.subplots(len(top), 3, figsize=(13, 3.5 * len(top)), squeeze=False, constrained_layout=True)
    for row, track in enumerate(top):
        _, _, labels, _ = cavities_at_radius(distance, track["first_radius"])
        mask = labels == track["first_label"]
        depth = ndi.distance_transform_edt(mask, sampling=spacing)
        center = np.array(np.unravel_index(np.argmax(depth), mask.shape))
        tag = "SELECTED" if track["id"] == selected["id"] else "alternative"
        draw_views(axes[row], coarse, center, spacing,
                   f"ID {track['id']} ({tag}) | {track['steps_present']} steps", window,
                   masks=[(mask, GREEN, .65)], point=center)
    fig.suptitle("Candidate comparison: each row passes through that candidate's center")
    save_figure(fig, out / "04_candidate_comparison.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    draw_views(axes, interior_distance, point, spacing, "Distance inside selected cavity [mm]",
               window=(0, interior_distance.max()), point=point, cmap="magma")
    fig.suptitle(f"Deepest point: {interior_distance[tuple(point)]:.3g} mm from eroded cavity boundary")
    save_figure(fig, out / "05_interior_distance.png")
    original_float = point * factor + (factor - 1) / 2
    original_index = np.clip(np.rint(original_float).astype(int), 0, np.array(image.shape)-1)
    world = nib.affines.apply_affine(image.affine, original_index)
    # Read only three ORIGINAL-resolution slices; never the whole original scan.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, ax in enumerate(axes):
        key = [slice(None)] * 3
        key[axis] = int(original_index[axis])
        plane = np.asarray(image.dataobj[tuple(key)], dtype=np.float32).T
        others = [a for a in range(3) if a != axis]
        ax.imshow(plane, cmap="gray", origin="lower", vmin=window[0], vmax=window[1],
                  aspect=native_spacing[others[1]] / native_spacing[others[0]])
        ax.scatter(original_index[others[0]], original_index[others[1]], marker="+",
                   s=240, linewidths=2, color="#e33243")
        ax.set(title=f"Original CT: axis {axis} = {original_index[axis]}",
               xlabel=f"axis {others[0]} ({codes[others[0]]}), voxel index",
               ylabel=f"axis {others[1]} ({codes[others[1]]}), voxel index")
    intensity = float(np.asarray(image.dataobj[tuple(original_index)]))
    if not np.isfinite(intensity) or intensity > threshold:
        warnings.append("Chosen original voxel is invalid or above bone threshold; reject/check this result.")
    fig.suptitle(f"Final point on ORIGINAL CT | 0-based (i, j, k) = {tuple(original_index.tolist())}\n"
                 "Arrays remain in the original NIfTI axis order (not necessarily axial/coronal/sagittal).")
    save_figure(fig, out / "06_FINAL_original_cross_sections.png")
    # Save coarse cavity with its correct block-center affine, useful in Slicer.
    cavity_image = nib.Nifti1Image(cavity.astype(np.uint8), coarse_affine)
    cavity_image.header.set_xyzt_units(unit if unit in ("mm", "meter", "micron") else "mm")
    nib.save(cavity_image, str(out / "selected_cavity_COARSE.nii.gz"))
    result = dict(status="candidate_requires_visual_check", input=str(path.resolve()),
        voxel_index_ijk_0based=original_index.tolist(),
        world_coordinates=world.tolist(), world_unit=unit or "unknown",
        world_coordinates_mm=(world * scale_to_mm).tolist(),
        array_axis_directions=list(codes), coarse_index=point.tolist(), block_factor=factor,
        coarse_shape=list(coarse.shape), coarse_spacing_mm=spacing.tolist(),
        threshold=threshold, original_point_intensity=intensity, selected_candidate_id=selected["id"],
        selected_dilation_radius_mm=selected["first_radius"], steps_present=selected["steps_present"],
        depth_in_eroded_cavity_mm=float(interior_distance[tuple(point)]),
        candidate_count=len(tracks), warnings=warnings,
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        note="Geometric heuristic, no anatomical guarantee. Coarse cavity is eroded, not an endocast.")
    (out / "result.json").write_text(json.dumps(result, indent=2))
    for warning in warnings:
        print(f"NOTE: {warning}")
    print(f"\nSelected candidate: {selected['id']}")
    print(f"Original voxel (i, j, k), ZERO-BASED: {tuple(original_index.tolist())}")
    print(f"NIfTI world coordinates ({unit}): {world.tolist()}")
    print(f"Finished in {result['elapsed_seconds']} s. Open: {out / '06_FINAL_original_cross_sections.png'}")


if __name__ == "__main__":
    main()

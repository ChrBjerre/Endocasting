# Endocasting from Micro-CT scans

This repository contains Python scripts for extracting digital endocasts from micro-CT scans of animal skulls (stored as NIfTI volumes). It was developed as part of our bachelor thesis at the Technical University of Denmark (DTU).

**Authors:** Christian Bjerregaard and Aske Rove

**Last updated:** June 10, 2025

---

## What does it do?

These scripts process a NIfTI file (`.nii`) of a micro-CT scan to create a mesh-based digital endocast of an animal skull's endocranial cavity.

The main script, `script.py`, utilizes:

* Optional PCA-based resampling of volume data.
* Automatic selection of the highest-ranked center candidate, with manual
  selection available in PCA mode.
* Iterative graph-cut methods to refine and generate meshes representing the endocranial cavity.

---

## How to use

1. Ensure you have Python installed, along with required libraries (see scripts).
2. Replace the paths in `script.py` with your file paths:

```python
path = "your_file_path.nii"          # Input NIfTI file path
out_path = "output_directory/"        # Output path for meshes
```

3. Optionally, adjust parameters like `num_iters` and `step_sizes`.
4. Run the script:

```bash
python script.py
```

Set `use_pca_resampling` in `script.py`:

* `False` (default): skip both menus, automatically select the highest-ranked candidate
  from the bottleneck/enclosure method, and run endocasting on the original input.
  Coordinates and distances remain in original voxel units; anisotropic inputs
  are not made isotropic in this mode.
* `True`: use the PCA-axis GUI, resample, then manually select the center
  using the existing menu. Endocasting uses the PCA-resampled volume.

PCA resampling uses bounded chunks (262,144 output voxels by default). Set
`chunk_size` in the `PCA_resample_GUI` call to adjust that memory budget.

Mesh expansion remeshes by default (`remesh = True`) once halfway through each stage, rounding up:
`[10, 5, 3, 2]` remeshes after `[5, 3, 2, 1]` completed iterations. Set
`remesh = False` in `script.py` to disable it. Both mesh drivers support this flag.
The step runs after smoothing and before export/continued expansion; the existing
between-stage subdivision remains. It uses three passes of
[PyMeshLab isotropic remeshing](https://pymeshlab.readthedocs.io/en/latest/filter_list.html#meshing-isotropic-explicit-remeshing),
targeting the current average edge length with surface reprojection and a local
surface-deviation limit of 10% of that edge length. Vertex counts can change;
surface geometry is only approximately preserved. Install the added dependency
in the environment used to run the pipeline with `python -m pip install pymeshlab`.
Run the focused checks with `python -m unittest discover -s tests -p test_remeshing.py -v`.
The schedule check uses pipeline stubs; the geometry check requires both PyGEL
and PyMeshLab and reports a skip if either is unavailable.

---

## Inspect skull openings without mesh expansion

Run the new inspection script using the input path already configured in `script.py`:

```bash
python test_hole_detection.py
```

It reads that path without executing `script.py`, performs the same Otsu thresholding
and largest-bone-component cleanup, chooses the existing highest-ranked center,
and runs a two-marker watershed. It does not resample the scan or run mesh expansion.
To use a different scan or supply an approximate center in array coordinates `(i, j, k)`:

```bash
python test_hole_detection.py "path/to/skull.nii" --center 120 140 100
```

Try the small, known two-opening example first:

```bash
python test_hole_detection.py --synthetic
```

The main Matplotlib window devotes most of its space to a large, rotatable 3D skull.
Only the selected opening is highlighted in orange by default. Use **Previous / Next**
or the keyboard's left/right arrows to inspect candidates **largest first**.
**Focus opening** zooms to the candidate and samples nearby bone more finely;
navigation then follows each opening in that focused view. **Whole skull** restores
the overview. The mouse wheel also zooms. **Slices...** opens three full-resolution
slice views in a separate window, linked to the selected opening, with independent
slice sliders. **Inspect seed** shows the center used by watershed.

The size histogram shows all found candidates in grey and the retained candidates
in blue. Choose **Area** or **Diameter** (equivalent diameter), then move **Minimum**
to exclude candidates strictly below that size from inspection. Clicking the
histogram also places the cutoff. The orange line marks the selected candidate;
the pink line marks the cutoff. Wide size distributions use a logarithmic size axis.
Switching area/diameter converts the cutoff and preserves the retained candidates.
If everything is excluded, lower the cutoff to resume inspection. The histogram
keeps the full distribution visible, and reports/label volumes still contain all
detected candidates; filtering changes inspection only, not the detection results.

Other retained points and the cavity surface can be enabled explicitly. Skull
opacity remains adjustable. The 3D preview uses at most 8,000 skull points,
4,000 cavity points and 1,000 points for the selected opening. It does not build
full-resolution opening meshes or draw hundreds of text labels. Thin bone is kept
by pooling occupied blocks onto a display grid of roughly 96 voxels per axis.
Detection, measurements and slices remain at full resolution. For a lighter
preview, run `python test_hole_detection.py --max-points 2000`; `--surface-step`
can also set the display block stride explicitly. These options only affect drawing.
A GUI-capable Matplotlib backend is required for interaction.

To inspect an actual surface, run `python test_hole_detection.py --mesh`, or enable
**Local bone mesh**. The **View:** label confirms whether a mesh is actually displayed.
The default renderer is **VTK/OpenGL**, which draws the surface on the GPU instead
of projecting and sorting every triangle through Matplotlib. It opens a linked
GPU mesh window in a **separate process**; native graphics objects are never
embedded in or shared with the Matplotlib/Qt inspector. Keep the histogram
controls in the inspector and rotate in the GPU window. Left/right arrows work
in that window too. The inspector receives snapshots after selection changes and
after rotation; the live mesh stays in the GPU window.
Rotation reuses uploaded geometry and does not redraw the histogram or re-extract
the surface. `--mesh-renderer matplotlib` explicitly selects the slower software
renderer if needed. VTK is required for the default mesh viewer.
This focuses on the
selected candidate and extracts a shaded triangle mesh directly from the original
cleaned bone voxels using [marching cubes](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes)
at level 0.5 and step size 1. There is no voxel pooling or hole filling.
It automatically hides **Opening overlay**
and starts opaque so that the colored candidate patch does not cover the passage;
you can re-enable the overlay for location reference or adjust opacity yourself.
Previous/Next follows candidates with local meshes; **Whole skull** returns to
the lightweight point overview. Mesh mode stays a surface during rotation.

To keep interaction practical, the mesh crop can shrink around the candidate
while retaining the opening's bounding box plus a margin. The viewer draws at most
60,000 triangles by default. The UI reports both vertex and triangle counts;
the limit is a **triangle** count. Use `--mesh-max-faces 120000` for a larger
surface (typically around 60,000 vertices). If the minimum crop exceeds the limit,
[VTK topology-preserving decimation](https://vtk.org/doc/nightly/html/classvtkDecimatePro.html)
simplifies the display surface with topology preservation enabled, splitting
disabled and crop boundary vertices protected. This approximates surface geometry;
the original volume, opening labels and measurements are unchanged. If the limit still cannot be met
while preserving topology, the viewer explicitly says **Mesh unavailable** and
does not substitute a point cloud. The crop sides are artificial cuts, not skull
openings; inspect the candidate near the center, and use **Slices...** to confirm.
This is a mesh of the bone surface, not an expanded endocast. Anatomical passages
through bone can exist in a watertight mesh; they do not require open triangle edges.

The automatic-center warning about `Escape analysis uses stride ...` concerns
center ranking, whose coarse grid can hide narrow openings. The opening watershed
still uses the original grid. Check that the seed is inside the endocranium in
the slices (**Inspect seed**, then **Slices...**); `--center I J K` uses a supplied seed and skips ranking.
The number of candidate interface components is not a count of confirmed anatomical
openings: small fragments and segmentation/crop effects can produce extra candidates.

Reports (`openings.json`, `openings.csv`) and a snapshot (`overview.png`) are saved
under the ignored `results/hole-detection/` directory. Use `--output` to choose a
different directory, `--no-show` for reports/snapshot only, or `--save-volumes` to
also export `regions.nii.gz` and `openings.nii.gz` with the input affine. Dependencies
are NumPy, SciPy, scikit-image, Matplotlib and (for NIfTI input/export) nibabel;
the synthetic example works without nibabel. No PyGEL, PyMeshLab or PyVista is needed.

The reusable detector takes a binary bone array and never edits it:

```python
from hole_detection import detect_openings

result = detect_openings(bone, center=(120, 140, 100), spacing=(0.2, 0.2, 0.4))
for opening in result.openings:
    print(opening.id, opening.centroid_ijk, opening.area, opening.equivalent_diameter)
```

The detector computes the [empty-space Euclidean distance transform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html)
with voxel spacing, then [watershed](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed)
on its negative, seeded at the center and all empty boundary voxels. Flooding and
cavity/exterior contacts use 6-connectivity. Contact voxels on the cavity side are
grouped with 26-connectivity to keep diagonal parts of an interface sheet together.
`result.regions` uses `1` for the cavity, `2` for exterior and `0` for bone or
unseeded air; `result.opening_labels` assigns a positive ID to each opening.

Locations are reported in voxel coordinates, spacing-scaled coordinates relative
to voxel `(0,0,0)`, and (in the script reports) NIfTI world coordinates. Areas sum
the shared cavity/exterior voxel-face areas in physical units squared; equivalent
diameter is `2*sqrt(area/pi)`. NIfTI spatial units are used as declared in its header;
unknown units are labeled `units`, not assumed to be millimeters. Voxel axes must
be orthogonal; rotated/flipped axes are supported, sheared affines are rejected.
The viewer uses spacing-scaled voxel axes, not anatomical world orientation.

This basic method returns candidate interfaces, not guaranteed anatomical foramina.
Voxel stair steps affect area estimates, touching candidates can merge, and the
seed, crop, or very wide exits can move the watershed divide. No hole filling,
closing or extra detection algorithm is applied. A sealed cavity returns no
openings. The analysis runs on the full volume and needs several volume-sized
arrays; display coarsening does not reduce watershed memory usage.

Run the focused geometric and viewer checks with:

```bash
python -m unittest discover -s tests -p test_opening_detection.py -v
python -m unittest discover -s tests -p test_skull_mesh.py -v
python -m unittest discover -s tests -p test_gpu_mesh_viewer.py -v
```

The GPU test module runs process-bridge checks without opening a renderer.
Native OpenGL tests are opt-in via `ENDOCASTING_RUN_GL_TESTS=1`; they are skipped
by default so ordinary test runs do not open native graphics windows.

---

## Inspect watershed steps in 2D

For a simple view of the intermediate volumes, run:

```bash
python test_watershed_steps.py
```

It reads the scan path from `script.py`, applies the same cleanup and automatic
center ranking, then shows four linked slice panels: **cleaned bone**, **positive
distance transform**, **seed markers**, and **final watershed regions**. It reuses
the detector's watershed calculation, without computing opening components or
running mesh expansion. The distance shown is positive; watershed uses its negative.

Choose **i / j / k** and move **Slice** to scroll all panels together. **Seed slice**
returns to the center on the chosen axis. **DT display max** adjusts distance-map
contrast only. Blue marks the cavity region, orange the exterior, white the bone,
and black any unseeded air. Everything is Matplotlib 2D; no VTK or mesh renderer
is loaded. The volume is processed once; browsing does not rerun watershed.

```bash
python test_watershed_steps.py "path/to/skull.nii" --center 120 140 100
python test_watershed_steps.py --synthetic
python test_watershed_steps.py --synthetic --no-show
```

`--center` skips automatic seed selection. `--no-show` uses the non-interactive
backend and saves `results/watershed-steps.png`; `--output path.png` chooses another
PNG path. Intermediate fields stay in memory for inspection: a 500-cubed scan
retains about 1.4 GB across the bone, distance, marker and region arrays, in addition
to watershed's temporary working memory. NIfTI input uses the existing nibabel
dependency; the synthetic example does not need it.

```bash
python -m unittest discover -s tests -p test_watershed_inspection.py -v
```

## Files included

* `script.py`: Main execution script.
* `preprocessing.py`: Initial data preparation functions.
* `PCA_resample.py`: PCA-based resampling methods.
* `find_center_point.py`: Functions for determining suitable seed points.
* `cavity_ranking.py`: Boundary escape-clearance and directional-enclosure ranking.
* `center_point_debug.py`: Orthogonal slices, candidate scores, and a synthetic debug mode.
* `graph_cut_funcs.py`: Iterative mesh extraction and refinement functions.
* `hole_detection.py`: Reusable two-marker watershed opening detector.
* `hole_detection_viewer.py`: Interactive 3D and linked slice inspection.
* `skull_mesh.py`: Full-resolution bone-surface and bounded local mesh extraction.
* `gpu_mesh_viewer.py`: Separate-process OpenGL mesh viewer and inspection links.
* `test_hole_detection.py`: Standalone cleanup, center selection and opening inspection.
* `test_watershed_steps.py`: Simple linked 2D views of the watershed stages.

Center candidates now rank by cavity enclosure rather than convex-hull membership.
The highest-ranked candidate appears first in the existing selection GUI. See
[center-point detection](CENTER_POINT_DETECTION.md) for batch use, parameters,
coarse-grid limitations and validation. Try the headless synthetic example with
`python center_point_debug.py --synthetic --no-show`.

---
* `Visualization/`: Contains various scripts used for visualizations in our thesis.
---

Feel free to contact us for further details or questions regarding the usage of these scripts.



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

## Files included

* `script.py`: Main execution script.
* `preprocessing.py`: Initial data preparation functions.
* `PCA_resample.py`: PCA-based resampling methods.
* `find_center_point.py`: Functions for determining suitable seed points.
* `cavity_ranking.py`: Boundary escape-clearance and directional-enclosure ranking.
* `center_point_debug.py`: Orthogonal slices, candidate scores, and a synthetic debug mode.
* `graph_cut_funcs.py`: Iterative mesh extraction and refinement functions.

Center candidates now rank by cavity enclosure rather than convex-hull membership.
The highest-ranked candidate appears first in the existing selection GUI. See
[center-point detection](CENTER_POINT_DETECTION.md) for batch use, parameters,
coarse-grid limitations and validation. Try the headless synthetic example with
`python center_point_debug.py --synthetic --no-show`.

---
* `Visualization/`: Contains various scripts used for visualizations in our thesis.
---

Feel free to contact us for further details or questions regarding the usage of these scripts.



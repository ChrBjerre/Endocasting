# Endocasting from Micro-CT scans

This repository contains Python scripts for extracting digital endocasts from micro-CT scans of animal skulls (stored as NIfTI volumes). It was developed as part of our bachelor thesis at the Technical University of Denmark (DTU).

**Authors:** Christian Bjerregaard and Aske Rove

**Last updated:** June 10, 2025

---

## What does it do?

These scripts process a NIfTI file (`.nii`) of a micro-CT scan to create a mesh-based digital endocast of an animal skull's endocranial cavity.

The main script, `script.py`, utilizes:

* PCA-based resampling of volume data.
* Identification of local maxima to determine a suitable center point.
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

PCA resampling uses bounded chunks (262,144 output voxels by default). Set
`chunk_size` in the `PCA_resample_GUI` call to adjust that memory budget.

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



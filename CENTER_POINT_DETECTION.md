# Enclosure-based center-point ranking

The original path segmented bone with Otsu and the largest connected component,
computed a Euclidean distance transform (EDT) of non-bone, smoothed it, multiplied
by a convex-hull mask, and found peaks with a fixed 50-voxel exclusion radius.
A hull encloses space between skull projections as well as real cavities. It
does not distinguish endocranium from an open region with a broad connection to
external air. Large exterior peaks can therefore dominate, especially in an
elongated skull.

## What changed

Bone segmentation and the original, voxel-unit Euclidean distance transform are
unchanged. The default center finder no longer builds a convex hull. It proposes
up to 64 strong peaks using a smoothed field, then evaluates escape using the
**unsmoothed** field. Bone remains zero clearance; smoothing cannot create
traversable paths through it.

For every candidate, `D_escape` is the maximum, over paths to any of the six image
faces, of the minimum distance along that path. A sealed region has escape zero;
a point with an unobstructed path has escape equal to its peak clearance.

This is computed for all candidates together with boundary-seeded grayscale
reconstruction by dilation. Boundary distances are the seed, and the EDT is the
upper mask. Values propagate between neighbours and are clipped to local
clearance, which implements the max-min path recurrence. The compiled
[scikit-image reconstruction routine](https://scikit-image.org/docs/0.25.x/api/skimage.morphology.html#skimage.morphology.reconstruction)
sorts levels and performs propagation; it does not run a connected-component
analysis for each candidate/threshold. Default connectivity is 6 neighbours.

Optional rays use 96 approximately uniform Fibonacci-sphere directions, sampled
every half voxel on the original bone mask. Each stops at bone or the image
boundary. We do **not** stop at arbitrary boundary-connected air: that would also
stop rays inside a cavity connected through a foramen. The fraction hitting bone
is an approximate directional enclosure score, not an exact surface ray trace.

The transparent ranking is:

```text
bottleneck = D_peak - D_escape
score = bottleneck * ((1 - enclosure_weight) + enclosure_weight * enclosure)
```

Default weight is 0.5. With rays disabled, score equals bottleneck. Thus peak
clearance rewards larger cavities while escape penalizes wide exits; rays can
reduce the result by up to half at the default weight. No anatomical coordinates
or divergent ratios are used. Ties prefer larger peaks, then lexicographic array
indices for reproducibility. A zero-bottleneck list raises a warning; its first
entry is only a fallback, not positive evidence of an endocranium.

## Coordinates, approximation and memory

All coordinates use **original NIfTI array order `(i, j, k)`**, zero-based, without
transposing to `(z, y, x)` or canonicalizing orientation. All distances and ray
directions use original **voxel units**, as in the existing downstream mesh
sampling. On anisotropic input these are not physical millimetres. Prefer the
isotropic PCA-resampled input for the pipeline. The debug JSON separately reports
world coordinates using the full NIfTI affine.

To bound reconstruction's substantial native sorting/workspace allocations, the
default analysis grid has at most 160 cells along its longest axis. Larger EDTs
are minimum-pooled over integer blocks. This preserves zero-clearance thin bone
instead of dropping it by strided sampling, but **can close narrow openings**.
Escape clearance is then a conservative lower bound and bottleneck scores can be
inflated, including in exterior space. Results carry an explicit `stride`, issue
a warning and are labelled approximate in debug output. Check finer resolutions
when acceptance depends on a narrow foramen; coarse rankings are not guaranteed
to agree with exact rankings.

Each proposed coarse block is mapped back to the maximum original EDT voxel
within that block. Returned points are original integer indices in non-bone,
and their radii are the actual unsmoothed original EDT values. This differs from
the old smoothed-EDT radius and is intentional. No score is ever passed as a mesh
radius. A coarse block is only locally refined; a peak omitted from the shortlist
cannot be recovered by this refinement.

The full-resolution EDT is computed once. Coarsening and candidate refinement
precede reconstruction, and the full EDT is released if a separate coarse grid
exists. The full bone mask remains for rays. Debug records retain only analysis
distance and escape grids, plus candidate arrays. Reconstruction is global and
must not be naively tiled: escape paths can cross tile boundaries. **The original
loading, segmentation and EDT still have full-volume memory costs**; bounding
reconstruction does not make this an out-of-core pipeline. Very large CTs may
still need a separately validated preprocessing/downsampling strategy.

## Existing pipeline and batch use

Existing calls still work, returning `(local_max, local_max_values, points)`:

```python
local_max, radii, cloud = find_center_points(pca_path)
center_point, min_dist = select_local_max(cloud, local_max, radii)
```

The GUI starts on the highest-ranked candidate and still permits manual selection.
For an automatic batch choice without visualization:

```python
local_max, radii, cloud, scores = find_center_points(
    pca_path, return_scores=True, n_rays=96, max_grid_size=160)
center_point, min_dist = local_max[0], radii[0]
```

`get_candidate_points(bone, mesh_mask=None, sigma=3, ...)` also retains its
two-value return unless `return_scores=True`. A passed hull mask is ignored in
the new mode. The old hull method remains available with
`find_center_points(path, method="legacy")`, or
`get_candidate_points(bone, hull, method="legacy")`. Legacy hull dependencies
are imported only when used. `alpha_value` pertains only to legacy mode (where
the existing hull implementation still uses `alpha=0`). `subsampling_factor`
controls only the GUI point cloud in the new mode. `ENDOTEST.py` is unchanged.

## Visual debugging

Run in `endocasts` from the repository root:

```powershell
python center_point_debug.py --synthetic --no-show --output center-debug-synthetic
python center_point_debug.py "C:\data\skull_PCA_resampled.nii" --output center-debug-skull
```

Use `--no-show` for a headless run and `--candidate-rank 2` to inspect another
candidate's slices. Results are written as `slices.png`, `scores.png`,
`candidates.csv`, and `result.json`; reusing the same directory overwrites those
files. The slice figure shows original bone, distance and escape fields through
the inspected candidate in all three planes, with every candidate projected and
labelled by rank. Projected markers need not lie on the displayed slice. The
score table/CSV lists every candidate and all score components.

The cyan contour is the candidate component in `D > D_escape`: the cavity just
disconnected from the boundary. It illustrates the escape threshold, not an
explicit optimal path or a final endocranium segmentation. At stride > 1, field
pixels represent pooled blocks, whereas the bone panels and markers use original
voxel coordinates. Diagnostics live outside the ranking module and are not
created during normal batch/pipeline calls.

## Parameters to tune and expected failures

| Parameter | Default | Effect |
| --- | ---: | --- |
| `sigma` | 3 | Peak-proposal smoothing in original voxel units; does not smooth escape geometry. |
| `min_distance` | 10 | Separation of shortlisted peaks, in original voxels. |
| `max_candidates` | 64 | Cap before scoring; increase if many exterior/sinus peaks crowd out the cavity. |
| `max_grid_size` | 160 | Increase to preserve smaller openings; `None` in Python or `--max-grid-size 0` in CLI for exact analysis. |
| `connectivity` | 1 | 6 neighbours; 2/3 gives 18/26 and allows more diagonal escape paths. |
| `n_rays` / CLI `--rays` | 96 | Set 0 to disable; roughly 50–200 is suitable for a secondary heuristic. |
| `enclosure_weight` | 0.5 | Ray-score influence; 0 gives bottleneck-only ordering. |

Large sinuses or other enclosed cavities can outrank the endocranium; geometry
alone does not establish anatomy. A damaged skull or a crop cutting through the
braincase can create a genuinely wide escape and penalize the true target. The
unchanged largest-bone-component step can discard detached bones. Coarse pooling
and 6-connectivity can artificially seal narrow/diagonal gaps; ray sampling can
miss tiny holes and is not exactly rotation-invariant. Cropping away useful
exterior air also changes the quantity being measured. Review the scores and
slices rather than treating the top rank as anatomical ground truth.

## Validation performed

```powershell
python -m unittest discover -s tests -v
```

Ten tests passed in `endocasts`. They include an independent exhaustive
threshold/flood-fill oracle for exact escape values with 6/18/26 connectivity;
two competing exits and a sealed cavity; a smaller narrow-neck cavity outranking
a larger exterior pocket; shell/exterior ray scores; input immutability; coarse
mapping and true original radii; the coarse lower-bound property; an axis
permutation; cropped-open air and boundary seeds; empty-input handling and the
unchanged pipeline tuple without hull invocation.

The 65-cubed synthetic shell selected `(32,32,32)`: peak 14.9666, escape 3.0,
bottleneck 11.9666. Exterior peaks of 35.2278 received bottleneck zero. Ranking
took about 0.14 seconds locally (one synthetic run, not a full-scan benchmark).
An additional fresh-process 129-cubed shell run selected `(64,64,64)` from nine
candidates at stride 1 in 1.44 seconds, with a Windows process peak working set
of 278.8 MiB (including imports and fixture construction). These are synthetic
observations, not estimates of wolf-scan runtime or RAM.
The figures were generated headlessly and visually inspected. A saved synthetic
NIfTI with translated, rotated and anisotropic affine also ran through the real
Otsu/largest-component loader and debug CLI, retaining the voxel center and
correctly mapping its world coordinate. No real CT scan was processed. Wolf
skulls, different crops and finer-grid stability still require real-data checks.

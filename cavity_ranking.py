"""Geometric cavity ranking in NumPy/NIfTI array-index coordinates (i, j, k).

Distances are in original voxel units, as in the mesh sampling pipeline. They
are not millimetres; no array axes are swapped or affine transforms applied.
"""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import reconstruction


@dataclass
class CavityCandidates:
    points: np.ndarray
    peak: np.ndarray
    escape: np.ndarray
    bottleneck: np.ndarray
    enclosure: np.ndarray
    score: np.ndarray
    grid_points: np.ndarray
    stride: int
    analysis_distance: np.ndarray
    escape_field: np.ndarray
    connectivity: int


def boundary_seed(field):
    """Seed *all six faces*, including cavities cut open by the image crop."""
    seed = np.zeros_like(field)
    for axis in range(3):
        for end in (0, -1):
            face = [slice(None)] * 3
            face[axis] = end
            seed[tuple(face)] = field[tuple(face)]
    return seed


def escape_clearance(distance, connectivity=1):
    """Maximum, over paths to a boundary, of the minimum clearance on the path.

    Grayscale reconstruction propagates boundary values but clips each step to
    the local distance. At convergence it solves the max-min path problem for
    all voxels together, without repeated threshold/component labelling.
    Zero-distance bone blocks every positive-clearance path. Sealed air has
    escape clearance zero. Connectivity 1/2/3 means 6/18/26 neighbours.
    """
    distance = np.asarray(distance)
    if distance.ndim != 3 or not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError("distance must be a finite nonnegative 3D field")
    if connectivity not in (1, 2, 3):
        raise ValueError("connectivity must be 1, 2 or 3")
    return reconstruction(boundary_seed(distance), distance, method="dilation",
                          footprint=ndi.generate_binary_structure(3, connectivity))


def directional_enclosure(bone, points, n_rays=96, step=0.5):
    """Fraction of Fibonacci-sphere rays hitting bone before leaving the image.

    Rays are in voxel-coordinate space, consistent with the EDT. Half-voxel
    sampling approximates intersection with voxel cells (not surface meshes).
    Boundary-connected air is NOT an early stopping criterion: it includes the
    true cavity whenever a foramen is open. Only a hit or image exit stops a ray.
    """
    if not isinstance(n_rays, (int, np.integer)) or n_rays < 1:
        raise ValueError("n_rays must be a positive integer")
    if not np.isfinite(step) or step <= 0 or step > 0.5:
        raise ValueError("ray step must be in (0, 0.5]")
    k = np.arange(n_rays)
    z = 1 - 2 * (k + 0.5) / n_rays
    angle = k * np.pi * (3 - np.sqrt(5))
    radius = np.sqrt(1 - z * z)
    directions = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    limit = np.asarray(bone.shape)
    scores = []
    for point in points:
        active = np.ones(n_rays, dtype=bool)
        hits = 0
        # The image diagonal bounds any ray starting inside the volume.
        for t in np.arange(step, np.linalg.norm(limit) + step, step):
            ids = np.flatnonzero(active)
            if not len(ids):
                break
            indices = np.floor(point + t * directions[ids] + 0.5).astype(int)
            inside = np.all((indices >= 0) & (indices < limit), axis=1)
            active[ids[~inside]] = False
            ids, indices = ids[inside], indices[inside]
            hit = bone[tuple(indices.T)]
            hits += np.count_nonzero(hit)
            active[ids[hit]] = False
        scores.append(hits / n_rays)
    return np.asarray(scores)


def rank_cavity_candidates(bone, *, sigma=3, min_distance=10, max_candidates=64,
                           max_grid_size=160, connectivity=1, n_rays=96,
                           enclosure_weight=0.5):
    """Rank a bounded peak shortlist by (D_peak - D_escape) * ray multiplier.

    score = bottleneck * ((1 - weight) + weight * enclosure). With n_rays=0,
    the score is simply bottleneck. This rewards a large cavity *and* a narrow
    exit, without a divergent ratio at sealed cavities. Ties prefer larger peaks.

    Full-resolution bone segmentation and EDT are retained. If needed, minimum
    pooling bounds the reconstruction grid; every coarse cell uses its *lowest*
    clearance. This does not erase thin bone but can close small openings and
    underestimate escape clearance. max_grid_size=None requests exact analysis.
    Coarse peak blocks are refined to the largest original EDT value in the
    block, so returned indices and radii always refer to the original image.
    """
    bone = np.asarray(bone, dtype=bool)
    if bone.ndim != 3 or min(bone.shape) < 3:
        raise ValueError("bone must be a 3D mask with at least 3 voxels per axis")
    if not bone.any() or bone.all():
        raise ValueError("Center detection requires both bone and non-bone voxels")
    if not np.isfinite(sigma) or sigma < 0 or not np.isfinite(min_distance) or min_distance < 1:
        raise ValueError("sigma must be nonnegative and min_distance must be >= 1")
    if not isinstance(max_candidates, (int, np.integer)) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    if max_grid_size is not None and (not isinstance(max_grid_size, (int, np.integer)) or max_grid_size < 3):
        raise ValueError("max_grid_size must be None or an integer >= 3")
    if not isinstance(n_rays, (int, np.integer)) or n_rays < 0:
        raise ValueError("n_rays must be a nonnegative integer")
    if not 0 <= enclosure_weight <= 1:
        raise ValueError("enclosure_weight must be in [0, 1]")

    distance = ndi.distance_transform_edt(~bone)
    stride = 1 if max_grid_size is None else max(1, int(np.ceil(max(bone.shape) / max_grid_size)))
    analysis = distance
    if stride > 1:
        warnings.warn(f"Escape analysis uses stride {stride}; narrow openings may close. "
                      "Compare with a finer grid before accepting the automatic seed.", stacklevel=2)
        for axis in range(3):
            analysis = np.minimum.reduceat(analysis, np.arange(0, analysis.shape[axis], stride), axis=axis)

    # Smooth only for proposing peaks; bone cannot become traversable in the
    # unsmoothed clearance field used to evaluate their escape routes.
    smooth = ndi.gaussian_filter(analysis, sigma=sigma / stride)
    peaks = peak_local_max(smooth, min_distance=max(1, int(np.ceil(min_distance / stride))),
                           threshold_abs=0.1, exclude_border=1,
                           labels=(analysis > 0).astype(np.uint8), num_peaks=max_candidates)
    del smooth
    if not len(peaks):
        raise ValueError("No candidate peaks found; reduce sigma/min_distance or use a finer analysis grid")
    points = []
    for peak in peaks:
        start = peak * stride
        region = tuple(slice(int(a), min(int(a + stride), n)) for a, n in zip(start, bone.shape))
        block = distance[region]
        points.append(start + np.asarray(np.unravel_index(np.argmax(block), block.shape)))
    points = np.asarray(points, dtype=int)
    peak_values = distance[tuple(points.T)]
    if stride > 1:
        del distance  # Retain only the bounded analysis grid beyond peak refinement.
    escape_field = escape_clearance(analysis, connectivity)
    escape = escape_field[tuple(peaks.T)]
    bottleneck = np.maximum(peak_values - escape, 0)
    enclosure = directional_enclosure(bone, points, n_rays) if n_rays else np.full(len(points), np.nan)
    score = bottleneck * ((1 - enclosure_weight) + enclosure_weight * enclosure) if n_rays else bottleneck.copy()
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0], -peak_values, -score))
    if np.max(bottleneck) == 0:
        warnings.warn("No positive enclosure bottleneck found; the first candidate is only a fallback.", stacklevel=2)
    return CavityCandidates(points[order], peak_values[order], escape[order], bottleneck[order],
                            enclosure[order], score[order], peaks[order], stride,
                            analysis, escape_field, connectivity)


def cavity_above_escape(result, index=0):
    """One diagnostic component just ABOVE escape, not an endocast segmentation."""
    allowed = result.analysis_distance > result.escape[index]
    seed = np.zeros_like(allowed)
    point = tuple(result.grid_points[index])
    if allowed[point]:
        seed[point] = True
    return ndi.binary_propagation(seed, structure=ndi.generate_binary_structure(3, result.connectivity),
                                  mask=allowed)

"""Two-marker distance-transform watershed for openings in a binary skull.

Array coordinates are (i, j, k), with no axis swaps. Spacing is in that same
order. This module needs only NumPy, SciPy and scikit-image; it performs no I/O
and never edits the input volume.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed


@dataclass(frozen=True)
class Opening:
    id: int
    centroid_ijk: tuple[float, float, float]
    centroid_physical: tuple[float, float, float]
    voxel_count: int
    area: float
    equivalent_diameter: float


@dataclass
class HoleDetection:
    regions: np.ndarray
    opening_labels: np.ndarray
    openings: list[Opening]
    center_ijk: tuple[int, int, int]
    spacing: tuple[float, float, float]


@dataclass
class WatershedStages:
    distance: np.ndarray
    markers: np.ndarray
    regions: np.ndarray
    center_ijk: tuple[int, int, int]
    spacing: tuple[float, float, float]


def _neighbor_slices(axis):
    """Pairs of face-sharing voxel views, without wrapping at image edges."""
    lower, upper = [slice(None)] * 3, [slice(None)] * 3
    lower[axis], upper[axis] = slice(None, -1), slice(1, None)
    return tuple(lower), tuple(upper)


def compute_watershed_stages(bone, center, spacing=(1.0, 1.0, 1.0)):
    """Return the positive EDT, seed markers and final 6-connected watershed.

    This is the same calculation used by detect_openings. Coordinates are
    array indices (i,j,k); spacing is along those axes. Center rounding is
    nearest voxel, half upwards. The input is never edited. Keeping this
    record retains the intermediate volume-sized buffers for inspection.
    """
    bone = np.asarray(bone)
    if bone.ndim != 3 or min(bone.shape) < 3:
        raise ValueError("bone must be a 3D binary volume with at least 3 voxels per axis")
    if bone.dtype != bool and not np.all((bone == 0) | (bone == 1)):
        raise ValueError("bone must contain only 0 (empty) and 1 (bone)")
    if not bone.any():
        raise ValueError("bone must contain some bone")
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing must contain three finite positive voxel sizes")
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center must contain three finite array coordinates (i, j, k)")
    if np.any(center < 0) or np.any(center > np.asarray(bone.shape) - 1):
        raise ValueError("center is outside the volume")
    seed = np.floor(center + 0.5).astype(int)
    if np.any(seed == 0) or np.any(seed == np.asarray(bone.shape) - 1):
        raise ValueError("center must be off the volume boundary, away from exterior markers")
    if bone[tuple(seed)]:
        raise ValueError("center is in bone; supply an empty endocranium voxel")

    empty = bone == 0
    markers = np.zeros(bone.shape, dtype=np.int32)
    for axis in range(3):
        for end in (0, -1):
            face = [slice(None)] * 3
            face[axis] = end
            face = tuple(face)
            markers[face] = 2 * empty[face]
    if not np.any(markers):
        raise ValueError("No empty boundary voxels are available for the exterior marker")
    markers[tuple(seed)] = 1

    distance = ndi.distance_transform_edt(empty, sampling=spacing)
    distance *= -1  # Reuse the EDT buffer instead of allocating another 3D field.
    regions = watershed(distance, markers, connectivity=1, mask=empty)
    distance *= -1  # Display positive clearance; watershed used its negative.
    return WatershedStages(distance, markers, regions,
                           tuple(int(v) for v in seed), tuple(float(v) for v in spacing))


def detect_openings(bone, center, spacing=(1.0, 1.0, 1.0)):
    """Locate candidate cavity/exterior interfaces using a seeded watershed.

    Parameters
    ----------
    bone : 3D array of bool or 0/1
        One denotes bone; zero denotes empty space. Not modified.
    center : sequence of three numbers
        Approximate endocranium seed in array indices (i, j, k). Rounded to
        the nearest voxel (half upwards); must be inside empty space and off
        the image boundary. An invalid seed is rejected rather than moved.
    spacing : sequence of three positive numbers
        Physical voxel sizes along (i, j, k), assuming orthogonal voxel axes.

    Returns
    -------
    HoleDetection
        regions: 1 = seed's cavity, 2 = exterior, 0 = bone or unseeded air.
        opening_labels: positive component IDs on the cavity side of each
        interface, zero elsewhere. Face-sharing (6-connected) paths are used
        for flooding and interfaces; interface components use 26-connectivity
        so diagonally touching parts of a sheet stay together.

        Each opening has its mean voxel location, its location multiplied by
        spacing (relative to voxel (0,0,0), NOT a NIfTI world coordinate), and
        approximate area in spacing units squared. Area sums cavity/exterior
        voxel-face areas, counting each face once. Equivalent diameter is
        2*sqrt(area/pi). These are voxelized interface measurements, not an
        exact anatomical aperture area or a maximum inscribed diameter.

    A sealed cavity returns no openings. Different openings that touch under
    26-connectivity merge. Seed placement, crop, segmentation and wide exits
    can shift the watershed interface; no hole filling or filtering is applied.
    """
    stages = compute_watershed_stages(bone, center, spacing)
    regions, seed, spacing = stages.regions, stages.center_ijk, stages.spacing
    del stages  # Normal detection does not retain the distance/marker buffers.

    # Keep only contacts between the TWO seeded regions: bone boundaries and
    # isolated, unseeded pockets of air must not become candidate openings.
    interface = np.zeros(regions.shape, dtype=bool)
    for axis in range(3):
        lower, upper = _neighbor_slices(axis)
        interface[lower] |= (regions[lower] == 1) & (regions[upper] == 2)
        interface[upper] |= (regions[upper] == 1) & (regions[lower] == 2)
    opening_labels, count = ndi.label(interface, structure=np.ones((3, 3, 3), dtype=bool))
    del interface
    result = HoleDetection(regions, opening_labels, [],
                           tuple(int(v) for v in seed), tuple(float(v) for v in spacing))
    if count == 0:
        return result

    # Measure each shared face exactly once; a voxel may own several faces.
    areas = np.zeros(count + 1, dtype=float)
    for axis in range(3):
        lower, upper = _neighbor_slices(axis)
        face_area = np.prod(spacing) / spacing[axis]
        for inside, outside in ((lower, upper), (upper, lower)):
            contact = (regions[inside] == 1) & (regions[outside] == 2)
            ids = opening_labels[inside][contact]
            areas += np.bincount(ids, minlength=count + 1) * face_area
    del contact, ids

    # Local component boxes avoid allocating a full-volume coordinate grid.
    for index, box in enumerate(ndi.find_objects(opening_labels), start=1):
        coordinates = np.argwhere(opening_labels[box] == index)
        centroid = coordinates.mean(axis=0) + np.array([s.start for s in box])
        result.openings.append(Opening(
            id=index,
            centroid_ijk=tuple(float(v) for v in centroid),
            centroid_physical=tuple(float(v) for v in centroid * spacing),
            voxel_count=len(coordinates),
            area=float(areas[index]),
            equivalent_diameter=float(2 * np.sqrt(areas[index] / np.pi)),
        ))
    return result

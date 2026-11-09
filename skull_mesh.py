"""Surface extraction for visual inspection of the original binary bone mask."""

from dataclasses import dataclass

import numpy as np
from skimage.measure import marching_cubes


@dataclass
class BoneMesh:
    vertices: np.ndarray
    faces: np.ndarray
    bounds: np.ndarray
    within_budget: bool = True
    source_faces: int = 0


def bone_surface_mesh(bone, spacing=(1, 1, 1), origin=(0, 0, 0)):
    """Extract the 0.5 isosurface at full voxel resolution, without editing bone.

    Coordinates are spacing-scaled array indices plus origin (in voxel indices).
    There is no pooling, smoothing, filling, decimation or face subsampling.
    Surfaces that reach an array/crop face are left open at that face: those
    edges are crop cuts, not evidence of an anatomical opening.
    """
    bone = np.asarray(bone)
    spacing, origin = np.asarray(spacing, dtype=float), np.asarray(origin, dtype=float)
    if bone.ndim != 3 or min(bone.shape) < 2:
        raise ValueError("bone must be a 3D binary array with at least two voxels per axis")
    if not np.all((bone == 0) | (bone == 1)):
        raise ValueError("bone must contain only 0 and 1")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing must contain three finite positive voxel sizes")
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin must contain three finite voxel coordinates")
    bounds = np.stack((origin * spacing, (origin + np.asarray(bone.shape) - 1) * spacing))
    if not bone.any() or bone.all():
        return BoneMesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int32), bounds)
    vertices, faces, _, _ = marching_cubes(bone, level=0.5, spacing=spacing,
                                           step_size=1, allow_degenerate=False)
    vertices = vertices + origin * spacing
    return BoneMesh(vertices, faces, bounds)


def local_bone_mesh(bone, center, spacing, *, half_width, min_half_width, max_faces=60000):
    """Shrink the spatial crop if needed; never remove triangles to meet a budget.

    Widths are physical half-widths. The caller supplies a minimum that contains
    the selected opening plus a margin. If that crop still exceeds max_faces,
    within_budget=False lets the caller simplify the surface or report the limit.
    """
    if not isinstance(max_faces, (int, np.integer)) or max_faces < 1:
        raise ValueError("max_faces must be a positive integer")
    spacing, center = np.asarray(spacing), np.asarray(center)
    width = max(float(half_width), float(min_half_width))
    while True:
        lower = np.maximum(0, np.floor(center - width / spacing)).astype(int)
        upper = np.minimum(bone.shape, np.ceil(center + width / spacing + 1)).astype(int)
        roi = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
        mesh = bone_surface_mesh(bone[roi], spacing, origin=lower)
        mesh.within_budget = len(mesh.faces) <= max_faces
        if mesh.within_budget or width <= min_half_width:
            return mesh
        width = max(float(min_half_width), width * 0.7)


def mesh_to_polydata(mesh):
    """Copy triangle geometry into VTK; shared by display and display reduction."""
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(mesh.vertices, dtype=np.float64), deep=True))
    triangles = np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)).astype(np.int64)
    cells = vtkCellArray()
    cells.SetCells(len(mesh.faces), numpy_to_vtkIdTypeArray(triangles.ravel(), deep=True))
    surface = vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(cells)
    return surface


def simplify_bone_mesh(mesh, max_faces=60000):
    """Reduce display triangles using VTK with topology and crop boundaries fixed.

    This is a display approximation of the extracted surface. Source vertices,
    faces and voxel data are untouched. Preserving topology can prevent reaching
    the requested target; within_budget reports that rather than deleting faces.
    VTK is imported only if reduction is needed.
    """
    if not isinstance(max_faces, (int, np.integer)) or max_faces < 1:
        raise ValueError("max_faces must be positive")
    if len(mesh.faces) <= max_faces:
        return BoneMesh(mesh.vertices, mesh.faces, mesh.bounds, True, mesh.source_faces)
    from vtkmodules.vtkFiltersCore import vtkDecimatePro
    from vtkmodules.util.numpy_support import vtk_to_numpy

    surface = mesh_to_polydata(mesh)
    decimate = vtkDecimatePro()
    decimate.SetInputData(surface)
    decimate.SetTargetReduction(1 - max_faces / len(mesh.faces))
    decimate.PreserveTopologyOn()
    decimate.SplittingOff()
    decimate.BoundaryVertexDeletionOff()
    decimate.Update()
    output = decimate.GetOutput()
    vertices = vtk_to_numpy(output.GetPoints().GetData()).copy()
    faces = vtk_to_numpy(output.GetPolys().GetData()).reshape(-1, 4)[:, 1:].copy()
    return BoneMesh(vertices, faces, mesh.bounds.copy(), len(faces) <= max_faces,
                    source_faces=mesh.source_faces or len(mesh.faces))

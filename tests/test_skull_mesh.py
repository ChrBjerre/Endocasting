"""Check that bone meshing preserves synthetic openings and original voxel data."""

import unittest
import importlib.util

import numpy as np

from skull_mesh import bone_surface_mesh, local_bone_mesh, simplify_bone_mesh


def shell(openings):
    bone = np.zeros((19, 19, 19), dtype=bool)
    bone[4:15, 4:15, 4:15] = True
    bone[6:13, 6:13, 6:13] = False
    if openings:
        bone[10:, 8:11, 8:11] = False
    if openings == 2:
        bone[8:11, :10, 8:11] = False
    return bone


class SkullMeshTests(unittest.TestCase):
    def test_full_resolution_mesh_preserves_opening_topology(self):
        # A sealed shell has two closed surfaces (Euler 4). One passage joins
        # them (Euler 2); a second passage makes a handle (Euler 0). All these
        # bone surfaces are watertight: foramina need not be mesh boundary edges.
        for opening_count, expected_euler in ((0, 4), (1, 2), (2, 0)):
            with self.subTest(openings=opening_count):
                bone = shell(opening_count)
                original = bone.copy()
                bone.flags.writeable = False
                mesh = bone_surface_mesh(bone)
                faces = mesh.faces
                edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                                 faces[:, [2, 0]])), axis=1)
                edges, counts = np.unique(edges, axis=0, return_counts=True)
                self.assertTrue(np.all(counts == 2))
                self.assertEqual(len(mesh.vertices) - len(edges) + len(faces), expected_euler)
                self.assertTrue(np.isfinite(mesh.vertices).all())
                np.testing.assert_array_equal(bone, original)

    def test_spacing_and_crop_origin_transform_vertices(self):
        bone = shell(1)
        baseline = bone_surface_mesh(bone)
        transformed = bone_surface_mesh(bone, spacing=(2, 3, 4), origin=(10, 20, 30))
        np.testing.assert_allclose(transformed.vertices, (baseline.vertices + [10, 20, 30]) * [2, 3, 4])
        np.testing.assert_array_equal(transformed.faces, baseline.faces)

    def test_local_preview_shrinks_crop_to_meet_budget_without_changing_voxels(self):
        bone = np.random.default_rng(12).random((43, 43, 43)) > 0.7
        original = bone.copy()
        mesh = local_bone_mesh(bone, (21, 21, 21), (1, 1, 1),
                               half_width=15, min_half_width=3, max_faces=1000)
        self.assertTrue(mesh.within_budget)
        self.assertLessEqual(len(mesh.faces), 1000)
        self.assertLess(np.max(mesh.bounds[1] - mesh.bounds[0]), 30)
        np.testing.assert_array_equal(bone, original)

    def test_impossible_budget_reports_limit_instead_of_discarding_faces(self):
        bone = shell(1)
        mesh = local_bone_mesh(bone, (14, 9, 9), (1, 1, 1),
                               half_width=8, min_half_width=4, max_faces=10)
        self.assertFalse(mesh.within_budget)
        self.assertGreater(len(mesh.faces), 10)
        lower, upper = mesh.bounds[0].astype(int), mesh.bounds[1].astype(int) + 1
        roi = tuple(slice(a, b) for a, b in zip(lower, upper))
        reference = bone_surface_mesh(bone[roi], origin=lower)
        np.testing.assert_array_equal(mesh.faces, reference.faces)
        np.testing.assert_allclose(mesh.vertices, reference.vertices)

    def test_constant_crops_return_empty_mesh_and_invalid_spacing_is_rejected(self):
        for occupied in (False, True):
            mesh = bone_surface_mesh(np.full((4, 4, 4), occupied))
            self.assertEqual(mesh.vertices.shape, (0, 3))
            self.assertEqual(mesh.faces.shape, (0, 3))
        with self.assertRaisesRegex(ValueError, "spacing"):
            bone_surface_mesh(shell(1), spacing=(1, 0, 1))

    @unittest.skipUnless(importlib.util.find_spec("vtkmodules"), "Display reduction requires VTK")
    def test_display_reduction_preserves_passages_and_source_arrays(self):
        for opening_count, expected_euler in ((0, 4), (1, 2), (2, 0)):
            with self.subTest(openings=opening_count):
                original = bone_surface_mesh(shell(opening_count))
                vertices, faces = original.vertices.copy(), original.faces.copy()
                reduced = simplify_bone_mesh(original, max_faces=300)
                self.assertTrue(reduced.within_budget)
                self.assertLessEqual(len(reduced.faces), 300)
                self.assertEqual(reduced.source_faces, len(original.faces))
                triangles = reduced.faces
                edges = np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                                 triangles[:, [2, 0]])), axis=1)
                edges, counts = np.unique(edges, axis=0, return_counts=True)
                self.assertTrue(np.all(counts == 2))
                self.assertEqual(len(reduced.vertices) - len(edges) + len(triangles), expected_euler)
                np.testing.assert_array_equal(original.vertices, vertices)
                np.testing.assert_array_equal(original.faces, faces)

    @unittest.skipUnless(importlib.util.find_spec("vtkmodules"), "Display reduction requires VTK")
    def test_display_reduction_preserves_crop_boundary_edges(self):
        bone = np.zeros((23, 23, 23), dtype=bool)
        bone[:11] = True  # Open planar surface terminating at crop faces.
        original = bone_surface_mesh(bone)
        reduced = simplify_bone_mesh(original, max_faces=300)

        def boundary_segments(mesh):
            faces = mesh.faces
            edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
            edges, counts = np.unique(edges, axis=0, return_counts=True)
            return sorted(tuple(sorted(tuple(point) for point in mesh.vertices[edge]))
                          for edge in edges[counts == 1])

        self.assertTrue(reduced.within_budget)
        self.assertEqual(boundary_segments(original), boundary_segments(reduced))


if __name__ == "__main__":
    unittest.main()

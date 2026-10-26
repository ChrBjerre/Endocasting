"""Check midpoint wiring without CT data, and native remeshing when installed."""

import ast
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


def functions(**dependencies):
    path = Path(__file__).resolve().parents[1] / "graph_cut_funcs.py"
    tree = ast.parse(path.read_bytes())
    names = {"get_faces", "remesh_uniform", "iterative_mesh", "iterative_mesh_every_iter"}
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(np=np, **dependencies)
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class RemeshingTests(unittest.TestCase):
    def test_midpoints_and_rebuilt_sampling_in_both_drivers(self):
        for name in ("iterative_mesh", "iterative_mesh_every_iter"):
            for enabled in (True, False):
                with self.subTest(driver=name, remesh=enabled):
                    state = dict(mesh=object(), updates=-1, neighbours=None, remesh_at=[])

                    def neighbours(mesh):
                        self.assertIs(mesh, state["mesh"])
                        state["neighbours"] = mesh

                    def samples(mesh, *args, **kwargs):
                        self.assertIs(mesh, state["mesh"])
                        self.assertIs(mesh, state["neighbours"])

                    def shift(*args):
                        state["updates"] += 1

                    def remesh(mesh):
                        self.assertIs(mesh, state["mesh"])
                        state["remesh_at"].append(state["updates"])
                        state["mesh"] = object()  # Force subsequent calls to use the new mesh.
                        return state["mesh"]

                    no_op = lambda *a, **kw: None
                    namespace = functions(
                        nib=SimpleNamespace(load=lambda p: SimpleNamespace(
                            shape=(20, 20, 20), header=SimpleNamespace(get_zooms=lambda: (1, 1, 1)))),
                        hmesh=SimpleNamespace(taubin_smooth=no_op, loop_subdivide=no_op),
                        prepare_nifti_interpolator=no_op, create_icosphere=lambda *a, **kw: state["mesh"],
                        create_neighbor_list=neighbours, create_sample_grid_vec_new=samples,
                        sample_in_nifti=no_op, graph_cut_segmentation=lambda *a, **kw: (None, 0),
                        first_indices_mask=no_op, get_mask_coordinates=lambda *a: (None, None),
                        shift_mesh_vertices=shift, create_mesh_ply=no_op, create_binary_nifti_mask=no_op)
                    namespace["remesh_uniform"] = remesh
                    with contextlib.redirect_stdout(io.StringIO()):
                        namespace[name](np.zeros(3), 3, "unused", "unused",
                                        num_iters=[10, 5, 3, 2], step_sizes=[1, 1, .5, .5], remesh=enabled)
                    self.assertEqual(state["remesh_at"], [5, 13, 17, 19] if enabled else [])
                    self.assertEqual(state["updates"], 20)

    @unittest.skipUnless(importlib.util.find_spec("pygel3d") and importlib.util.find_spec("pymeshlab"),
                         "Native geometry test requires pygel3d and pymeshlab")
    def test_native_remeshing_improves_edge_uniformity(self):
        from pygel3d import hmesh
        namespace = functions(hmesh=hmesh)
        vertices = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0],
                             [-1, 0, 0], [0, -1, 0]], dtype=float)
        faces = np.array([[0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 2],
                          [1, 3, 2], [1, 4, 3], [1, 5, 4], [1, 2, 5]], dtype=np.int32)
        mesh = hmesh.Manifold.from_triangles(vertices, faces)
        for _ in range(2):
            hmesh.loop_subdivide(mesh)
        mesh.cleanup()
        positions = mesh.positions()
        positions /= np.linalg.norm(positions, axis=1, keepdims=True)
        # Redistribute sphere vertices towards one pole to create uneven triangles.
        z = positions[:, 2].copy()
        new_z = (z + 0.6) / (1 + 0.6 * z)
        interior = np.abs(z) < 1
        positions[interior, :2] *= np.sqrt((1 - new_z[interior]**2) / (1 - z[interior]**2))[:, None]
        positions[:, 2] = new_z

        def edge_lengths(m):
            triangles = namespace["get_faces"](m)
            edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                                            triangles[:, [2, 0]]]), axis=1)
            edges, counts = np.unique(edges, axis=0, return_counts=True)
            np.testing.assert_array_equal(counts, 2)  # Closed triangular surface.
            return np.linalg.norm(m.positions()[edges[:, 0]] - m.positions()[edges[:, 1]], axis=1)

        before = edge_lengths(mesh)
        result = namespace["remesh_uniform"](mesh)
        after = edge_lengths(result)
        self.assertLess(after.std() / after.mean(), before.std() / before.mean())
        self.assertTrue(np.isfinite(result.positions()).all())
        np.testing.assert_array_equal(list(result.vertices()), np.arange(len(result.positions())))
        print(f"Synthetic edge-length CV: {before.std()/before.mean():.3f} -> {after.std()/after.mean():.3f}")


if __name__ == "__main__":
    unittest.main()

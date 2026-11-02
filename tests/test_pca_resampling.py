"""Regression checks for PCA interpolation without launching the GUI."""

import ast
import contextlib
import gc
import io
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates


def load_resampler():
    source = Path(__file__).resolve().parents[1] / "PCA_resample.py"
    tree = ast.parse(source.read_bytes())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"rotated_points", "resample_nifti"}]
    namespace = dict(np=np, nib=nib, gc=gc,
                     RegularGridInterpolator=RegularGridInterpolator)
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["resample_nifti"]


class PCAResamplingTests(unittest.TestCase):
    def setUp(self):
        self.resample = load_resampler()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.spacing = np.array([0.6, 0.8, 1.1])
        self.shape = (17, 19, 21)
        i, j, k = np.indices(self.shape)
        self.data = (-1024 + 7*i + 11*j + 13*k).astype(np.int16)
        self.data[5:12, 6:14, 7:15] += 2200
        corners = np.array(np.meshgrid(*[(0, n-1) for n in self.shape],
                                      indexing="ij")).reshape(3, -1).T
        self.points = corners * self.spacing
        angle = 0.37
        c, s = np.cos(angle), np.sin(angle)
        self.U = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])

    def run_scan(self, name, affine, chunk_size=137):
        path = self.root / name
        nib.save(nib.Nifti1Image(self.data, affine), path)
        before = path.read_bytes()
        # Use the actual stored spacing, as the GUI does.
        spacing = np.array(nib.load(path).header.get_zooms())
        points = self.points / self.spacing * spacing
        with contextlib.redirect_stdout(io.StringIO()):
            self.resample(str(path), points, self.U, chunk_size=chunk_size)
        self.assertEqual(before, path.read_bytes(), "Source scan was modified")
        result = nib.load(str(path).replace(".nii", "_PCA_resampled.nii"))
        return result.get_fdata(), result.affine, points, spacing

    def test_translated_rotated_and_flipped_world_affines_preserve_intensities(self):
        reference = self.run_scan("reference.nii", np.diag([*self.spacing, 1]))
        for name, rotation in [
            ("translated.nii", np.eye(3)),
            ("rotated_flipped.nii.gz", np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])),
        ]:
            with self.subTest(name=name):
                affine = np.eye(4)
                affine[:3, :3] = rotation @ np.diag(self.spacing)
                affine[:3, 3] = [-54.4414, -121.05, 62.7129]
                actual, output_affine, _, _ = self.run_scan(name, affine)
                np.testing.assert_array_equal(actual, reference[0])
                np.testing.assert_array_equal(output_affine, reference[1])
                self.assertGreater(actual.max(), 1000)
                self.assertEqual(actual.min(), -1024)

    def test_rotated_anisotropic_scan_matches_independent_voxel_interpolation(self):
        actual, affine, points, spacing = self.run_scan(
            "oracle.nii", np.diag([*self.spacing, 1]))
        indices = np.indices(actual.shape).reshape(3, -1).T
        pca_coords = indices @ affine[:3, :3].T + affine[:3, 3]
        voxel_coords = (pca_coords @ self.U.T + points.mean(axis=0)) / spacing
        expected = map_coordinates(self.data.astype(np.float32), voxel_coords.T,
                                   order=1, mode="constant", cval=-1024,
                                   prefilter=False).astype(np.float16).astype(np.float32)
        # NIfTI rounds affine entries to float32; omit queries on the boundary.
        near_boundary = np.any((np.abs(voxel_coords) < 1e-4) |
                               (np.abs(voxel_coords - (np.array(self.shape)-1)) < 1e-4), axis=1)
        np.testing.assert_allclose(actual.ravel()[~near_boundary],
                                   expected[~near_boundary], atol=2, rtol=0)
        outside = np.any((voxel_coords < -1e-4) |
                         (voxel_coords > np.array(self.shape)-1+1e-4), axis=1)
        self.assertTrue(outside.any())
        np.testing.assert_array_equal(actual.ravel()[outside], -1024)

    def test_chunk_size_does_not_change_the_volume(self):
        affine = np.diag([*self.spacing, 1])
        small = self.run_scan("small.nii", affine, chunk_size=31)
        large = self.run_scan("large.nii", affine, chunk_size=1_000_000)
        np.testing.assert_array_equal(small[0], large[0])
        np.testing.assert_array_equal(small[1], large[1])


if __name__ == "__main__":
    unittest.main()

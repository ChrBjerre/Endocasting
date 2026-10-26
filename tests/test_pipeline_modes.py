"""Check script routing without opening a GUI or processing a CT scan."""

import ast
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import Mock


class PipelineModeTests(unittest.TestCase):
    def test_original_automatic_and_pca_manual_modes(self):
        for use_pca in (False, True):
            with self.subTest(use_pca_resampling=use_pca):
                source = Path(__file__).resolve().parents[1] / "script.py"
                tree = ast.parse(source.read_bytes())
                settings = {"path", "out_path", "num_iters", "step_sizes",
                            "remesh", "use_pca_resampling"}
                # Substitute settings and external operations, keeping the actual
                # script's branching and propagation of coordinates/path intact.
                tree.body = [node for node in tree.body
                             if not isinstance(node, (ast.Import, ast.ImportFrom))
                             and not (isinstance(node, ast.Assign) and any(
                                 isinstance(target, ast.Name) and target.id in settings
                                 for target in node.targets))]
                candidates, radii, cloud = [[1, 2, 3], [7, 8, 9]], [4.0, 2.0], []
                resample = Mock()
                find = Mock(return_value=(candidates, radii, cloud))
                select = Mock(return_value=(candidates[1], radii[1]))
                endocast = Mock()
                namespace = dict(path="input.nii", out_path="output/", remesh=False,
                                 num_iters=[10, 5, 3, 2], step_sizes=[1, 1, .5, .5],
                                 use_pca_resampling=use_pca, PCA_resample_GUI=resample,
                                 find_center_points=find, select_local_max=select,
                                 iterative_mesh_every_iter=endocast)
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(tree, str(source), "exec"), namespace)
                volume = "input_PCA_resampled.nii" if use_pca else "input.nii"
                find.assert_called_once_with(volume)
                if use_pca:
                    resample.assert_called_once_with("input.nii")
                    select.assert_called_once_with(cloud, candidates, radii)
                else:
                    resample.assert_not_called()
                    select.assert_not_called()
                chosen = 1 if use_pca else 0
                endocast.assert_called_once_with(
                    candidates[chosen], radii[chosen], "output/", volume,
                    num_iters=[10, 5, 3, 2], step_sizes=[1, 1, .5, .5], remesh=False)


if __name__ == "__main__":
    unittest.main()

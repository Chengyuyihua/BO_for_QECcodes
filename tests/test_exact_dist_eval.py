# multiprocessing is used for the timeout of exact distance evaluation, and it doesn't work well with notebooks

import numpy as np
import qldpc


import sys
from pathlib import Path

# hack to access classes outside of test directory
current_file_path = Path(__file__).resolve()
path_root = current_file_path.parent.parent

if str(path_root) not in sys.path:
    sys.path.insert(0, str(path_root))

from bayesian_optimization.objective_function import ObjectiveFunction
from code_construction.code_construction import CSSCode


# -----------------------------------------------------------------------


class DummyCodeConstructor:
    def __init__(self):
        self.n = 1
        self.k = 1

    def construct(self, matrices):
        return CSSCode(matrices[0], matrices[1])


if __name__ == "__main__":
    print("Initializing...")
    cc = DummyCodeConstructor()
    obj = ObjectiveFunction(
        code_constructor=cc, code_eval_metric="distance", dist_timeout=100
    )

    small_code = qldpc.codes.SteaneCode()
    small_code_matrices = np.array([small_code.matrix_x, small_code.matrix_z])
    small_known_distance = 3

    print("Running objective function...")
    _, d = obj.forward(small_code_matrices)

    print(f"Calculated distance: {d}")
    assert d == small_known_distance
    print("Exact distance test passed!")

import unittest

import numpy as np

import torch

from utils import fullKTD, rtkpsvd_numpy, rtkpsvd_torch


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

class TestCompareNumpyTorch(unittest.TestCase):
    def test_fullktd_numpy_vs_torch(self):
        rng = np.random.default_rng(0)
        A1 = rng.normal(size=(2, 2))
        A2 = rng.normal(size=(2, 2))
        s = np.array([1.25])
        np_out = fullKTD([[A1], [A2]], s, R=1)
        tA1 = torch.tensor(A1, dtype=torch.float64)
        tA2 = torch.tensor(A2, dtype=torch.float64)
        ts = torch.tensor(s, dtype=torch.float64)
        torch_out = fullKTD([[tA1], [tA2]], ts, R=1).detach().cpu().numpy()
        diff = _mean_abs_diff(np_out, torch_out)
        print(f"fullKTD mean abs diff (numpy vs torch): {diff:.6e}")
        self.assertLess(diff, 1e-6)

    def test_rtkpsvd_numpy_vs_torch_reconstruction(self):
        rng = np.random.default_rng(1)
        A1 = rng.normal(size=(2, 2))
        A2 = rng.normal(size=(2, 2))
        A = np.kron(A2, A1)
        n = [2, 2, 2, 2]
        B_np, sig_np = rtkpsvd_numpy(A, n, R=None)
        recon_np = fullKTD(B_np, sig_np, R=len(B_np[0]))
        tA = torch.tensor(A, dtype=torch.float64)
        B_t, sig_t = rtkpsvd_torch(tA, n, R=None)
        recon_t = fullKTD(B_t, sig_t, R=len(B_t[0])).detach().cpu().numpy()
        diff = _mean_abs_diff(recon_np, recon_t)
        print(f"rtkpsvd reconstruction mean abs diff (numpy vs torch): {diff:.6e}")
        self.assertLess(diff, 1e-5)


if __name__ == "__main__":
    unittest.main()

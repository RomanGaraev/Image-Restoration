import unittest
from typing import Tuple

import numpy as np

import torch

from utils import (
    fullKTD,
    rtkpsvd_numpy,
    rtkpsvd_torch,
    _rttr1svds_numpy,
    _rttr1svds_torch,
    _basicrSVDd_numpy,
    _basicrSVDd_torch,
    _svd_econ_numpy,
    _svd_econ_torch,
)


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def _mean_abs_diff_aligned(a: np.ndarray, b: np.ndarray) -> Tuple[float, str]:
    if a.shape == b.shape:
        return _mean_abs_diff(a, b), "direct"
    if a.size != b.size:
        raise ValueError(f"Shape mismatch with different sizes: {a.shape} vs {b.shape}")
    a_c = a.reshape(b.shape, order="C")
    a_f = a.reshape(b.shape, order="F")
    diff_c = _mean_abs_diff(a_c, b)
    diff_f = _mean_abs_diff(a_f, b)
    if diff_c <= diff_f:
        return diff_c, "reshape C"
    return diff_f, "reshape F"
def _log_sign_flips(name: str, mat_np: np.ndarray, mat_t: np.ndarray, max_log: int = 10) -> None:
    cols = min(mat_np.shape[1], mat_t.shape[1])
    flips = []
    for i in range(cols):
        dot = float(np.dot(mat_np[:, i], mat_t[:, i]))
        if dot < 0:
            flips.append(i)
            if len(flips) >= max_log:
                break
    if flips:
        print(f"{name} sign flips at columns: {flips}")
    else:
        print(f"{name} sign flips: none")

class TestCompareNumpyTorch(unittest.TestCase):
    def test_fullktd_numpy_vs_torch(self):
        rng = np.random.default_rng(0)
        # Use larger factors to better mimic image-like dimensions without exploding memory.
        A1 = rng.normal(size=(32, 32, 3))
        A2 = rng.normal(size=(32, 32, 3))
        s = np.array([1.25, 0.76])
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
        A1 = rng.normal(size=(20, 32,3))
        A2 = rng.normal(size=(32, 20,1))
        A = np.kron(A2, A1)
        tA = torch.tensor(A, dtype=torch.float64)
        
        n = [20, 32,3, 20, 32, 1]
        rank = 3
        B_np, sig_np = rtkpsvd_numpy(A, n, R=rank)
        recon_np = fullKTD(B_np, sig_np, R=min(rank, len(B_np[0])))
        
        B_t, sig_t = rtkpsvd_torch(tA, n, R=rank)
        recon_t = fullKTD(B_t, sig_t, R=min(rank, len(B_t[0]))).detach().cpu().numpy()
        err_np, mode_np = _mean_abs_diff_aligned(A, recon_np)
        err_t, mode_t = _mean_abs_diff_aligned(A, recon_t)
        
        print(f"rtkpsvd reconstruction mean abs diff (numpy vs A): {err_np:.6e} ({mode_np})")
        print(f"rtkpsvd reconstruction mean abs diff (torch vs A): {err_t:.6e} ({mode_t})")

        self.assertLess(err_np, 1e-4)
        self.assertLess(err_t, 1e-4)

    def test_rttr1svds_numpy_vs_torch_sigmas(self):
        rng = np.random.default_rng(2)
        A = rng.normal(size=(6, 4, 5))
        R = 3
        U_np, S_np, V_np, sig_np = _rttr1svds_numpy(A, R)
        tA = torch.tensor(A, dtype=torch.float32)
        U_t, S_t, V_t, sig_t = _rttr1svds_torch(tA, R)
        diff_sig = _mean_abs_diff(sig_np, sig_t.detach().cpu().numpy())
        print(f"rttr1svds sigmas mean abs diff (numpy vs torch): {diff_sig:.6e}")
        self.assertLess(diff_sig, 1e-5)

    def test_basicrSVDd_numpy_vs_torch_reconstruction(self):
        rng = np.random.default_rng(3)
        A = rng.normal(size=(64, 40))
        k = 5
        B = rng.standard_normal((A.shape[1], k + 5))
        U_np, s_np, V_np = _basicrSVDd_numpy(A, k, p=2, B=B)
        tA = torch.tensor(A, dtype=torch.float64)
        tB = torch.tensor(B, dtype=torch.float64)

        U_t, s_t, V_t = _basicrSVDd_torch(tA, k, p=2, B=tB)
        recon_np = U_np @ np.diag(s_np) @ V_np.T
        recon_t = (U_t @ torch.diag(s_t) @ V_t.T).detach().cpu().numpy()
        diff = _mean_abs_diff(recon_np, recon_t)
        print(f"basicrSVDd reconstruction mean abs diff (numpy vs torch): {diff:.6e}")
        self.assertLess(diff, 1e-5)

    def test_basicrSVDd_internal_steps_numpy_vs_torch(self):
        rng = np.random.default_rng(6)
        A = rng.normal(size=(64, 40))
        k = 6
        B = rng.standard_normal((A.shape[1], k + 5))
        # NumPy internal steps (including power iterations)
        C_np = A @ B
        Q_np, _ = np.linalg.qr(C_np, mode="reduced")
        for _ in range(2):
            Q_np, _ = np.linalg.qr(A.T @ Q_np, mode="reduced")
            Q_np, _ = np.linalg.qr(A @ Q_np, mode="reduced")
        B_small_np = Q_np.T @ A
        Ut_np, s_np, Vh_np = np.linalg.svd(B_small_np, full_matrices=False)
        U_np = Q_np @ Ut_np
        U_np = U_np[:, :k]
        s_np = s_np[:k]
        V_np = Vh_np[:k, :].T

        # Torch internal steps (including power iterations)
        tA = torch.tensor(A, dtype=torch.float64)
        tB = torch.tensor(B, dtype=torch.float64)
        C_t = tA @ tB
        Q_t, _ = torch.linalg.qr(C_t, mode="reduced")
        for _ in range(2):
            Q_t, _ = torch.linalg.qr(tA.T @ Q_t, mode="reduced")
            Q_t, _ = torch.linalg.qr(tA @ Q_t, mode="reduced")
        B_small_t = Q_t.T @ tA
        Ut_t, s_t, Vh_t = torch.linalg.svd(B_small_t, full_matrices=False)
        U_t = Q_t @ Ut_t
        U_t = U_t[:, :k]
        s_t = s_t[:k]
        V_t = Vh_t[:k, :].transpose(-2, -1)

        diff_Q = _mean_abs_diff(Q_np, Q_t.detach().cpu().numpy())
        diff_U = _mean_abs_diff(U_np, U_t.detach().cpu().numpy())
        diff_s = _mean_abs_diff(s_np, s_t.detach().cpu().numpy())
        diff_V = _mean_abs_diff(V_np, V_t.detach().cpu().numpy())
        recon_t = (U_t @ torch.diag(s_t) @ V_t.T).detach().cpu().numpy()
        recon_np = U_np @ np.diag(s_np) @ V_np.T
        diff = _mean_abs_diff(recon_np, recon_t)
        
        print(f"basicrSVDd internal Q mean abs diff: {diff_Q:.6e}")
        print(f"basicrSVDd internal U mean abs diff: {diff_U:.6e}")
        print(f"basicrSVDd internal s mean abs diff: {diff_s:.6e}")
        print(f"basicrSVDd internal V mean abs diff: {diff_V:.6e}")
        
        print(f"basicrSVDd  mean abs diff (numpy vs torch): {diff:.6e}")
        self.assertLess(diff, 1e-6)

    def test_svd_econ_numpy_vs_torch_reconstruction(self):
        rng = np.random.default_rng(4)
        A = rng.normal(size=(50, 30))
        U_np, s_np, V_np = _svd_econ_numpy(A)
        tA = torch.tensor(A, dtype=torch.float64)
        U_t, s_t, V_t = _svd_econ_torch(tA)
        _log_sign_flips("svd U", U_np, U_t.detach().cpu().numpy())
        _log_sign_flips("svd V", V_np, V_t.detach().cpu().numpy())
        recon_np = U_np @ np.diag(s_np) @ V_np.T
        recon_t = (U_t @ torch.diag(s_t) @ V_t.T).detach().cpu().numpy()
        diff = _mean_abs_diff(recon_np, recon_t)
        print(f"svd_econ reconstruction mean abs diff (numpy vs torch): {diff:.6e}")
        self.assertLess(diff, 1e-6)
    
    def test_qr_numpy_vs_torch_reconstruction(self):
        rng = np.random.default_rng(5)
        A = rng.normal(size=(64, 40))
        Q_np, R_np = np.linalg.qr(A, mode="reduced")
        tA = torch.tensor(A, dtype=torch.float64)
        Q_t, R_t = torch.linalg.qr(tA, mode="reduced")
        _log_sign_flips("qr Q", Q_np, Q_t.detach().cpu().numpy())
        recon_np = Q_np @ R_np
        recon_t = (Q_t @ R_t).detach().cpu().numpy()
        diff = _mean_abs_diff(recon_np, recon_t)
        print(f"qr reconstruction mean abs diff (numpy vs torch): {diff:.6e}")
        self.assertLess(diff, 1e-6)


if __name__ == "__main__":
    unittest.main()

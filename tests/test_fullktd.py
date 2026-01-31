import unittest

import numpy as np

try:
    import torch
    HAS_TORCH = hasattr(torch, "tensor")
except ModuleNotFoundError:
    import sys
    import types

    HAS_TORCH = False
    torch = types.ModuleType("torch")

if not HAS_TORCH:
    import sys
    import types

    if not isinstance(torch, types.ModuleType):
        torch = types.ModuleType("torch")

    def _is_tensor(_value):
        return False

    torch.is_tensor = _is_tensor
    sys.modules["torch"] = torch

try:
    import skimage  # noqa: F401
except ModuleNotFoundError:
    import sys
    import types

    skimage_stub = types.ModuleType("skimage")
    metrics_stub = types.ModuleType("skimage.metrics")

    def _missing_skimage(*_args, **_kwargs):
        raise RuntimeError("skimage is not installed in the test environment.")

    metrics_stub.peak_signal_noise_ratio = _missing_skimage
    skimage_stub.metrics = metrics_stub
    sys.modules["skimage"] = skimage_stub
    sys.modules["skimage.metrics"] = metrics_stub

try:
    import torchvision  # noqa: F401
except ModuleNotFoundError:
    import sys
    import types

    torchvision_stub = types.ModuleType("torchvision")
    ops_stub = types.ModuleType("torchvision.ops")

    def _missing_torchvision(*_args, **_kwargs):
        raise RuntimeError("torchvision is not installed in the test environment.")

    ops_stub.box_iou = _missing_torchvision
    torchvision_stub.ops = ops_stub
    sys.modules["torchvision"] = torchvision_stub
    sys.modules["torchvision.ops"] = ops_stub

try:
    import matplotlib  # noqa: F401
except ModuleNotFoundError:
    import sys
    import types

    matplotlib_stub = types.ModuleType("matplotlib")
    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    patches_stub = types.ModuleType("matplotlib.patches")

    class _MissingMatplotlib:
        def __getattr__(self, _name):
            raise RuntimeError("matplotlib is not installed in the test environment.")

    pyplot_stub.__dict__["__getattr__"] = _MissingMatplotlib().__getattr__

    class Rectangle:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("matplotlib is not installed in the test environment.")

    patches_stub.Rectangle = Rectangle
    matplotlib_stub.pyplot = pyplot_stub
    matplotlib_stub.patches = patches_stub
    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub
    sys.modules["matplotlib.patches"] = patches_stub

from utils import fullKTD


def tkron_bruteforce(A, B):
    """Reference tkron implementation: C{(i2,i1),...} = B{i2,...} * A{i1,...}."""
    A_arr = np.asarray(A)
    B_arr = np.asarray(B)
    n1 = list(A_arr.shape)
    n2 = list(B_arr.shape)
    d = max(len(n1), len(n2))
    if len(n1) < d:
        n1.extend([1] * (d - len(n1)))
        A_arr = A_arr.reshape(n1)
    if len(n2) < d:
        n2.extend([1] * (d - len(n2)))
        B_arr = B_arr.reshape(n2)
    out_shape = [n1[i] * n2[i] for i in range(d)]
    C = np.zeros(out_shape, dtype=np.result_type(A_arr, B_arr))
    for a_idx in np.ndindex(*n1):
        for b_idx in np.ndindex(*n2):
            out_idx = tuple(a_idx[i] * n2[i] + b_idx[i] for i in range(d))
            C[out_idx] = A_arr[a_idx] * B_arr[b_idx]
    return C


class TestFullKTD(unittest.TestCase):
    def test_fullktd_numpy_single_term_matrix(self):
        A1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        A2 = np.array([[0.5, -1.0], [2.0, 3.0]])
        s = np.array([2.0])
        expected = s[0] * tkron_bruteforce(A2, A1)
        result = fullKTD([[A1], [A2]], s, R=1)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(expected, s[0] * np.kron(A2, A1), rtol=1e-6, atol=1e-6)

    def test_fullktd_numpy_multi_term_tensor3(self):
        rng = np.random.default_rng(0)
        A1_0 = rng.normal(size=(2, 1, 2))
        A2_0 = rng.normal(size=(1, 2, 1))
        A1_1 = rng.normal(size=(2, 1, 2))
        A2_1 = rng.normal(size=(1, 2, 1))
        s = np.array([1.5, -0.75])
        A = [[A1_0, A1_1], [A2_0, A2_1]]
        expected = (
            s[0] * tkron_bruteforce(A2_0, A1_0)
            + s[1] * tkron_bruteforce(A2_1, A1_1)
        )
        result = fullKTD(A, s, R=2)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_fullktd_numpy_R_truncation(self):
        s = np.array([1.0, 2.0])
        A1 = [np.array([[1.0]]), np.array([[2.0]])]
        A2 = [np.array([[3.0]]), np.array([[4.0]])]
        result = fullKTD([A1, A2], s, R=1)
        expected = s[0] * tkron_bruteforce(A2[0], A1[0])
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    @unittest.skipIf(not HAS_TORCH, "torch is not installed")
    def test_fullktd_torch_single_term(self):
        A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        A2 = torch.tensor([[0.5, -1.0], [2.0, 3.0]])
        s = torch.tensor([2.0])
        result = fullKTD([[A1], [A2]], s, R=1)
        expected = s.item() * tkron_bruteforce(A2.numpy(), A1.numpy())
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

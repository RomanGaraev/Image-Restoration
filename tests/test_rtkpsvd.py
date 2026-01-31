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

from utils import fullKTD, rtkpsvd_numpy, rtkpsvd_torch


class TestRTKPSVD(unittest.TestCase):
    def test_rtkpsvd_numpy_reconstruction_rank1_matrix(self):
        A1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        A2 = np.array([[0.0, 1.0], [1.0, 0.0]])
        A = np.kron(A2, A1)
        n = [2, 2, 2, 2]
        B, sigmas = rtkpsvd_numpy(A, n)
        recon = fullKTD(B, sigmas, R=len(B[0]))
        np.testing.assert_allclose(recon, A, rtol=1e-5, atol=1e-5)
        self.assertEqual(len(B), 2)
        self.assertEqual(B[0][0].shape, A1.shape)
        self.assertEqual(B[1][0].shape, A2.shape)

    @unittest.skipIf(not HAS_TORCH, "torch is not installed")
    def test_rtkpsvd_torch_reconstruction_rank1_matrix(self):
        A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        A2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        A = torch.kron(A2, A1)
        n = [2, 2, 2, 2]
        B, sigmas = rtkpsvd_torch(A, n)
        recon = fullKTD(B, sigmas, R=len(B[0]))
        np.testing.assert_allclose(recon.detach().cpu().numpy(), A.cpu().numpy(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

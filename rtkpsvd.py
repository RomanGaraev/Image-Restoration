"""Python implementation of the RTKPSVD algorithm backed by PyTorch.

This module ports the MATLAB implementation of the reduced tensor-based
Kronecker product singular value decomposition (RTKPSVD) and the supporting
tensor-train rank-1 SVD utilities that appear in the original TKPSVD toolbox.
"""

from __future__ import annotations

from math import ceil
from typing import List, Optional, Sequence, Tuple

import torch


Array = torch.Tensor


def _ensure_tensor(data, device: Optional[torch.device] = None) -> Array:
    """Convert arbitrary input to a torch.float32 tensor on the requested device."""
    tensor = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
    if device is not None:
        tensor = tensor.to(device)
    if tensor.dtype != torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


def _reshape_fortran(tensor: Array, shape: Sequence[int]) -> Array:
    """Replicate MATLAB/NumPy column-major reshape semantics using PyTorch."""
    if tensor.ndim == 0:
        return tensor.reshape(shape)
    if len(shape) == 0:
        return tensor.reshape(shape)
    perm_in = tuple(range(tensor.ndim - 1, -1, -1))
    permuted = tensor.permute(perm_in).contiguous()
    new_shape = tuple(reversed(tuple(shape)))
    reshaped = permuted.reshape(new_shape)
    perm_out = tuple(range(len(shape) - 1, -1, -1))
    return reshaped.permute(perm_out).contiguous()


def _spacing(value: Array) -> Array:
    """Return the spacing to the next floating point number (NumPy-style)."""
    if value.numel() == 0:
        return torch.tensor(0.0, dtype=value.dtype, device=value.device)
    finfo = torch.finfo(value.dtype)
    return torch.where(
        value == 0,
        torch.full_like(value, finfo.tiny),
        torch.abs(value) * finfo.eps,
    )


def _repeat_with_ones(vec: Array, repeat: int) -> Array:
    """Construct np.kron(vec, ones(repeat)) without leaving torch."""
    if repeat <= 1:
        return vec
    ones_vec = torch.ones(repeat, dtype=vec.dtype, device=vec.device)
    return torch.kron(vec, ones_vec)


def _infer_dtype_device(*seqs) -> Tuple[torch.dtype, torch.device]:
    """Infer dtype/device from tensors stored in nested sequences."""
    for seq in seqs:
        if not seq:
            continue
        for item in seq:
            if isinstance(item, torch.Tensor):
                return item.dtype, item.device
    return torch.float32, torch.device("cpu")


def _prod_range(values: Sequence[int], start: int, end: int) -> int:
    """Product of values[start:end] with MATLAB-style inclusive indexing."""
    if start > end:
        return 1
    prod = 1
    for idx in range(start, end + 1):
        prod *= int(values[idx])
    return prod


def basicrSVDd(A: Array, k: int, p: int) -> Tuple[Array, Array, Array]:
    """Randomized truncated SVD used in the sparse RTTR1 routines."""
    A_t = _ensure_tensor(A)
    device = A_t.device
    dtype = A_t.dtype
    work_device = torch.device("cpu") if device.type == "mps" else device
    if work_device != device:
        A_work = A_t.to(work_device)
    else:
        A_work = A_t
    s = 5
    m, n = A_work.shape
    k = int(k)
    if k <= 0:
        return (
            torch.empty((m, 0), dtype=dtype, device=device),
            torch.empty((0,), dtype=dtype, device=device),
            torch.empty((n, 0), dtype=dtype, device=device),
        )
    oversample = k + s
    B = torch.randn((n, oversample), dtype=dtype, device=work_device)
    C = A_work @ B
    Q, _ = torch.linalg.qr(C, mode="reduced")
    for _ in range(p):
        Q, _ = torch.linalg.qr(A_work.transpose(-2, -1) @ Q, mode="reduced")
        Q, _ = torch.linalg.qr(A_work @ Q, mode="reduced")
    B_small = Q.transpose(-2, -1) @ A_work
    Ut, svals, Vh = torch.linalg.svd(B_small, full_matrices=False)
    if k > Ut.shape[1] or k > Vh.shape[0]:
        raise ValueError("Requested rank exceeds the available rank in basicrSVDd (MATLAB-style behavior).")
    U = Q @ Ut[:, :k]
    svals = svals[:k]
    V = Vh[:k, :].transpose(-2, -1)
    if work_device != device:
        U = U.to(device)
        svals = svals.to(device)
        V = V.to(device)
    return U, svals, V


def ttr1svdz(A: Array) -> Tuple[List[Optional[Array]], List[Optional[Array]], List[Optional[Array]], Array]:
    """Reduced tensor-train rank-1 SVD using dense SVDs."""
    tensor = _ensure_tensor(A)
    dtype = tensor.dtype
    device = tensor.device
    n = [None] + list(tensor.shape)
    d = len(n) - 1
    r_vals = [min(n[i], _prod_range(n, i + 1, d)) for i in range(1, d)]
    r = [None] + r_vals
    length_r = len(r) - 1
    totalsvd = 1
    if length_r >= 1:
        for i in range(2, length_r + 1):
            totalsvd += _prod_range(r, 1, i - 1)
    nleaf = _prod_range(r, 1, length_r) if length_r else 1
    U = [None] * (totalsvd + 1)
    S = [None] * (totalsvd + 1)
    V = [None] * (totalsvd + 1)
    rows = n[1]
    cols = _prod_range(n, 2, d)
    mat = _reshape_fortran(tensor, (rows, cols))
    Ut, svals, Vh = torch.linalg.svd(mat, full_matrices=False)
    U[1] = Ut
    S[1] = svals
    V[1] = Vh.transpose(-2, -1)
    counter = 2
    which_v = 1
    repeat = int(nleaf / len(S[1])) if len(S[1]) else 1
    sigmas = _repeat_with_ones(S[1], repeat)
    prod_dims = 1
    for val in n[1:]:
        prod_dims *= int(val)
    tol = torch.tensor(float(prod_dims), dtype=dtype, device=device) * _spacing(torch.max(S[which_v]))
    for i in range(1, length_r):
        Slevel: List[Array] = []
        num_nodes = _prod_range(r, 1, i)
        for j in range(1, num_nodes + 1):
            col = r[i] if j % r[i] == 0 else j % r[i]
            current_S = S[which_v]
            has_values = current_S is not None and current_S.numel() > 0
            current_V = V[which_v]
            if has_values and current_V is not None and current_S[col - 1] > tol:
                vec = current_V[:, col - 1]
                rows = n[i + 1]
                cols = _prod_range(n, i + 2, d)
                T = _reshape_fortran(vec, (rows, cols))
                Ut, svals, Vh = torch.linalg.svd(T, full_matrices=False)
                U[counter] = Ut
                S[counter] = svals
                V[counter] = Vh.transpose(-2, -1)
                Slevel.append(svals)
            else:
                zeros_len = min(n[i + 1], _prod_range(n, i + 2, d))
                Slevel.append(torch.zeros(zeros_len, dtype=dtype, device=device))
            counter += 1
            if j % n[i] == 0:
                which_v += 1
        if Slevel:
            Slevel_vec = torch.cat(Slevel)
            repeat = int(nleaf / Slevel_vec.numel()) if Slevel_vec.numel() else 1
            Slevel_vec = _repeat_with_ones(Slevel_vec, repeat)
            sigmas = sigmas * Slevel_vec
    return U, S, V, sigmas


def rttr1svds(A: Array, R: int) -> Tuple[List[Optional[Array]], List[Optional[Array]], List[Optional[Array]], Array]:
    """Sparse tensor-train rank-1 SVD that keeps at most R terms per node."""
    tensor = _ensure_tensor(A)
    dtype = tensor.dtype
    device = tensor.device
    n = [None] + list(tensor.shape)
    d = len(n) - 1
    r_vals = [min(n[i], R) for i in range(1, d)]
    r = [None] + r_vals
    length_r = len(r) - 1
    totalsvd = 1
    if length_r >= 1:
        for i in range(2, length_r + 1):
            totalsvd += _prod_range(r, 1, i - 1)
    nleaf = _prod_range(r, 1, length_r) if length_r else 1
    U = [None] * (totalsvd + 1)
    S = [None] * (totalsvd + 1)
    V = [None] * (totalsvd + 1)
    rows = n[1]
    cols = _prod_range(n, 2, d)
    mat = _reshape_fortran(tensor, (rows, cols))
    Ut, St, Vt = basicrSVDd(mat, R, 1)
    U[1] = Ut
    S[1] = St
    V[1] = Vt
    counter = 2
    which_v = 1
    repeat = int(nleaf / len(S[1])) if len(S[1]) else 1
    sigmas = _repeat_with_ones(S[1], repeat)
    prod_dims = 1
    for val in n[1:]:
        prod_dims *= int(val)
    tol = torch.tensor(float(prod_dims), dtype=dtype, device=device) * _spacing(torch.max(S[which_v]))
    for i in range(1, length_r):
        Slevel: List[Array] = []
        num_nodes = _prod_range(r, 1, i)
        for j in range(1, num_nodes + 1):
            col = r[i] if j % r[i] == 0 else j % r[i]
            current_S = S[which_v]
            has_values = current_S is not None and current_S.numel() > 0
            current_V = V[which_v]
            if has_values and current_V is not None and current_S[col - 1] > tol:
                vec = current_V[:, col - 1]
                rows = n[i + 1]
                cols = _prod_range(n, i + 2, d)
                T = _reshape_fortran(vec, (rows, cols))
                Ut, St, Vt = basicrSVDd(T, R, 1)
                U[counter] = Ut
                S[counter] = St
                V[counter] = Vt
                Slevel.append(S[counter])
            else:
                Slevel.append(torch.zeros(R, dtype=dtype, device=device))
            counter += 1
            length_current = len(S[which_v]) if S[which_v] is not None else 0
            if length_current > 0 and j % length_current == 0:
                which_v += 1
        if Slevel:
            Slevel_vec = torch.cat(Slevel)
            repeat = int(nleaf / Slevel_vec.numel()) if Slevel_vec.numel() else 1
            Slevel_vec = _repeat_with_ones(Slevel_vec, repeat)
            sigmas = sigmas * Slevel_vec
    return U, S, V, sigmas


def leave2ind(sigma_indices: Sequence[int], n: Sequence[int]) -> torch.Tensor:
    """Replicates the MATLAB leave2ind helper."""
    sigma = [int(idx) for idx in sigma_indices]
    d = len(n)
    if d == 0:
        return torch.zeros((0, 0), dtype=torch.int64)
    if d == 2:
        indices = torch.ones((len(sigma), 2), dtype=torch.int64)
        if len(sigma) > 0:
            indices[:, 1] = torch.arange(1, len(sigma) + 1, dtype=torch.int64)
        return indices
    r = [1] * d
    nodesperlevel = [1] * d
    endI = [1] * d
    for i in range(1, d):
        prod_n = 1
        for val in n[i:]:
            prod_n *= val
        r[i] = min(n[i - 1], prod_n)
        prod_r = 1
        for val in r[: i + 1]:
            prod_r *= val
        nodesperlevel[i] = prod_r
        endI[i] = sum(nodesperlevel[: i + 1])
    endI = endI[:-2]
    k = sigma.copy()
    l = [0] * len(sigma)
    indices = torch.zeros((len(sigma), 2 * (d - 1)), dtype=torch.int64)
    for j in range(len(sigma)):
        if k[j] % r[-1] == 0:
            k[j] = k[j] // r[-1]
            l[j] = r[-1]
        else:
            l[j] = k[j] % r[-1]
            k[j] = int(ceil(k[j] / r[-1]))
        indices[j, 0] = endI[-1] + k[j]
        indices[j, 1] = l[j]
    endI = endI[:-1]
    r = r[:-1]
    t_counter = 2
    for _ in range(len(endI)):
        for j in range(len(sigma)):
            if k[j] % r[-1] == 0:
                k[j] = k[j] // r[-1]
                l[j] = r[-1]
            else:
                l[j] = k[j] % r[-1]
                k[j] = int(ceil(k[j] / r[-1]))
            indices[j, t_counter] = endI[-1] + k[j]
            indices[j, t_counter + 1] = l[j]
        t_counter += 2
        endI = endI[:-1]
        r = r[:-1]
    for j in range(len(sigma)):
        if k[j] % r[-1] == 0:
            l[j] = r[-1]
        else:
            l[j] = k[j] % r[-1]
        indices[j, t_counter] = 1
        indices[j, t_counter + 1] = l[j]
    return indices


def UV2Uhat(
    U: Sequence[Optional[Array]],
    V: Sequence[Optional[Array]],
    sigma_indices: Sequence[int],
    n: Sequence[int],
    n2: Optional[Sequence[int]] = None,
) -> List[Array]:
    """Convert TTr1 output matrices into CANDECOMP mode matrices."""
    sigma_indices = list(map(int, sigma_indices))
    d = len(n)
    if d == 0:
        return []
    indices = leave2ind(sigma_indices, n)
    dtype, device = _infer_dtype_device(U, V)
    if n2 is not None:
        rows = list(n2)
        cols = list(n)
        Uhat = [torch.zeros((rows[i], cols[i]), dtype=dtype, device=device) for i in range(d)]
    else:
        rows = list(n)
        n_terms = len(sigma_indices)
        Uhat = [torch.zeros((rows[i], n_terms), dtype=dtype, device=device) for i in range(d)]
    num_terms = len(sigma_indices)
    total_cols = indices.shape[1] if indices.numel() else 0
    for term in range(num_terms):
        row = indices[term]
        for j in range(1, d):
            cell_col = total_cols - 2 * j
            vec_col = total_cols - (2 * j - 1)
            cell_idx = int(row[cell_col].item())
            vec_idx = int(row[vec_col].item())
            if cell_idx >= len(U) or U[cell_idx] is None:
                raise ValueError("UV2Uhat expected populated U entries (MATLAB would error on missing data).")
            if vec_idx <= 0 or vec_idx > U[cell_idx].shape[1]:
                raise ValueError("UV2Uhat received an out-of-range column index for U (MATLAB-style behavior).")
            Uhat[j - 1][:, term] = U[cell_idx][:, vec_idx - 1]
        cell_idx = int(row[0].item())
        vec_idx = int(row[1].item())
        if cell_idx >= len(V) or V[cell_idx] is None:
            raise ValueError("UV2Uhat expected populated V entries (MATLAB would error on missing data).")
        if vec_idx <= 0 or vec_idx > V[cell_idx].shape[1]:
            raise ValueError("UV2Uhat received an out-of-range column index for V (MATLAB-style behavior).")
        Uhat[d - 1][:, term] = V[cell_idx][:, vec_idx - 1]
    return Uhat


def rtkpsvd(A: Array, n: Sequence[int], R: Optional[int] = None):
    """Tensor-based Kronecker Product SVD implemented in PyTorch."""
    tensor = _ensure_tensor(A)
    dtype = tensor.dtype
    device = tensor.device
    n = list(map(int, n))
    dim = tensor.shape
    d = len(dim)
    if len(n) % d != 0:
        raise ValueError("Not all factors have the same order.")
    numfac = len(n) // d
    noe: List[int] = []
    for i in range(d):
        noe.extend(n[i::d])
    tensor = _reshape_fortran(tensor, noe)
    permuted: List[int] = []
    for i in range(1, numfac + 1):
        permuted.extend(list(range(i, d * numfac + 1, numfac)))
    axes = [idx - 1 for idx in permuted]
    tensor = tensor.permute(axes).contiguous()
    n_perm = [noe[idx - 1] for idx in permuted]
    n2: List[int] = []
    for i in range(numfac):
        block = n_perm[i * d : (i + 1) * d]
        prod = 1
        for val in block:
            prod *= val
        n2.append(prod)
    tensor = _reshape_fortran(tensor, n2)
    if R is not None:
        U, S, V, sigmas = rttr1svds(tensor, R)
    else:
        U, S, V, sigmas = ttr1svdz(tensor)
    if sigmas.numel() == 0:
        raise ValueError("ttr1svd returned no singular values (mirroring MATLAB behavior).")
    prod_dims = 1
    for val in n2:
        prod_dims *= int(val)
    tol = torch.tensor(float(prod_dims), dtype=dtype, device=device) * _spacing(torch.abs(sigmas[0]))
    idx = torch.nonzero(sigmas > tol, as_tuple=False).squeeze(-1) + 1
    sigma_indices = idx.cpu().tolist()
    if R is not None:
        reduced_dims = [R] * len(n2)
        Uhat = UV2Uhat(U, V, sigma_indices, reduced_dims, n2)
    else:
        Uhat = UV2Uhat(U, V, sigma_indices, n2)
    num_terms = Uhat[0].shape[1] if Uhat else 0
    B: List[List[Array]] = []
    for j in range(numfac):
        block_dims = n[j * d : (j + 1) * d]
        row: List[Array] = []
        for i in range(num_terms):
            vec = Uhat[j][:, i]
            reshaped = _reshape_fortran(vec, block_dims)
            row.append(reshaped)
        B.append(row)
    return B, sigmas

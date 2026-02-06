from collections import defaultdict
from typing import Dict, List, Optional, Union, Sequence, Tuple
from regex import B
from skimage.metrics import peak_signal_noise_ratio
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
from matplotlib import patches
from pathlib import Path
import torch
import numpy as np


def calculate_accuracy(labels, predictions):
    correct = 0
    for pred, label in zip(predictions, labels):
        if pred.item() == label.item():
            correct += 1
    return 100.0 * correct / labels.size(0)


def compare_images(clean_img, adv_image, restored_img, clean_pred: str, adv_pred: str, restored_pred:str):
    print(f"PSNR between clean and adversarial samples: {peak_signal_noise_ratio(clean_img, adv_image)}")
    print(f"PSNR between clean and restored samples: {peak_signal_noise_ratio(clean_img, restored_img)}")

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    assert clean_img.shape == restored_img.shape
    for ind, (name, img) in enumerate(zip(["Original" + f" ({clean_pred})",
                                           "Adversarial" + f" ({adv_pred})",
                                           "Restored" + f" ({restored_pred})"],
                                          [clean_img, adv_image, restored_img])):
        ax[ind].imshow(img.transpose(1, 2, 0))
        ax[ind].axis('off')
        ax[ind].set_title(name)
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def get_preferred_device() -> torch.device:
    # First check MPS availability
    if torch.backends.mps.is_available():
        return torch.device("mps")
    # Then check CUDA availability
    elif torch.cuda.is_available():
        return torch.device("cuda")
    # Default to CPU if neither is available
    else:
        return torch.device("cpu")

class DetectionMetricTracker:
    """Tracks simple precision/recall/F1 statistics for detection outputs."""

    def __init__(self, iou_threshold: float = 0.5, score_threshold: float = 0.25):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.matches = 0
        self.total_predictions = 0
        self.total_targets = 0
        self.class_predictions: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
        self.gt_counts: Dict[int, int] = defaultdict(int)

    def update(self, predictions: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]) -> None:
        for pred, target in zip(predictions, targets):
            boxes = pred["boxes"]
            scores = pred["scores"]
            labels = pred["labels"]
            mask = scores >= self.score_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            self.total_predictions += boxes.size(0)

            gt_boxes = target.get("boxes", torch.zeros((0, 4), device=boxes.device))
            gt_labels = target.get("labels", torch.zeros((0,), dtype=torch.long, device=boxes.device))
            self.total_targets += gt_labels.size(0)
            gt_labels_list = gt_labels.detach().long().cpu().tolist() if gt_labels.numel() else []
            for label_id in gt_labels_list:
                self.gt_counts[int(label_id)] += 1

            if boxes.numel() == 0 or gt_boxes.numel() == 0:
                continue

            iou_matrix = box_iou(boxes, gt_boxes)
            matched = torch.zeros(gt_boxes.size(0), dtype=torch.bool, device=boxes.device)
            order = torch.argsort(scores, descending=True)
            for idx in order:
                best_iou, gt_idx = iou_matrix[idx].max(dim=0)
                class_id = int(labels[idx].item())
                score = float(scores[idx].item())
                if best_iou >= self.iou_threshold and not matched[gt_idx] and labels[idx] == gt_labels[gt_idx]:
                    matched[gt_idx] = True
                    self.matches += 1
                    self.class_predictions[class_id].append((score, 1))
                else:
                    self.class_predictions[class_id].append((score, 0))

    def compute(self) -> Dict[str, float]:
        precision = self.matches / self.total_predictions if self.total_predictions else 0.0
        recall = self.matches / self.total_targets if self.total_targets else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        ap_per_class: Dict[int, float] = {}
        for class_id, total_gt in self.gt_counts.items():
            preds = self.class_predictions.get(class_id)
            if not preds or total_gt == 0:
                continue
            sorted_preds = sorted(preds, key=lambda item: item[0], reverse=True)
            tp_cum = 0
            fp_cum = 0
            precisions: List[float] = []
            recalls: List[float] = []
            for _, is_tp in sorted_preds:
                if is_tp:
                    tp_cum += 1
                else:
                    fp_cum += 1
                recall_point = tp_cum / total_gt
                precision_point = tp_cum / (tp_cum + fp_cum)
                precisions.append(precision_point)
                recalls.append(recall_point)
            ap = 0.0
            for recall_threshold in (i / 10 for i in range(11)):
                precisions_at_recall = [p for p, r in zip(precisions, recalls) if r >= recall_threshold]
                best_precision = max(precisions_at_recall) if precisions_at_recall else 0.0
                ap += best_precision
            ap_per_class[class_id] = ap / 11.0
        map_value = sum(ap_per_class.values()) / len(ap_per_class) if ap_per_class else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "map": map_value}

def _resolve_label_text(labels: Optional[Sequence[int]], idx: int,
                        class_names: Optional[Union[Dict[int, str], List[str]]]) -> str:
    """Map label indices to display text using provided class names, if available."""
    if labels is None:
        return ""
    try:
        label_id = int(labels[idx])
    except (IndexError, TypeError, ValueError):
        return ""
    if class_names is None:
        return str(label_id)
    if isinstance(class_names, dict):
        return class_names.get(label_id, str(label_id))
    if isinstance(class_names, (list, tuple)):
        if 0 <= label_id < len(class_names):
            return class_names[label_id]
    if hasattr(class_names, "__getitem__"):
        try:
            return class_names[label_id]
        except Exception:
            pass
    return str(label_id)


def visualize_detections(image_batch: torch.Tensor,
                         predictions: List[Dict[str, torch.Tensor]],
                         targets: Optional[List[Dict[str, torch.Tensor]]],
                         title_prefix: str,
                         batch_index: int,
                         save_dir: Optional[Path] = None,
                         class_names: Optional[Union[Dict[int, str], List[str]]] = None) -> None:
    """Render predicted (red) and ground-truth (blue) boxes on the supplied images."""
    if image_batch is None or image_batch.numel() == 0:
        return
    base_dir = save_dir or Path("data/detection_visualization")
    batch_dir = base_dir / f"batch_{batch_index:04d}" / title_prefix.lower()
    batch_dir.mkdir(parents=True, exist_ok=True)
    cpu_images = image_batch.detach().cpu()
    num_samples = min(len(cpu_images), len(predictions))
    for idx in range(num_samples):
        fig, ax = plt.subplots(figsize=(6, 6))
        image = cpu_images[idx].permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(image)

        if targets is not None and idx < len(targets):
            gt_boxes = targets[idx].get("boxes")
            gt_labels = targets[idx].get("labels")
            if gt_boxes is not None and gt_boxes.numel() > 0:
                labels_iter = gt_labels.cpu().tolist() if gt_labels is not None and gt_labels.numel() else []
                for box_idx, box in enumerate(gt_boxes.cpu().numpy()):
                    x1, y1, x2, y2 = box.tolist()
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                             linewidth=1.5, edgecolor="blue", facecolor="none")
                    ax.add_patch(rect)
                    label_text = _resolve_label_text(labels_iter, box_idx, class_names)
                    if label_text:
                        ax.text(x1, max(y1 - 3, 0), label_text,
                                color="blue", fontsize=8,
                                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

        pred_boxes = predictions[idx].get("boxes")
        pred_labels = predictions[idx].get("labels")
        if pred_boxes is not None and pred_boxes.numel() > 0:
            labels_iter = pred_labels.cpu().tolist() if pred_labels is not None and pred_labels.numel() else []
            for box_idx, box in enumerate(pred_boxes.cpu().numpy()):
                x1, y1, x2, y2 = box.tolist()
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=1.5, edgecolor="red", facecolor="none")
                ax.add_patch(rect)
                label_text = _resolve_label_text(labels_iter, box_idx, class_names)
                if label_text:
                    ax.text(x1, max(y1 - 3, 0), label_text,
                            color="red", fontsize=8,
                            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

        #ax.set_title(f"{title_prefix} sample {idx}")
        ax.axis("off")
        plt.tight_layout()
        filename = batch_dir / f"sample{idx}.png"
        fig.savefig(filename, dpi=200, bbox_inches="tight")
        #plt.show(block=False)
        #plt.pause(0.001)
        plt.close(fig)


def _flatten_fortran_numpy(array: np.ndarray) -> np.ndarray:
    # MATLAB A(:) uses column-major order.
    return np.reshape(array, -1, order="F")


def _reshape_fortran_numpy(array: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    # Mirror MATLAB/Fortran reshape semantics.
    return np.reshape(array, tuple(int(dim) for dim in shape), order="F")


def _flatten_fortran_torch(tensor: torch.Tensor) -> torch.Tensor:
    # Emulate Fortran-order vectorization for torch tensors.
    if tensor.ndim <= 1:
        return tensor.reshape(-1)
    dims = list(range(tensor.ndim - 1, -1, -1))
    return tensor.permute(dims).contiguous().reshape(-1)


def _reshape_fortran_torch(tensor: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    # Inverse of _flatten_fortran_torch for a given target shape.
    shape = [int(dim) for dim in shape]
    if len(shape) <= 1:
        return tensor.reshape(shape)
    reversed_shape = list(reversed(shape))
    reshaped = tensor.reshape(reversed_shape)
    perm = list(range(len(shape) - 1, -1, -1))
    return reshaped.permute(perm)


def _tkron_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Replicates tkron.m: C{(i2,i1),...} = B{i2,...} * A{i1,...}.
    A_arr = np.asarray(A)
    B_arr = np.asarray(B)
    n1 = list(A_arr.shape)
    n2 = list(B_arr.shape)
    d = max(len(n1), len(n2))
    if len(n1) < d:
        n1.extend([1] * (d - len(n1)))
    if len(n2) < d:
        n2.extend([1] * (d - len(n2)))
    a_vec = _flatten_fortran_numpy(A_arr)
    b_vec = _flatten_fortran_numpy(B_arr)
    kron_vec = np.kron(a_vec, b_vec)
    kron_tensor = _reshape_fortran_numpy(kron_vec, n2 + n1)
    perm = []
    for i in range(d):
        perm.extend([i, i + d])
    kron_tensor = np.transpose(kron_tensor, perm)
    out_shape = [n1[i] * n2[i] for i in range(d)]
    kron_vec = _flatten_fortran_numpy(kron_tensor)
    return _reshape_fortran_numpy(kron_vec, out_shape)


def _tkron_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Torch equivalent of tkron.m with column-major semantics.
    A_t = A if torch.is_tensor(A) else torch.as_tensor(A)
    B_t = B if torch.is_tensor(B) else torch.as_tensor(B)
    B_t = B_t.to(device=A_t.device, dtype=A_t.dtype)
    n1 = list(A_t.shape)
    n2 = list(B_t.shape)
    d = max(len(n1), len(n2))
    if len(n1) < d:
        n1.extend([1] * (d - len(n1)))
    if len(n2) < d:
        n2.extend([1] * (d - len(n2)))
    a_vec = _flatten_fortran_torch(A_t)
    b_vec = _flatten_fortran_torch(B_t)
    kron_vec = torch.kron(a_vec, b_vec)
    kron_tensor = _reshape_fortran_torch(kron_vec, n2 + n1)
    perm = []
    for i in range(d):
        perm.extend([i, i + d])
    kron_tensor = kron_tensor.permute(perm)
    out_shape = [n1[i] * n2[i] for i in range(d)]
    kron_vec = _flatten_fortran_torch(kron_tensor)
    return _reshape_fortran_torch(kron_vec, out_shape)


def _mkron_numpy(*tensors: np.ndarray) -> np.ndarray:
    # Multiway Kronecker product applied left-to-right.
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    if len(tensors) < 2:
        raise ValueError("Need at least two tensors for Kronecker product.")
    result = tensors[0]
    for tensor in tensors[1:]:
        result = _tkron_numpy(result, tensor)
    return result


def _mkron_torch(*tensors: torch.Tensor) -> torch.Tensor:
    # Multiway Kronecker product applied left-to-right.
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    if len(tensors) < 2:
        raise ValueError("Need at least two tensors for Kronecker product.")
    result = tensors[0]
    for tensor in tensors[1:]:
        result = _tkron_torch(result, tensor)
    return result


def fullKTD(A, s, R: Optional[int] = None):
    """Reconstruct tensor from TKPSVD factors (MATLAB fullKTD equivalent)."""
    if A is None or len(A) < 2:
        raise ValueError("A must contain at least two factor lists.")
    if s is None:
        raise ValueError("s must be provided.")
    num_terms = min(len(A[0]), len(A[1]))
    if torch.is_tensor(s):
        s_len = int(s.numel())
    elif isinstance(s, np.ndarray):
        s_len = int(s.size)
    else:
        try:
            s_len = len(s)
        except TypeError:
            s_len = 1
    num_terms = min(num_terms, s_len)
    if R is None:
        R = num_terms
    if R > num_terms:
        raise ValueError(f"R={R} exceeds available terms ({num_terms}).")

    use_torch = torch.is_tensor(s)
    if not use_torch:
        for row in A[:2]:
            for item in row:
                if torch.is_tensor(item):
                    use_torch = True
                    break
            if use_torch:
                break

    if use_torch:
        ref = None
        for row in A[:2]:
            for item in row:
                if torch.is_tensor(item):
                    ref = item
                    break
            if ref is not None:
                break
        if ref is None and torch.is_tensor(s):
            ref = s
        if ref is None:
            ref = torch.as_tensor(A[0][0])
        s_tensor = s if torch.is_tensor(s) else torch.as_tensor(s, device=ref.device, dtype=ref.dtype)
        # Accumulate sum_i s[i] * mkron(A[1][i], A[0][i]) using torch tensors.
        result = None
        for i in range(R):
            a2 = A[1][i]
            a1 = A[0][i]
            a2 = a2 if torch.is_tensor(a2) else torch.as_tensor(a2, device=ref.device, dtype=ref.dtype)
            a1 = a1 if torch.is_tensor(a1) else torch.as_tensor(a1, device=ref.device, dtype=ref.dtype)
            term = _mkron_torch(a2, a1)
            term = term * s_tensor[i]
            result = term if result is None else result + term
        return result

    s_arr = np.asarray(s)
    # NumPy fall-back for list/array inputs.
    result = None
    for i in range(R):
        term = _mkron_numpy(A[1][i], A[0][i])
        term = term * s_arr[i]
        result = term if result is None else result + term
    return result


def _prod_int(values: Sequence[int]) -> int:
    # Integer product with defensive casting.
    prod = 1
    for val in values:
        prod *= int(val)
    return int(prod)


def _svd_econ_numpy(matrix: np.ndarray):
    # NumPy economy SVD wrapper to mirror MATLAB outputs.
    U, svals, Vh = np.linalg.svd(matrix, full_matrices=False)
    return U, svals, Vh.T


def _svd_econ_torch(matrix: torch.Tensor):
    # Torch economy SVD wrapper (returns V, not Vh).
    U, svals, Vh = torch.linalg.svd(matrix, full_matrices=False)
    return U, svals, Vh.transpose(-2, -1)


def _basicrSVDd_numpy(
    A: np.ndarray,
    k: int,
    p: int,
    *,
    B: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Randomized SVD (Halko et al.) used by the MATLAB code for sparse trees.
    A = np.asarray(A)
    m, n = A.shape
    k = int(k)
    if k <= 0:
        return np.empty((m, 0)), np.empty((0,)), np.empty((n, 0))
    s = 5
    if B is None:
        rng = np.random.default_rng()
        B = rng.standard_normal((n, k + s))
    else:
        B = np.asarray(B)
    C = A @ B
    Q, _ = np.linalg.qr(C, mode="reduced")
    for _ in range(int(p)):
        Q, _ = np.linalg.qr(A.T @ Q, mode="reduced")
        Q, _ = np.linalg.qr(A @ Q, mode="reduced")
    B_small = Q.T @ A
    Ut, svals, Vh = np.linalg.svd(B_small, full_matrices=False)
    U = Q @ Ut
    U = U[:, :k]
    svals = svals[:k]
    V = Vh[:k, :].T
    return U, svals, V


def _basicrSVDd_torch(
    A: torch.Tensor,
    k: int,
    p: int,
    B: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Torch variant of basicrSVDd.m for randomized SVD.
    A_t = A if torch.is_tensor(A) else torch.as_tensor(A)
    m, n = A_t.shape
    k = int(k)
    if k <= 0:
        empty = torch.empty((m, 0), device=A_t.device, dtype=A_t.dtype)
        return empty, torch.empty((0,), device=A_t.device, dtype=A_t.dtype), torch.empty((n, 0), device=A_t.device, dtype=A_t.dtype)
    s = 5
    if B is None:
        rng = np.random.default_rng()

        B = rng.standard_normal((n, k + s))
        B = torch.tensor(B,device=A_t.device, dtype=A_t.dtype)
        #B = torch.randn((n, k + s), device=A_t.device, dtype=A_t.dtype)
    C = A_t @ B
    Q, _ = torch.linalg.qr(C, mode="reduced")
    for _ in range(int(p)):
        Q, _ = torch.linalg.qr(A_t.T @ Q, mode="reduced")
        Q, _ = torch.linalg.qr(A_t @ Q, mode="reduced")
    B_small = Q.T @ A_t
    Ut, svals, Vh = torch.linalg.svd(B_small, full_matrices=False)
    U = Q @ Ut
    U = U[:, :k]
    svals = svals[:k]
    V = Vh[:k, :].transpose(-2, -1)
    return U, svals, V


def _spacing_torch(value: torch.Tensor) -> torch.Tensor:
    # Torch equivalent of numpy.spacing for a scalar tensor.
    return torch.nextafter(value, value + torch.ones_like(value)) - value


def _leave2ind(sigma_indices: Sequence[int], n: Sequence[int]) -> np.ndarray:
    # Port of leave2ind.m (1-based indices).
    d = len(n)
    sigma_indices = [int(val) for val in sigma_indices]
    if d == 2:
        indices = np.ones((len(sigma_indices), 2), dtype=int)
        indices[:, 1] = np.arange(1, len(sigma_indices) + 1)
        return indices

    r = [1] * d
    nodesperlevel = [1] * d
    endI = [1] * d
    for i in range(1, d):
        r[i] = min(int(n[i - 1]), _prod_int(n[i:]))
        nodesperlevel[i] = _prod_int(r[: i + 1])
        endI[i] = sum(nodesperlevel[: i + 1])

    endI = endI[:-2]
    k_vals = sigma_indices[:]
    l_vals = [0] * len(sigma_indices)
    indices = np.zeros((len(sigma_indices), 2 * (len(n) - 1)), dtype=int)

    for j in range(len(sigma_indices)):
        if k_vals[j] % r[-1] == 0:
            k_vals[j] = k_vals[j] // r[-1]
            l_vals[j] = r[-1]
        else:
            l_vals[j] = k_vals[j] % r[-1]
            k_vals[j] = int(np.ceil(k_vals[j] / r[-1]))
        indices[j, 0] = endI[-1] + k_vals[j]
        indices[j, 1] = l_vals[j]

    endI = endI[:-1]
    r = r[:-1]
    t_counter = 2
    for _ in range(len(endI)):
        for j in range(len(sigma_indices)):
            if k_vals[j] % r[-1] == 0:
                k_vals[j] = k_vals[j] // r[-1]
                l_vals[j] = r[-1]
            else:
                l_vals[j] = k_vals[j] % r[-1]
                k_vals[j] = int(np.ceil(k_vals[j] / r[-1]))
            indices[j, t_counter] = endI[-1] + k_vals[j]
            indices[j, t_counter + 1] = l_vals[j]
        t_counter += 2
        endI = endI[:-1]
        r = r[:-1]

    for j in range(len(sigma_indices)):
        if k_vals[j] % r[-1] == 0:
            l_vals[j] = r[-1]
        else:
            l_vals[j] = k_vals[j] % r[-1]
        indices[j, t_counter] = 1
        indices[j, t_counter + 1] = l_vals[j]
    return indices


def _UV2Uhat_numpy(
    U: np.ndarray,
    V: np.ndarray,
    sigma_indices: Sequence[int],
    n: Sequence[int],
    n2: Optional[Sequence[int]] = None,
) -> List[np.ndarray]:
    # Convert TTr1 tree indices into mode matrices (NumPy).
    indices = _leave2ind(sigma_indices, n2 if n2 is not None else n)
    d = len(n2) if n2 is not None else len(n)
    sample = next((arr for arr in U if arr is not None), None)
    dtype = sample.dtype if sample is not None else float
    num_terms = len(sigma_indices)
    if n2 is not None:
        Uhat = [np.zeros((int(n2[i]), int(n[i])), dtype=dtype) for i in range(d)]
    else:
        Uhat = [np.zeros((int(n[i]), num_terms), dtype=dtype) for i in range(d)]
    total_cols = indices.shape[1] if indices.size else 0
    for term in range(num_terms):
        row = indices[term]
        for j in range(1, d):
            cell_col = total_cols - 2 * j
            vec_col = total_cols - (2 * j - 1)
            cell_idx = int(row[cell_col]) - 1
            vec_idx = int(row[vec_col]) - 1
            Uhat[j - 1][:, term] = U[cell_idx][:, vec_idx]
        cell_idx = int(row[0]) - 1
        vec_idx = int(row[1]) - 1
        Uhat[d - 1][:, term] = V[cell_idx][:, vec_idx]
    return Uhat


def _UV2Uhat_torch(
    U: torch.Tensor,
    V: torch.Tensor,
    sigma_indices: Sequence[int],
    n: Sequence[int],
    n2: Optional[Sequence[int]] = None,
) -> List[torch.Tensor]:
    # Convert TTr1 tree indices into mode matrices (Torch).
    indices = _leave2ind(sigma_indices, n2 if n2 is not None else n)
    d = len(n2) if n2 is not None else len(n)
    sample = next((arr for arr in U if arr is not None), None)
    if sample is None:
        sample = next((arr for arr in V if arr is not None), None)
    if sample is None:
        raise ValueError("UV2Uhat_torch expected non-empty U or V tensors.")
    num_terms = len(sigma_indices)
    if n2 is not None:
        Uhat = [torch.zeros((int(n2[i]), int(n[i])), device=sample.device, dtype=sample.dtype) for i in range(d)]
    else:
        Uhat = [torch.zeros((int(n[i]), num_terms), device=sample.device, dtype=sample.dtype) for i in range(d)]
    total_cols = indices.shape[1] if indices.size else 0
    for term in range(num_terms):
        row = indices[term]
        for j in range(1, d):
            cell_col = total_cols - 2 * j
            vec_col = total_cols - (2 * j - 1)
            cell_idx = int(row[cell_col]) - 1
            vec_idx = int(row[vec_col]) - 1
            Uhat[j - 1][:, term] = U[cell_idx][:, vec_idx]
        cell_idx = int(row[0]) - 1
        vec_idx = int(row[1]) - 1
        Uhat[d - 1][:, term] = V[cell_idx][:, vec_idx]
    return Uhat


def _ttr1svdz_numpy(A: np.ndarray):
    # Full TTr1SVD with pruning of numerically zero branches (NumPy).
    data = np.asarray(A)
    n = list(data.shape)
    d = len(n)
    if d == 0:
        raise ValueError("Input tensor must have at least one dimension.")
    r = [min(n[i], _prod_int(n[i + 1 :])) for i in range(d - 1)]
    totalsvd = 1
    for i in range(1, len(r)):
        totalsvd += _prod_int(r[:i])
    nleaf = _prod_int(r) if r else 1
    U: List[Optional[np.ndarray]] = [None] * totalsvd
    S: List[Optional[np.ndarray]] = [None] * totalsvd
    V: List[Optional[np.ndarray]] = [None] * totalsvd
    first = _reshape_fortran_numpy(data, (n[0], _prod_int(n[1:])))
    U0, s0, V0 = _svd_econ_numpy(first)
    U[0] = U0
    S[0] = s0
    V[0] = V0
    counter = 1
    whichvcounter = 0
    sigmas = np.kron(s0, np.ones(int(nleaf / len(s0))))
    tol = _prod_int(n) * np.spacing(np.max(s0))
    for level in range(len(r) - 1):
        Slevel: List[np.ndarray] = []
        level_count = _prod_int(r[: level + 1])
        for j in range(level_count):
            j1 = j + 1
            col = r[level] if j1 % r[level] == 0 else j1 % r[level]
            S_current = S[whichvcounter]
            if S_current is not None and col <= len(S_current) and S_current[col - 1] > tol:
                vec = V[whichvcounter][:, col - 1]
                next_shape = (n[level + 1], _prod_int(n[level + 2 :]))
                mat = _reshape_fortran_numpy(vec, next_shape)
                Ut, st, Vt = _svd_econ_numpy(mat)
                U[counter] = Ut
                S[counter] = st
                V[counter] = Vt
                Slevel.append(st)
            else:
                Slevel.append(np.zeros(min(n[level + 1], _prod_int(n[level + 2 :])), dtype=data.dtype))
            counter += 1
            if j1 % n[level] == 0:
                whichvcounter += 1
        Slevel_vec = np.concatenate(Slevel) if Slevel else np.zeros((0,), dtype=data.dtype)
        if Slevel_vec.size:
            Slevel_vec = np.kron(Slevel_vec, np.ones(int(nleaf / len(Slevel_vec))))
            sigmas = sigmas * Slevel_vec
    return U, S, V, sigmas


def _ttr1svdz_torch(A: torch.Tensor):
    # Full TTr1SVD with pruning of numerically zero branches (Torch).
    data = A if torch.is_tensor(A) else torch.as_tensor(A)
    n = list(data.shape)
    d = len(n)
    if d == 0:
        raise ValueError("Input tensor must have at least one dimension.")
    r = [min(n[i], _prod_int(n[i + 1 :])) for i in range(d - 1)]
    totalsvd = 1
    for i in range(1, len(r)):
        totalsvd += _prod_int(r[:i])
    nleaf = _prod_int(r) if r else 1
    U: List[Optional[torch.Tensor]] = [None] * totalsvd
    S: List[Optional[torch.Tensor]] = [None] * totalsvd
    V: List[Optional[torch.Tensor]] = [None] * totalsvd
    first = _reshape_fortran_torch(data.contiguous(), (n[0], _prod_int(n[1:])))
    U0, s0, V0 = _svd_econ_torch(first)
    U[0] = U0
    S[0] = s0
    V[0] = V0
    counter = 1
    whichvcounter = 0
    sigmas = torch.kron(s0, torch.ones(int(nleaf / len(s0)), device=data.device, dtype=data.dtype))
    tol = _prod_int(n) * _spacing_torch(torch.max(s0))
    for level in range(len(r) - 1):
        Slevel: List[torch.Tensor] = []
        level_count = _prod_int(r[: level + 1])
        for j in range(level_count):
            j1 = j + 1
            col = r[level] if j1 % r[level] == 0 else j1 % r[level]
            S_current = S[whichvcounter]
            if S_current is not None and col <= int(S_current.numel()) and S_current[col - 1] > tol:
                vec = V[whichvcounter][:, col - 1]
                next_shape = (n[level + 1], _prod_int(n[level + 2 :]))
                mat = _reshape_fortran_torch(vec, next_shape)
                Ut, st, Vt = _svd_econ_torch(mat)
                U[counter] = Ut
                S[counter] = st
                V[counter] = Vt
                Slevel.append(st)
            else:
                zeros_len = min(n[level + 1], _prod_int(n[level + 2 :]))
                Slevel.append(torch.zeros(zeros_len, device=data.device, dtype=data.dtype))
            counter += 1
            if j1 % n[level] == 0:
                whichvcounter += 1
        if Slevel:
            Slevel_vec = torch.cat(Slevel)
            Slevel_vec = torch.kron(Slevel_vec, torch.ones(int(nleaf / len(Slevel_vec)), device=data.device, dtype=data.dtype))
            sigmas = sigmas * Slevel_vec
    return U, S, V, sigmas


def _rttr1svds_numpy(A: np.ndarray, R: int):
    # Sparse/randomized TTr1SVD variant used by rtkpsvd (NumPy).
    data = np.asarray(A)
    n = list(data.shape)
    d = len(n)
    if d == 0:
        raise ValueError("Input tensor must have at least one dimension.")
    r = [min(n[i], int(R)) for i in range(d - 1)]
    totalsvd = 1
    for i in range(1, len(r)):
        totalsvd += _prod_int(r[:i])
    nleaf = _prod_int(r) if r else 1
    U: List[Optional[np.ndarray]] = [None] * totalsvd
    S: List[Optional[np.ndarray]] = [None] * totalsvd
    V: List[Optional[np.ndarray]] = [None] * totalsvd
    first = _reshape_fortran_numpy(data, (n[0], _prod_int(n[1:])))
    U0, s0, V0 = _basicrSVDd_numpy(first, int(R), 1)
    U[0] = U0
    S[0] = s0
    V[0] = V0
    counter = 1
    whichvcounter = 0
    sigmas = np.kron(s0, np.ones(int(nleaf / len(s0))))
    tol = _prod_int(n) * np.spacing(np.max(s0))
    for level in range(len(r) - 1):
        Slevel: List[np.ndarray] = []
        level_count = _prod_int(r[: level + 1])
        for j in range(level_count):
            j1 = j + 1
            col = r[level] if j1 % r[level] == 0 else j1 % r[level]
            S_current = S[whichvcounter]
            if S_current is not None and col <= len(S_current) and S_current[col - 1] > tol:
                vec = V[whichvcounter][:, col - 1]
                next_shape = (n[level + 1], _prod_int(n[level + 2 :]))
                mat = _reshape_fortran_numpy(vec, next_shape)
                Ut, st, Vt = _basicrSVDd_numpy(mat, int(R), 1)
                U[counter] = Ut
                S[counter] = st
                V[counter] = Vt
                Slevel.append(st)
            else:
                Slevel.append(np.zeros(int(R), dtype=data.dtype))
            counter += 1
            if j1 % len(S_current) == 0:
                whichvcounter += 1
        Slevel_vec = np.concatenate(Slevel) if Slevel else np.zeros((0,), dtype=data.dtype)
        if Slevel_vec.size:
            Slevel_vec = np.kron(Slevel_vec, np.ones(int(nleaf / len(Slevel_vec))))
            sigmas = sigmas * Slevel_vec
    return U, S, V, sigmas


def _rttr1svds_torch(A: torch.Tensor, R: int):
    # Sparse/randomized TTr1SVD variant used by rtkpsvd (Torch).
    data = A if torch.is_tensor(A) else torch.as_tensor(A)
    n = list(data.shape)
    d = len(n)
    if d == 0:
        raise ValueError("Input tensor must have at least one dimension.")
    r = [min(n[i], int(R)) for i in range(d - 1)]
    totalsvd = 1
    for i in range(1, len(r)):
        totalsvd += _prod_int(r[:i])
    nleaf = _prod_int(r) if r else 1
    U: List[Optional[torch.Tensor]] = [None] * totalsvd
    S: List[Optional[torch.Tensor]] = [None] * totalsvd
    V: List[Optional[torch.Tensor]] = [None] * totalsvd
    first = _reshape_fortran_torch(data.contiguous(), (n[0], _prod_int(n[1:])))
    U0, s0, V0 = _basicrSVDd_torch(first, int(R), 1)
    U[0] = U0
    S[0] = s0
    V[0] = V0
    counter = 1
    whichvcounter = 0
    sigmas = torch.kron(s0, torch.ones(int(nleaf / len(s0)), device=data.device, dtype=data.dtype))
    tol = _prod_int(n) * _spacing_torch(torch.max(s0))
    for level in range(len(r) - 1):
        Slevel: List[torch.Tensor] = []
        level_count = _prod_int(r[: level + 1])
        for j in range(level_count):
            j1 = j + 1
            col = r[level] if j1 % r[level] == 0 else j1 % r[level]
            S_current = S[whichvcounter]
            if S_current is not None and col <= int(S_current.numel()) and S_current[col - 1] > tol:
                vec = V[whichvcounter][:, col - 1]
                next_shape = (n[level + 1], _prod_int(n[level + 2 :]))
                mat = _reshape_fortran_torch(vec, next_shape)
                Ut, st, Vt = _basicrSVDd_torch(mat, int(R), 1)
                U[counter] = Ut
                S[counter] = st
                V[counter] = Vt
                Slevel.append(st)
            else:
                Slevel.append(torch.zeros(int(R), device=data.device, dtype=data.dtype))
            counter += 1
            if j1 % int(S_current.numel()) == 0:
                whichvcounter += 1
        if Slevel:
            Slevel_vec = torch.cat(Slevel)
            Slevel_vec = torch.kron(Slevel_vec, torch.ones(int(nleaf / len(Slevel_vec)), device=data.device, dtype=data.dtype))
            sigmas = sigmas * Slevel_vec
    return U, S, V, sigmas


def rtkpsvd_numpy(A: np.ndarray, n: Sequence[int], R: Optional[int] = None):
    # NumPy implementation of rtkpsvd.m (TKPSVD with optional sparse TTr1).
    data = np.asarray(A)
    n = [int(val) for val in n]
    dim = data.shape
    d = len(dim)
    if len(n) % d != 0:
        raise ValueError("Not all factors have the same order.")
    numfac = len(n) // d
    noe: List[int] = []
    for i in range(d):
        noe.extend(n[i::d])
    data = _reshape_fortran_numpy(data, noe)
    permuted: List[int] = []
    for i in range(numfac):
        permuted.extend(list(range(i, d * numfac, numfac)))
    data = np.transpose(data, permuted)
    n_perm = [noe[idx] for idx in permuted]
    n2: List[int] = []
    for i in range(numfac):
        block = n_perm[i * d : (i + 1) * d]
        n2.append(_prod_int(block))
    data = _reshape_fortran_numpy(data, n2)
    if R is not None:
        U, S, V, sigmas = _rttr1svds_numpy(data, int(R))
    else:
        U, S, V, sigmas = _ttr1svdz_numpy(data)
    if sigmas.size == 0:
        raise ValueError("rtkpsvd returned no singular values.")
    tol = _prod_int(n2) * np.spacing(np.abs(sigmas[0]))
    sigma_indices = (np.nonzero(sigmas > tol)[0] + 1).tolist()
    if R is not None:
        reduced_dims = [int(R)] * len(n2)
        Uhat = _UV2Uhat_numpy(U, V, sigma_indices, reduced_dims, n2)
    else:
        Uhat = _UV2Uhat_numpy(U, V, sigma_indices, n2)
    num_terms = Uhat[0].shape[1] if Uhat else 0
    B: List[List[np.ndarray]] = []
    for j in range(numfac):
        block_dims = n[j * d : (j + 1) * d]
        row: List[np.ndarray] = []
        for i in range(num_terms):
            vec = Uhat[j][:, i]
            reshaped = _reshape_fortran_numpy(vec, block_dims)
            row.append(reshaped)
        B.append(row)
    return B, sigmas


def rtkpsvd_torch(A: torch.Tensor, n: Sequence[int], R: Optional[int] = None):
    # Torch implementation of rtkpsvd.m (TKPSVD with optional sparse TTr1).
    data = A if torch.is_tensor(A) else torch.as_tensor(A)
    n = [int(val) for val in n]
    dim = list(data.shape)
    d = len(dim)
    if len(n) % d != 0:
        raise ValueError("Not all factors have the same order.")
    numfac = len(n) // d
    noe: List[int] = []
    for i in range(d):
        noe.extend(n[i::d])
    data = _reshape_fortran_torch(data.contiguous(), noe)
    permuted: List[int] = []
    for i in range(numfac):
        permuted.extend(list(range(i, d * numfac, numfac)))
    data = data.permute(permuted).contiguous()
    n_perm = [noe[idx] for idx in permuted]
    n2: List[int] = []
    for i in range(numfac):
        block = n_perm[i * d : (i + 1) * d]
        n2.append(_prod_int(block))
    data = _reshape_fortran_torch(data, n2)
    if R is not None:
        U, S, V, sigmas = _rttr1svds_torch(data, int(R))
    else:
        U, S, V, sigmas = _ttr1svdz_torch(data)
    if sigmas.numel() == 0:
        raise ValueError("rtkpsvd returned no singular values.")
    tol = _prod_int(n2) * _spacing_torch(torch.abs(sigmas[0]))
    sigma_indices = (sigmas > tol).nonzero(as_tuple=False).squeeze(-1) + 1
    sigma_indices_list = [int(val.item()) for val in sigma_indices]
    if R is not None:
        reduced_dims = [int(R)] * len(n2)
        Uhat = _UV2Uhat_torch(U, V, sigma_indices_list, reduced_dims, n2)
    else:
        Uhat = _UV2Uhat_torch(U, V, sigma_indices_list, n2)
    num_terms = Uhat[0].shape[1] if Uhat else 0
    B: List[List[torch.Tensor]] = []
    for j in range(numfac):
        block_dims = n[j * d : (j + 1) * d]
        row: List[torch.Tensor] = []
        for i in range(num_terms):
            vec = Uhat[j][:, i]
            reshaped = _reshape_fortran_torch(vec, block_dims)
            row.append(reshaped)
        B.append(row)
    return B, sigmas


def rtkpsvd(A, n: Sequence[int], R: Optional[int] = None):
    # Dispatch to NumPy or Torch implementation based on input type.
    if torch.is_tensor(A):
        return rtkpsvd_torch(A, n, R)
    return rtkpsvd_numpy(A, n, R)

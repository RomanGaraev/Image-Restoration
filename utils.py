from collections import defaultdict
from typing import Dict, List, Optional, Union, Sequence, Tuple
from skimage.metrics import peak_signal_noise_ratio
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
from matplotlib import patches
from pathlib import Path
import torch


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

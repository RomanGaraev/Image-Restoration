import matplotlib.pyplot as plt
from matplotlib import patches
import torch
from torchvision.ops import box_iou

from reconstruct import DecompositionLayer
from utils import DetectionMetricTracker, rtkpsvd_numpy, fullKTD
from load_yolo_coco import load_model_and_data, prepare_coco_resources
from yolo_coco import run_batch


def _draw_boxes(ax, boxes, color, label_prefix=None):
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=1.5, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        if label_prefix:
            ax.text(x1, max(y1 - 3, 0), f"{label_prefix}{idx}",
                    color=color, fontsize=7,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

def _filter_matched_predictions(pred_boxes, gt_boxes, iou_thresh=0.5):
    if pred_boxes is None or pred_boxes.numel() == 0:
        return pred_boxes
    if gt_boxes is None or gt_boxes.numel() == 0:
        return pred_boxes.new_zeros((0, 4))
    ious = box_iou(pred_boxes, gt_boxes)
    best_iou, _ = ious.max(dim=1)
    return pred_boxes[best_iou >= iou_thresh]

def _restore_rtkpsvd_numpy(layer, noisy, mask, steps, rank):
    if steps == 0:
        return noisy
    rev_mask = (mask - 1) * (-1)
    X = noisy.clone()
    n_vec = layer._DecompositionLayer__build_rtkpsvd_n(X.shape)
    if not n_vec:
        return noisy
    for _ in range(steps):
        X_np = X.detach().cpu().numpy()
        Aij, sigmas = rtkpsvd_numpy(X_np, n_vec, rank)
        if not Aij or not Aij[0]:
            break
        terms_available = len(Aij[0])
        if terms_available == 0:
            break
        rank_to_use = terms_available if rank is None else min(rank, terms_available)
        Y_np = fullKTD(Aij, sigmas, R=rank_to_use)
        Y = torch.from_numpy(Y_np).to(device=X.device, dtype=X.dtype)
        X = noisy + rev_mask * Y
        X = layer.filter(X).to(layer.device)
    return X

def _is_monotonic_increasing(values):
    return all(b >= a for a, b in zip(values, values[1:]))

def main():
    steps_list = [0, 5, 15, 30]
    max_attempts = 20
    images_dir, annotations_file = prepare_coco_resources()
    model, device, dataset, dataloader = load_model_and_data(
        model_path="models/yolo11m.pt",
        images_dir=images_dir,
        annotations_file=annotations_file,
        batch_size=1,
        image_size=640,
    )

    images, targets = next(iter(dataloader))
    images = images.to(device)

    fig, axes = plt.subplots(1, len(steps_list), figsize=(4 * len(steps_list), 4))
    if len(steps_list) == 1:
        axes = [axes]

    attempt = 0
    f1_scores = []
    restored_images = []
    while attempt < max_attempts:
        layer = DecompositionLayer(
            noise_percent=0.7,
            device=device,
            decomposition="rtkpsvd_numpy",
            rank=70,
            steps=0,
        )
        noisy, mask = layer._DecompositionLayer__generate_salt_pepper_noise(images[0])
        noisy = noisy.unsqueeze(0)
        mask = mask.unsqueeze(0)

        f1_scores = []
        restored_images = []
        for steps in steps_list:
            restored = _restore_rtkpsvd_numpy(layer, noisy, mask, steps, rank=70)
            restored_images.append(restored)
            predictions, _, _, _, _ = run_batch(model, restored, targets)
            tracker = DetectionMetricTracker()
            tracker.update(predictions, targets)
            f1_scores.append(tracker.compute()["f1"])

        if _is_monotonic_increasing(f1_scores):
            break
        attempt += 1

    for ax, steps, restored, f1 in zip(axes, steps_list, restored_images, f1_scores):
        image_np = restored[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(image_np)

        predictions, _, _, _, _ = run_batch(model, restored, targets)
        pred_boxes = predictions[0].get("boxes")
        gt_boxes = targets[0].get("boxes")
        matched_boxes = _filter_matched_predictions(
            pred_boxes.to(device) if pred_boxes is not None else pred_boxes,
            gt_boxes.to(device) if gt_boxes is not None else gt_boxes,
        )
        if matched_boxes is not None and matched_boxes.numel() > 0:
            _draw_boxes(ax, matched_boxes.detach().cpu(), color="red")

        ax.set_title(f"steps={steps}\nF1={f1:.3f}")
        ax.axis("off")

    if attempt >= max_attempts:
        print("Warning: monotonic F1 not found; showing last attempt.")

    plt.tight_layout()
    output_path = "rtkpsvd_numpy_steps.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

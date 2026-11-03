import argparse
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from load_yolo_coco import (
    COCODetectionDataset,
    build_label_mapping,
    coco_collate_fn,
    prepare_coco_resources,
)
from reconstruct import DecompositionLayer
from rfdetr import RFDETRLarge
from rfdetr.assets.coco_classes import COCO_CLASSES
from rfdetr.utilities import NestedTensor
from utils import DetectionMetricTracker


def _tensor_to_pil(image_chw: torch.Tensor) -> Image.Image:
    image_hwc = image_chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0)
    image_uint8 = (image_hwc * 255.0).to(torch.uint8).numpy()
    return Image.fromarray(image_uint8)


def _predict_batch_rfdetr(
    model: RFDETRLarge,
    images: torch.Tensor,
    threshold: float,
) -> List[Dict[str, torch.Tensor]]:
    predictions: List[Dict[str, torch.Tensor]] = []
    for image in images:
        detections = model.predict(_tensor_to_pil(image), threshold=threshold)

        if detections is None or getattr(detections, "xyxy", None) is None:
            predictions.append(
                {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.long),
                }
            )
            continue

        boxes = torch.as_tensor(detections.xyxy, dtype=torch.float32)
        scores = torch.as_tensor(detections.confidence, dtype=torch.float32)
        labels = torch.as_tensor(detections.class_id, dtype=torch.long)
        predictions.append({"boxes": boxes, "scores": scores, "labels": labels})
    return predictions


def _get_rfdetr_device(model: RFDETRLarge) -> torch.device:
    """Return the device that actually holds the detector weights."""
    core_model = model.model.model
    parameter = next(core_model.parameters(), None)
    if parameter is not None:
        return parameter.device

    buffer = next(core_model.buffers(), None)
    if buffer is not None:
        return buffer.device

    device = getattr(model.model, "device", torch.device("cpu"))
    return device if isinstance(device, torch.device) else torch.device(device)


def _move_rfdetr_model_to_device(model: RFDETRLarge, device: str | torch.device) -> torch.device:
    """Move RF-DETR's lazily placed PyTorch module before direct forward calls."""
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")

    model.model.model.to(target_device)
    # Keep RF-DETR's context in sync for its public predict() method.
    model.model.device = target_device
    return _get_rfdetr_device(model)


def _get_rfdetr_resolution(model: RFDETRLarge) -> int | tuple[int, int] | None:
    """Read the resolution used by RF-DETR's public prediction context.

    RF-DETR stores this on ``model.model`` (its ModelContext), not on the
    public RFDETRLarge wrapper. Using that same value keeps the differentiable
    attack forward pass aligned with ``model.predict`` used for metrics.
    """
    model_context = getattr(model, "model", None)
    model_config = getattr(model, "model_config", None)
    for resolution in (
        getattr(model_context, "resolution", None),
        getattr(model_config, "resolution", None),
        # Compatibility fallbacks for older RF-DETR versions/wrappers.
        getattr(model, "resolution", None),
        getattr(model, "imgsz", None),
        getattr(model, "image_size", None),
    ):
        if isinstance(resolution, int) and not isinstance(resolution, bool):
            return resolution
        if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
            return int(resolution[0]), int(resolution[1])
    return None


def _rfdetr_outputs(model: RFDETRLarge, images: torch.Tensor) -> Dict[str, torch.Tensor]:
    device = _get_rfdetr_device(model)
    means_value = getattr(model, "means", [0.485, 0.456, 0.406])
    stds_value = getattr(model, "stds", [0.229, 0.224, 0.225])
    means = torch.as_tensor(means_value, device=device, dtype=images.dtype).view(1, 3, 1, 1)
    stds = torch.as_tensor(stds_value, device=device, dtype=images.dtype).view(1, 3, 1, 1)
    normalized = (images.to(device) - means) / stds

    resolution = _get_rfdetr_resolution(model)

    if isinstance(resolution, int):
        target_size = (resolution, resolution)
    elif isinstance(resolution, (tuple, list)) and len(resolution) == 2:
        target_size = (int(resolution[0]), int(resolution[1]))
    else:
        target_size = normalized.shape[-2:]

    if tuple(normalized.shape[-2:]) != tuple(target_size):
        resized = F.interpolate(
            normalized,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    else:
        resized = normalized

    def _extract_outputs(outputs_obj) -> Dict[str, torch.Tensor]:
        outputs_norm = outputs_obj
        if isinstance(outputs_norm, tuple):
            outputs_norm = outputs_norm[-1]
        if not isinstance(outputs_norm, dict) or "pred_logits" not in outputs_norm:
            raise RuntimeError("RF-DETR outputs do not contain logits compatible with attacks.")
        return outputs_norm

    core_model = model.model.model
    batch_size, _, height, width = resized.shape
    mask = torch.zeros((batch_size, height, width), dtype=torch.bool, device=resized.device)
    samples = NestedTensor(resized, mask)
    try:
        return _extract_outputs(core_model(samples))
    except (TypeError, AttributeError):
        # Compatibility fallback for versions that only accept tensor input.
        return _extract_outputs(core_model(resized))


@torch.no_grad()
def _vanishing_diagnostics(model: RFDETRLarge, images: torch.Tensor) -> Dict[str, float]:
    """Measure the exact objective that the RF-DETR vanishing attack minimizes."""
    model_core = model.model.model
    was_training = model_core.training
    model_core.eval()
    try:
        logits = _rfdetr_outputs(model, images)["pred_logits"]
        probabilities = logits.sigmoid()
        return {
            "zero_target_bce": float(
                F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="mean").item()
            ),
            "mean_max_class_probability": float(probabilities.amax(dim=-1).mean().item()),
        }
    finally:
        if was_training:
            model_core.train()

def _build_rfdetr_attack_loss(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    attack_name: str,
    attack_targets: Dict[str, torch.Tensor],
) -> torch.Tensor:
    if attack_name == "vanishing":
        # RF-DETR uses a sigmoid class head with no background/no-object logit.
        # Driving every class logit toward zero is its equivalent of assigning
        # all anchor objectness targets to background in the original TOG code.
        return F.binary_cross_entropy_with_logits(
            pred_logits,
            torch.zeros_like(pred_logits),
            reduction="sum",
        )

    if attack_name == "fabrication":
        # A query's maximum class logit is the RF-DETR counterpart to an
        # objectness logit. Targeting it to one fabricates high-confidence
        # detections without imposing a particular class.
        objectness_surrogate = pred_logits.amax(dim=-1)
        return F.binary_cross_entropy_with_logits(
            objectness_surrogate,
            torch.ones_like(objectness_surrogate),
            reduction="sum",
        )

    query_mask = attack_targets["query_mask"]
    if not query_mask.any():
        return pred_logits.sum() * 0.0

    target_labels = attack_targets["labels"]
    target_boxes = attack_targets["boxes"]
    classification_loss = F.cross_entropy(
        pred_logits[query_mask], target_labels[query_mask], reduction="sum"
    )
    localization_loss = F.l1_loss(
        pred_boxes[query_mask], target_boxes[query_mask], reduction="sum"
    )

    if attack_name in {"untargeted_fgsm", "untargeted_multistep_fgsm"}:
        # TOG-untargeted minimizes the negative clean-detection loss, thereby
        # increasing the loss of the clean query labels and boxes.
        return -(classification_loss + localization_loss)

    if attack_name in {"mislabeling_ml", "mislabeling_ll"}:
        # The target labels/boxes are frozen from the clean image, as in TOG.
        return classification_loss + localization_loss

    raise ValueError(f"Unsupported attack '{attack_name}'")


def _build_rfdetr_attack_targets(
    model: RFDETRLarge,
    images: torch.Tensor,
    attack_name: str,
    conf_threshold: float,
) -> Dict[str, torch.Tensor]:
    """Freeze clean-image query targets before the TOG optimization loop."""
    model_core = model.model.model
    was_training = model_core.training
    model_core.eval()
    with torch.no_grad():
        clean_outputs = _rfdetr_outputs(model, images)
    if was_training:
        model_core.train()

    clean_logits = clean_outputs["pred_logits"].detach()
    clean_boxes = clean_outputs["pred_boxes"].detach()
    query_mask = clean_logits.sigmoid().amax(dim=-1) > conf_threshold
    labels = clean_logits.argmax(dim=-1)

    if attack_name == "mislabeling_ll":
        labels = clean_logits.argmin(dim=-1)
    elif attack_name == "mislabeling_ml":
        class_probs = torch.softmax(clean_logits, dim=-1)
        masked_logits = clean_logits.masked_fill(class_probs > conf_threshold, float("-inf"))
        labels = masked_logits.argmax(dim=-1)
        fallback = torch.isinf(masked_logits).all(dim=-1)
        labels = torch.where(fallback, clean_logits.argmin(dim=-1), labels)

    return {
        "query_mask": query_mask,
        "labels": labels.detach(),
        "boxes": clean_boxes,
    }


def _compute_rfdetr_gradient(
    model: RFDETRLarge,
    images: torch.Tensor,
    attack_name: str,
    attack_targets: Dict[str, torch.Tensor],
) -> torch.Tensor:
    model_core = model.model.model
    was_training = model_core.training
    model_core.eval()

    inputs = images.detach().clone().to(_get_rfdetr_device(model))
    inputs.requires_grad_(True)
    outputs = _rfdetr_outputs(model, inputs)
    loss = _build_rfdetr_attack_loss(
        outputs["pred_logits"],
        outputs["pred_boxes"],
        attack_name,
        attack_targets,
    )

    model_core.zero_grad(set_to_none=True)
    if inputs.grad is not None:
        inputs.grad.zero_()
    loss.backward()
    grad = inputs.grad.detach().clone()
    model_core.zero_grad(set_to_none=True)

    if was_training:
        model_core.train()
    return grad


def _run_rfdetr_attack(
    model: RFDETRLarge,
    images: torch.Tensor,
    attack_name: str,
    eps: float,
    eps_iter: float,
    n_iter: int,
    conf_threshold: float,
) -> torch.Tensor:
    x_query = images.detach().clone().to(_get_rfdetr_device(model))
    original = x_query.detach()
    iterations = 1 if attack_name == "untargeted_fgsm" else n_iter
    attack_targets = _build_rfdetr_attack_targets(
        model, original, attack_name, conf_threshold
    )

    # Use the same random-start projected L-infinity procedure as yolo_coco.py.
    eta = torch.empty_like(original).uniform_(-eps, eps)
    x_adv = torch.clamp(original + eta, 0.0, 1.0)
    for _ in range(iterations):
        grad = _compute_rfdetr_gradient(model, x_adv, attack_name, attack_targets)
        # The reference TOG implementation minimizes every attack objective.
        # Keep the one-step helper's epsilon-sized update for FGSM.
        step_size = eps if attack_name == "untargeted_fgsm" else eps_iter
        x_adv = x_adv - step_size * grad.sign()
        x_adv = torch.max(torch.min(x_adv, original + eps), original - eps).clamp(0.0, 1.0).detach()
    return x_adv.detach()


def _build_attack(
    attack_name: str,
    rfdetr_model: RFDETRLarge,
    eps: float,
    eps_iter: float,
    n_iter: int,
    conf_threshold: float,
) -> Callable[[torch.Tensor, List[Dict[str, torch.Tensor]]], torch.Tensor]:
    supported = {
        "untargeted_fgsm",
        "untargeted_multistep_fgsm",
        "vanishing",
        "fabrication",
        "mislabeling_ml",
        "mislabeling_ll",
    }
    if attack_name not in supported:
        raise ValueError(f"Unsupported attack '{attack_name}'")
    return lambda images, targets: _run_rfdetr_attack(
        model=rfdetr_model,
        images=images,
        attack_name=attack_name,
        eps=eps,
        eps_iter=eps_iter,
        n_iter=n_iter,
        conf_threshold=conf_threshold,
    )


def _create_dataloader(
    images_dir: str,
    annotations_file: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = COCODetectionDataset(
        images_dir=images_dir, annotations_file=annotations_file, image_size=image_size
    )
    label_mapping = build_label_mapping(dataset.coco, COCO_CLASSES)
    dataset.set_label_mapping(label_mapping)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=coco_collate_fn,
    )


def _evaluate_single_attack(
    dataloader: DataLoader,
    rfdetr_model: RFDETRLarge,
    device: torch.device,
    attack_name: str,
    args: argparse.Namespace,
) -> None:
    attack_fn = _build_attack(
        attack_name,
        rfdetr_model,
        args.eps,
        args.eps_iter,
        args.n_iter,
        args.det_threshold,
    )
    decompose = DecompositionLayer(
        noise_percent=args.noise_percent,
        device=device,
        decomposition=args.decomposition,
        rank=args.rank,
        steps=args.steps,
    )

    meter_clean = DetectionMetricTracker(
        iou_threshold=args.iou_threshold, score_threshold=args.score_threshold
    )
    meter_adv = DetectionMetricTracker(
        iou_threshold=args.iou_threshold, score_threshold=args.score_threshold
    )
    meter_restored = DetectionMetricTracker(
        iou_threshold=args.iou_threshold, score_threshold=args.score_threshold
    )
    vanishing_clean_bce: List[float] = []
    vanishing_adv_bce: List[float] = []
    vanishing_clean_confidence: List[float] = []
    vanishing_adv_confidence: List[float] = []

    if args.max_batches > 0:
        total = min(len(dataloader), args.max_batches)
    else:
        total = len(dataloader)

    progress = tqdm(dataloader, total=total, desc=f"Attack={attack_name}")
    for batch_index, (images, targets) in enumerate(progress):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break

        images = images.to(device)
        images_adv = attack_fn(images, targets)
        images_restored = decompose(images_adv.detach().clone()).to(device)

        if attack_name == "vanishing":
            clean_diagnostics = _vanishing_diagnostics(rfdetr_model, images)
            adversarial_diagnostics = _vanishing_diagnostics(rfdetr_model, images_adv)
            vanishing_clean_bce.append(clean_diagnostics["zero_target_bce"])
            vanishing_adv_bce.append(adversarial_diagnostics["zero_target_bce"])
            vanishing_clean_confidence.append(clean_diagnostics["mean_max_class_probability"])
            vanishing_adv_confidence.append(adversarial_diagnostics["mean_max_class_probability"])

        predictions_clean = _predict_batch_rfdetr(
            rfdetr_model, images, threshold=args.det_threshold
        )
        predictions_adv = _predict_batch_rfdetr(
            rfdetr_model, images_adv, threshold=args.det_threshold
        )
        predictions_restored = _predict_batch_rfdetr(
            rfdetr_model, images_restored, threshold=args.det_threshold
        )

        meter_clean.update(predictions_clean, targets)
        meter_adv.update(predictions_adv, targets)
        meter_restored.update(predictions_restored, targets)

        f1_clean = meter_clean.compute()["f1"]
        f1_adv = meter_adv.compute()["f1"]
        f1_restored = meter_restored.compute()["f1"]
        status = f"F1 clean={f1_clean:.4f} adv={f1_adv:.4f} restored={f1_restored:.4f}"
        if attack_name == "vanishing":
            status += (
                f" BCE {clean_diagnostics['zero_target_bce']:.4f}"
                f"->{adversarial_diagnostics['zero_target_bce']:.4f}"
            )
        progress.set_postfix_str(status)

    print(f"Attack: {attack_name}")
    print(f"RF-DETR clean metrics: {meter_clean.compute()}")
    print(f"RF-DETR adversarial metrics: {meter_adv.compute()}")
    print(f"RF-DETR restored metrics: {meter_restored.compute()}")
    print(f"Decomposition params: {decompose.get_metadata()}")
    if attack_name == "vanishing" and vanishing_clean_bce:
        mean_clean_bce = sum(vanishing_clean_bce) / len(vanishing_clean_bce)
        mean_adv_bce = sum(vanishing_adv_bce) / len(vanishing_adv_bce)
        mean_clean_confidence = sum(vanishing_clean_confidence) / len(vanishing_clean_confidence)
        mean_adv_confidence = sum(vanishing_adv_confidence) / len(vanishing_adv_confidence)
        batches_decreased = sum(
            adv_bce < clean_bce for clean_bce, adv_bce in zip(vanishing_clean_bce, vanishing_adv_bce)
        )
        print(
            "RF-DETR vanishing diagnostic "
            f"(direct {_get_rfdetr_resolution(rfdetr_model)}px forward): "
            f"zero-target BCE {mean_clean_bce:.6f}->{mean_adv_bce:.6f}; "
            f"mean max class probability {mean_clean_confidence:.6f}->{mean_adv_confidence:.6f}; "
            f"BCE decreased in {batches_decreased}/{len(vanishing_clean_bce)} batches."
        )
        if mean_adv_bce >= mean_clean_bce:
            print("WARNING: the vanishing objective did not decrease on average; do not interpret the metrics as a successful vanishing attack.")


def evaluate(args: argparse.Namespace) -> None:
    images_dir, annotations_file = prepare_coco_resources(
        split=args.coco_split, annotation=args.coco_annotation
    )
    dataloader = _create_dataloader(
        images_dir=images_dir,
        annotations_file=annotations_file,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    rfdetr_kwargs = {}
    if args.device is not None:
        rfdetr_kwargs["device"] = args.device
    if args.rfdetr_weights:
        rfdetr_kwargs["pretrain_weights"] = args.rfdetr_weights
    rfdetr_model = RFDETRLarge(**rfdetr_kwargs)
    # RF-DETR places its module lazily for public inference methods.  This script
    # calls the underlying module directly to compute attack gradients, so place
    # those weights explicitly before creating CUDA input tensors.
    requested_device = args.device if args.device is not None else _get_rfdetr_device(rfdetr_model)
    device = _move_rfdetr_model_to_device(rfdetr_model, requested_device)
    if args.attack == "all":
        attack_names = [
            "untargeted_fgsm",
            "untargeted_multistep_fgsm",
            "vanishing",
            "fabrication",
            "mislabeling_ml",
            "mislabeling_ll",
        ]
    else:
        attack_names = [args.attack]

    for attack_name in attack_names:
        _evaluate_single_attack(
            dataloader=dataloader,
            rfdetr_model=rfdetr_model,
            device=device,
            attack_name=attack_name,
            args=args,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RFDETRLarge on COCO under TOG-style attacks generated with RFDETRLarge."
    )
    parser.add_argument("--rfdetr-weights", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--coco-split", type=str, default="val2017")
    parser.add_argument("--coco-annotation", type=str, default="instances_val2017.json")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=250)
    parser.add_argument("--attack", type=str, default="vanishing", choices=[
        "untargeted_fgsm",
        "untargeted_multistep_fgsm",
        "vanishing",
        "fabrication",
        "mislabeling_ml",
        "mislabeling_ll",
        "all",
    ])
    parser.add_argument("--eps", type=float, default=8 / 255.0)
    parser.add_argument("--eps-iter", type=float, default=2 / 255.0)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--det-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.25)

    parser.add_argument("--decomposition", type=str, default="tucker",
                        choices=["svd", "tucker", "rtkpsvd", "rtkpsvd_numpy"])
    parser.add_argument("--rank", type=int, default=70)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--noise-percent", type=float, default=0.7)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

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
from rfdetr.util.coco_classes import COCO_CLASSES
from rfdetr.util.misc import NestedTensor
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
    device = getattr(model.model, "device", torch.device("cpu"))
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _rfdetr_logits(model: RFDETRLarge, images: torch.Tensor) -> torch.Tensor:
    device = _get_rfdetr_device(model)
    means_value = getattr(model, "means", [0.485, 0.456, 0.406])
    stds_value = getattr(model, "stds", [0.229, 0.224, 0.225])
    means = torch.as_tensor(means_value, device=device, dtype=images.dtype).view(1, 3, 1, 1)
    stds = torch.as_tensor(stds_value, device=device, dtype=images.dtype).view(1, 3, 1, 1)
    normalized = (images.to(device) - means) / stds

    resolution = getattr(model, "resolution", None)
    if resolution is None:
        resolution = getattr(model, "imgsz", None)
    if resolution is None:
        resolution = getattr(model, "image_size", None)

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

    def _extract_pred_logits(outputs_obj) -> torch.Tensor:
        outputs_norm = outputs_obj
        if isinstance(outputs_norm, tuple):
            outputs_norm = outputs_norm[-1]
        if isinstance(outputs_norm, dict):
            if "pred_logits" in outputs_norm:
                return outputs_norm["pred_logits"]
            if "logits" in outputs_norm:
                return outputs_norm["logits"]
        raise RuntimeError("RF-DETR outputs do not contain logits compatible with attacks.")

    core_model = model.model.model
    batch_size, _, height, width = resized.shape
    mask = torch.zeros((batch_size, height, width), dtype=torch.bool, device=resized.device)
    samples = NestedTensor(resized, mask)
    try:
        return _extract_pred_logits(core_model(samples))
    except (TypeError, AttributeError):
        # Compatibility fallback for versions that only accept tensor input.
        return _extract_pred_logits(core_model(resized))


def _build_rfdetr_attack_loss(
    pred_logits: torch.Tensor,
    attack_name: str,
    conf_threshold: float = 0.25,
) -> torch.Tensor:
    if pred_logits.shape[-1] <= 1:
        raise RuntimeError("RF-DETR logits do not contain foreground classes.")

    foreground_logits = pred_logits[..., :-1]
    no_object_logits = pred_logits[..., -1]
    foreground_scores = foreground_logits.max(dim=-1).values

    if attack_name in {"untargeted_fgsm", "untargeted_multistep_fgsm"}:
        # Push predictions toward the no-object token and away from foreground classes.
        return (no_object_logits - foreground_scores).mean()

    if attack_name == "vanishing":
        # Stronger suppression of foreground confidence.
        return (no_object_logits - foreground_logits.logsumexp(dim=-1)).mean()

    if attack_name == "fabrication":
        # Increase foreground confidence and suppress no-object confidence.
        return (foreground_logits.logsumexp(dim=-1) - no_object_logits).mean()

    if attack_name in {"mislabeling_ml", "mislabeling_ll"}:
        probs = torch.softmax(foreground_logits, dim=-1)
        conf = probs.max(dim=-1).values
        active = conf > conf_threshold
        if not active.any():
            return pred_logits.sum() * 0.0

        logits_active = foreground_logits[active]
        if attack_name == "mislabeling_ll":
            target_class = logits_active.argmin(dim=-1)
        else:
            if logits_active.shape[-1] == 1:
                target_class = torch.zeros(logits_active.shape[0], dtype=torch.long, device=logits_active.device)
            else:
                top2 = torch.topk(logits_active, k=2, dim=-1).indices
                target_class = top2[:, 1]
        # Minimize CE toward wrong classes via gradient ascent on negative CE.
        return -F.cross_entropy(logits_active, target_class, reduction="mean")

    raise ValueError(f"Unsupported attack '{attack_name}'")


def _compute_rfdetr_gradient(
    model: RFDETRLarge,
    images: torch.Tensor,
    attack_name: str,
) -> torch.Tensor:
    model_core = model.model.model
    was_training = model_core.training
    model_core.eval()

    inputs = images.detach().clone().to(_get_rfdetr_device(model))
    inputs.requires_grad_(True)
    pred_logits = _rfdetr_logits(model, inputs)
    loss = _build_rfdetr_attack_loss(pred_logits, attack_name)

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
) -> torch.Tensor:
    x_query = images.detach().clone().to(_get_rfdetr_device(model))

    if attack_name == "untargeted_fgsm":
        grad = _compute_rfdetr_gradient(model, x_query, attack_name)
        return torch.clamp(x_query + eps * grad.sign(), 0.0, 1.0).detach()

    eta = torch.empty_like(x_query).uniform_(-eps, eps)
    x_adv = torch.clamp(x_query + eta, 0.0, 1.0)
    for _ in range(n_iter):
        grad = _compute_rfdetr_gradient(model, x_adv, attack_name)
        x_adv = torch.clamp(x_adv + eps_iter * grad.sign(), 0.0, 1.0)
        eta = torch.clamp(x_adv - x_query, -eps, eps)
        x_adv = torch.clamp(x_query + eta, 0.0, 1.0)
    return x_adv.detach()


def _build_attack(
    attack_name: str,
    rfdetr_model: RFDETRLarge,
    eps: float,
    eps_iter: float,
    n_iter: int,
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
    attack_fn = _build_attack(attack_name, rfdetr_model, args.eps, args.eps_iter, args.n_iter)
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
        progress.set_postfix_str(
            f"F1 clean={f1_clean:.4f} adv={f1_adv:.4f} restored={f1_restored:.4f}"
        )

    print(f"Attack: {attack_name}")
    print(f"RF-DETR clean metrics: {meter_clean.compute()}")
    print(f"RF-DETR adversarial metrics: {meter_adv.compute()}")
    print(f"RF-DETR restored metrics: {meter_restored.compute()}")
    print(f"Decomposition params: {decompose.get_metadata()}")


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
    device = _get_rfdetr_device(rfdetr_model)
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
    parser.add_argument("--device", type=str, default=None)
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

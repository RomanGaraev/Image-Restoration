from utils import DetectionMetricTracker, visualize_detections
from reconstruct import DecompositionLayer

from load_yolo_coco import DEFAULT_MODEL_NAME, load_model_and_data, prepare_coco_resources
from typing import Dict, List, Optional, Tuple
from torchvision.ops import box_iou
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.utils.loss import FocalLoss
from ultralytics.utils.loss import BboxLoss
from ultralytics.models.yolo import YOLO
from ultralytics.utils import nms


def run_batch(model: YOLO,
              images: torch.Tensor,
              true_labels: Optional[List[Dict[str, torch.Tensor]]] = None,
              auto_backward: bool = False,
              loss_type: str = "bbox") -> Tuple[List[Dict[str, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    device = getattr(model, "device", images.device)
    images = images.to(device).detach()
    images.requires_grad_(True)

    detector = model.model
    was_training = detector.training
    detector.eval()

    preds = detector(images)
    if isinstance(preds, tuple):
        preds = preds[0]
    preds_perm = preds.permute(0, 2, 1).contiguous()
    pred_xywh = preds_perm[..., :4]
    half_wh = pred_xywh[..., 2:] / 2
    pred_bboxes = torch.empty_like(pred_xywh)
    pred_bboxes[..., 0] = pred_xywh[..., 0] - half_wh[..., 0]
    pred_bboxes[..., 1] = pred_xywh[..., 1] - half_wh[..., 1]
    pred_bboxes[..., 2] = pred_xywh[..., 0] + half_wh[..., 0]
    pred_bboxes[..., 3] = pred_xywh[..., 1] + half_wh[..., 1]
    pred_bboxes = pred_bboxes.float()

    if was_training:
        detector.train()

    loss = None
    loss_items = None
    grad = None

    if true_labels is not None:
        batch_size, num_preds, _ = pred_bboxes.shape
        target_bboxes = torch.zeros_like(pred_bboxes)
        target_scores = torch.zeros((batch_size, num_preds, 1), device=device, dtype=pred_bboxes.dtype)
        fg_mask = torch.zeros((batch_size, num_preds), dtype=torch.bool, device=device)
        match_iou_threshold = 0.1

        for i, target in enumerate(true_labels):
            gt_boxes = target.get("boxes")
            if gt_boxes is None or gt_boxes.numel() == 0:
                continue
            gt_boxes = gt_boxes.to(device=device, dtype=pred_bboxes.dtype)
            ious = box_iou(pred_bboxes[i], gt_boxes)
            best_iou, best_idx = ious.max(dim=1)
            mask = best_iou > match_iou_threshold
            if mask.any():
                target_bboxes[i, mask] = gt_boxes[best_idx[mask]]
                target_scores[i, mask, 0] = best_iou[mask]
                fg_mask[i, mask] = True

        if loss_type == "bbox":
            bbox_loss = BboxLoss(reg_max=1).to(device)
            pred_dist = pred_bboxes.new_zeros((batch_size, num_preds, 4))
            anchor_points = pred_bboxes.new_zeros((num_preds, 2))
            target_scores_sum = target_scores.sum().clamp(min=1.0)

            if fg_mask.any():
                loss_iou, loss_dfl = bbox_loss(
                    pred_dist,
                    pred_bboxes,
                    anchor_points,
                    target_bboxes,
                    target_scores,
                    target_scores_sum,
                    fg_mask,
                )
                loss = loss_iou + loss_dfl
                loss_items = torch.stack((loss_iou.detach(), loss_dfl.detach()))
            else:
                loss = pred_bboxes.sum() * 0.0
                loss_items = torch.zeros(2, device=device)
        elif loss_type == "focal":
            focal_loss = FocalLoss().to(device)
            pred_objectness = preds_perm[..., 4:5]
            loss = focal_loss(pred_objectness, target_scores)
            loss_items = torch.stack((loss.detach(), torch.tensor(0.0, device=device, dtype=loss.dtype)))
        else:
            raise ValueError(f"Unsupported loss_type '{loss_type}'. Expected 'bbox' or 'focal'.")

        if auto_backward:
            detector.zero_grad(set_to_none=True)
            if images.grad is not None:
                images.grad.zero_()
            loss.backward()
            grad = images.grad.detach().clone()

    det_outputs = nms.non_max_suppression(
        preds.detach().float(),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=300,
        nc=getattr(detector.model[-1], "nc", 0),
    )

    predictions: List[Dict[str, torch.Tensor]] = []
    for det in det_outputs:
        if det is None or det.numel() == 0:
            predictions.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.long),
            })
            continue
        predictions.append({
            "boxes": det[:, :4].detach().cpu(),
            "scores": det[:, 4].detach().cpu(),
            "labels": det[:, 5].to(torch.long).detach().cpu(),
        })

    return predictions, grad, loss, loss_items, images


def build_object_vanishing_loss(preds_perm: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy on objectness logits with the zero mask used for vanishing attacks."""
    objectness_logits = preds_perm[..., 4:5]
    target_object_mask = torch.zeros_like(objectness_logits)
    return F.binary_cross_entropy_with_logits(objectness_logits, target_object_mask, reduction="sum")


def compute_object_vanishing_gradient(model: YOLO, images: torch.Tensor) -> torch.Tensor:
    """Compute gradients that suppress objectness predictions similar to the Keras implementation."""
    device = getattr(model, "device", images.device)
    detector = model.model
    was_training = detector.training
    detector.eval()

    inputs = images.detach().clone().to(device)
    inputs.requires_grad_(True)
    preds = detector(inputs)
    if isinstance(preds, tuple):
        preds = preds[0]
    preds_perm = preds.permute(0, 2, 1).contiguous()
    loss = build_object_vanishing_loss(preds_perm)

    detector.zero_grad(set_to_none=True)
    if inputs.grad is not None:
        inputs.grad.zero_()
    loss.backward()
    grad = inputs.grad.detach().clone()

    if was_training:
        detector.train()
    detector.zero_grad(set_to_none=True)
    return grad


def tog_untargeted_fgsm(model: YOLO, x_query, targets, eps=8/255.):
    _, data_grad, _, _, _ = run_batch(model, x_query, targets, auto_backward=True, loss_type="bbox")
    x_adv = torch.clamp(x_query + eps * data_grad.sign(), 0.0, 1.0)
    return x_adv.detach()


def tog_untargeted_multistep_fgsm(model: YOLO, x_query, targets, n_iter=10, eps=8/255., eps_iter=8/255.):
    eta = np.random.uniform(-eps, eps, size=x_query.shape).astype(np.float32)
    eta = torch.tensor(eta).to(x_query.device)
    x_adv = torch.clip(x_query + eta, 0.0, 1.0)
    for iter in range(n_iter):      
        _, data_grad, _, _, _ = run_batch(model, x_adv, targets, auto_backward=True)
        x_adv = torch.clamp(x_query + eps_iter * data_grad.sign(), 0.0, 1.0)
        eta = torch.clip(x_adv - x_query, -eps, eps)
        x_adv = torch.clip(x_query + eta, 0.0, 1.0)
    return x_adv.detach()


def tog_vanishing(model: YOLO, x_query, targets, n_iter=10, eps=8/255., eps_iter=8/255.):
    eta = np.random.uniform(-eps, eps, size=x_query.shape).astype(np.float32)
    eta = torch.tensor(eta, device=x_query.device)
    x_adv = torch.clamp(x_query + eta, 0.0, 1.0)
    for _ in range(n_iter):
        grad = compute_object_vanishing_gradient(model, x_adv)
        x_adv = torch.clamp(x_adv + eps_iter * grad.sign(), 0.0, 1.0)
        eta = torch.clamp(x_adv - x_query, -eps, eps)
        x_adv = torch.clamp(x_query + eta, 0.0, 1.0)
    return x_adv.detach()


if __name__ == "__main__":
    images_dir, annotations_file = prepare_coco_resources()
    model, device, dataset, dataloader = load_model_and_data(model_path="models/yolo11m.pt",
                                                             images_dir=images_dir,
                                                             annotations_file=annotations_file,
                                                             batch_size=1,
                                                             image_size=640)
    decompose = DecompositionLayer(noise_percent=0.7,
                                   device=device,
                                   decomposition='rtkpsvd',
                                   rank=70,
                                   steps=60)
    
    meter = DetectionMetricTracker()
    meter_adv = DetectionMetricTracker()
    meter_restored = DetectionMetricTracker()
    
    test_len = 250
    batch_index = 0
    dataloader_iterator = tqdm(dataloader, total=test_len)
    for images, targets in dataloader_iterator:
        images = images.to(device) 
        adv_pred = tog_untargeted_multistep_fgsm(model, images, targets)
        
        predictions, _, _, _, _ = run_batch(model, images, targets)
        predictions_adv, _, _, _, _ = run_batch(model, adv_pred, targets)
        
        restored = decompose(adv_pred.detach().clone()).to(device)
        predictions_restored, _, _, _, _ = run_batch(model, restored, targets)   
        
        meter.update(predictions, targets)
        meter_adv.update(predictions_adv, targets)
        meter_restored.update(predictions_restored, targets)

        visualize_detections(images, predictions, targets, "Original", batch_index, class_names=model.names)
        visualize_detections(adv_pred, predictions_adv, targets, "Adversarial", batch_index, class_names=model.names)
        visualize_detections(restored, predictions_restored, targets, "Restored", batch_index, class_names=model.names)
        batch_index += 1
        #diff = (adv_pred.to('cpu') - images.to('cpu')).view(images.size(0), -1)
        #linf_norm = max(abs(diff).max(dim=1)[0])
        #l2_norm = diff.norm(p=2,dim=1).mean()
        #print(f"Average L_inf norm: {linf_norm}; Average L_2 norm: {l2_norm}")
        if dataloader_iterator.n > test_len:
            break
        f1_adv = meter_adv.compute()['f1']
        f1_restored = meter_restored.compute()['f1']
        f1_original = meter.compute()['f1']
        postfix = f"Adv {f1_adv} | Restored {f1_restored} | Original {f1_original}"
        dataloader_iterator.set_postfix_str(postfix)
    print(f"Denoising metrics: {decompose.get_metadata()}")
    print(f"Original images metrics: {meter.compute()}")
    print(f"Adversarial images metrics: {meter_adv.compute()}")
    print(f"Restored images metrics: {meter_restored.compute()}")

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from tqdm import tqdm

from reconstruct import DecompositionLayer
from utils import get_preferred_device


TargetType = Union[int, torch.Tensor, Dict[str, torch.Tensor], None]


@dataclass
class AttackSampleResult:
    index: int
    success: bool
    attempts_used: int
    original_pred: Optional[int]
    final_pred: Optional[int]
    target_label: Optional[int]
    clean_correct: bool
    linf_norm: float
    l2_norm: float


class DecompositionAttack:
    """
    Apply DecompositionLayer with increasing `steps` until prediction becomes wrong
    or max_attempts is reached.

    Supported tasks:
    - classification: model(image_batch) -> logits
    - detection: Ultralytics YOLO style model(image_batch, verbose=False)
    """

    def __init__(
        self,
        noise_percent: float = 0.7,
        decomposition: str = "svd",
        rank: int = 70,
        max_attempts: int = 30,
        task_type: str = "classification",
        device: Optional[Union[str, torch.device]] = None,
        score_threshold: float = 0.25,
        epsilon: float = 0.03,
        limited: bool = False,
    ):
        if max_attempts <= 0:
            raise ValueError(f"max_attempts must be > 0, got {max_attempts}")
        if task_type not in {"classification", "detection"}:
            raise ValueError("task_type must be either 'classification' or 'detection'")
        if epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {epsilon}")

        self.noise_percent = noise_percent
        self.decomposition = decomposition
        self.rank = rank
        self.max_attempts = max_attempts
        self.task_type = task_type
        self.score_threshold = score_threshold
        self.epsilon = epsilon
        self.limited = limited 
        self.device = torch.device(device) if device is not None else get_preferred_device()

        self.model = None
        self.dataset = None

    def set_model(self, model: Any)-> None:
        self.model = model
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def set_dataset(self, dataset: Any) -> None:
        self.dataset = dataset

    def _extract_target_label(self, target: TargetType) -> Optional[int]:
        if target is None:
            return None

        if isinstance(target, int):
            return int(target)

        if torch.is_tensor(target):
            if target.numel() == 0:
                return None
            return int(target.flatten()[0].item())

        if isinstance(target, dict):
            labels = target.get("labels")
            if labels is None or not torch.is_tensor(labels) or labels.numel() == 0:
                return None
            return int(labels.flatten()[0].item())

        return None

    def _predict_classification_label(self, image: torch.Tensor) -> int:
        with torch.no_grad():
            logits = self.model(image.unsqueeze(0).to(self.device))
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            pred = int(torch.argmax(logits, dim=1).item())
        return pred

    def _predict_detection_label(self, image: torch.Tensor) -> Optional[int]:
        with torch.no_grad():
            results = self.model(image.unsqueeze(0).to(self.device), verbose=False)

        if not results:
            return None

        first = results[0]
        boxes = getattr(first, "boxes", None)
        if boxes is None:
            return None
        if boxes.cls is None or boxes.conf is None or boxes.cls.numel() == 0:
            return None

        keep = boxes.conf >= self.score_threshold
        if keep.any():
            conf = boxes.conf[keep]
            cls = boxes.cls[keep]
        else:
            conf = boxes.conf
            cls = boxes.cls

        best_idx = int(torch.argmax(conf).item())
        return int(cls[best_idx].item())

    def _predict_label(self, image: torch.Tensor) -> Optional[int]:
        if self.task_type == "classification":
            return self._predict_classification_label(image)
        return self._predict_detection_label(image)

    @staticmethod
    def _unpack_dataset_item(item: Any) -> Tuple[torch.Tensor, TargetType]:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            return item[0], item[1]
        return item, None

    def _single_image_attack(self, image: torch.Tensor, target: TargetType, index: int) -> AttackSampleResult:
        original_image = image.detach().clone()
        target_label = self._extract_target_label(target)
        clean_pred = self._predict_label(image)

        # If the clean sample is already incorrect (or target missing), return quickly.
        if target_label is None:
            return AttackSampleResult(
                index=index,
                success=False,
                attempts_used=0,
                original_pred=clean_pred,
                final_pred=clean_pred,
                target_label=None,
                clean_correct=False,
                linf_norm=0.0,
                l2_norm=0.0,
            )

        if clean_pred != target_label:
            return AttackSampleResult(
                index=index,
                success=False,
                attempts_used=0,
                original_pred=clean_pred,
                final_pred=clean_pred,
                target_label=target_label,
                clean_correct=False,
                linf_norm=0.0,
                l2_norm=0.0,
            )

        final_pred = clean_pred
        success = False
        attempts_used = 0
        last_incorrect_pred: Optional[int] = None
        attacked_image = original_image

        for attempt in range(1, self.max_attempts + 1):
            decompose = DecompositionLayer(
                noise_percent=self.noise_percent,
                decomposition=self.decomposition,
                rank=self.rank,
                steps=attempt,
                device=self.device,
            )

            # DecompositionLayer expects a batch tensor and mutates by index internally.
            noisy = decompose(image.unsqueeze(0).detach().clone().to(self.device))[0]
            if self.limited:
                noisy = torch.clamp(noisy, min=original_image - self.epsilon, max=original_image + self.epsilon)
            pred = self._predict_label(noisy)
            attempts_used = attempt

            if pred != target_label:
                success = True
                last_incorrect_pred = pred
                final_pred = pred
                attacked_image = noisy.detach().clone()
                continue

            # Stop once we reach the correct class after at least one incorrect prediction.
            if success:
                break

        if last_incorrect_pred is not None:
            final_pred = last_incorrect_pred
        diff = (attacked_image - original_image).detach()
        linf_norm = diff.abs().max().item()
        l2_norm = diff.reshape(-1).norm(p=2).item()

        return AttackSampleResult(
            index=index,
            success=success,
            attempts_used=attempts_used,
            original_pred=clean_pred,
            final_pred=final_pred,
            target_label=target_label,
            clean_correct=True,
            linf_norm=linf_norm,
            l2_norm=l2_norm,
        )

    def run(
        self,
        max_samples: Optional[int] = None,
        indices: Optional[Sequence[int]] = None,
        require_clean_correct: bool = True,
    ) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not set. Call set_model(...) first.")
        if self.dataset is None:
            raise RuntimeError("Dataset is not set. Call set_dataset(...) first.")

        if indices is None:
            total = len(self.dataset)
            if max_samples is not None:
                total = min(total, max_samples)
            indices = list(range(total))
        else:
            indices = list(indices)
            if max_samples is not None:
                indices = indices[:max_samples]

        attack_indices = indices
        skipped_missing_target = 0
        skipped_clean_wrong = 0

        if require_clean_correct:
            attack_indices = []
            prefilter_iterator = tqdm(indices, desc="Prefilter clean-correct", unit="img")

            for idx in prefilter_iterator:
                image, target = self._unpack_dataset_item(self.dataset[idx])
                image = image.to(self.device)
                target_label = self._extract_target_label(target)
                if target_label is None:
                    skipped_missing_target += 1
                    continue
                clean_pred = self._predict_label(image)
                if clean_pred != target_label:
                    skipped_clean_wrong += 1
                    continue
                attack_indices.append(idx)

        results: List[AttackSampleResult] = []
        iterator = tqdm(attack_indices, desc="DecompositionAttack", unit="img")
        for idx in iterator:
            image, target = self._unpack_dataset_item(self.dataset[idx])
            image = image.to(self.device)
            result = self._single_image_attack(image=image, target=target, index=idx)
            results.append(result)

        valid = [r for r in results if r.target_label is not None]
        successes = [r for r in valid if r.success]

        success_rate = (len(successes) / len(valid)) if valid else 0.0
        avg_attempts_success = (
            float(sum(r.attempts_used for r in successes) / len(successes)) if successes else 0.0
        )
        mean_linf_norm = (
            float(sum(r.linf_norm for r in successes) / len(successes)) if successes else 0.0
        )
        mean_l2_norm = (
            float(sum(r.l2_norm for r in successes) / len(successes)) if successes else 0.0
        )

        return {
            "task_type": self.task_type,
            "max_attempts": self.max_attempts,
            "num_samples_requested": len(indices),
            "num_samples_attacked": len(results),
            "num_valid_samples": len(valid),
            "num_successes": len(successes),
            "success_rate": success_rate,
            "avg_attempts_success": avg_attempts_success,
            "mean_linf_norm": mean_linf_norm,
            "mean_l2_norm": mean_l2_norm,
            "require_clean_correct": require_clean_correct,
            "num_skipped_missing_target": skipped_missing_target,
            "num_skipped_clean_wrong": skipped_clean_wrong,
            "results": results,
        }
        
if __name__ == "__main__":
    from models import load_regular_model_imagenet
    from testing_datasets import ImageNetValidationDataset
    from decomposition_attack import DecompositionAttack

    model = load_regular_model_imagenet("resnet50")
    dataset = ImageNetValidationDataset(root_dir="data/imagenet val (small)", model_id="resnet50")

    attack = DecompositionAttack(
        noise_percent=0.3,
        decomposition="svd",
        rank=70,
        max_attempts=30,
        task_type="classification",
        epsilon=0.2,
        limited=False,
    )
    attack.set_model(model)
    attack.set_dataset(dataset)

    summary = attack.run(max_samples=50)
    print(summary["success_rate"], summary["avg_attempts_success"], summary["num_successes"], 
          summary["num_valid_samples"], summary["mean_l2_norm"], summary["mean_linf_norm"])

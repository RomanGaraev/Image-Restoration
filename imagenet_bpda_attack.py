"""Evaluate KTD purification against an adaptive advertorch BPDA-PGD attack.

The attack perturbs pixel-space inputs in [0, 1] against the whole defended
pipeline: normalization -> DecompositionLayer -> classifier.  BPDA keeps the
actual decomposition in the forward pass and substitutes the identity map in
the backward pass.  EOT can average several randomized decomposition forwards
at every PGD iteration.

Example:
    pip install git+https://github.com/BorealisAI/advertorch.git
    python imagenet_bpda_attack.py --dataset-name imagenet --model-id resnet50 \
        --method tucker --noise-percent 0.6 --pgd-steps 40 --eot-samples 2

    # Automatically selects CUDA, then Apple MPS, then CPU.
    python imagenet_bpda_attack.py --device auto
"""

import argparse
import json
import sys
from collections.abc import Iterable
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from tqdm import tqdm


try:
    from advertorch.attacks import LinfPGDAttack
    from advertorch.bpda import BPDAWrapper
except ImportError as error:
    raise ImportError(
        "advertorch is not installed for the Python interpreter running this script: "
        f"{sys.executable}\n\n"
        "Install it into that exact interpreter, then rerun:\n"
        f"  {sys.executable} -m pip install git+https://github.com/BorealisAI/advertorch.git\n\n"
        "Verify the installation with:\n"
        f"  {sys.executable} -c \"import advertorch; print(advertorch.__file__)\""
    ) from error

from models import load_regular_model_cifar, load_regular_model_imagenet
from reconstruct import DecompositionLayer
from testing_datasets import CIFAR10Dataset, ImageNetValidationDataset
from utils import calculate_accuracy, compare_images


class NormalizeLayer(nn.Module):
    """Normalize pixel-space images for an ImageNet-pretrained classifier."""

    def __init__(self, mean: Iterable[float], std: Iterable[float]) -> None:
        super().__init__()
        self.register_buffer(
            "mean", torch.as_tensor(list(mean), dtype=torch.float32).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.as_tensor(list(std), dtype=torch.float32).view(1, -1, 1, 1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean) / self.std


def denormalize(images: torch.Tensor, mean: Iterable[float], std: Iterable[float]) -> torch.Tensor:
    """Convert normalized images to clamped pixel space."""
    mean_tensor = torch.as_tensor(list(mean), device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
    std_tensor = torch.as_tensor(list(std), device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
    return (images * std_tensor + mean_tensor).clamp(0.0, 1.0)


def resolve_device(requested_device: str) -> torch.device:
    """Resolve ``auto`` to CUDA, MPS, or CPU, in that order."""
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but PyTorch cannot access the Apple MPS backend.")
    return device


class BPDAClassifier(nn.Module):
    """Classifier defended by real KTD forward passes and BPDA identity gradients."""

    def __init__(
        self, normalize: nn.Module, defense: nn.Module, classifier: nn.Module, eot_samples: int
    ) -> None:
        super().__init__()
        self.normalize = normalize
        if eot_samples < 1:
            raise ValueError("eot_samples must be at least one.")
        self.bpda_defense = BPDAWrapper(defense, forwardsub=lambda x: x)
        self.classifier = classifier
        self.eot_samples = eot_samples

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # BPDA's custom autograd Function must not expose a view to in-place
        # classifier operations. SafeDecomposition returns independent storage.
        normalized_images = self.normalize(images)
        logits = [
            self.classifier(self.bpda_defense(normalized_images).clone()) for _ in range(self.eot_samples)
        ]
        # Keep the EOT dimension: EOT is an expectation of the *loss*, not
        # of the logits before the nonlinear cross-entropy is calculated.
        return torch.stack(logits, dim=0)  # [EOT, batch, classes]


class EOTCrossEntropyLoss(nn.Module):
    """Cross-entropy averaged over exact randomized defense forwards."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 2:
            return functional.cross_entropy(logits, labels, reduction="sum")
        if logits.ndim != 3:
            raise RuntimeError("Expected classifier logits shaped [batch, classes] or [EOT, batch, classes].")
        return torch.stack(
            [functional.cross_entropy(eot_logits, labels, reduction="sum") for eot_logits in logits]
        ).mean()


class SafeDecomposition(nn.Module):
    """Isolate DecompositionLayer's in-place batch writes from autograd inputs."""

    def __init__(self, defense: DecompositionLayer) -> None:
        super().__init__()
        self.defense = defense

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # DecompositionLayer assigns into ``images[ind]`` internally.  Cloning
        # both sides avoids mutating a BPDA custom-Function input or returning
        # a view from that function.
        return self.defense(images.clone()).clone()


def _load_dataset(dataset_name: str, model_id: str):
    dataset_name = dataset_name.lower()
    if dataset_name == "imagenet":
        return (
            ImageNetValidationDataset(root_dir="data/imagenet val (small)", model_id=model_id),
            load_regular_model_imagenet,
        )
    if dataset_name == "cifar10":
        return (
            CIFAR10Dataset(root_dir="data/cifar10", model_id=model_id, train=False, download=True),
            load_regular_model_cifar,
        )
    raise ValueError("dataset_name must be 'imagenet' or 'cifar10'.")


def _accuracy(labels: torch.Tensor, logits: torch.Tensor) -> float:
    return calculate_accuracy(labels, logits.argmax(dim=1))


def evaluate_bpda(
    model_id: str = "resnet50",
    dataset_name: str = "imagenet",
    method: str = "tucker",
    noise_percents: Iterable[float] = (0.6,),
    rank: int = 70,
    decompose_steps: int = 30,
    device: str = "auto",
    batch_size: int = 8,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    pgd_steps: int = 40,
    eot_samples: int = 1,
    random_start: bool = True,
    max_batches: int = -1,
    visualize: bool = False,
) -> Dict[float, Dict[str, float]]:
    """Run pixel-space adaptive Linf BPDA-PGD against KTD purification.

    Returns one metric dictionary per noise percentage. All accuracies are
    percentages, while ``mean_linf`` is measured in pixel-space [0, 1].
    """
    run_device = resolve_device(device)
    print(f"Using device: {run_device}")

    dataset, model_loader = _load_dataset(dataset_name, model_id)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        # Preserve one image order for each noise setting. This makes a
        # truncated sweep compare Tucker parameters on the same samples.
        shuffle=False,
        num_workers=0,
        pin_memory=run_device.type == "cuda",
    )

    classifier = model_loader(model_id).to(run_device).eval()
    # We attack inputs, not model parameters. This avoids storing parameter gradients.
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)

    normalize = NormalizeLayer(dataset.mean, dataset.std).to(run_device).eval()
    bare_model = nn.Sequential(normalize, classifier).to(run_device).eval()
    results: Dict[float, Dict[str, float]] = {}

    for noise_percent in noise_percents:
        decomposition = DecompositionLayer(
            noise_percent=noise_percent,
            device=run_device,
            decomposition=method,
            rank=rank,
            steps=decompose_steps,
        ).to(run_device).eval()
        defense = SafeDecomposition(decomposition).to(run_device).eval()

        # Exact defended model used only for reporting metrics.
        defended_model = nn.Sequential(normalize, defense, classifier).to(run_device).eval()
        # Adaptive attack model: real defense forward, identity BPDA backward.
        bpda_model = BPDAClassifier(normalize, defense, classifier, eot_samples).to(run_device).eval()
        adversary = LinfPGDAttack(
            predict=bpda_model,
            loss_fn=EOTCrossEntropyLoss(),
            eps=eps,
            nb_iter=pgd_steps,
            eps_iter=alpha,
            rand_init=random_start,
            clip_min=0.0,
            clip_max=1.0,
            targeted=False,
        )

        measurements: Dict[str, List[float]] = {
            "clean": [],
            "clean_defended": [],
            "adversarial": [],
            "adversarial_defended": [],
            "linf": [],
        }
        total = len(dataloader) if max_batches < 0 else min(len(dataloader), max_batches)
        progress = tqdm(dataloader, total=total, desc=f"BPDA rho={noise_percent}")

        for batch_index, (normalized_images, labels) in enumerate(progress):
            if max_batches >= 0 and batch_index >= max_batches:
                break

            labels = labels.to(run_device, non_blocking=True)
            images = denormalize(normalized_images.to(run_device, non_blocking=True), dataset.mean, dataset.std)
            adversarial_images = adversary.perturb(images, labels)

            with torch.no_grad():
                clean_logits = bare_model(images)
                clean_defended_logits = defended_model(images)
                adversarial_logits = bare_model(adversarial_images)
                adversarial_defended_logits = defended_model(adversarial_images)

            measurements["clean"].append(_accuracy(labels, clean_logits))
            measurements["clean_defended"].append(_accuracy(labels, clean_defended_logits))
            measurements["adversarial"].append(_accuracy(labels, adversarial_logits))
            measurements["adversarial_defended"].append(_accuracy(labels, adversarial_defended_logits))
            measurements["linf"].extend(
                (adversarial_images - images).abs().flatten(1).amax(dim=1).detach().cpu().tolist()
            )

            progress.set_postfix(
                clean=f"{np.mean(measurements['clean']):.2f}%",
                adversarial=f"{np.mean(measurements['adversarial']):.2f}%",
                defended_clean=f"{np.mean(measurements['clean_defended']):.2f}%",
                defended_adv=f"{np.mean(measurements['adversarial_defended']):.2f}%",
            )

            if visualize and batch_index == 0:
                with torch.no_grad():
                    reconstructed_pixels = defense(adversarial_images)
                compare_images(
                    images[0].detach().cpu().numpy(),
                    adversarial_images[0].detach().cpu().numpy(),
                    reconstructed_pixels[0].detach().cpu().numpy(),
                    dataset.label(int(clean_logits[0].argmax().item())),
                    dataset.label(int(adversarial_logits[0].argmax().item())),
                    dataset.label(int(adversarial_defended_logits[0].argmax().item())),
                )

        summary = {
            "clean_accuracy": float(np.mean(measurements["clean"])) if measurements["clean"] else 0.0,
            "clean_defended_accuracy": (
                float(np.mean(measurements["clean_defended"])) if measurements["clean_defended"] else 0.0
            ),
            "adversarial_accuracy": float(np.mean(measurements["adversarial"])) if measurements["adversarial"] else 0.0,
            "adversarial_defended_accuracy": (
                float(np.mean(measurements["adversarial_defended"]))
                if measurements["adversarial_defended"]
                else 0.0
            ),
            "mean_linf": float(np.mean(measurements["linf"])) if measurements["linf"] else 0.0,
        }
        results[float(noise_percent)] = summary
        print(f"rho={noise_percent}: {json.dumps(summary, indent=2)}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DecompositionLayer with an adaptive advertorch BPDA-PGD attack."
    )
    parser.add_argument("--dataset-name", choices=["imagenet", "cifar10"], default="imagenet")
    parser.add_argument("--model-id", default="resnet50")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (CUDA → MPS → CPU), cuda, mps, cpu, or a specific device such as cuda:0",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--method", choices=["svd", "tucker", "rtkpsvd", "rtkpsvd_numpy"], default="tucker")
    parser.add_argument("--noise-percent", type=float, nargs="+", default=[0.6])
    parser.add_argument("--rank", type=int, default=70)
    parser.add_argument("--decompose-steps", type=int, default=30)
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=40)
    parser.add_argument("--eot-samples", type=int, default=1)
    parser.add_argument("--no-random-start", action="store_true")
    parser.add_argument("--max-batches", type=int, default=-1, help="Use a positive value for a short debug run.")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--log-file", default="")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    experiment_results = evaluate_bpda(
        model_id=arguments.model_id,
        dataset_name=arguments.dataset_name,
        method=arguments.method,
        noise_percents=arguments.noise_percent,
        rank=arguments.rank,
        decompose_steps=arguments.decompose_steps,
        device=arguments.device,
        batch_size=arguments.batch_size,
        eps=arguments.eps,
        alpha=arguments.alpha,
        pgd_steps=arguments.pgd_steps,
        eot_samples=arguments.eot_samples,
        random_start=not arguments.no_random_start,
        max_batches=arguments.max_batches,
        visualize=arguments.visualize,
    )
    attack_description = {
        "attack": "advertorch.BPDAWrapper + advertorch.LinfPGDAttack",
        "eps": arguments.eps,
        "alpha": arguments.alpha,
        "pgd_steps": arguments.pgd_steps,
        "eot_samples": arguments.eot_samples,
        "random_start": not arguments.no_random_start,
    }
    print(json.dumps({"attack": attack_description, "results": experiment_results}, indent=2))

    if arguments.log_file:
        with open(arguments.log_file, "w", encoding="utf-8") as log_file:
            json.dump({"attack": attack_description, "results": experiment_results}, log_file, indent=2)
            log_file.write("\n")

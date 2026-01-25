"""Reusable loader utilities for Ultralytics YOLO models and the COCO dataset."""

import argparse
import os
import shutil
import tempfile
import urllib.request
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.ops import box_iou
from ultralytics import YOLO

from utils import get_preferred_device


DEFAULT_COCO_ROOT = os.path.join("data", "coco")
DEFAULT_COCO_SPLIT = "val2017"
DEFAULT_COCO_ANNOTATION = "instances_val2017.json"
DEFAULT_MODEL_NAME = "models/yolo11m.pt"


COCO_IMAGE_ZIPS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "test2017": "http://images.cocodataset.org/zips/test2017.zip",
}

COCO_ANNOTATION_ZIPS = {
    "instances_train2017.json": ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                                  "annotations/instances_train2017.json"),
    "instances_val2017.json": ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                                "annotations/instances_val2017.json"),
    "captions_train2017.json": ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                                 "annotations/captions_train2017.json"),
    "captions_val2017.json": ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                               "annotations/captions_val2017.json"),
    "person_keypoints_train2017.json": ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                                         "annotations/person_keypoints_train2017.json"),
    "person_keypoints_val2017.json": ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                                       "annotations/person_keypoints_val2017.json"),
}


def _download_file(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {url} -> {dest_path}")
    with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def get_default_coco_paths(split: str = DEFAULT_COCO_SPLIT,
                           annotation: str = DEFAULT_COCO_ANNOTATION) -> Tuple[str, str]:
    images_dir = os.path.join(DEFAULT_COCO_ROOT, split)
    annotations_dir = os.path.join(DEFAULT_COCO_ROOT, "annotations")
    annotations_file = os.path.join(annotations_dir, annotation)
    return images_dir, annotations_file


def prepare_coco_resources(split: str = DEFAULT_COCO_SPLIT,
                           annotation: str = DEFAULT_COCO_ANNOTATION) -> Tuple[str, str]:
    images_dir, annotations_file = get_default_coco_paths(split=split, annotation=annotation)
    _ensure_coco_images_exist(images_dir)
    _ensure_coco_annotations_exist(annotations_file)
    return images_dir, annotations_file


def _ensure_coco_images_exist(images_dir: str) -> None:
    if os.path.isdir(images_dir) and os.listdir(images_dir):
        return
    folder = os.path.basename(os.path.normpath(images_dir))
    url = COCO_IMAGE_ZIPS.get(folder)
    if url is None:
        raise FileNotFoundError(f"COCO images directory not found: {images_dir}. Provide a valid path or use one of {list(COCO_IMAGE_ZIPS.keys())}.")
    parent = os.path.dirname(images_dir)
    os.makedirs(parent, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, f"{folder}.zip")
        _download_file(url, zip_path)
        print(f"Extracting {zip_path} to {parent}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(parent)
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Failed to extract COCO images into {images_dir}")


def _ensure_coco_annotations_exist(annotations_file: str) -> None:
    if os.path.isfile(annotations_file):
        return
    filename = os.path.basename(annotations_file)
    if filename not in COCO_ANNOTATION_ZIPS:
        raise FileNotFoundError(f"Annotation file {annotations_file} not found and no download rule available. Provide a valid COCO annotation file.")
    url, member = COCO_ANNOTATION_ZIPS[filename]
    target_dir = os.path.dirname(annotations_file)
    os.makedirs(target_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, os.path.basename(url))
        _download_file(url, zip_path)
        print(f"Extracting {member} from {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            if member not in zf.namelist():
                raise FileNotFoundError(f"File {member} not found inside archive {zip_path}")
            zf.extract(member, tmp_dir)
        extracted_path = os.path.join(tmp_dir, member)
        os.makedirs(os.path.dirname(annotations_file), exist_ok=True)
        shutil.move(extracted_path, annotations_file)


def _normalize_label(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


def build_label_mapping(coco_api: COCO, model_names: Union[Dict[int, str], Iterable[str]]) -> Dict[int, int]:
    """Create a robust mapping from COCO category ids to YOLO class indices."""
    if isinstance(model_names, dict):
        iterator = model_names.items()
    else:
        iterator = enumerate(list(model_names))

    normalized = {_normalize_label(name): idx for idx, name in iterator}
    mapping = {}
    missing = []
    for cat_id, cat in coco_api.cats.items():
        key = _normalize_label(cat["name"])
        if key in normalized:
            mapping[cat_id] = normalized[key]
        else:
            missing.append(cat["name"])
    if missing:
        print(f"Warning: {len(missing)} COCO categories could not be matched to YOLO names: {missing}")
    return mapping


class COCODetectionDataset(Dataset):
    def __init__(self,
                 images_dir: str,
                 annotations_file: str,
                 image_size: int = 640,
                 label_mapping: Optional[Dict[int, int]] = None):
        _ensure_coco_images_exist(images_dir)
        _ensure_coco_annotations_exist(annotations_file)

        self.images_dir = images_dir
        self.coco = COCO(annotations_file)
        self.ids = sorted(self.coco.imgs.keys())
        self.image_size = image_size
        self.transform = transforms.ToTensor()
        self.label_mapping = label_mapping or {}

    def set_label_mapping(self, mapping: Dict[int, int]) -> None:
        self.label_mapping = mapping

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.images_dir, img_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        image = self.transform(image)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        boxes, labels = [], []
        for ann in annotations:
            category_id = ann["category_id"]
            mapped_label = self.label_mapping.get(category_id, category_id)
            x, y, w, h = ann["bbox"]
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            boxes.append([x1, y1, x2, y2])
            labels.append(mapped_label)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
            "image_id": torch.tensor(img_id),
        }
        return image, target


def coco_collate_fn(batch):
    images = torch.stack([sample[0] for sample in batch], dim=0)
    targets = [sample[1] for sample in batch]
    return images, targets


def load_yolo_model(model_path: str, device: Optional[torch.device] = None) -> Tuple[YOLO, torch.device]:
    device = device or get_preferred_device()
    print(f"Loading YOLO model '{model_path}' on {device}")
    model = YOLO(model_path)
    model.to(device)
    return model, device


def create_coco_dataloader(images_dir: str,
                           annotations_file: str,
                           image_size: int,
                           batch_size: int,
                           label_mapping: Optional[Dict[int, int]] = None,
                           num_workers: int = 2,
                           shuffle: bool = False,
                           pin_memory: bool = True) -> Tuple[COCODetectionDataset, DataLoader]:
    dataset = COCODetectionDataset(images_dir,
                                   annotations_file,
                                   image_size=image_size,
                                   label_mapping=label_mapping)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            collate_fn=coco_collate_fn)
    return dataset, dataloader


def load_model_and_data(model_path: str,
                        images_dir: str,
                        annotations_file: str,
                        batch_size: int,
                        image_size: int,
                        num_workers: int = 2) -> Tuple[YOLO, torch.device, COCODetectionDataset, DataLoader]:
    model, device = load_yolo_model(model_path)
    dataset = COCODetectionDataset(images_dir, annotations_file, image_size=image_size)
    label_mapping = build_label_mapping(dataset.coco, model.names)
    dataset.set_label_mapping(label_mapping)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=coco_collate_fn)
    return model, device, dataset, dataloader


def inspect_predictions(predictions: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]):
    """Print a quick IoU summary between predicted and ground-truth boxes."""
    if not predictions:
        return
    pred = predictions[0]
    target = targets[0]
    if pred["boxes"].numel() == 0 or target["boxes"].numel() == 0:
        print("No overlapping boxes to compare in the first batch.")
        return
    iou = box_iou(pred["boxes"].cpu(), target["boxes"].cpu())
    print(f"IoU matrix between predictions and targets for sample 0:\n{iou}")


def _run_sanity_check(model_path: str,
                      split: str,
                      annotation_name: str,
                      batch_size: int,
                      image_size: int) -> None:
    images_dir, annotations_file = prepare_coco_resources(split=split, annotation=annotation_name)
    model, device, _, dataloader = load_model_and_data(model_path,
                                                       images_dir,
                                                       annotations_file,
                                                       batch_size,
                                                       image_size)
    model.info()
    images, targets = next(iter(dataloader))
    images = images.to(device)
    print("Running a sanity-check forward pass...")
    with torch.no_grad():
        results = model(images, verbose=False)
    predictions = []
    for result in results:
        if result.boxes is None or result.boxes.xyxy.numel() == 0:
            predictions.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                "labels": torch.zeros((0,), dtype=torch.long, device=device),
            })
            continue
        predictions.append({
            "boxes": result.boxes.xyxy.to(device),
            "scores": result.boxes.conf.to(device),
            "labels": result.boxes.cls.to(torch.long).to(device),
        })
    print(f"Produced {sum(pred['boxes'].size(0) for pred in predictions)} boxes for the first batch")
    inspect_predictions(predictions, targets)


def parse_args():
    parser = argparse.ArgumentParser(description="Load YOLO/COCO resources or run the evaluation pipeline")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Ultralytics model path or name")
    parser.add_argument("--split", type=str, default=DEFAULT_COCO_SPLIT, help="COCO split to download/use (train2017, val2017, test2017)")
    parser.add_argument("--annotation-name", type=str, default=DEFAULT_COCO_ANNOTATION, help="COCO annotation filename to download/use")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size used for sanity check or evaluation")
    parser.add_argument("--image-size", type=int, default=640, help="Image size used for preprocessing")
    parser.add_argument("--mode", choices=["sanity", "evaluate"], default="sanity",
                        help="sanity: run a single forward pass; evaluate: run yolo_coco_test with prepared paths")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Std dev of additive noise for evaluation")
    parser.add_argument("--noise-levels", type=float, nargs="+",
                        default=[0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help="Noise percentages fed to DecompositionLayer during evaluation")
    parser.add_argument("--method", choices=["svd", "tucker"], default="svd")
    parser.add_argument("--rank", type=int, default=70)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--visualize", action="store_true", help="Show qualitative results during evaluation")
    parser.add_argument("--max-visualizations", type=int, default=5, help="Maximum batches to visualize")
    parser.add_argument("--attack-type", choices=["none", "patch", "tog"], default="none")
    parser.add_argument("--attack-score-threshold", type=float, default=0.25,
                        help="Score threshold shared by the attack routines")
    parser.add_argument("--tog-iters", type=int, default=10, help="Number of steps for the TOG untargeted attack")
    parser.add_argument("--tog-eps", type=float, default=8 / 255., help="Max perturbation for the TOG attack")
    parser.add_argument("--tog-eps-iter", type=float, default=2 / 255., help="Step size for the TOG attack")
    parser.add_argument("--patch-path", type=str, default=None)
    parser.add_argument("--patch-target-class", type=str, default="person")
    parser.add_argument("--patch-size", type=int, nargs=2, metavar=("PATCH_H", "PATCH_W"), default=[64, 64])
    parser.add_argument("--patch-min-area", type=float, default=4096.0)
    parser.add_argument("--patch-max-rois", type=int, default=1)
    parser.add_argument("--patch-score-threshold", type=float, default=0.25)
    parser.add_argument("--patch-overlap-threshold", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _run_sanity_check(model_path=args.model,
                      split=args.split,
                      annotation_name=args.annotation_name,
                      batch_size=args.batch_size,
                      image_size=args.image_size)

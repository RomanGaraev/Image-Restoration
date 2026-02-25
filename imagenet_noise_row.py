import argparse
import random

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

from reconstruct import DecompositionLayer
from testing_datasets import ImageNetValidationDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load one ImageNet image, apply salt-pepper noise at multiple levels, and save a row figure."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="data/imagenet val (small)",
        help="Path to ImageNet validation subset root.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Image index to use. If omitted, a random image is chosen.",
    )
    parser.add_argument(
        "--noise-levels",
        type=int,
        nargs=4,
        default=[20, 45, 70, 90],
        help="Exactly 4 integer noise levels in percent.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="imagenet_noise_row.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="resnet50",
        help="Model id used only to pick dataset preprocessing (resnet50 by default).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def tensor_to_pil(image: torch.Tensor, dataset: ImageNetValidationDataset) -> Image.Image:
    restored = dataset.inverse_transform(image.cpu()).clamp(0.0, 1.0)
    return to_pil_image(restored)


def main():
    args = parse_args()

    random.seed(args.seed)
    np_rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    dataset = ImageNetValidationDataset(root_dir=args.root_dir, model_id=args.model_id)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found in dataset root: {args.root_dir}")

    if args.index is None:
        image_index = np_rng.randrange(len(dataset))
    else:
        if args.index < 0 or args.index >= len(dataset):
            raise IndexError(f"--index must be in [0, {len(dataset) - 1}], got {args.index}")
        image_index = args.index

    image, _ = dataset[image_index]

    tiles = []
    for i, noise_percent in enumerate(args.noise_levels, start=1):
        if noise_percent < 0 or noise_percent > 100:
            raise ValueError(f"Noise percent must be in [0, 100], got {noise_percent}")

        layer = DecompositionLayer(noise_percent=noise_percent / 100.0)
        noisy, _ = layer._DecompositionLayer__generate_salt_pepper_noise(image.clone())
        tiles.append((tensor_to_pil(noisy, dataset), f"({i})"))

    spacing = 16
    subtitle_h = 28
    widths = [img.width for img, _ in tiles]
    heights = [img.height for img, _ in tiles]
    canvas_w = sum(widths) + spacing * (len(tiles) - 1)
    canvas_h = max(heights) + subtitle_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color="white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    x = 0
    for img, subtitle in tiles:
        canvas.paste(img, (x, 0))
        bbox = draw.textbbox((0, 0), subtitle, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x + (img.width - text_w) // 2
        text_y = max(heights) + 6
        draw.text((text_x, text_y), subtitle, fill="black", font=font)
        x += img.width + spacing

    canvas.save(args.output)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geobuild.data.dataset import build_dataloader, build_dataset
from geobuild.utils.config import load_config, output_path_from_config, resolve_path


def tensor_stats(name: str, value: torch.Tensor) -> None:
    print(
        f"{name}: shape={list(value.shape)} dtype={value.dtype} "
        f"min={float(value.min()):.6g} max={float(value.max()):.6g}"
    )


def print_sample(sample: dict[str, Any]) -> None:
    print("Sample")

    for key in ("image", "mask", "boundary", "corner", "center", "offset"):
        tensor_stats(key, sample[key])

    print(f"image_id: {sample['image_id']}")


def print_batch(batch: dict[str, Any]) -> None:
    print("Batch")

    for key in ("image", "mask", "boundary", "corner", "center", "offset"):
        print(f"{key}: {list(batch[key].shape)}")

    print(f"valid_mask: {list(batch['valid_mask'].shape)}")
    print(f"original_size: {batch['original_size']}")
    padded_height = int(batch["image"].shape[-2])
    padded_width = int(batch["image"].shape[-1])
    print(f"padded_batch_size: {(padded_height, padded_width)}")
    print(f"image_id: {list(batch['image_id'])}")


def to_hwc_image(image: torch.Tensor) -> np.ndarray:
    return image.detach().cpu().permute(1, 2, 0).numpy()


def to_hw(target: torch.Tensor) -> np.ndarray:
    return target.detach().cpu().squeeze(0).numpy()


def offset_magnitude(offset: torch.Tensor) -> np.ndarray:
    offset_array = offset.detach().cpu().numpy()
    return np.sqrt(offset_array[0] ** 2 + offset_array[1] ** 2)


def batch_sample(batch: dict[str, Any], index: int) -> dict[str, Any]:
    original_height, original_width = batch["original_size"][index]
    cropped = {
        "image_id": str(batch["image_id"][index]),
        "original_size": (int(original_height), int(original_width)),
    }

    for key in ("image", "mask", "boundary", "corner", "center", "offset"):
        cropped[key] = batch[key][
            index,
            :,
            :original_height,
            :original_width,
        ]

    cropped["valid_mask"] = batch["valid_mask"][
        index,
        :,
        :original_height,
        :original_width,
    ]

    return cropped


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    minimum = float(array.min()) if array.size else 0.0
    maximum = float(array.max()) if array.size else 0.0

    if maximum <= minimum:
        return np.zeros(array.shape, dtype=np.uint8)

    normalized = (array - minimum) / (maximum - minimum)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def colorize(array: np.ndarray, colors: list[tuple[int, int, int]]) -> np.ndarray:
    values = normalize_to_uint8(array).astype(np.float32) / 255.0
    scaled = values * float(len(colors) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.clip(lower + 1, 0, len(colors) - 1)
    weight = scaled - lower.astype(np.float32)

    palette = np.asarray(colors, dtype=np.float32)
    rgb = palette[lower] * (1.0 - weight[..., None]) + palette[upper] * weight[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def panel_image(data: np.ndarray, mode: str) -> Image.Image:
    if mode == "rgb":
        array = np.clip(data * 255.0, 0, 255).astype(np.uint8)
    elif mode == "gray":
        gray = normalize_to_uint8(data)
        array = np.repeat(gray[:, :, None], 3, axis=2)
    elif mode == "viridis":
        array = colorize(
            data,
            [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
        )
    elif mode == "magma":
        array = colorize(
            data,
            [(0, 0, 4), (73, 16, 108), (183, 55, 121), (249, 142, 8), (252, 253, 191)],
        )
    else:
        raise ValueError(f"Unsupported preview mode: {mode}")

    return Image.fromarray(array, mode="RGB")


def titled_panel(title: str, image: Image.Image) -> Image.Image:
    title_height = 28
    panel = Image.new("RGB", (image.width, image.height + title_height), "white")
    panel.paste(image, (0, title_height))

    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), title, fill="black")

    return panel


def save_preview(sample: dict[str, Any], output_path: Path) -> None:
    height, width = sample["original_size"]
    print(
        "Preview crop: "
        f"image_id={sample['image_id']} original_size={(height, width)}"
    )

    panels = [
        ("Image", to_hwc_image(sample["image"]), "rgb"),
        ("Mask", to_hw(sample["mask"]), "gray"),
        ("Boundary", to_hw(sample["boundary"]), "gray"),
        ("Corner", to_hw(sample["corner"]), "viridis"),
        ("Center", to_hw(sample["center"]), "viridis"),
        ("Offset magnitude", offset_magnitude(sample["offset"]), "magma"),
    ]

    rendered = [
        titled_panel(title, panel_image(data, mode))
        for title, data, mode in panels
    ]

    panel_width = max(panel.width for panel in rendered)
    panel_height = max(panel.height for panel in rendered)
    grid = Image.new("RGB", (panel_width * 3, panel_height * 2), "white")

    for index, panel in enumerate(rendered):
        x = (index % 3) * panel_width
        y = (index // 3) * panel_height
        grid.paste(panel, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"Saved preview to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, root=ROOT)

    dataset = build_dataset(config, "train")
    print(f"Dataset length: {len(dataset)}")

    sample = dataset[0]
    print_sample(sample)

    dataloader = build_dataloader(config, "train")
    batch = next(iter(dataloader))
    print_batch(batch)

    output_path = (
        resolve_path(args.out, root=ROOT)
        if args.out is not None
        else output_path_from_config(config, "dataloader_check", root=ROOT)
    )
    save_preview(batch_sample(batch, 0), output_path)


if __name__ == "__main__":
    main()

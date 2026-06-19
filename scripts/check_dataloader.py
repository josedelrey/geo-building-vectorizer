import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

from geobuild.data.dataset import (
    BuildingFootprintDataset,
    build_dataloader,
    build_dataset,
    collate_samples,
)
from geobuild.data.records import ImageRecord
from geobuild.utils.config import load_config, output_path_from_config, resolve_path


TARGET_CHANNELS = {
    "mask": 1,
    "boundary": 1,
    "corner": 1,
    "center": 1,
    "offset": 2,
    "instance": 1,
}
TARGET_KEYS = tuple(TARGET_CHANNELS)


def tensor_stats(name: str, value: torch.Tensor) -> None:
    print(
        f"{name}: shape={list(value.shape)} dtype={value.dtype} "
        f"min={float(value.min()):.6g} max={float(value.max()):.6g}"
    )


def assert_tensor(
    name: str,
    value: Any,
    expected_dtype: torch.dtype,
    expected_channels: int,
    ndim: int,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")

    if value.dtype != expected_dtype:
        raise TypeError(f"{name} dtype must be {expected_dtype}, got {value.dtype}")

    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}")

    channel_dim = 0 if ndim == 3 else 1
    actual_channels = int(value.shape[channel_dim])

    if actual_channels != expected_channels:
        raise ValueError(
            f"{name} must have {expected_channels} channels, got {actual_channels}"
        )


def assert_sample(sample: dict[str, Any]) -> None:
    assert_tensor("image", sample["image"], torch.float32, 3, ndim=3)

    for key, channels in TARGET_CHANNELS.items():
        if key in sample:
            assert_tensor(key, sample[key], torch.float32, channels, ndim=3)

    height = int(sample["image"].shape[-2])
    width = int(sample["image"].shape[-1])

    for key in TARGET_KEYS:
        if key not in sample:
            continue

        actual_size = tuple(int(dim) for dim in sample[key].shape[-2:])

        if actual_size != (height, width):
            raise ValueError(
                f"{key} spatial shape must match image {(height, width)}, "
                f"got {actual_size}"
            )


def assert_binary_tensor(name: str, value: torch.Tensor) -> None:
    unique_values = torch.unique(value)
    allowed = (unique_values == 0) | (unique_values == 1)

    if not bool(torch.all(allowed)):
        raise ValueError(
            f"{name} must contain only 0 and 1, got {unique_values.tolist()}"
        )


def assert_padded_region_zero(
    name: str,
    value: torch.Tensor,
    original_size: list[tuple[int, int]],
) -> None:
    for index, (height, width) in enumerate(original_size):
        bottom = value[index, :, height:, :]
        right = value[index, :, :, width:]

        if bottom.numel() > 0 and bool(torch.any(bottom != 0)):
            raise ValueError(
                f"{name} padded bottom region is non-zero for batch index {index}"
            )

        if right.numel() > 0 and bool(torch.any(right != 0)):
            raise ValueError(
                f"{name} padded right region is non-zero for batch index {index}"
            )


def assert_batch(batch: dict[str, Any]) -> None:
    assert_tensor("batch.image", batch["image"], torch.float32, 3, ndim=4)

    for key, channels in TARGET_CHANNELS.items():
        if key in batch:
            assert_tensor(f"batch.{key}", batch[key], torch.float32, channels, ndim=4)

    assert_tensor("batch.valid_mask", batch["valid_mask"], torch.float32, 1, ndim=4)

    batch_size = int(batch["image"].shape[0])
    padded_height = int(batch["image"].shape[-2])
    padded_width = int(batch["image"].shape[-1])

    if padded_height % 32 != 0 or padded_width % 32 != 0:
        raise ValueError(
            "Padded batch height and width must be multiples of 32, got "
            f"{(padded_height, padded_width)}"
        )

    original_size = [
        (int(height), int(width))
        for height, width in batch["original_size"]
    ]

    if len(original_size) != batch_size:
        raise ValueError(
            f"original_size length must be {batch_size}, got {len(original_size)}"
        )

    if len(batch["image_id"]) != batch_size:
        raise ValueError(
            f"image_id length must be {batch_size}, got {len(batch['image_id'])}"
        )

    assert_binary_tensor("valid_mask", batch["valid_mask"])

    for index, (height, width) in enumerate(original_size):
        expected_sum = float(height * width)
        actual_sum = float(batch["valid_mask"][index].sum())

        if actual_sum != expected_sum:
            raise ValueError(
                f"valid_mask sum for batch index {index} must be {expected_sum}, "
                f"got {actual_sum}"
            )

    for key in ("image", *[key for key in TARGET_KEYS if key in batch]):
        assert_padded_region_zero(key, batch[key], original_size)


def find_sample_indices_by_size(
    dataset: BuildingFootprintDataset,
    sizes: list[tuple[int, int]],
) -> list[int]:
    wanted = {size: None for size in sizes}

    for index in range(len(dataset)):
        size = dataset.record_size(index)

        if size in wanted and wanted[size] is None:
            wanted[size] = index

        if all(value is not None for value in wanted.values()):
            break

    missing = [size for size, index in wanted.items() if index is None]

    if missing:
        raise ValueError(f"Could not find samples with sizes: {missing}")

    return [int(wanted[size]) for size in sizes]


def check_forced_mixed_size_batch(dataset: BuildingFootprintDataset) -> None:
    sizes = [(300, 300), (512, 512), (650, 650)]
    indices = find_sample_indices_by_size(dataset, sizes)
    samples = [dataset[index] for index in indices]
    batch = collate_samples(samples)
    assert_batch(batch)

    expected_padded_size = (672, 672)
    actual_padded_size = (
        int(batch["image"].shape[-2]),
        int(batch["image"].shape[-1]),
    )

    if actual_padded_size != expected_padded_size:
        raise ValueError(
            f"Forced mixed-size batch padded size must be {expected_padded_size}, "
            f"got {actual_padded_size}"
        )

    original_size = [
        (int(height), int(width))
        for height, width in batch["original_size"]
    ]

    if original_size != sizes:
        raise ValueError(
            f"Forced mixed-size batch original_size must be {sizes}, "
            f"got {original_size}"
        )

    print("Forced mixed-size batch")
    print(f"indices: {indices}")
    print(f"original_size: {original_size}")
    print(f"padded_batch_size: {actual_padded_size}")


def print_sample(sample: dict[str, Any]) -> None:
    print("Sample")

    for key in ("image", *[key for key in TARGET_KEYS if key in sample]):
        tensor_stats(key, sample[key])

    print(f"image_id: {sample['image_id']}")


def print_batch(batch: dict[str, Any]) -> None:
    print("Batch")

    for key in ("image", *[key for key in TARGET_KEYS if key in batch]):
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


def zeros_like_image_channel(sample: dict[str, Any]) -> np.ndarray:
    height, width = sample["original_size"]
    return np.zeros((int(height), int(width)), dtype=np.float32)


def batch_sample(batch: dict[str, Any], index: int) -> dict[str, Any]:
    original_height, original_width = batch["original_size"][index]
    cropped = {
        "image_id": str(batch["image_id"][index]),
        "original_size": (int(original_height), int(original_width)),
    }

    for key in ("image", *[key for key in TARGET_KEYS if key in batch]):
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


def record_by_image_id(dataset: BuildingFootprintDataset) -> dict[str, ImageRecord]:
    return {str(record.image_id): record for record in dataset._records}


def draw_polygon_lines(
    image: Image.Image,
    record: ImageRecord | None,
    color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    if record is None:
        return image

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    for polygon in record.polygons:
        points = [(float(x), float(y)) for x, y in polygon.exterior]

        if not points:
            continue

        draw.line(points + [points[0]], fill=color, width=1)

    return overlay


def titled_panel(title: str, image: Image.Image) -> Image.Image:
    title_height = 28
    panel = Image.new("RGB", (image.width, image.height + title_height), "white")
    panel.paste(image, (0, title_height))

    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), title, fill="black")

    return panel


def save_preview(
    sample: dict[str, Any],
    output_path: Path,
    record: ImageRecord | None = None,
) -> None:
    height, width = sample["original_size"]
    print(
        "Preview crop: "
        f"image_id={sample['image_id']} original_size={(height, width)}"
    )

    panels = [titled_panel("Image", panel_image(to_hwc_image(sample["image"]), "rgb"))]

    if "mask" in sample:
        panels.append(titled_panel("Mask", panel_image(to_hw(sample["mask"]), "gray")))

    if "boundary" in sample:
        panels.append(
            titled_panel("Boundary", panel_image(to_hw(sample["boundary"]), "gray"))
        )

    if "corner" in sample:
        panels.append(
            titled_panel(
                "Corner",
                draw_polygon_lines(
                    panel_image(to_hw(sample["corner"]), "viridis"),
                    record,
                ),
            )
        )

    if "center" in sample:
        panels.append(
            titled_panel(
                "Center",
                draw_polygon_lines(
                    panel_image(to_hw(sample["center"]), "viridis"),
                    record,
                ),
            )
        )

    if "offset" in sample:
        panels.append(
            titled_panel(
                "Offset magnitude",
                panel_image(offset_magnitude(sample["offset"]), "magma"),
            )
        )

    if len(panels) == 1:
        panels.append(
            titled_panel(
                "No active targets",
                panel_image(zeros_like_image_channel(sample), "gray"),
            )
        )

    panel_width = max(panel.width for panel in panels)
    panel_height = max(panel.height for panel in panels)
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    grid = Image.new("RGB", (panel_width * columns, panel_height * rows), "white")

    for index, panel in enumerate(panels):
        x = (index % 3) * panel_width
        y = (index // 3) * panel_height
        grid.paste(panel, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"Saved preview to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, root=ROOT)

    dataset = build_dataset(config, args.split)
    print(f"Split: {args.split}")
    print(f"Dataset length: {len(dataset)}")

    if args.split == "train":
        check_forced_mixed_size_batch(dataset)

    sample = dataset[0]
    assert_sample(sample)
    print_sample(sample)

    dataloader = build_dataloader(config, args.split)
    batch = next(iter(dataloader))
    assert_batch(batch)
    print_batch(batch)

    if args.out is not None:
        output_path = resolve_path(args.out, root=ROOT)
    else:
        output_path = output_path_from_config(
            config,
            "dataloader_check",
            root=ROOT,
            split=args.split,
        )

    preview_sample = batch_sample(batch, 0)
    records = record_by_image_id(dataset)
    record = records.get(str(preview_sample["image_id"]))
    save_preview(preview_sample, output_path, record=record)


if __name__ == "__main__":
    main()

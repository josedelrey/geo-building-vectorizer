import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geobuild.data.dataset import BuildingFootprintDataset
from geobuild.data.transforms import EvalTransform, SafeAugTransform
from geobuild.utils.config import (
    load_config,
    manifest_path_from_config,
    resolve_path,
    target_config_from_config,
)


TransformFn = Callable[[dict[str, Any]], dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/unet_baseline.yaml")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="outputs/safeaug_visual_check")
    parser.add_argument("--sample-indices", nargs="*", type=int, default=None)
    return parser.parse_args()


def identity(sample: dict[str, Any]) -> dict[str, Any]:
    return sample


def copy_raw_sample(sample: dict[str, Any]) -> dict[str, Any]:
    copied = {}

    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = value

    return copied


def contiguous_sample(sample: dict[str, Any]) -> dict[str, Any]:
    converted = {}

    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            converted[key] = np.ascontiguousarray(value)
        else:
            converted[key] = value

    return converted


def forced_transforms() -> list[tuple[str, TransformFn]]:
    aug = SafeAugTransform()

    return [
        ("noaug", lambda sample: sample),
        ("hflip", aug._horizontal_flip),
        ("vflip", aug._vertical_flip),
        ("rot90_k1", lambda sample: aug._rot90(sample, 1)),
        ("rot90_k2", lambda sample: aug._rot90(sample, 2)),
        ("rot90_k3", lambda sample: aug._rot90(sample, 3)),
    ]


def tensor_sample(sample: dict[str, Any], transform_fn: TransformFn) -> dict[str, Any]:
    raw = copy_raw_sample(sample)
    raw = transform_fn(raw)
    raw = contiguous_sample(raw)
    return EvalTransform()(raw)


def to_hwc(image: Any) -> np.ndarray:
    return image.detach().cpu().permute(1, 2, 0).numpy()


def to_hw(value: Any) -> np.ndarray:
    return value.detach().cpu().squeeze(0).numpy()


def offset_array(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy()


def rgb_image(image: np.ndarray) -> Image.Image:
    array = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def normalize_heatmap(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    maximum = float(np.nanmax(value)) if value.size else 0.0

    if maximum <= 0.0:
        return np.zeros_like(value, dtype=np.float32)

    return np.clip(value / maximum, 0.0, 1.0)


def colorize_heatmap(value: np.ndarray, mode: str) -> np.ndarray:
    normalized = normalize_heatmap(value)

    if mode == "magma":
        colors = np.asarray(
            [(0, 0, 4), (73, 16, 108), (183, 55, 121), (249, 142, 8), (252, 253, 191)],
            dtype=np.float32,
        )
    else:
        colors = np.asarray(
            [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
            dtype=np.float32,
        )

    scaled = normalized * float(len(colors) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.clip(lower + 1, 0, len(colors) - 1)
    weight = scaled - lower.astype(np.float32)
    rgb = colors[lower] * (1.0 - weight[..., None]) + colors[upper] * weight[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def blend_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> Image.Image:
    base = np.clip(image * 255.0, 0, 255).astype(np.float32)
    overlay = np.asarray(color, dtype=np.float32)
    active = mask > 0.5
    base[active] = (1.0 - alpha) * base[active] + alpha * overlay
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def blend_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    mode: str,
    alpha_scale: float = 0.75,
) -> Image.Image:
    base = np.clip(image * 255.0, 0, 255).astype(np.float32)
    normalized = normalize_heatmap(heatmap)
    color = colorize_heatmap(heatmap, mode).astype(np.float32)
    alpha = (normalized * alpha_scale)[..., None]
    blended = (1.0 - alpha) * base + alpha * color
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def draw_offset_panel(
    image: np.ndarray,
    mask: np.ndarray,
    offset: np.ndarray,
    normalize_offset: bool,
) -> Image.Image:
    panel = blend_mask(image, mask, color=(255, 255, 255), alpha=0.35)
    draw = ImageDraw.Draw(panel)
    height, width = mask.shape
    stride = max(8, min(height, width) // 24)

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            if mask[y, x] <= 0.5:
                continue

            dx = float(offset[0, y, x])
            dy = float(offset[1, y, x])

            if normalize_offset:
                dx *= float(width)
                dy *= float(height)

            end_x = x + dx
            end_y = y + dy
            draw.line((x, y, end_x, end_y), fill=(0, 255, 255), width=1)
            draw.ellipse(
                (end_x - 1.5, end_y - 1.5, end_x + 1.5, end_y + 1.5),
                fill=(0, 255, 255),
            )

    return panel


def titled_panel(title: str, image: Image.Image) -> Image.Image:
    title_height = 28
    panel = Image.new("RGB", (image.width, image.height + title_height), "white")
    panel.paste(image, (0, title_height))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), title, fill="black")
    return panel


def sample_stats(sample: dict[str, Any]) -> dict[str, Any]:
    mask = to_hw(sample["mask"])
    center = to_hw(sample["center"])
    corner = to_hw(sample["corner"])
    offset = offset_array(sample["offset"])

    return {
        "mask_foreground_pixels": int(mask.sum()),
        "max_center": float(center.max()) if center.size else 0.0,
        "max_corner": float(corner.max()) if corner.size else 0.0,
        "max_abs_offset": float(np.max(np.abs(offset))) if offset.size else 0.0,
        "offset_has_nan_or_inf": bool(not np.isfinite(offset).all()),
    }


def save_grid(
    sample: dict[str, Any],
    output_path: Path,
    title: str,
    normalize_offset: bool,
) -> dict[str, Any]:
    image = to_hwc(sample["image"])
    mask = to_hw(sample["mask"])
    boundary = to_hw(sample["boundary"])
    corner = to_hw(sample["corner"])
    center = to_hw(sample["center"])
    offset = offset_array(sample["offset"])

    panels = [
        titled_panel(f"{title}: RGB image", rgb_image(image)),
        titled_panel("Image + mask", blend_mask(image, mask, (255, 0, 0), 0.45)),
        titled_panel(
            "Image + boundary",
            blend_mask(image, boundary, (255, 255, 0), 0.7),
        ),
        titled_panel("Image + corner heatmap", blend_heatmap(image, corner, "viridis")),
        titled_panel("Image + center heatmap", blend_heatmap(image, center, "magma")),
        titled_panel(
            "Offset quiver over mask",
            draw_offset_panel(image, mask, offset, normalize_offset),
        ),
    ]

    panel_width = max(panel.width for panel in panels)
    panel_height = max(panel.height for panel in panels)
    grid = Image.new("RGB", (panel_width * 3, panel_height * 2), "white")

    for index, panel in enumerate(panels):
        x = (index % 3) * panel_width
        y = (index // 3) * panel_height
        grid.paste(panel, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return sample_stats(sample)


def parse_sample_indices(
    args: argparse.Namespace,
    dataset: BuildingFootprintDataset,
) -> list[int]:
    if args.sample_indices is not None and len(args.sample_indices) > 0:
        indices = [int(index) for index in args.sample_indices]
    else:
        count = min(int(args.num_samples), len(dataset))
        indices = list(range(count))

    for index in indices:
        if index < 0 or index >= len(dataset):
            raise IndexError(
                f"Sample index {index} out of range for dataset length {len(dataset)}"
            )

    return indices


def write_summary(
    output_dir: Path,
    config_path: str,
    split: str,
    sample_indices: list[int],
    transform_names: list[str],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        f"config: {config_path}",
        f"split: {split}",
        f"sample_indices: {sample_indices}",
        f"output_dir: {output_dir}",
        f"transforms_visualized: {transform_names}",
        "",
        "images:",
    ]

    for row in rows:
        lines.extend(
            [
                f"- file: {row['file']}",
                f"  sample_index: {row['sample_index']}",
                f"  transform: {row['transform']}",
                f"  image_id: {row['image_id']}",
                f"  mask_foreground_pixels: {row['mask_foreground_pixels']}",
                f"  max_center_heatmap_value: {row['max_center']:.6g}",
                f"  max_corner_heatmap_value: {row['max_corner']:.6g}",
                f"  max_abs_offset_value: {row['max_abs_offset']:.6g}",
                f"  offset_contains_nan_or_inf: {row['offset_has_nan_or_inf']}",
            ]
        )

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config, root=ROOT)
    target_config = target_config_from_config(config)
    output_dir = resolve_path(args.output_dir, root=ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = BuildingFootprintDataset(
        manifest_path=manifest_path_from_config(config, args.split, root=ROOT),
        target_config=target_config,
        transform=identity,
    )
    indices = parse_sample_indices(args, dataset)
    transforms = forced_transforms()
    normalize_offset = bool(target_config["normalize_offset"])

    summary_rows = []

    for sample_number, dataset_index in enumerate(indices):
        raw_sample = dataset[dataset_index]

        for transform_name, transform_fn in transforms:
            rendered_sample = tensor_sample(raw_sample, transform_fn)
            output_path = output_dir / f"sample_{sample_number:03d}_{transform_name}.png"
            stats = save_grid(
                rendered_sample,
                output_path,
                title=f"sample {dataset_index} {transform_name}",
                normalize_offset=normalize_offset,
            )
            summary_rows.append(
                {
                    **stats,
                    "file": output_path.name,
                    "sample_index": dataset_index,
                    "transform": transform_name,
                    "image_id": str(raw_sample["image_id"]),
                }
            )
            print(f"Saved {output_path}")

    write_summary(
        output_dir=output_dir,
        config_path=args.config,
        split=args.split,
        sample_indices=indices,
        transform_names=[name for name, _ in transforms],
        rows=summary_rows,
    )
    print(f"Saved {output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()

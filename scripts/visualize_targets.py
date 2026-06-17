import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geobuild.data.rasterize import TargetBundle, rasterize_record, summarize_targets
from geobuild.data.records import ImageRecord


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    return config


def load_record(manifest: Path, index: int) -> ImageRecord:
    with manifest.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return ImageRecord.from_dict(json.loads(line))

    raise IndexError(f"Index {index} not found in {manifest}")


def target_config(config: dict[str, Any]) -> dict[str, Any]:
    targets = config["targets"]
    corner = targets["corner"]
    center_method = targets["center"]["method"]

    if center_method != "distance_transform":
        raise ValueError(
            "Unsupported targets.center.method: "
            f"{center_method!r}. Only 'distance_transform' is supported."
        )

    return {
        "boundary_width": int(targets["boundary_width"]),
        "corner_radius": int(corner["radius"]),
        "corner_sigma": float(corner["sigma"]),
        "corner_source": corner["source"],
        "corner_simplify_tolerance": float(corner["simplify_tolerance"]),
        "corner_cumulative_turn_angle_degrees": float(
            corner["cumulative_turn_angle_degrees"]
        ),
        "center_radius": int(targets["center"]["radius"]),
        "center_sigma": float(targets["center"]["sigma"]),
        "center_method": center_method,
        "normalize_offset": bool(targets["offset"]["normalize"]),
    }


def print_summary(record: ImageRecord, targets: TargetBundle) -> None:
    print(f"image_id: {record.image_id}")
    print(f"split: {record.split}")
    print(f"image_size: {record.width}x{record.height}")
    print(f"num_polygons: {len(record.polygons)}")

    for key, value in summarize_targets(targets).items():
        print(f"{key}: {value}")


def draw_polygon_lines(
    ax: plt.Axes,
    record: ImageRecord,
    color: str | None = None,
) -> None:
    for polygon in record.polygons:
        xs = [point[0] for point in polygon.exterior]
        ys = [point[1] for point in polygon.exterior]

        if len(xs) == 0:
            continue

        xs.append(xs[0])
        ys.append(ys[0])

        ax.plot(xs, ys, color=color, linewidth=1)


def draw_polygon_overlay(ax: plt.Axes, image: Image.Image, record: ImageRecord) -> None:
    ax.imshow(image, aspect="equal")
    draw_polygon_lines(ax, record)

    ax.set_title("Image + polygons")
    set_image_axes(ax, record)


def set_image_axes(ax: plt.Axes, record: ImageRecord) -> None:
    ax.set_xlim(0, record.width)
    ax.set_ylim(record.height, 0)
    ax.axis("off")


def add_image_panel(
    ax: plt.Axes,
    data: np.ndarray,
    title: str,
    record: ImageRecord,
    cmap: str = "viridis",
) -> None:
    ax.imshow(data, cmap=cmap, aspect="equal")
    ax.set_title(title)
    set_image_axes(ax, record)


def draw_corner_panel(
    ax: plt.Axes,
    targets: TargetBundle,
    record: ImageRecord,
) -> None:
    ax.imshow(targets.corner, cmap="viridis", aspect="equal")
    draw_polygon_lines(ax, record, color="white")
    ax.set_title("Corner heatmap")
    set_image_axes(ax, record)


def draw_center_panel(
    ax: plt.Axes,
    targets: TargetBundle,
    record: ImageRecord,
) -> None:
    ax.imshow(targets.center, cmap="viridis", aspect="equal")
    draw_polygon_lines(ax, record, color="white")
    ax.set_title("Center heatmap")
    set_image_axes(ax, record)


def offset_preview(
    targets: TargetBundle,
    width: int,
    height: int,
    normalize_offset: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = targets.offset[0].copy()
    dy = targets.offset[1].copy()

    if normalize_offset:
        dx *= float(width)
        dy *= float(height)

    magnitude = np.sqrt(dx**2 + dy**2)
    magnitude[targets.mask == 0] = 0.0

    return magnitude, dx, dy


def draw_offset_panel(
    ax: plt.Axes,
    targets: TargetBundle,
    record: ImageRecord,
    normalize_offset: bool,
    stride: int,
) -> None:
    magnitude, dx, dy = offset_preview(
        targets,
        record.width,
        record.height,
        normalize_offset,
    )
    ax.imshow(magnitude, cmap="magma", aspect="equal")

    if stride > 0:
        ys = np.arange(0, record.height, stride)
        xs = np.arange(0, record.width, stride)
        grid_x, grid_y = np.meshgrid(xs, ys)
        mask_samples = targets.mask[grid_y, grid_x] > 0

        if np.any(mask_samples):
            ax.quiver(
                grid_x[mask_samples],
                grid_y[mask_samples],
                dx[grid_y, grid_x][mask_samples],
                dy[grid_y, grid_x][mask_samples],
                color="white",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.003,
            )

    ax.set_title("Offset magnitude")
    set_image_axes(ax, record)


def create_figure(
    image: Image.Image,
    record: ImageRecord,
    targets: TargetBundle,
    normalize_offset: bool,
    offset_stride: int,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    flat_axes = axes.ravel()

    draw_polygon_overlay(flat_axes[0], image, record)
    add_image_panel(flat_axes[1], targets.mask, "Mask", record, cmap="gray")
    add_image_panel(flat_axes[2], targets.boundary, "Boundary", record, cmap="gray")
    draw_corner_panel(flat_axes[3], targets, record)
    draw_center_panel(flat_axes[4], targets, record)
    draw_offset_panel(
        flat_axes[5],
        targets,
        record,
        normalize_offset,
        offset_stride,
    )

    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--offset-stride", type=int, default=24)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    params = target_config(config)

    manifest = ROOT / args.manifest
    record = load_record(manifest, args.index)
    image = Image.open(record.image_path).convert("RGB")

    targets = rasterize_record(
        record,
        boundary_width=params["boundary_width"],
        corner_radius=params["corner_radius"],
        corner_sigma=params["corner_sigma"],
        corner_source=params["corner_source"],
        corner_simplify_tolerance=params["corner_simplify_tolerance"],
        corner_cumulative_turn_angle_degrees=params[
            "corner_cumulative_turn_angle_degrees"
        ],
        center_radius=params["center_radius"],
        center_sigma=params["center_sigma"],
        normalize_offset=params["normalize_offset"],
    )
    print_summary(record, targets)

    fig = create_figure(
        image,
        record,
        targets,
        normalize_offset=params["normalize_offset"],
        offset_stride=args.offset_stride,
    )

    if args.out:
        output_path = ROOT / args.out
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved visualization to: {output_path}")

    if args.show or not args.out:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

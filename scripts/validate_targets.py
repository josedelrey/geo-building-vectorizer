import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geobuild.data.rasterize import TargetBundle, rasterize_record, summarize_targets
from geobuild.data.records import ImageRecord
from geobuild.utils.config import load_config, resolve_path, target_config_from_config


@dataclass
class RecordValidation:
    manifest_index: int
    image_id: int | str
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ValidationTotals:
    processed_samples: int = 0
    total_polygons: int = 0
    total_mask_pixels: int = 0
    total_boundary_pixels: int = 0
    total_instances_non_empty: int = 0
    non_empty_images: int = 0
    empty_records: int = 0
    records_with_warnings: int = 0
    records_with_errors: int = 0
    max_abs_offset: float = 0.0


def iter_records(
    manifest: Path,
    stride: int,
    max_samples: int | None,
) -> Iterator[tuple[int, ImageRecord]]:
    processed = 0

    with manifest.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index % stride != 0:
                continue

            if max_samples is not None and processed >= max_samples:
                break

            processed += 1
            yield index, ImageRecord.from_dict(json.loads(line))


def count_records(
    manifest: Path,
    stride: int,
    max_samples: int | None,
) -> int:
    count = 0

    with manifest.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index % stride != 0 or not line.strip():
                continue

            if max_samples is not None and count >= max_samples:
                break

            count += 1

    return count


def _array_has_nan(array: np.ndarray) -> bool:
    if not np.issubdtype(array.dtype, np.floating):
        return False

    return bool(np.isnan(array).any())


def _invalid_binary_values(array: np.ndarray) -> np.ndarray:
    values = np.unique(array)
    return values[~np.isin(values, [0, 1])]


def validate_shapes(
    record: ImageRecord,
    targets: TargetBundle,
    result: RecordValidation,
) -> None:
    height = int(record.height)
    width = int(record.width)
    expected_shapes = {
        "mask": (height, width),
        "boundary": (height, width),
        "corner": (height, width),
        "center": (height, width),
        "offset": (2, height, width),
        "instance": (height, width),
    }

    for name, expected_shape in expected_shapes.items():
        actual_shape = getattr(targets, name).shape

        if actual_shape != expected_shape:
            result.errors.append(
                f"{name} shape is {actual_shape}, expected {expected_shape}"
            )


def validate_dtypes(targets: TargetBundle, result: RecordValidation) -> None:
    expected_dtypes = {
        "mask": np.uint8,
        "boundary": np.uint8,
        "corner": np.float32,
        "center": np.float32,
        "offset": np.float32,
        "instance": np.int32,
    }

    for name, expected_dtype in expected_dtypes.items():
        actual_dtype = getattr(targets, name).dtype

        if actual_dtype != expected_dtype:
            result.errors.append(
                f"{name} dtype is {actual_dtype}, expected {np.dtype(expected_dtype)}"
            )


def validate_values(
    record: ImageRecord,
    targets: TargetBundle,
    summary: dict[str, Any],
    normalize_offset: bool,
    result: RecordValidation,
) -> None:
    for name, array in targets.to_dict().items():
        if _array_has_nan(array):
            result.errors.append(f"{name} contains NaN")

    bad_mask_values = _invalid_binary_values(targets.mask)

    if bad_mask_values.size > 0:
        result.errors.append(f"mask has non-binary values: {bad_mask_values.tolist()}")

    bad_boundary_values = _invalid_binary_values(targets.boundary)

    if bad_boundary_values.size > 0:
        result.errors.append(
            f"boundary has non-binary values: {bad_boundary_values.tolist()}"
        )

    if targets.instance.size and int(np.min(targets.instance)) < 0:
        result.errors.append("instance ids contain negative values")

    mask_pixels = int(summary["mask_pixels"])
    boundary_pixels = int(summary["boundary_pixels"])
    num_instances = int(summary["num_instances"])

    if len(record.polygons) == 0:
        result.warnings.append("record has no polygons")
    elif mask_pixels == 0:
        result.warnings.append("record has polygons but rasterized mask is empty")

    if num_instances > 0 and float(summary["center_max"]) <= 0.0:
        result.warnings.append("record has instances but center target is empty")

    if num_instances > 0:
        inside_offset = targets.offset[:, targets.mask > 0]

        if inside_offset.size == 0 or not np.any(np.abs(inside_offset) > 1e-6):
            result.warnings.append("record has instances but zero offset inside mask")

    outside_offset = targets.offset[:, targets.mask == 0]

    if outside_offset.size > 0:
        outside_max = float(np.max(np.abs(outside_offset)))

        if outside_max > 1e-6:
            result.errors.append(
                f"offset outside mask is non-zero; max abs={outside_max:.6g}"
            )

    if np.any((targets.instance > 0) & (targets.mask == 0)):
        result.errors.append("instance pixels are not a subset of mask pixels")

    if mask_pixels > 4 and boundary_pixels == 0:
        result.warnings.append("mask is non-empty but boundary target is empty")

    max_abs_offset = float(np.max(np.abs(targets.offset))) if targets.offset.size else 0.0

    if normalize_offset:
        if max_abs_offset > 1.05:
            result.warnings.append(
                f"unusually high normalized offset value: {max_abs_offset:.6g}"
            )
    else:
        max_expected = float(max(record.width, record.height))

        if max_abs_offset > max_expected:
            result.warnings.append(
                f"unusually high pixel offset value: {max_abs_offset:.6g}"
            )


def validate_record(
    manifest_index: int,
    record: ImageRecord,
    params: dict[str, Any],
) -> tuple[RecordValidation, dict[str, Any] | None, TargetBundle | None]:
    result = RecordValidation(
        manifest_index=manifest_index,
        image_id=record.image_id,
    )

    try:
        targets = rasterize_record(record, **params)
        summary = summarize_targets(targets)
    except Exception as exc:
        result.errors.append(f"rasterization failed: {type(exc).__name__}: {exc}")
        return result, None, None

    validate_shapes(record, targets, result)
    validate_dtypes(targets, result)
    validate_values(
        record,
        targets,
        summary,
        normalize_offset=params["normalize_offset"],
        result=result,
    )

    return result, summary, targets


def update_totals(
    totals: ValidationTotals,
    record: ImageRecord,
    result: RecordValidation,
    summary: dict[str, Any] | None,
    targets: TargetBundle | None,
) -> None:
    totals.processed_samples += 1
    totals.total_polygons += len(record.polygons)

    if len(record.polygons) == 0:
        totals.empty_records += 1

    if result.warnings:
        totals.records_with_warnings += 1

    if result.errors:
        totals.records_with_errors += 1

    if summary is not None:
        totals.total_mask_pixels += int(summary["mask_pixels"])
        totals.total_boundary_pixels += int(summary["boundary_pixels"])

        if int(summary["num_instances"]) > 0:
            totals.non_empty_images += 1
            totals.total_instances_non_empty += int(summary["num_instances"])

    if targets is not None and targets.offset.size:
        totals.max_abs_offset = max(
            totals.max_abs_offset,
            float(np.max(np.abs(targets.offset))),
        )


def format_example(result: RecordValidation, messages: list[str]) -> str:
    return (
        f"index={result.manifest_index} image_id={result.image_id}: "
        + "; ".join(messages)
    )


def print_examples(title: str, examples: list[str]) -> None:
    print(f"{title}:")

    if not examples:
        print("  none")
        return

    for example in examples:
        print(f"  - {example}")


def print_summary(
    manifest: Path,
    totals: ValidationTotals,
    warning_examples: list[str],
    error_examples: list[str],
) -> None:
    average_instances = 0.0

    if totals.non_empty_images > 0:
        average_instances = totals.total_instances_non_empty / totals.non_empty_images

    print("Target validation summary")
    print(f"manifest: {manifest}")
    print(f"processed_samples: {totals.processed_samples}")
    print(f"total_polygons: {totals.total_polygons}")
    print(f"total_mask_pixels: {totals.total_mask_pixels}")
    print(f"total_boundary_pixels: {totals.total_boundary_pixels}")
    print(f"average_instances_per_non_empty_image: {average_instances:.4f}")
    print(f"empty_records: {totals.empty_records}")
    print(f"records_with_warnings: {totals.records_with_warnings}")
    print(f"records_with_errors: {totals.records_with_errors}")
    print(f"maximum_absolute_offset_value: {totals.max_abs_offset:.6g}")
    print_examples("first_warning_examples", warning_examples)
    print_examples("first_error_examples", error_examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.stride < 1:
        raise ValueError(f"--stride must be at least 1, got {args.stride}")

    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError(f"--max-samples must be non-negative, got {args.max_samples}")

    config = load_config(args.config, root=ROOT)
    config.setdefault("targets", {})["active"] = "all"
    params = target_config_from_config(config)
    manifest = resolve_path(args.manifest, root=ROOT)

    totals = ValidationTotals()
    warning_examples: list[str] = []
    error_examples: list[str] = []
    total_records = count_records(
        manifest,
        stride=args.stride,
        max_samples=args.max_samples,
    )
    records = iter_records(
        manifest,
        stride=args.stride,
        max_samples=args.max_samples,
    )

    for manifest_index, record in tqdm(
        records,
        total=total_records,
        desc="Validating targets",
        ascii=True,
        unit="record",
    ):
        result, summary, targets = validate_record(manifest_index, record, params)
        update_totals(totals, record, result, summary, targets)

        if result.warnings and len(warning_examples) < 5:
            warning_examples.append(format_example(result, result.warnings))

        if result.errors and len(error_examples) < 5:
            error_examples.append(format_example(result, result.errors))

    print_summary(manifest, totals, warning_examples, error_examples)

    if args.fail_on_error and totals.records_with_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

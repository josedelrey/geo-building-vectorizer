import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from geobuild.eval.matching import ObjectMatch
from geobuild.eval.raster_metrics import RasterImageMetrics, RasterSplitMetrics
from geobuild.eval.vector_metrics import VectorSplitMetrics


PER_IMAGE_FIELDS = [
    "image_id",
    "height",
    "width",
    "num_gt",
    "num_pred",
    "mask_iou",
    "dice",
    "precision",
    "recall",
    "boundary_f1_from_mask",
    "boundary_precision_from_mask",
    "boundary_recall_from_mask",
    "tp",
    "fp",
    "fn",
    "tn",
    "pred_positive_pixels",
    "gt_positive_pixels",
    "pred_boundary_pixels",
    "gt_boundary_pixels",
    "tp_50",
    "fp_50",
    "fn_50",
    "f1_50",
    "mean_matched_iou_50",
]

PER_OBJECT_FIELDS = [
    "image_id",
    "status",
    "iou_threshold",
    "iou",
    "pred_index",
    "gt_index",
    "pred_score",
    "pred_source",
    "pred_source_id",
    "gt_id",
    "gt_area",
    "gt_vertex_count",
    "pred_area",
]


def build_summary(
    *,
    experiment: str,
    split: str,
    vectorizer: str,
    raster_metrics: RasterSplitMetrics,
    vector_metrics: VectorSplitMetrics,
    context: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "experiment": experiment,
        "split": split,
        "vectorizer": vectorizer,
        "num_images": int(raster_metrics.num_images),
        "num_gt": int(vector_metrics.num_gt),
        "num_pred": int(vector_metrics.num_pred),
        "mask_iou": float(raster_metrics.mask_iou),
        "dice": float(raster_metrics.dice),
        "precision": float(raster_metrics.precision),
        "recall": float(raster_metrics.recall),
        "boundary_f1": float(raster_metrics.boundary_f1_from_mask),
        "boundary_f1_from_mask": float(raster_metrics.boundary_f1_from_mask),
        "f1_50": float(vector_metrics.f1_50),
        "ap50": float(vector_metrics.ap50),
        "ap75": float(vector_metrics.ap75),
        "mean_matched_iou_50": float(vector_metrics.mean_matched_iou_50),
        "invalid_polygon_ratio": float(vector_metrics.invalid_polygon_ratio),
        "mean_vertex_count": float(vector_metrics.mean_vertex_count),
        "mean_area_error_rel": float(vector_metrics.mean_area_error_rel),
        "mean_perimeter_error_rel": float(vector_metrics.mean_perimeter_error_rel),
    }
    summary.update(context)
    return summary


def write_summary_json(summary: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def write_per_image_metrics_csv(
    metrics: list[RasterImageMetrics] | list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    rows = [
        metric.to_dict() if hasattr(metric, "to_dict") else dict(metric)
        for metric in metrics
    ]
    _write_csv(rows, output_path, PER_IMAGE_FIELDS)


def write_per_object_matches_csv(
    matches: list[ObjectMatch],
    output_path: str | Path,
) -> None:
    rows = [match.to_dict() for match in matches]
    _write_csv(rows, output_path, PER_OBJECT_FIELDS)


def _write_csv(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    fieldnames: list[str],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, Path):
        return str(value)

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

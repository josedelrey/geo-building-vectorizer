import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from geobuild.data.rasterize import rasterize_record
from geobuild.data.records import ImageRecord
from geobuild.eval.boundary_metrics import (
    BoundaryCounts,
    boundary_counts_from_masks,
    boundary_metrics_from_counts,
)


@dataclass(frozen=True)
class RasterConfusionCounts:
    tp: int
    fp: int
    fn: int
    tn: int


@dataclass(frozen=True)
class RasterImageMetrics:
    image_id: str
    height: int
    width: int
    mask_threshold: float
    boundary_tolerance_px: int
    mask_iou: float
    dice: float
    precision: float
    recall: float
    boundary_f1_from_mask: float
    boundary_precision_from_mask: float
    boundary_recall_from_mask: float
    tp: int
    fp: int
    fn: int
    tn: int
    pred_positive_pixels: int
    gt_positive_pixels: int
    matched_pred_boundary_pixels: int
    matched_gt_boundary_pixels: int
    pred_boundary_pixels: int
    gt_boundary_pixels: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "height": int(self.height),
            "width": int(self.width),
            "mask_threshold": float(self.mask_threshold),
            "boundary_tolerance_px": int(self.boundary_tolerance_px),
            "mask_iou": float(self.mask_iou),
            "dice": float(self.dice),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "boundary_f1_from_mask": float(self.boundary_f1_from_mask),
            "boundary_precision_from_mask": float(self.boundary_precision_from_mask),
            "boundary_recall_from_mask": float(self.boundary_recall_from_mask),
            "tp": int(self.tp),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "tn": int(self.tn),
            "pred_positive_pixels": int(self.pred_positive_pixels),
            "gt_positive_pixels": int(self.gt_positive_pixels),
            "matched_pred_boundary_pixels": int(self.matched_pred_boundary_pixels),
            "matched_gt_boundary_pixels": int(self.matched_gt_boundary_pixels),
            "pred_boundary_pixels": int(self.pred_boundary_pixels),
            "gt_boundary_pixels": int(self.gt_boundary_pixels),
        }


@dataclass(frozen=True)
class RasterSplitMetrics:
    num_images: int
    mask_threshold: float
    boundary_tolerance_px: int
    mask_iou: float
    dice: float
    precision: float
    recall: float
    boundary_f1_from_mask: float
    boundary_precision_from_mask: float
    boundary_recall_from_mask: float
    tp: int
    fp: int
    fn: int
    tn: int
    pred_positive_pixels: int
    gt_positive_pixels: int
    matched_pred_boundary_pixels: int
    matched_gt_boundary_pixels: int
    pred_boundary_pixels: int
    gt_boundary_pixels: int
    per_image: list[RasterImageMetrics] = field(default_factory=list)

    def to_dict(self, include_per_image: bool = False) -> dict[str, Any]:
        data = {
            "num_images": int(self.num_images),
            "mask_threshold": float(self.mask_threshold),
            "boundary_tolerance_px": int(self.boundary_tolerance_px),
            "mask_iou": float(self.mask_iou),
            "dice": float(self.dice),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "boundary_f1_from_mask": float(self.boundary_f1_from_mask),
            "boundary_precision_from_mask": float(self.boundary_precision_from_mask),
            "boundary_recall_from_mask": float(self.boundary_recall_from_mask),
            "tp": int(self.tp),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "tn": int(self.tn),
            "pred_positive_pixels": int(self.pred_positive_pixels),
            "gt_positive_pixels": int(self.gt_positive_pixels),
            "matched_pred_boundary_pixels": int(self.matched_pred_boundary_pixels),
            "matched_gt_boundary_pixels": int(self.matched_gt_boundary_pixels),
            "pred_boundary_pixels": int(self.pred_boundary_pixels),
            "gt_boundary_pixels": int(self.gt_boundary_pixels),
        }

        if include_per_image:
            data["per_image"] = [metrics.to_dict() for metrics in self.per_image]

        return data


def rasterize_gt_mask(
    record: ImageRecord,
    raster_config: dict[str, Any] | None = None,
) -> np.ndarray:
    params = {} if raster_config is None else dict(raster_config)
    params["active_targets"] = {"mask"}
    targets = rasterize_record(record, **params)

    if targets.mask is None:
        raise RuntimeError(f"Rasterization did not return mask for {record.image_id!r}")

    return np.ascontiguousarray(targets.mask.astype(bool))


def mask_metrics_from_arrays(
    mask_prob: np.ndarray,
    gt_mask: np.ndarray,
    mask_threshold: float = 0.5,
    boundary_tolerance_px: int = 2,
) -> tuple[RasterConfusionCounts, BoundaryCounts]:
    probability = _as_mask_probability(mask_prob)
    target = _as_binary_mask(gt_mask)

    if probability.shape != target.shape:
        raise ValueError(
            "mask_prob and gt_mask must have the same shape, got "
            f"{probability.shape} and {target.shape}"
        )

    prediction = probability >= float(mask_threshold)
    counts = confusion_counts(prediction, target)
    boundary_counts = boundary_counts_from_masks(
        prediction,
        target,
        tolerance_px=int(boundary_tolerance_px),
    )
    return counts, boundary_counts


def evaluate_raster_image(
    record: ImageRecord,
    prediction_record: dict[str, Any],
    raster_config: dict[str, Any] | None = None,
    metrics_config: dict[str, Any] | None = None,
) -> RasterImageMetrics:
    prediction_image_id = prediction_record.get("image_id")

    if prediction_image_id is not None and str(prediction_image_id) != str(record.image_id):
        raise ValueError(
            "Prediction image_id does not match ImageRecord: "
            f"{prediction_image_id!r} != {record.image_id!r}"
        )

    metrics_settings = _metrics_settings(metrics_config)
    mask_threshold = float(
        metrics_settings.get("mask_threshold", metrics_settings.get("threshold", 0.5))
    )
    boundary_tolerance_px = int(metrics_settings.get("boundary_tolerance_px", 2))
    mask_prob = load_mask_probability(prediction_record)
    gt_mask = rasterize_gt_mask(record, raster_config=raster_config)
    counts, boundary_counts = mask_metrics_from_arrays(
        mask_prob,
        gt_mask,
        mask_threshold=mask_threshold,
        boundary_tolerance_px=boundary_tolerance_px,
    )
    raster_metrics = raster_metrics_from_counts(counts)
    boundary_metrics = boundary_metrics_from_counts(boundary_counts)

    return RasterImageMetrics(
        image_id=str(record.image_id),
        height=int(record.height),
        width=int(record.width),
        mask_threshold=mask_threshold,
        boundary_tolerance_px=boundary_tolerance_px,
        mask_iou=raster_metrics["mask_iou"],
        dice=raster_metrics["dice"],
        precision=raster_metrics["precision"],
        recall=raster_metrics["recall"],
        boundary_f1_from_mask=boundary_metrics.boundary_f1,
        boundary_precision_from_mask=boundary_metrics.boundary_precision,
        boundary_recall_from_mask=boundary_metrics.boundary_recall,
        tp=counts.tp,
        fp=counts.fp,
        fn=counts.fn,
        tn=counts.tn,
        pred_positive_pixels=int(counts.tp + counts.fp),
        gt_positive_pixels=int(counts.tp + counts.fn),
        matched_pred_boundary_pixels=boundary_counts.matched_pred,
        matched_gt_boundary_pixels=boundary_counts.matched_gt,
        pred_boundary_pixels=boundary_counts.pred_boundary_pixels,
        gt_boundary_pixels=boundary_counts.gt_boundary_pixels,
    )


def evaluate_raster_split(
    records: list[ImageRecord],
    prediction_records: list[dict[str, Any]],
    raster_config: dict[str, Any] | None = None,
    metrics_config: dict[str, Any] | None = None,
) -> RasterSplitMetrics:
    predictions_by_image_id = {
        str(record["image_id"]): record
        for record in prediction_records
    }
    per_image = []

    for record in records:
        image_id = str(record.image_id)

        if image_id not in predictions_by_image_id:
            raise KeyError(f"Missing prediction record for image_id={image_id!r}")

        per_image.append(
            evaluate_raster_image(
                record,
                predictions_by_image_id[image_id],
                raster_config=raster_config,
                metrics_config=metrics_config,
            )
        )

    return aggregate_raster_metrics(per_image)


def aggregate_raster_metrics(
    per_image: list[RasterImageMetrics],
) -> RasterSplitMetrics:
    if not per_image:
        return RasterSplitMetrics(
            num_images=0,
            mask_threshold=0.5,
            boundary_tolerance_px=2,
            mask_iou=1.0,
            dice=1.0,
            precision=1.0,
            recall=1.0,
            boundary_f1_from_mask=1.0,
            boundary_precision_from_mask=1.0,
            boundary_recall_from_mask=1.0,
            tp=0,
            fp=0,
            fn=0,
            tn=0,
            pred_positive_pixels=0,
            gt_positive_pixels=0,
            matched_pred_boundary_pixels=0,
            matched_gt_boundary_pixels=0,
            pred_boundary_pixels=0,
            gt_boundary_pixels=0,
            per_image=[],
        )

    _validate_consistent_metric_settings(per_image)
    counts = RasterConfusionCounts(
        tp=sum(item.tp for item in per_image),
        fp=sum(item.fp for item in per_image),
        fn=sum(item.fn for item in per_image),
        tn=sum(item.tn for item in per_image),
    )
    boundary_counts = BoundaryCounts(
        matched_pred=sum(item.matched_pred_boundary_pixels for item in per_image),
        pred_boundary_pixels=sum(item.pred_boundary_pixels for item in per_image),
        matched_gt=sum(item.matched_gt_boundary_pixels for item in per_image),
        gt_boundary_pixels=sum(item.gt_boundary_pixels for item in per_image),
    )
    raster_metrics = raster_metrics_from_counts(counts)
    boundary_metrics = boundary_metrics_from_counts(boundary_counts)

    return RasterSplitMetrics(
        num_images=len(per_image),
        mask_threshold=float(per_image[0].mask_threshold),
        boundary_tolerance_px=int(per_image[0].boundary_tolerance_px),
        mask_iou=raster_metrics["mask_iou"],
        dice=raster_metrics["dice"],
        precision=raster_metrics["precision"],
        recall=raster_metrics["recall"],
        boundary_f1_from_mask=boundary_metrics.boundary_f1,
        boundary_precision_from_mask=boundary_metrics.boundary_precision,
        boundary_recall_from_mask=boundary_metrics.boundary_recall,
        tp=counts.tp,
        fp=counts.fp,
        fn=counts.fn,
        tn=counts.tn,
        pred_positive_pixels=int(counts.tp + counts.fp),
        gt_positive_pixels=int(counts.tp + counts.fn),
        matched_pred_boundary_pixels=boundary_counts.matched_pred,
        matched_gt_boundary_pixels=boundary_counts.matched_gt,
        pred_boundary_pixels=boundary_counts.pred_boundary_pixels,
        gt_boundary_pixels=boundary_counts.gt_boundary_pixels,
        per_image=per_image,
    )


def confusion_counts(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> RasterConfusionCounts:
    prediction = _as_binary_mask(pred_mask)
    target = _as_binary_mask(gt_mask)

    if prediction.shape != target.shape:
        raise ValueError(
            f"pred_mask and gt_mask shapes differ: {prediction.shape} and {target.shape}"
        )

    return RasterConfusionCounts(
        tp=int(np.count_nonzero(prediction & target)),
        fp=int(np.count_nonzero(prediction & ~target)),
        fn=int(np.count_nonzero(~prediction & target)),
        tn=int(np.count_nonzero(~prediction & ~target)),
    )


def raster_metrics_from_counts(counts: RasterConfusionCounts) -> dict[str, float]:
    mask_iou = _ratio_empty_perfect(
        counts.tp,
        counts.tp + counts.fp + counts.fn,
    )
    dice = _ratio_empty_perfect(
        2 * counts.tp,
        2 * counts.tp + counts.fp + counts.fn,
    )
    precision = _ratio_with_empty_perfect(
        counts.tp,
        counts.tp + counts.fp,
        counts.tp + counts.fn,
    )
    recall = _ratio_with_empty_perfect(
        counts.tp,
        counts.tp + counts.fn,
        counts.tp + counts.fp,
    )

    return {
        "mask_iou": mask_iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


def load_prediction_records(predictions: str | Path) -> list[dict[str, Any]]:
    path = Path(predictions)
    manifest_path = path / "predictions.jsonl" if path.is_dir() else path

    if not manifest_path.exists():
        raise FileNotFoundError(f"Prediction manifest does not exist: {manifest_path}")

    records = []

    with manifest_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in prediction manifest {manifest_path} "
                    f"at line {line_number}"
                ) from exc

    return records


def load_mask_probability(prediction_record: dict[str, Any]) -> np.ndarray:
    npz_path = Path(prediction_record["npz_path"])

    if not npz_path.exists():
        raise FileNotFoundError(f"Prediction NPZ file does not exist: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        if "mask_prob" in data.files:
            mask_prob = data["mask_prob"]
        elif "mask" in data.files:
            mask_prob = data["mask"]
        else:
            raise KeyError(
                f"Prediction NPZ {npz_path} has no 'mask_prob' or 'mask' output"
            )

    return _as_mask_probability(mask_prob)


def _as_mask_probability(mask_prob: np.ndarray) -> np.ndarray:
    array = np.asarray(mask_prob, dtype=np.float32)

    if array.ndim == 3 and int(array.shape[0]) == 1:
        array = array[0]

    if array.ndim != 2:
        raise ValueError(f"mask_prob must have shape [H, W], got {array.shape}")

    return np.ascontiguousarray(array)


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)

    if array.ndim == 3 and int(array.shape[0]) == 1:
        array = array[0]

    if array.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {array.shape}")

    return np.ascontiguousarray(array.astype(bool))


def _ratio_empty_perfect(numerator: int, denominator: int) -> float:
    if int(denominator) == 0:
        return 1.0

    return float(numerator) / float(denominator)


def _ratio_with_empty_perfect(
    numerator: int,
    denominator: int,
    other_denominator: int,
) -> float:
    if int(denominator) == 0:
        return 1.0 if int(other_denominator) == 0 else 0.0

    return float(numerator) / float(denominator)


def _metrics_settings(metrics_config: dict[str, Any] | None) -> dict[str, Any]:
    if metrics_config is None:
        return {}

    if "metrics" in metrics_config and isinstance(metrics_config["metrics"], dict):
        return dict(metrics_config["metrics"])

    return dict(metrics_config)


def _validate_consistent_metric_settings(per_image: list[RasterImageMetrics]) -> None:
    first = per_image[0]

    for item in per_image[1:]:
        if (
            float(item.mask_threshold) != float(first.mask_threshold)
            or int(item.boundary_tolerance_px) != int(first.boundary_tolerance_px)
        ):
            raise ValueError(
                "Cannot aggregate raster metrics with mixed metric settings: "
                f"first={(first.mask_threshold, first.boundary_tolerance_px)}, "
                f"got={(item.mask_threshold, item.boundary_tolerance_px)}"
            )

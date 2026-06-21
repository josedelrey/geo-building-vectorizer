import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from geobuild.eval.geometry import iter_polygons, repair_geometry, vertex_count
from geobuild.eval.ground_truth import GroundTruthPolygon, ground_truth_polygons_from_record
from geobuild.eval.matching import (
    AveragePrecisionResult,
    ObjectMatch,
    compute_ap50_ap75,
)


@dataclass(frozen=True)
class EvaluatedPredictedPolygon:
    image_id: str
    polygon: BaseGeometry | None
    score: float
    source: str
    source_id: int | str | None
    properties: dict[str, Any] = field(default_factory=dict)
    is_invalid: bool = False
    raw_geometry_type: str | None = None


@dataclass(frozen=True)
class VectorSplitMetrics:
    num_gt: int
    num_pred: int
    f1_50: float
    ap50: float
    ap75: float
    mean_matched_iou_50: float
    invalid_polygon_ratio: float
    mean_vertex_count: float
    mean_area_error_rel: float
    mean_perimeter_error_rel: float
    tp_50: int
    fp_50: int
    fn_50: int
    matches_50: list[ObjectMatch]
    ap50_result: AveragePrecisionResult
    ap75_result: AveragePrecisionResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_gt": int(self.num_gt),
            "num_pred": int(self.num_pred),
            "f1_50": float(self.f1_50),
            "ap50": float(self.ap50),
            "ap75": float(self.ap75),
            "mean_matched_iou_50": float(self.mean_matched_iou_50),
            "invalid_polygon_ratio": float(self.invalid_polygon_ratio),
            "mean_vertex_count": float(self.mean_vertex_count),
            "mean_area_error_rel": float(self.mean_area_error_rel),
            "mean_perimeter_error_rel": float(self.mean_perimeter_error_rel),
            "tp_50": int(self.tp_50),
            "fp_50": int(self.fp_50),
            "fn_50": int(self.fn_50),
        }


def load_predicted_polygons(path: str | Path) -> list[EvaluatedPredictedPolygon]:
    polygons_path = _polygons_jsonl_path(path)

    if not polygons_path.exists():
        raise FileNotFoundError(f"Predicted polygon JSONL does not exist: {polygons_path}")

    polygons = []

    with polygons_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in predicted polygons {polygons_path} "
                    f"at line {line_number}"
                ) from exc

            polygons.append(predicted_polygon_from_record(record))

    return polygons


def predicted_polygon_from_record(
    record: dict[str, Any],
) -> EvaluatedPredictedPolygon:
    raw_geometry = record.get("geometry")
    geometry = None
    raw_geometry_type = None

    if raw_geometry is not None:
        raw_geometry_type = str(raw_geometry.get("type", "")) if isinstance(raw_geometry, dict) else None

        try:
            geometry = shape(raw_geometry)
        except Exception:
            geometry = None

    is_invalid = (
        geometry is None
        or geometry.is_empty
        or not geometry.is_valid
        or float(geometry.area) <= 0.0
    )
    repaired = repair_geometry(geometry)
    score = _float(record.get("score", 0.0), default=0.0)

    return EvaluatedPredictedPolygon(
        image_id=str(record.get("image_id", "")),
        polygon=repaired,
        score=score,
        source=str(record.get("source", "")),
        source_id=record.get("source_id"),
        properties=dict(record.get("properties", {})),
        is_invalid=bool(is_invalid),
        raw_geometry_type=raw_geometry_type,
    )


def ground_truth_by_image(records: list[Any]) -> dict[str, list[GroundTruthPolygon]]:
    grouped: dict[str, list[GroundTruthPolygon]] = {}

    for record in records:
        grouped[str(record.image_id)] = ground_truth_polygons_from_record(record)

    return grouped


def predictions_by_image(
    predictions: list[EvaluatedPredictedPolygon],
) -> dict[str, list[EvaluatedPredictedPolygon]]:
    grouped: dict[str, list[EvaluatedPredictedPolygon]] = {}

    for prediction in predictions:
        grouped.setdefault(str(prediction.image_id), []).append(prediction)

    return grouped


def compute_vector_metrics(
    gt_polygons: list[GroundTruthPolygon],
    pred_polygons: list[EvaluatedPredictedPolygon],
) -> VectorSplitMetrics:
    ap_results = compute_ap50_ap75(gt_polygons, pred_polygons)
    ap50_result = ap_results["ap50"]
    ap75_result = ap_results["ap75"]
    tp_50 = int(ap50_result.true_positives)
    fp_50 = int(ap50_result.false_positives)
    fn_50 = int(ap50_result.false_negatives)
    f1_50 = _f1_from_counts(tp_50, fp_50, fn_50)
    matched_iou = [
        float(match.iou)
        for match in ap50_result.matches
        if match.status == "tp"
    ]
    invalid_count = sum(1 for prediction in pred_polygons if prediction.is_invalid)
    valid_predictions = [
        prediction
        for prediction in pred_polygons
        if not prediction.is_invalid and prediction.polygon is not None
    ]
    vertex_counts = [_geometry_vertex_count(prediction.polygon) for prediction in valid_predictions]
    area_errors, perimeter_errors = _matched_relative_errors(
        gt_polygons,
        pred_polygons,
        ap50_result.matches,
    )

    return VectorSplitMetrics(
        num_gt=len(gt_polygons),
        num_pred=len(pred_polygons),
        f1_50=f1_50,
        ap50=float(ap50_result.ap),
        ap75=float(ap75_result.ap),
        mean_matched_iou_50=_mean_or_zero(matched_iou),
        invalid_polygon_ratio=(
            float(invalid_count) / float(len(pred_polygons))
            if pred_polygons
            else 0.0
        ),
        mean_vertex_count=_mean_or_zero(vertex_counts),
        mean_area_error_rel=_mean_or_zero(area_errors),
        mean_perimeter_error_rel=_mean_or_zero(perimeter_errors),
        tp_50=tp_50,
        fp_50=fp_50,
        fn_50=fn_50,
        matches_50=ap50_result.matches,
        ap50_result=ap50_result,
        ap75_result=ap75_result,
    )


def _matched_relative_errors(
    gt_polygons: list[GroundTruthPolygon],
    pred_polygons: list[EvaluatedPredictedPolygon],
    matches: list[ObjectMatch],
) -> tuple[list[float], list[float]]:
    area_errors = []
    perimeter_errors = []

    for match in matches:
        if match.status != "tp" or match.gt_index is None or match.pred_index is None:
            continue

        gt = gt_polygons[int(match.gt_index)]
        pred = pred_polygons[int(match.pred_index)]

        if pred.polygon is None:
            continue

        gt_area = float(gt.polygon.area)
        gt_perimeter = float(gt.polygon.length)

        if gt_area > 0.0:
            area_errors.append(abs(float(pred.polygon.area) - gt_area) / gt_area)

        if gt_perimeter > 0.0:
            perimeter_errors.append(
                abs(float(pred.polygon.length) - gt_perimeter) / gt_perimeter
            )

    return area_errors, perimeter_errors


def _geometry_vertex_count(geometry: BaseGeometry | None) -> int:
    if geometry is None:
        return 0

    return sum(vertex_count(polygon) for polygon in iter_polygons(geometry))


def _polygons_jsonl_path(path: str | Path) -> Path:
    path = Path(path)

    if path.is_dir():
        return path / "polygons.jsonl"

    return path


def _mean_or_zero(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0

    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * int(tp) + int(fp) + int(fn)

    if denominator == 0:
        return 1.0

    return float(2 * int(tp)) / float(denominator)


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

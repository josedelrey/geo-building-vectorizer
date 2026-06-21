from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry.base import BaseGeometry

from geobuild.eval.geometry import repair_geometry


@dataclass(frozen=True)
class ObjectMatch:
    image_id: str
    status: str
    iou_threshold: float
    iou: float
    pred_index: int | None
    gt_index: int | None
    pred_score: float | None
    pred_source: str | None
    pred_source_id: int | str | None
    gt_id: int | str | None
    gt_area: float | None
    gt_vertex_count: int | None
    pred_area: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "status": self.status,
            "iou_threshold": float(self.iou_threshold),
            "iou": float(self.iou),
            "pred_index": self.pred_index,
            "gt_index": self.gt_index,
            "pred_score": self.pred_score,
            "pred_source": self.pred_source,
            "pred_source_id": self.pred_source_id,
            "gt_id": self.gt_id,
            "gt_area": self.gt_area,
            "gt_vertex_count": self.gt_vertex_count,
            "pred_area": self.pred_area,
        }


@dataclass(frozen=True)
class ImageMatchResult:
    image_id: str
    iou_threshold: float
    matches: list[ObjectMatch]
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class AveragePrecisionResult:
    iou_threshold: float
    ap: float
    num_gt: int
    num_predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    matches: list[ObjectMatch] = field(default_factory=list)


def polygon_iou(a: Any, b: Any) -> float:
    if not isinstance(a, BaseGeometry) or not isinstance(b, BaseGeometry):
        return 0.0

    a = repair_geometry(a)
    b = repair_geometry(b)

    if a is None or b is None:
        return 0.0

    if a.is_empty or b.is_empty or float(a.area) <= 0.0 or float(b.area) <= 0.0:
        return 0.0

    try:
        intersection_area = float(a.intersection(b).area)
        union_area = float(a.union(b).area)
    except Exception:
        a = repair_geometry(a)
        b = repair_geometry(b)

        if a is None or b is None:
            return 0.0

        try:
            intersection_area = float(a.intersection(b).area)
            union_area = float(a.union(b).area)
        except Exception:
            return 0.0

    if union_area <= 0.0:
        return 0.0

    return max(0.0, min(1.0, intersection_area / union_area))


def match_image_predictions(
    gt_polygons: list[Any],
    pred_polygons: list[Any],
    iou_threshold: float,
) -> ImageMatchResult:
    threshold = float(iou_threshold)
    image_id = _image_id(gt_polygons, pred_polygons)
    valid_gt = _valid_indexed_targets(gt_polygons)
    indexed_pred = list(enumerate(pred_polygons))
    matched_gt_indices: set[int] = set()
    matches: list[ObjectMatch] = []

    sorted_predictions = sorted(
        indexed_pred,
        key=lambda item: (-_score(item[1]), item[0]),
    )

    for pred_index, prediction in sorted_predictions:
        best_gt_index = None
        best_gt = None
        best_iou = 0.0

        for gt_index, gt in valid_gt:
            if gt_index in matched_gt_indices:
                continue

            iou = polygon_iou(_polygon(prediction), _polygon(gt))

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
                best_gt = gt

        if best_gt_index is not None and best_gt is not None and best_iou >= threshold:
            matched_gt_indices.add(best_gt_index)
            matches.append(
                _match_record(
                    status="tp",
                    iou_threshold=threshold,
                    iou=best_iou,
                    pred_index=pred_index,
                    prediction=prediction,
                    gt_index=best_gt_index,
                    gt=best_gt,
                    image_id=image_id,
                )
            )
        else:
            matches.append(
                _match_record(
                    status="fp",
                    iou_threshold=threshold,
                    iou=best_iou,
                    pred_index=pred_index,
                    prediction=prediction,
                    gt_index=None,
                    gt=None,
                    image_id=image_id,
                )
            )

    for gt_index, gt in valid_gt:
        if gt_index in matched_gt_indices:
            continue

        matches.append(
            _match_record(
                status="fn",
                iou_threshold=threshold,
                iou=0.0,
                pred_index=None,
                prediction=None,
                gt_index=gt_index,
                gt=gt,
                image_id=image_id,
            )
        )

    true_positives = sum(1 for match in matches if match.status == "tp")
    false_positives = sum(1 for match in matches if match.status == "fp")
    false_negatives = sum(1 for match in matches if match.status == "fn")

    return ImageMatchResult(
        image_id=image_id,
        iou_threshold=threshold,
        matches=matches,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def compute_split_average_precision(
    gt_polygons: list[Any],
    pred_polygons: list[Any],
    iou_threshold: float,
) -> AveragePrecisionResult:
    threshold = float(iou_threshold)
    gt_by_image = _group_by_image(_valid_indexed_targets(gt_polygons))
    matched_gt_by_image: dict[str, set[int]] = defaultdict(set)
    sorted_predictions = sorted(
        list(enumerate(pred_polygons)),
        key=lambda item: (-_score(item[1]), item[0]),
    )
    prediction_matches: list[ObjectMatch] = []
    tp_flags = []
    fp_flags = []

    for pred_index, prediction in sorted_predictions:
        image_id = str(_value(prediction, "image_id", ""))
        best_gt_index = None
        best_gt = None
        best_iou = 0.0

        for gt_index, gt in gt_by_image.get(image_id, []):
            if gt_index in matched_gt_by_image[image_id]:
                continue

            iou = polygon_iou(_polygon(prediction), _polygon(gt))

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
                best_gt = gt

        if best_gt_index is not None and best_gt is not None and best_iou >= threshold:
            matched_gt_by_image[image_id].add(best_gt_index)
            tp_flags.append(1.0)
            fp_flags.append(0.0)
            prediction_matches.append(
                _match_record(
                    status="tp",
                    iou_threshold=threshold,
                    iou=best_iou,
                    pred_index=pred_index,
                    prediction=prediction,
                    gt_index=best_gt_index,
                    gt=best_gt,
                    image_id=image_id,
                )
            )
        else:
            tp_flags.append(0.0)
            fp_flags.append(1.0)
            prediction_matches.append(
                _match_record(
                    status="fp",
                    iou_threshold=threshold,
                    iou=best_iou,
                    pred_index=pred_index,
                    prediction=prediction,
                    gt_index=None,
                    gt=None,
                    image_id=image_id,
                )
            )

    false_negative_matches = []

    for image_id, image_gt in sorted(gt_by_image.items()):
        for gt_index, gt in image_gt:
            if gt_index in matched_gt_by_image[image_id]:
                continue

            false_negative_matches.append(
                _match_record(
                    status="fn",
                    iou_threshold=threshold,
                    iou=0.0,
                    pred_index=None,
                    prediction=None,
                    gt_index=gt_index,
                    gt=gt,
                    image_id=image_id,
                )
            )

    num_gt = sum(len(items) for items in gt_by_image.values())
    num_predictions = len(sorted_predictions)

    if num_predictions == 0:
        precision = []
        recall = []
        ap = 0.0
    else:
        tp_cumsum = np.cumsum(np.asarray(tp_flags, dtype=np.float64))
        fp_cumsum = np.cumsum(np.asarray(fp_flags, dtype=np.float64))
        precision_array = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-12)
        recall_array = (
            tp_cumsum / float(num_gt)
            if num_gt > 0
            else np.zeros_like(tp_cumsum)
        )
        precision = precision_array.tolist()
        recall = recall_array.tolist()
        ap = _average_precision(recall_array, precision_array) if num_gt > 0 else 0.0

    true_positives = int(sum(tp_flags))
    false_positives = int(sum(fp_flags))
    false_negatives = int(num_gt - true_positives)

    return AveragePrecisionResult(
        iou_threshold=threshold,
        ap=float(ap),
        num_gt=int(num_gt),
        num_predictions=int(num_predictions),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        matches=[*prediction_matches, *false_negative_matches],
    )


def compute_ap50_ap75(
    gt_polygons: list[Any],
    pred_polygons: list[Any],
) -> dict[str, AveragePrecisionResult]:
    return {
        "ap50": compute_split_average_precision(
            gt_polygons,
            pred_polygons,
            iou_threshold=0.50,
        ),
        "ap75": compute_split_average_precision(
            gt_polygons,
            pred_polygons,
            iou_threshold=0.75,
        ),
    }


def _average_precision(
    recall: np.ndarray,
    precision: np.ndarray,
) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])

    changing_points = np.where(mrec[1:] != mrec[:-1])[0]
    return float(
        np.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1])
    )


def _valid_indexed_targets(items: list[Any]) -> list[tuple[int, Any]]:
    valid = []

    for index, item in enumerate(items):
        polygon = repair_geometry(_polygon(item))

        if polygon is None or polygon.is_empty or float(polygon.area) <= 0.0:
            continue

        valid.append((index, item))

    return valid


def _group_by_image(items: list[tuple[int, Any]]) -> dict[str, list[tuple[int, Any]]]:
    grouped: dict[str, list[tuple[int, Any]]] = defaultdict(list)

    for index, item in items:
        grouped[str(_value(item, "image_id", ""))].append((index, item))

    return grouped


def _image_id(gt_polygons: list[Any], pred_polygons: list[Any]) -> str:
    if gt_polygons:
        return str(_value(gt_polygons[0], "image_id", ""))

    if pred_polygons:
        return str(_value(pred_polygons[0], "image_id", ""))

    return ""


def _match_record(
    status: str,
    iou_threshold: float,
    iou: float,
    pred_index: int | None,
    prediction: Any | None,
    gt_index: int | None,
    gt: Any | None,
    image_id: str,
) -> ObjectMatch:
    return ObjectMatch(
        image_id=image_id,
        status=status,
        iou_threshold=float(iou_threshold),
        iou=float(iou),
        pred_index=pred_index,
        gt_index=gt_index,
        pred_score=_score(prediction) if prediction is not None else None,
        pred_source=(
            str(_value(prediction, "source", ""))
            if prediction is not None and _value(prediction, "source", None) is not None
            else None
        ),
        pred_source_id=(
            _value(prediction, "source_id", None) if prediction is not None else None
        ),
        gt_id=_value(gt, "gt_id", None) if gt is not None else None,
        gt_area=_safe_area(gt, preferred=_value(gt, "area", None)),
        gt_vertex_count=(
            int(_value(gt, "vertex_count", 0))
            if gt is not None and _value(gt, "vertex_count", None) is not None
            else None
        ),
        pred_area=_safe_area(prediction),
    )


def _polygon(item: Any) -> BaseGeometry | None:
    value = _value(item, "polygon", item)

    if value is None or isinstance(value, BaseGeometry):
        return value

    return None


def _score(item: Any) -> float:
    if item is None:
        return 0.0

    try:
        return float(_value(item, "score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _value(item: Any, name: str, default: Any) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)

    return getattr(item, name, default)


def _safe_area(item: Any | None, preferred: Any = None) -> float | None:
    if item is None:
        return None

    if preferred is not None:
        try:
            return float(preferred)
        except (TypeError, ValueError):
            pass

    polygon = repair_geometry(_polygon(item))

    if polygon is None:
        return None

    return float(polygon.area)

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BoundaryCounts:
    matched_pred: int
    pred_boundary_pixels: int
    matched_gt: int
    gt_boundary_pixels: int


@dataclass(frozen=True)
class BoundaryMetrics:
    boundary_f1: float
    boundary_precision: float
    boundary_recall: float
    matched_pred: int
    pred_boundary_pixels: int
    matched_gt: int
    gt_boundary_pixels: int


def binary_boundary(mask: np.ndarray) -> np.ndarray:
    mask_bool = _as_binary_mask(mask)

    if not np.any(mask_bool):
        return np.zeros(mask_bool.shape, dtype=bool)

    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask_bool & ~eroded


def boundary_counts_from_masks(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_px: int = 2,
) -> BoundaryCounts:
    pred_boundary = binary_boundary(pred_mask)
    gt_boundary = binary_boundary(gt_mask)
    return boundary_counts(pred_boundary, gt_boundary, tolerance_px=tolerance_px)


def boundary_counts(
    pred_boundary: np.ndarray,
    gt_boundary: np.ndarray,
    tolerance_px: int = 2,
) -> BoundaryCounts:
    pred_boundary = _as_binary_mask(pred_boundary)
    gt_boundary = _as_binary_mask(gt_boundary)

    if pred_boundary.shape != gt_boundary.shape:
        raise ValueError(
            "pred_boundary and gt_boundary must have the same shape, got "
            f"{pred_boundary.shape} and {gt_boundary.shape}"
        )

    tolerance_px = int(tolerance_px)

    if tolerance_px < 0:
        raise ValueError(f"tolerance_px must be non-negative, got {tolerance_px}")

    pred_pixels = int(np.count_nonzero(pred_boundary))
    gt_pixels = int(np.count_nonzero(gt_boundary))
    dilated_gt = dilate_binary(gt_boundary, tolerance_px)
    dilated_pred = dilate_binary(pred_boundary, tolerance_px)

    return BoundaryCounts(
        matched_pred=int(np.count_nonzero(pred_boundary & dilated_gt)),
        pred_boundary_pixels=pred_pixels,
        matched_gt=int(np.count_nonzero(gt_boundary & dilated_pred)),
        gt_boundary_pixels=gt_pixels,
    )


def boundary_metrics_from_counts(counts: BoundaryCounts) -> BoundaryMetrics:
    precision = _ratio_with_empty_perfect(
        counts.matched_pred,
        counts.pred_boundary_pixels,
        other_denominator=counts.gt_boundary_pixels,
    )
    recall = _ratio_with_empty_perfect(
        counts.matched_gt,
        counts.gt_boundary_pixels,
        other_denominator=counts.pred_boundary_pixels,
    )
    f1 = _f1(precision, recall)

    return BoundaryMetrics(
        boundary_f1=f1,
        boundary_precision=precision,
        boundary_recall=recall,
        matched_pred=int(counts.matched_pred),
        pred_boundary_pixels=int(counts.pred_boundary_pixels),
        matched_gt=int(counts.matched_gt),
        gt_boundary_pixels=int(counts.gt_boundary_pixels),
    )


def boundary_f1_from_masks(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_px: int = 2,
) -> BoundaryMetrics:
    counts = boundary_counts_from_masks(
        pred_mask,
        gt_mask,
        tolerance_px=tolerance_px,
    )
    return boundary_metrics_from_counts(counts)


def dilate_binary(mask: np.ndarray, radius_px: int) -> np.ndarray:
    mask_bool = _as_binary_mask(mask)

    if int(radius_px) <= 0:
        return mask_bool

    kernel_size = int(radius_px) * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)

    if array.ndim == 3 and int(array.shape[0]) == 1:
        array = array[0]

    if array.ndim != 2:
        raise ValueError(f"Expected a 2D binary mask, got shape {array.shape}")

    return np.ascontiguousarray(array.astype(bool))


def _ratio_with_empty_perfect(
    numerator: int,
    denominator: int,
    other_denominator: int,
) -> float:
    if int(denominator) == 0:
        return 1.0 if int(other_denominator) == 0 else 0.0

    return float(numerator) / float(denominator)


def _f1(precision: float, recall: float) -> float:
    denominator = float(precision) + float(recall)

    if denominator <= 0.0:
        return 0.0

    return 2.0 * float(precision) * float(recall) / denominator

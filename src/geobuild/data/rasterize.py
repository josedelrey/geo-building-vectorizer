from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from geobuild.data.records import ImageRecord


@dataclass
class TargetBundle:
    mask: np.ndarray
    boundary: np.ndarray
    corner: np.ndarray
    center: np.ndarray
    offset: np.ndarray
    instance: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            "mask": self.mask,
            "boundary": self.boundary,
            "corner": self.corner,
            "center": self.center,
            "offset": self.offset,
            "instance": self.instance,
        }


def _clip_points(
    points: list[list[float]],
    width: int,
    height: int,
) -> np.ndarray:
    if not points:
        return np.empty((0, 2), dtype=np.int32)

    array = np.asarray(points, dtype=np.float32)

    if array.ndim != 2 or array.shape[1] != 2:
        return np.empty((0, 2), dtype=np.int32)

    finite = np.isfinite(array).all(axis=1)
    array = array[finite]

    if len(array) == 0:
        return np.empty((0, 2), dtype=np.int32)

    array[:, 0] = np.clip(array[:, 0], 0, width - 1)
    array[:, 1] = np.clip(array[:, 1], 0, height - 1)

    return np.rint(array).astype(np.int32)


def _has_polygon_area(points: np.ndarray) -> bool:
    if len(points) < 3:
        return False

    if len(np.unique(points, axis=0)) < 3:
        return False

    return abs(cv2.contourArea(points.reshape((-1, 1, 2)))) > 0.0


def _make_instance_mask(
    exterior: np.ndarray,
    holes: list[list[list[float]]],
    width: int,
    height: int,
) -> np.ndarray:
    instance_mask = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(instance_mask, [exterior.reshape((-1, 1, 2))], 1)

    for hole in holes:
        clipped_hole = _clip_points(hole, width, height)

        if not _has_polygon_area(clipped_hole):
            continue

        cv2.fillPoly(instance_mask, [clipped_hole.reshape((-1, 1, 2))], 0)

    return instance_mask


def _draw_gaussian(
    target: np.ndarray,
    center_x: float,
    center_y: float,
    radius: int,
    sigma: float,
) -> None:
    if radius < 0 or sigma <= 0.0:
        return

    height, width = target.shape
    x_min = max(0, int(np.floor(center_x)) - radius)
    x_max = min(width - 1, int(np.ceil(center_x)) + radius)
    y_min = max(0, int(np.floor(center_y)) - radius)
    y_max = min(height - 1, int(np.ceil(center_y)) + radius)

    if x_min > x_max or y_min > y_max:
        return

    ys, xs = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
    squared_distance = (xs.astype(np.float32) - center_x) ** 2 + (
        ys.astype(np.float32) - center_y
    ) ** 2
    blob = np.exp(-squared_distance / (2.0 * sigma * sigma)).astype(np.float32)

    target[y_min : y_max + 1, x_min : x_max + 1] = np.maximum(
        target[y_min : y_max + 1, x_min : x_max + 1],
        blob,
    )


def _instance_center(instance_mask: np.ndarray) -> tuple[int, int] | None:
    if not np.any(instance_mask):
        return None

    distance = cv2.distanceTransform(instance_mask, cv2.DIST_L2, 5)
    _, max_value, _, max_location = cv2.minMaxLoc(distance)

    if max_value <= 0.0:
        return None

    center_x, center_y = max_location
    return int(center_x), int(center_y)


def rasterize_record(
    record: ImageRecord,
    boundary_width: int = 3,
    corner_radius: int = 4,
    corner_sigma: float = 2.0,
    center_radius: int = 5,
    center_sigma: float = 2.5,
    normalize_offset: bool = True,
) -> TargetBundle:
    height = int(record.height)
    width = int(record.width)

    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size: width={width}, height={height}")

    mask = np.zeros((height, width), dtype=np.uint8)
    boundary = np.zeros((height, width), dtype=np.uint8)
    corner = np.zeros((height, width), dtype=np.float32)
    center = np.zeros((height, width), dtype=np.float32)
    offset = np.zeros((2, height, width), dtype=np.float32)
    instance = np.zeros((height, width), dtype=np.int32)

    next_instance_id = 1

    for polygon in record.polygons:
        exterior = _clip_points(polygon.exterior, width, height)

        if not _has_polygon_area(exterior):
            continue

        instance_mask = _make_instance_mask(
            exterior,
            polygon.holes or [],
            width,
            height,
        )

        if not np.any(instance_mask):
            continue

        instance_id = next_instance_id
        next_instance_id += 1

        building_pixels = instance_mask.astype(bool)
        mask[building_pixels] = 1
        instance[building_pixels] = instance_id

        if boundary_width > 0:
            cv2.drawContours(
                boundary,
                [exterior.reshape((-1, 1, 2))],
                contourIdx=-1,
                color=1,
                thickness=int(boundary_width),
            )

        for x, y in exterior:
            _draw_gaussian(corner, float(x), float(y), corner_radius, corner_sigma)

        instance_center = _instance_center(instance_mask)

        if instance_center is None:
            continue

        center_x, center_y = instance_center
        _draw_gaussian(
            center,
            float(center_x),
            float(center_y),
            center_radius,
            center_sigma,
        )

        ys, xs = np.nonzero(building_pixels)
        offset_x = center_x - xs.astype(np.float32)
        offset_y = center_y - ys.astype(np.float32)

        if normalize_offset:
            offset_x /= float(width)
            offset_y /= float(height)

        offset[0, ys, xs] = offset_x
        offset[1, ys, xs] = offset_y

    return TargetBundle(
        mask=mask,
        boundary=boundary,
        corner=corner,
        center=center,
        offset=offset,
        instance=instance,
    )


def summarize_targets(targets: TargetBundle) -> dict[str, Any]:
    instance_ids = targets.instance[targets.instance > 0]

    return {
        "mask_pixels": int(np.count_nonzero(targets.mask)),
        "boundary_pixels": int(np.count_nonzero(targets.boundary)),
        "corner_max": float(np.max(targets.corner)) if targets.corner.size else 0.0,
        "center_max": float(np.max(targets.center)) if targets.center.size else 0.0,
        "num_instances": int(len(np.unique(instance_ids))),
        "offset_min": float(np.min(targets.offset)) if targets.offset.size else 0.0,
        "offset_max": float(np.max(targets.offset)) if targets.offset.size else 0.0,
        "has_nan": bool(
            np.isnan(targets.mask).any()
            or np.isnan(targets.boundary).any()
            or np.isnan(targets.corner).any()
            or np.isnan(targets.center).any()
            or np.isnan(targets.offset).any()
            or np.isnan(targets.instance).any()
        ),
    }

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

try:
    from shapely.errors import GEOSException
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
    from shapely.validation import make_valid
except ImportError as exc:
    raise ImportError(
        "Shapely is required for geometric polygon clipping in "
        "geobuild.data.rasterize. Install it with `pip install shapely` "
        "or add it to the project environment."
    ) from exc

from geobuild.data.records import ImageRecord, PolygonInstance


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


def _finite_points(points: list[list[float]]) -> list[tuple[float, float]]:
    if not points:
        return []

    array = np.asarray(points, dtype=np.float64)

    if array.ndim != 2 or array.shape[1] != 2:
        return []

    finite = np.isfinite(array).all(axis=1)
    array = array[finite]

    return [(float(x), float(y)) for x, y in array]


def _to_cv2_points(
    points: list[list[float]],
    width: int,
    height: int,
) -> np.ndarray:
    if not points:
        return np.empty((0, 2), dtype=np.int32)

    array = np.asarray(points, dtype=np.float64)

    if array.ndim != 2 or array.shape[1] != 2:
        return np.empty((0, 2), dtype=np.int32)

    finite = np.isfinite(array).all(axis=1)
    array = array[finite]

    if len(array) == 0:
        return np.empty((0, 2), dtype=np.int32)

    array[:, 0] = np.clip(array[:, 0], 0, width - 1)
    array[:, 1] = np.clip(array[:, 1], 0, height - 1)

    return np.rint(array).astype(np.int32)


def _iter_polygon_parts(geometry: object) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]

    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        polygons = []

        for part in geometry.geoms:
            polygons.extend(_iter_polygon_parts(part))

        return polygons

    return []


def _clip_polygon_to_tile(
    polygon: PolygonInstance,
    width: int,
    height: int,
) -> list[Polygon]:
    exterior = _finite_points(polygon.exterior)

    if len(exterior) < 3:
        return []

    holes = []

    for hole in polygon.holes or []:
        hole_points = _finite_points(hole)

        if len(hole_points) >= 3:
            holes.append(hole_points)

    try:
        geometry = Polygon(exterior, holes)

        if not geometry.is_valid:
            geometry = make_valid(geometry)

        tile_box = box(0.0, 0.0, float(width), float(height))
        clipped = geometry.intersection(tile_box)

        if not clipped.is_valid:
            clipped = make_valid(clipped)
    except (GEOSException, ValueError):
        return []

    return [
        part
        for part in _iter_polygon_parts(clipped)
        if not part.is_empty and part.area > 0.0
    ]


def _polygon_to_cv2_rings(
    polygon: Polygon,
    width: int,
    height: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    exterior = _to_cv2_points(list(polygon.exterior.coords[:-1]), width, height)
    holes = [
        _to_cv2_points(list(interior.coords[:-1]), width, height)
        for interior in polygon.interiors
    ]

    return exterior, holes


def _simplify_polygon_for_corners(geom: Polygon, tolerance: float) -> Polygon:
    if tolerance <= 0.0:
        return geom

    try:
        simplified = geom.simplify(tolerance, preserve_topology=True)
    except (GEOSException, ValueError):
        return geom

    if not isinstance(simplified, Polygon):
        return geom

    if simplified.is_empty or not simplified.is_valid:
        return geom

    return simplified


def _compute_turn_angle_degrees(
    prev_point: tuple[float, float],
    point: tuple[float, float],
    next_point: tuple[float, float],
) -> float:
    incoming = np.asarray(point, dtype=np.float64) - np.asarray(
        prev_point,
        dtype=np.float64,
    )
    outgoing = np.asarray(next_point, dtype=np.float64) - np.asarray(
        point,
        dtype=np.float64,
    )

    incoming_norm = float(np.linalg.norm(incoming))
    outgoing_norm = float(np.linalg.norm(outgoing))

    if incoming_norm <= 1e-8 or outgoing_norm <= 1e-8:
        return 0.0

    cosine = float(np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm))
    cosine = float(np.clip(cosine, -1.0, 1.0))

    return float(np.degrees(np.arccos(cosine)))


def _dedupe_ring_points(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    deduped = []

    for point in points:
        if not np.isfinite(point).all():
            continue

        if (
            deduped
            and np.linalg.norm(np.asarray(point) - np.asarray(deduped[-1])) <= 1e-8
        ):
            continue

        deduped.append(point)

    if len(deduped) > 1:
        first = np.asarray(deduped[0], dtype=np.float64)
        last = np.asarray(deduped[-1], dtype=np.float64)

        if np.linalg.norm(first - last) <= 1e-8:
            deduped.pop()

    return deduped


def _select_cumulative_turn_corners(
    points: list[tuple[float, float]],
    cumulative_turn_angle_degrees: float,
) -> list[tuple[float, float]]:
    points = _dedupe_ring_points(points)

    if len(points) == 0:
        return []

    if len(points) < 3:
        return points

    threshold = max(0.0, float(cumulative_turn_angle_degrees))

    if threshold <= 0.0:
        return points

    selected = []
    accumulated_turn = 0.0
    strongest_index = 0
    strongest_angle = -1.0

    for index, point in enumerate(points):
        prev_point = points[index - 1]
        next_point = points[(index + 1) % len(points)]
        angle = _compute_turn_angle_degrees(prev_point, point, next_point)
        accumulated_turn += angle

        if angle > strongest_angle:
            strongest_index = index
            strongest_angle = angle

        if accumulated_turn < threshold:
            continue

        point = points[index]
        selected.append(point)
        accumulated_turn = 0.0

    if selected:
        return selected

    return [points[strongest_index]]


def _get_corner_points_for_polygon(
    geom: Polygon,
    corner_source: str,
    simplify_tolerance: float,
    cumulative_turn_angle_degrees: float,
) -> list[tuple[float, float]]:
    if corner_source == "raw":
        corner_geom = geom
    elif corner_source == "simplified":
        corner_geom = _simplify_polygon_for_corners(geom, simplify_tolerance)
    else:
        raise ValueError(
            "Unsupported corner_source: "
            f"{corner_source!r}. Expected 'raw' or 'simplified'."
        )

    points = [
        (float(x), float(y))
        for x, y in list(corner_geom.exterior.coords[:-1])
        if np.isfinite([x, y]).all()
    ]

    return _select_cumulative_turn_corners(
        points,
        cumulative_turn_angle_degrees=cumulative_turn_angle_degrees,
    )


def _has_polygon_area(points: np.ndarray) -> bool:
    if len(points) < 3:
        return False

    if len(np.unique(points, axis=0)) < 3:
        return False

    return abs(cv2.contourArea(points.reshape((-1, 1, 2)))) > 0.0


def _make_instance_mask(
    exterior: np.ndarray,
    holes: list[np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    instance_mask = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(instance_mask, [exterior.reshape((-1, 1, 2))], 1)

    for hole in holes:
        if not _has_polygon_area(hole):
            continue

        cv2.fillPoly(instance_mask, [hole.reshape((-1, 1, 2))], 0)

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


def _point_inside_mask(instance_mask: np.ndarray, x: float, y: float) -> bool:
    height, width = instance_mask.shape
    pixel_x = int(np.rint(x))
    pixel_y = int(np.rint(y))

    if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
        return False

    return bool(instance_mask[pixel_y, pixel_x])


def _representative_point_center(
    polygon: Polygon,
    instance_mask: np.ndarray,
) -> tuple[float, float] | None:
    try:
        point = polygon.representative_point()
    except (GEOSException, ValueError):
        return None

    center_x = float(point.x)
    center_y = float(point.y)

    if not np.isfinite([center_x, center_y]).all():
        return None

    if not _point_inside_mask(instance_mask, center_x, center_y):
        return None

    return center_x, center_y


def _instance_center(
    instance_mask: np.ndarray,
    polygon: Polygon,
) -> tuple[float, float] | None:
    if not np.any(instance_mask):
        return None

    distance = cv2.distanceTransform(instance_mask, cv2.DIST_L2, 5)
    max_value = float(distance.max()) if distance.size else 0.0

    if max_value <= 0.0:
        return _representative_point_center(polygon, instance_mask)

    plateau_y, plateau_x = np.where(distance >= 0.99 * max_value)

    if plateau_x.size == 0:
        return _representative_point_center(polygon, instance_mask)

    center_x = float(plateau_x.mean())
    center_y = float(plateau_y.mean())

    if _point_inside_mask(instance_mask, center_x, center_y):
        return center_x, center_y

    fallback = _representative_point_center(polygon, instance_mask)

    if fallback is not None:
        return fallback

    squared_distance = (plateau_x.astype(np.float32) - center_x) ** 2 + (
        plateau_y.astype(np.float32) - center_y
    ) ** 2
    nearest_index = int(np.argmin(squared_distance))

    return float(plateau_x[nearest_index]), float(plateau_y[nearest_index])


def rasterize_record(
    record: ImageRecord,
    boundary_width: int = 3,
    corner_radius: int = 4,
    corner_sigma: float = 2.0,
    corner_source: str = "simplified",
    corner_simplify_tolerance: float = 1.0,
    corner_cumulative_turn_angle_degrees: float = 25.0,
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
        clipped_polygons = _clip_polygon_to_tile(polygon, width, height)

        for clipped_polygon in clipped_polygons:
            exterior, holes = _polygon_to_cv2_rings(clipped_polygon, width, height)

            if not _has_polygon_area(exterior):
                continue

            instance_mask = _make_instance_mask(
                exterior,
                holes,
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

            corner_points = _get_corner_points_for_polygon(
                clipped_polygon,
                corner_source=corner_source,
                simplify_tolerance=corner_simplify_tolerance,
                cumulative_turn_angle_degrees=corner_cumulative_turn_angle_degrees,
            )

            for x, y in corner_points:
                _draw_gaussian(corner, float(x), float(y), corner_radius, corner_sigma)

            instance_center = _instance_center(instance_mask, clipped_polygon)

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

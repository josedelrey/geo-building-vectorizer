from typing import Iterable

import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

try:
    from shapely.validation import make_valid
except ImportError:  # pragma: no cover - for older Shapely versions.
    make_valid = None


def contour_to_polygon(contour: np.ndarray) -> Polygon | None:
    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)

    if len(points) < 3:
        return None

    polygon = Polygon([(float(x), float(y)) for x, y in points])
    return repair_polygon(polygon)


def repair_polygon(polygon: Polygon) -> Polygon | None:
    if polygon.is_empty:
        return None

    geometry = polygon

    if not geometry.is_valid:
        if make_valid is not None:
            geometry = make_valid(geometry)

        candidate = largest_polygon(geometry)

        if candidate is None:
            geometry = polygon.buffer(0)

    candidate = largest_polygon(geometry)

    if candidate is None or candidate.is_empty or candidate.area <= 0:
        return None

    if not candidate.is_valid:
        candidate = candidate.buffer(0)
        candidate = largest_polygon(candidate)

    if candidate is None or candidate.is_empty or candidate.area <= 0:
        return None

    return candidate


def simplify_polygon(
    polygon: Polygon,
    tolerance: float,
) -> Polygon | None:
    tolerance = float(tolerance)

    if tolerance <= 0:
        return repair_polygon(polygon)

    simplified = polygon.simplify(tolerance, preserve_topology=True)
    return repair_polygon(simplified)


def largest_polygon(geometry: object) -> Polygon | None:
    polygons = list(iter_polygons(geometry))

    if not polygons:
        return None

    return max(polygons, key=lambda polygon: float(polygon.area))


def iter_polygons(geometry: object) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
        return

    if isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
        return

    if isinstance(geometry, GeometryCollection):
        for item in geometry.geoms:
            yield from iter_polygons(item)

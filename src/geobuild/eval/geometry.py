from collections.abc import Iterable

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry

from geobuild.data.records import PolygonInstance

try:
    from shapely.validation import make_valid
except ImportError:  # pragma: no cover - for older Shapely versions.
    make_valid = None


def image_bounds(width: int, height: int) -> Polygon:
    return box(0.0, 0.0, float(width), float(height))


def polygon_instance_to_geometry(instance: PolygonInstance) -> Polygon | None:
    exterior = _coordinate_ring(instance.exterior)

    if exterior is None:
        return None

    holes = []

    for hole in instance.holes:
        hole_ring = _coordinate_ring(hole)

        if hole_ring is not None:
            holes.append(hole_ring)

    try:
        return Polygon(exterior, holes=holes)
    except (TypeError, ValueError):
        return None


def clip_repair_split_polygon(
    geometry: BaseGeometry,
    width: int,
    height: int,
) -> list[Polygon]:
    repaired = repair_geometry(geometry)

    if repaired is None:
        return []

    try:
        clipped = repaired.intersection(image_bounds(width, height))
    except Exception:
        clipped = repair_geometry(repaired)

        if clipped is None:
            return []

        try:
            clipped = clipped.intersection(image_bounds(width, height))
        except Exception:
            return []

    clipped = repair_geometry(clipped)

    if clipped is None:
        return []

    return [
        polygon
        for polygon in iter_polygons(clipped)
        if not polygon.is_empty and float(polygon.area) > 0.0
    ]


def repair_geometry(geometry: BaseGeometry | None) -> BaseGeometry | None:
    if geometry is None or geometry.is_empty:
        return None

    repaired = geometry

    if not repaired.is_valid and make_valid is not None:
        try:
            repaired = make_valid(repaired)
        except Exception:
            repaired = geometry

    if not repaired.is_valid:
        try:
            repaired = repaired.buffer(0)
        except Exception:
            return None

    if repaired.is_empty:
        return None

    return repaired


def iter_polygons(geometry: BaseGeometry) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
        return

    if isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
        return

    if isinstance(geometry, GeometryCollection):
        for item in geometry.geoms:
            yield from iter_polygons(item)


def vertex_count(polygon: Polygon) -> int:
    count = max(0, len(polygon.exterior.coords) - 1)

    for interior in polygon.interiors:
        count += max(0, len(interior.coords) - 1)

    return count


def _coordinate_ring(points: object) -> list[tuple[float, float]] | None:
    if not isinstance(points, list) or len(points) < 3:
        return None

    ring = []

    for point in points:
        if not isinstance(point, list) or len(point) < 2:
            return None

        try:
            ring.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            return None

    if len(ring) < 3:
        return None

    return ring

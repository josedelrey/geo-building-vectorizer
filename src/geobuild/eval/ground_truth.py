import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon

from geobuild.data.records import ImageRecord
from geobuild.eval.geometry import (
    clip_repair_split_polygon,
    polygon_instance_to_geometry,
    vertex_count,
)


@dataclass(frozen=True)
class GroundTruthPolygon:
    image_id: str
    gt_id: int | str
    polygon: Polygon
    area: float
    vertex_count: int


def load_image_records(manifest_path: str | Path) -> list[ImageRecord]:
    path = Path(manifest_path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {path}")

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(ImageRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid ImageRecord in manifest {path} at line {line_number}"
                ) from exc

    return records


def load_ground_truth_polygons(
    manifest_path: str | Path,
) -> list[GroundTruthPolygon]:
    records = load_image_records(manifest_path)
    ground_truth = []

    for record in records:
        ground_truth.extend(ground_truth_polygons_from_record(record))

    return ground_truth


def ground_truth_polygons_from_record(
    record: ImageRecord,
) -> list[GroundTruthPolygon]:
    polygons = []

    for polygon_index, instance in enumerate(record.polygons):
        geometry = polygon_instance_to_geometry(instance)

        if geometry is None:
            continue

        clipped_polygons = clip_repair_split_polygon(
            geometry,
            width=int(record.width),
            height=int(record.height),
        )

        for part_index, polygon in enumerate(clipped_polygons):
            if polygon.is_empty or float(polygon.area) <= 0.0:
                continue

            polygons.append(
                GroundTruthPolygon(
                    image_id=str(record.image_id),
                    gt_id=_gt_id(instance.annotation_id, polygon_index, part_index),
                    polygon=polygon,
                    area=float(polygon.area),
                    vertex_count=vertex_count(polygon),
                )
            )

    return polygons


def _gt_id(
    annotation_id: Any,
    polygon_index: int,
    part_index: int,
) -> int | str:
    base_id = annotation_id if annotation_id is not None else polygon_index

    if part_index == 0:
        return base_id

    return f"{base_id}:{part_index}"

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from geobuild.data.records import ImageRecord, PolygonInstance


def load_coco_annotations(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = {"images", "annotations"}
    missing_keys = required_keys - set(data.keys())

    if missing_keys:
        raise ValueError(f"COCO file is missing keys: {sorted(missing_keys)}")

    return data


def _flat_polygon_to_points(flat_polygon: list[float]) -> list[list[float]]:
    if len(flat_polygon) < 6:
        return []

    if len(flat_polygon) % 2 != 0:
        flat_polygon = flat_polygon[:-1]

    return [
        [float(flat_polygon[i]), float(flat_polygon[i + 1])]
        for i in range(0, len(flat_polygon), 2)
    ]


def _parse_segmentation(annotation: dict[str, Any]) -> list[list[list[float]]]:
    segmentation = annotation.get("segmentation", [])

    if not isinstance(segmentation, list):
        return []

    polygons = []

    for item in segmentation:
        if not isinstance(item, list):
            continue

        points = _flat_polygon_to_points(item)

        if len(points) >= 3:
            polygons.append(points)

    return polygons


def _resolve_image_path(image_dir: Path, file_name: str) -> Path:
    direct_path = image_dir / file_name

    if direct_path.exists():
        return direct_path

    candidates = list(image_dir.rglob(Path(file_name).name))

    if len(candidates) == 1:
        return candidates[0]

    return direct_path


def build_image_records(
    annotation_file: str | Path,
    image_dir: str | Path,
    split: str,
) -> list[ImageRecord]:
    annotation_file = Path(annotation_file)
    image_dir = Path(image_dir)

    data = load_coco_annotations(annotation_file)

    annotations_by_image_id: dict[int | str, list[dict[str, Any]]] = defaultdict(list)

    for annotation in data["annotations"]:
        annotations_by_image_id[annotation["image_id"]].append(annotation)

    records = []

    for image_info in data["images"]:
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        image_path = _resolve_image_path(image_dir, file_name)

        polygons = []

        for annotation in annotations_by_image_id.get(image_id, []):
            parsed_polygons = _parse_segmentation(annotation)

            for exterior in parsed_polygons:
                polygons.append(
                    PolygonInstance(
                        exterior=exterior,
                        holes=[],
                        category_id=annotation.get("category_id"),
                        iscrowd=annotation.get("iscrowd", 0),
                        area=annotation.get("area"),
                        bbox=annotation.get("bbox"),
                        annotation_id=annotation.get("id"),
                    )
                )

        record = ImageRecord(
            image_id=image_id,
            image_path=str(image_path),
            width=int(image_info["width"]),
            height=int(image_info["height"]),
            split=split,
            polygons=polygons,
        )

        records.append(record)

    return records


def validate_records(records: list[ImageRecord]) -> dict[str, int]:
    stats = {
        "num_images": len(records),
        "num_missing_images": 0,
        "num_empty_images": 0,
        "num_polygons": 0,
        "num_invalid_polygons": 0,
        "num_out_of_bounds_points": 0,
    }

    for record in records:
        if not Path(record.image_path).exists():
            stats["num_missing_images"] += 1

        if len(record.polygons) == 0:
            stats["num_empty_images"] += 1

        stats["num_polygons"] += len(record.polygons)

        for polygon in record.polygons:
            if len(polygon.exterior) < 3:
                stats["num_invalid_polygons"] += 1
                continue

            for x, y in polygon.exterior:
                if x < 0 or y < 0 or x > record.width or y > record.height:
                    stats["num_out_of_bounds_points"] += 1

    return stats
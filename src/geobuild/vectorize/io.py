import json
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import mapping

from geobuild.vectorize.base import PredictionBundle, PredictedPolygon, Vectorizer


PROBABILITY_OUTPUT_ALIASES = {
    "mask": "mask_prob",
    "boundary": "boundary_prob",
    "corner": "corner_prob",
    "center": "center_prob",
}


def load_prediction_bundle(
    record_from_predictions_jsonl: dict[str, Any],
) -> PredictionBundle:
    record = dict(record_from_predictions_jsonl)
    npz_path = Path(record["npz_path"])

    if not npz_path.exists():
        raise FileNotFoundError(f"Prediction NPZ file does not exist: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {
            name: np.ascontiguousarray(data[name])
            for name in data.files
        }

    _add_probability_aliases(arrays)

    return PredictionBundle(
        image_id=str(record["image_id"]),
        height=int(record["height"]),
        width=int(record["width"]),
        arrays=arrays,
        metadata=record,
    )


def _add_probability_aliases(arrays: dict[str, np.ndarray]) -> None:
    for source_name, alias_name in PROBABILITY_OUTPUT_ALIASES.items():
        if alias_name not in arrays and source_name in arrays:
            arrays[alias_name] = arrays[source_name]


def validate_required_outputs(
    vectorizer: Vectorizer,
    prediction: PredictionBundle,
) -> None:
    required_outputs = set(vectorizer.required_outputs)
    available_outputs = set(prediction.arrays)
    missing = sorted(required_outputs - available_outputs)

    if missing:
        raise ValueError(
            f"Vectorizer {vectorizer.name!r} requires missing prediction outputs "
            f"for image_id={prediction.image_id!r}: {missing}. "
            f"Available outputs: {sorted(available_outputs)}"
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _polygon_record(prediction: PredictedPolygon) -> dict[str, Any]:
    return {
        "image_id": str(prediction.image_id),
        "geometry": mapping(prediction.polygon),
        "score": float(prediction.score),
        "source": str(prediction.source),
        "source_id": prediction.source_id,
        "properties": dict(prediction.properties),
    }


def save_polygons_jsonl(
    polygons: list[PredictedPolygon],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for polygon in polygons:
            f.write(json.dumps(_polygon_record(polygon), default=_json_default) + "\n")


def _feature_properties(polygon: PredictedPolygon) -> dict[str, Any]:
    properties = {
        "image_id": str(polygon.image_id),
        "score": float(polygon.score),
        "source": str(polygon.source),
        "source_id": polygon.source_id,
    }
    properties.update(dict(polygon.properties))
    return properties


def save_polygons_geojson(
    polygons: list[PredictedPolygon],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(polygon.polygon),
                "properties": _feature_properties(polygon),
            }
            for polygon in polygons
        ],
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(feature_collection, f, default=_json_default)
        f.write("\n")

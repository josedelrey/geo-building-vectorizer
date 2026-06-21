import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

from geobuild.utils.config import load_config, resolve_path
from geobuild.vectorize.contours import binary_mask_from_probability
from geobuild.vectorize.debug import save_debug_overlay
from geobuild.vectorize.io import (
    load_prediction_bundle,
    save_polygons_geojson,
    save_polygons_jsonl,
)
from geobuild.vectorize.registry import build_vectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--vectorizer-config", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def _prediction_manifest_path(predictions: str | Path) -> Path:
    path = Path(predictions)

    if path.is_dir():
        return path / "predictions.jsonl"

    return path


def _prediction_records(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prediction manifest does not exist: {manifest_path}")

    records = []

    with manifest_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in prediction manifest {manifest_path} "
                    f"at line {line_number}"
                ) from exc

    return records


def _safe_filename_part(value: Any) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(value)
    )
    safe = safe.strip("._")
    return (safe or "image")[:128]


def _max_debug_images(config: dict[str, Any]) -> int:
    vectorizer_config = config.get("vectorizer", {})
    debug_config = config.get("debug", {})
    return int(
        debug_config.get(
            "max_debug_images",
            vectorizer_config.get("max_debug_images", 0),
        )
    )


def _save_vectorizer_config(config: dict[str, Any], out_dir: Path) -> None:
    config_path = out_dir / "vectorizer_config.yaml"

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def main() -> None:
    args = parse_args()
    predictions_path = resolve_path(args.predictions, root=ROOT)
    manifest_path = _prediction_manifest_path(predictions_path)
    vectorizer_config = load_config(args.vectorizer_config, root=ROOT)
    out_dir = resolve_path(args.out, root=ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = build_vectorizer(vectorizer_config)
    records = _prediction_records(manifest_path)
    polygons = []
    debug_limit = _max_debug_images(vectorizer_config)
    debug_count = 0

    for record in tqdm(records, desc=f"Vectorize {vectorizer.name}"):
        prediction = load_prediction_bundle(record)
        image_polygons = vectorizer.vectorize(prediction)
        polygons.extend(image_polygons)

        if debug_count < debug_limit:
            mask_prob = prediction.arrays["mask_prob"]
            binary_mask = binary_mask_from_probability(
                mask_prob,
                vectorizer.mask_threshold,
            )
            debug_path = (
                out_dir
                / "debug"
                / f"{debug_count:06d}_{_safe_filename_part(prediction.image_id)}.png"
            )
            save_debug_overlay(mask_prob, binary_mask, image_polygons, debug_path)
            debug_count += 1

    save_polygons_jsonl(polygons, out_dir / "polygons.jsonl")
    save_polygons_geojson(polygons, out_dir / "predictions.geojson")
    _save_vectorizer_config(vectorizer_config, out_dir)

    print(
        f"Saved {len(polygons)} polygons for {len(records)} predictions to: {out_dir}"
    )


if __name__ == "__main__":
    main()

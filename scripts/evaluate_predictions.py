import argparse
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

from geobuild.eval.ground_truth import load_image_records
from geobuild.eval.matching import match_image_predictions
from geobuild.eval.raster_metrics import (
    evaluate_raster_split,
    load_prediction_records,
)
from geobuild.eval.report import (
    build_summary,
    write_per_image_metrics_csv,
    write_per_object_matches_csv,
    write_summary_json,
)
from geobuild.eval.vector_metrics import (
    compute_vector_metrics,
    ground_truth_by_image,
    load_predicted_polygons,
    predictions_by_image,
)
from geobuild.utils.config import (
    load_config,
    manifest_path_from_config,
    resolve_path,
    target_config_from_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--vectors", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument(
        "--metric-config",
        type=str,
        default=None,
        help="Optional YAML file whose metrics section overrides config metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, root=ROOT)
    metrics_config = _metrics_config(config, args.metric_config)
    predictions_path = resolve_path(args.predictions, root=ROOT)
    vectors_path = resolve_path(args.vectors, root=ROOT)
    out_dir = resolve_path(args.out, root=ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_path_from_config(config, args.split, root=ROOT)
    records = load_image_records(manifest_path)
    prediction_records = load_prediction_records(predictions_path)
    predicted_polygons = load_predicted_polygons(vectors_path)
    gt_by_image = ground_truth_by_image(records)
    pred_by_image = predictions_by_image(predicted_polygons)
    gt_polygons = [
        polygon
        for image_polygons in gt_by_image.values()
        for polygon in image_polygons
    ]

    raster_metrics = evaluate_raster_split(
        records,
        prediction_records,
        raster_config=target_config_from_config(config),
        metrics_config=metrics_config,
    )
    vector_metrics = compute_vector_metrics(gt_polygons, predicted_polygons)
    per_image_rows = _per_image_rows(
        raster_metrics.per_image,
        gt_by_image=gt_by_image,
        pred_by_image=pred_by_image,
    )
    summary = build_summary(
        experiment=str(config.get("experiment", {}).get("name", "")),
        split=str(args.split),
        vectorizer=_vectorizer_label(vectors_path, split=str(args.split)),
        raster_metrics=raster_metrics,
        vector_metrics=vector_metrics,
        context=_summary_context(
            config=config,
            split=str(args.split),
            manifest_path=manifest_path,
            predictions_path=predictions_path,
            vectors_path=vectors_path,
            out_dir=out_dir,
            prediction_records=prediction_records,
            vectorizer_config=_load_vectorizer_config(vectors_path),
            metrics_config=metrics_config,
            config_path=resolve_path(args.config, root=ROOT),
        ),
    )

    write_summary_json(summary, out_dir / "summary_metrics.json")
    write_per_image_metrics_csv(per_image_rows, out_dir / "per_image_metrics.csv")
    write_per_object_matches_csv(
        vector_metrics.matches_50,
        out_dir / "per_object_matches.csv",
    )

    print(f"Saved evaluation report to: {out_dir}")


def _per_image_rows(
    raster_per_image: list[Any],
    gt_by_image: dict[str, list[Any]],
    pred_by_image: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    rows = []

    for raster in raster_per_image:
        image_id = str(raster.image_id)
        gt_polygons = gt_by_image.get(image_id, [])
        pred_polygons = pred_by_image.get(image_id, [])
        match_result = match_image_predictions(gt_polygons, pred_polygons, 0.50)
        matched_iou = [
            float(match.iou)
            for match in match_result.matches
            if match.status == "tp"
        ]
        row = raster.to_dict()
        row.update(
            {
                "num_gt": len(gt_polygons),
                "num_pred": len(pred_polygons),
                "tp_50": match_result.true_positives,
                "fp_50": match_result.false_positives,
                "fn_50": match_result.false_negatives,
                "f1_50": _f1_from_counts(
                    match_result.true_positives,
                    match_result.false_positives,
                    match_result.false_negatives,
                ),
                "mean_matched_iou_50": (
                    sum(matched_iou) / len(matched_iou) if matched_iou else 0.0
                ),
            }
        )
        rows.append(row)

    return rows


def _summary_context(
    config: dict[str, Any],
    split: str,
    manifest_path: Path,
    predictions_path: Path,
    vectors_path: Path,
    out_dir: Path,
    prediction_records: list[dict[str, Any]],
    vectorizer_config: dict[str, Any],
    metrics_config: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    first_prediction = prediction_records[0] if prediction_records else {}
    vectorizer_settings = vectorizer_config.get("vectorizer", {})
    return {
        "checkpoint": first_prediction.get("checkpoint_path"),
        "checkpoint_path": first_prediction.get("checkpoint_path"),
        "prediction_experiment": first_prediction.get("experiment_name"),
        "vectorizer_name": vectorizer_settings.get("name"),
        "vectorizer_version": vectorizer_settings.get("version"),
        "mask_threshold": float(
            metrics_config.get(
                "mask_threshold",
                metrics_config.get(
                    "threshold",
                    vectorizer_settings.get("mask_threshold", 0.5),
                ),
            )
        ),
        "boundary_tolerance_px": int(metrics_config.get("boundary_tolerance_px", 2)),
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "predictions_path": str(predictions_path),
        "vectors_path": str(vectors_path),
        "output_path": str(out_dir),
        "split_name": split,
        "vectorizer_config_path": str(vectors_path / "vectorizer_config.yaml"),
        "iou_thresholds": [0.5, 0.75],
    }


def _metrics_config(
    config: dict[str, Any],
    metric_config_path: str | None,
) -> dict[str, Any]:
    metrics = dict(config.get("metrics", {}))

    if metric_config_path is None:
        return metrics

    override = load_config(metric_config_path, root=ROOT)
    override_metrics = override.get("metrics", override)

    if not isinstance(override_metrics, dict):
        raise ValueError(f"Metric config must be a mapping: {metric_config_path}")

    metrics.update(override_metrics)
    return metrics


def _load_vectorizer_config(vectors_path: Path) -> dict[str, Any]:
    config_path = vectors_path / "vectorizer_config.yaml"

    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data if isinstance(data, dict) else {}


def _vectorizer_label(vectors_path: Path, split: str) -> str:
    if vectors_path.name == str(split) and vectors_path.parent.name:
        return vectors_path.parent.name

    return vectors_path.name


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * int(tp) + int(fp) + int(fn)

    if denominator == 0:
        return 1.0

    return float(2 * int(tp)) / float(denominator)


if __name__ == "__main__":
    main()

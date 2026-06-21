import csv
import tempfile
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from geobuild.eval.ground_truth import load_image_records
from geobuild.eval.raster_metrics import evaluate_raster_split, load_prediction_records
from geobuild.eval.report import build_summary
from geobuild.eval.vector_metrics import (
    compute_vector_metrics,
    ground_truth_by_image,
    load_predicted_polygons,
)
from geobuild.utils.config import manifest_path_from_config, target_config_from_config
from geobuild.vectorize.io import (
    load_prediction_bundle,
    save_polygons_geojson,
    save_polygons_jsonl,
)
from geobuild.vectorize.registry import build_vectorizer


DEFAULT_SEARCH_GRID = {
    "mask_threshold": [0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
    "min_area_px": [10, 20, 40, 80],
    "simplify_tolerance": [0.5, 1.0, 1.5, 2.0],
}
DEFAULT_SELECTION_METRIC = "ap50"
TRIAL_COLUMNS = [
    "trial_index",
    "mask_threshold",
    "min_area_px",
    "simplify_tolerance",
    "selection_metric",
    "selection_value",
    "experiment",
    "split",
    "vectorizer",
    "num_images",
    "num_gt",
    "num_pred",
    "mask_iou",
    "dice",
    "precision",
    "recall",
    "boundary_f1",
    "ap50",
    "ap75",
    "f1_50",
    "mean_matched_iou_50",
    "invalid_polygon_ratio",
    "mean_vertex_count",
    "mean_area_error_rel",
    "mean_perimeter_error_rel",
]


@dataclass(frozen=True)
class TuningTrial:
    trial_index: int
    params: dict[str, Any]
    summary: dict[str, Any]

    def row(self, selection_metric: str) -> dict[str, Any]:
        row = {
            "trial_index": int(self.trial_index),
            "selection_metric": selection_metric,
            "selection_value": self.summary.get(selection_metric, ""),
        }
        row.update(self.params)

        for key in TRIAL_COLUMNS:
            if key not in row:
                row[key] = self.summary.get(key, "")

        return row


@dataclass(frozen=True)
class TuningResult:
    best_config: dict[str, Any]
    best_trial: TuningTrial
    trials: list[TuningTrial] = field(default_factory=list)


def tune_vectorizer(
    *,
    project_config: dict[str, Any],
    split: str,
    predictions_path: str | Path,
    vectorizer_config: dict[str, Any],
    output_dir: str | Path,
    selection_metric: str | None = None,
    root: str | Path,
) -> TuningResult:
    validate_tuning_split(split)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    selection_metric = str(selection_metric or _selection_metric(vectorizer_config))
    search_grid = _search_grid(vectorizer_config)
    trial_params = list(iter_grid(search_grid))

    if not trial_params:
        raise ValueError("Vectorizer tuning search grid produced no trials")

    prediction_records = load_prediction_records(predictions_path)
    manifest_path = manifest_path_from_config(project_config, split, root=root)
    image_records = load_image_records(manifest_path)
    raster_metrics = evaluate_raster_split(
        image_records,
        prediction_records,
        raster_config=target_config_from_config(project_config),
        metrics_config=project_config.get("metrics", {}),
    )
    gt_by_image = ground_truth_by_image(image_records)
    gt_polygons = [
        polygon
        for image_polygons in gt_by_image.values()
        for polygon in image_polygons
    ]
    trials = []
    tmp_root = output_path / "_trial_outputs"
    tmp_root.mkdir(parents=True, exist_ok=True)

    for trial_index, params in enumerate(
        tqdm(trial_params, desc="Tune vectorizer", unit="trial"),
        start=1,
    ):
        trial_config = apply_vectorizer_params(vectorizer_config, params)

        with tempfile.TemporaryDirectory(
            prefix=f"trial_{trial_index:03d}_",
            dir=tmp_root,
        ) as tmp_dir:
            vector_dir = Path(tmp_dir)
            vectorize_prediction_records(
                prediction_records=prediction_records,
                vectorizer_config=trial_config,
                output_dir=vector_dir,
            )
            predicted_polygons = load_predicted_polygons(vector_dir)
            vector_metrics = compute_vector_metrics(gt_polygons, predicted_polygons)
            summary = build_summary(
                experiment=str(project_config.get("experiment", {}).get("name", "")),
                split=str(split),
                vectorizer=_vectorizer_name(trial_config),
                raster_metrics=raster_metrics,
                vector_metrics=vector_metrics,
                context={
                    "selection_metric": selection_metric,
                    "trial_index": trial_index,
                    **params,
                },
            )
            trials.append(
                TuningTrial(
                    trial_index=trial_index,
                    params=dict(params),
                    summary=summary,
                )
            )

    best_trial = select_best_trial(trials, selection_metric)
    best_config = apply_vectorizer_params(vectorizer_config, best_trial.params)
    write_trials_csv(trials, output_path / "all_trials.csv", selection_metric)
    write_best_config(best_config, output_path / "best_config.yaml")

    try:
        tmp_root.rmdir()
    except OSError:
        pass

    return TuningResult(
        best_config=best_config,
        best_trial=best_trial,
        trials=trials,
    )


def validate_tuning_split(split: str) -> None:
    normalized = str(split).lower()

    if normalized in {"test", "test2"}:
        raise ValueError(f"Refusing to tune on held-out split: {split!r}")

    if normalized != "val":
        raise ValueError(
            f"Vectorizer tuning is validation-only; expected split 'val', got {split!r}"
        )


def vectorize_prediction_records(
    *,
    prediction_records: list[dict[str, Any]],
    vectorizer_config: dict[str, Any],
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vectorizer = build_vectorizer(vectorizer_config)
    polygons = []

    for record in prediction_records:
        prediction = load_prediction_bundle(record)
        polygons.extend(vectorizer.vectorize(prediction))

    save_polygons_jsonl(polygons, output_path / "polygons.jsonl")
    save_polygons_geojson(polygons, output_path / "predictions.geojson")
    write_best_config(vectorizer_config, output_path / "vectorizer_config.yaml")


def iter_grid(search_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(search_grid)
    values = [list(search_grid[key]) for key in keys]

    if any(len(items) == 0 for items in values):
        return []

    return [
        dict(zip(keys, combination))
        for combination in product(*values)
    ]


def apply_vectorizer_params(
    vectorizer_config: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    config = _deep_copy_mapping(vectorizer_config)
    config.setdefault("vectorizer", {})

    for key, value in params.items():
        config["vectorizer"][key] = value

    config.pop("search", None)
    config.pop("selection_metric", None)
    return config


def select_best_trial(
    trials: list[TuningTrial],
    selection_metric: str,
) -> TuningTrial:
    if not trials:
        raise ValueError("Cannot select best trial from an empty trial list")

    missing = [
        trial.trial_index
        for trial in trials
        if selection_metric not in trial.summary
    ]

    if missing:
        raise KeyError(
            f"Selection metric {selection_metric!r} missing from trials: {missing}"
        )

    return max(
        trials,
        key=lambda trial: (
            _metric_value(trial.summary, selection_metric),
            _metric_value(trial.summary, "ap50"),
            _metric_value(trial.summary, "f1_50"),
            _metric_value(trial.summary, "mask_iou"),
            -_metric_value(trial.summary, "invalid_polygon_ratio"),
            -int(trial.trial_index),
        ),
    )


def write_trials_csv(
    trials: list[TuningTrial],
    output_path: str | Path,
    selection_metric: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for trial in trials:
            writer.writerow(trial.row(selection_metric))


def write_best_config(
    config: dict[str, Any],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _search_grid(vectorizer_config: dict[str, Any]) -> dict[str, list[Any]]:
    raw_search = vectorizer_config.get("search", DEFAULT_SEARCH_GRID)

    if raw_search is None:
        raw_search = DEFAULT_SEARCH_GRID

    if not isinstance(raw_search, dict):
        raise TypeError("Vectorizer tuning search must be a mapping")

    return {
        str(key): list(value)
        for key, value in raw_search.items()
    }


def _selection_metric(vectorizer_config: dict[str, Any]) -> str:
    return str(vectorizer_config.get("selection_metric", DEFAULT_SELECTION_METRIC))


def _vectorizer_name(config: dict[str, Any]) -> str:
    vectorizer = config.get("vectorizer", {})
    name = str(vectorizer.get("name", ""))
    version = vectorizer.get("version")

    if version:
        return f"{name}_{version}"

    return name


def _deep_copy_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return yaml.safe_load(yaml.safe_dump(value, sort_keys=False))


def _metric_value(summary: dict[str, Any], name: str) -> float:
    try:
        return float(summary.get(name, float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")

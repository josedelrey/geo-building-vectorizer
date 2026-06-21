import csv
import json
import re
from pathlib import Path
from typing import Any


RASTER_ABLATION_COLUMNS = [
    "ID",
    "experiment",
    "split",
    "vectorizer",
    "mask_iou",
    "dice",
    "boundary_f1",
    "ap50",
    "ap75",
    "f1_50",
    "invalid_polygon_ratio",
    "mean_vertex_count",
]
COMMON_VECTORIZER_COLUMNS = RASTER_ABLATION_COLUMNS
BEST_PERMITTED_COLUMNS = [
    *RASTER_ABLATION_COLUMNS,
    "mean_matched_iou_50",
    "mean_area_error_rel",
    "mean_perimeter_error_rel",
]
GEOMETRY_QUALITY_COLUMNS = [
    "ID",
    "experiment",
    "split",
    "vectorizer",
    "mean_matched_iou_50",
    "invalid_polygon_ratio",
    "mean_vertex_count",
    "mean_area_error_rel",
    "mean_perimeter_error_rel",
    "num_gt",
    "num_pred",
]
TABLE_COLUMNS = {
    "raster_ablation": RASTER_ABLATION_COLUMNS,
    "common_vectorizer_ablation": COMMON_VECTORIZER_COLUMNS,
    "best_permitted_vectorizer_ablation": BEST_PERMITTED_COLUMNS,
    "geometry_quality": GEOMETRY_QUALITY_COLUMNS,
}


def load_summary_metrics(paths: list[str | Path]) -> list[dict[str, Any]]:
    summaries = []

    for path in paths:
        summary_path = Path(path)

        if not summary_path.exists():
            raise FileNotFoundError(f"Summary metrics file does not exist: {summary_path}")

        with summary_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Summary metrics must be a JSON object: {summary_path}")

        data.setdefault("summary_path", str(summary_path))
        summaries.append(data)

    return summaries


def build_table(
    summaries: list[dict[str, Any]],
    table_type: str,
) -> list[dict[str, Any]]:
    table_type = str(table_type)

    if table_type == "raster_ablation":
        selected = summaries
    elif table_type == "common_vectorizer_ablation":
        selected = common_vectorizer_rows(summaries)
    elif table_type == "best_permitted_vectorizer_ablation":
        selected = best_permitted_rows(summaries)
    elif table_type == "geometry_quality":
        selected = best_permitted_rows(summaries)
    else:
        raise ValueError(
            f"Unknown table type {table_type!r}; available: {sorted(TABLE_COLUMNS)}"
        )

    columns = TABLE_COLUMNS[table_type]
    return [
        _project_row(summary, columns)
        for summary in sort_summaries(selected)
    ]


def write_comparison_tables(
    summaries: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    for table_type, columns in TABLE_COLUMNS.items():
        rows = build_table(summaries, table_type)
        output_path = out_dir / f"{table_type}.csv"
        write_csv(rows, output_path, columns)
        written[table_type] = output_path

    return written


def write_csv(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    columns: list[str],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def common_vectorizer_rows(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_experiment = _by_experiment(summaries)

    if not by_experiment:
        return []

    common_vectorizers: set[str] | None = None

    for rows in by_experiment.values():
        vectorizers = {str(row.get("vectorizer", "")) for row in rows}
        common_vectorizers = (
            vectorizers
            if common_vectorizers is None
            else common_vectorizers & vectorizers
        )

    if not common_vectorizers:
        return []

    selected_vectorizer = sorted(common_vectorizers)[0]
    return [
        _best_row(
            [
                row
                for row in rows
                if str(row.get("vectorizer", "")) == selected_vectorizer
            ]
        )
        for rows in by_experiment.values()
    ]


def best_permitted_rows(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _best_row(rows)
        for rows in _by_experiment(summaries).values()
    ]


def sort_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        summaries,
        key=lambda summary: (
            _experiment_sort_key(str(summary.get("experiment", ""))),
            str(summary.get("split", "")),
            str(summary.get("vectorizer", "")),
        ),
    )


def _project_row(
    summary: dict[str, Any],
    columns: list[str],
) -> dict[str, Any]:
    row = {}

    for column in columns:
        if column == "ID":
            row[column] = experiment_id(summary)
        else:
            row[column] = summary.get(column, "")

    return row


def experiment_id(summary: dict[str, Any]) -> str:
    explicit = summary.get("ID", summary.get("id"))

    if explicit:
        return str(explicit).upper()

    experiment = str(summary.get("experiment", ""))
    match = re.match(r"^([a-zA-Z]+\d+)", experiment)

    if match:
        return match.group(1).upper()

    return ""


def _experiment_sort_key(experiment: str) -> tuple[int, str]:
    match = re.match(r"^[a-zA-Z]+(\d+)", experiment)

    if match:
        return int(match.group(1)), experiment

    return 10_000, experiment


def _by_experiment(
    summaries: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for summary in summaries:
        grouped.setdefault(str(summary.get("experiment", "")), []).append(summary)

    return grouped


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot select best row from an empty list")

    return max(
        rows,
        key=lambda row: (
            _metric_value(row, "ap50"),
            _metric_value(row, "f1_50"),
            _metric_value(row, "mask_iou"),
            -_metric_value(row, "invalid_polygon_ratio"),
            str(row.get("vectorizer", "")),
        ),
    )


def _metric_value(row: dict[str, Any], name: str) -> float:
    try:
        return float(row.get(name, float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")

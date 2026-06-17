from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path, root: str | Path = PROJECT_ROOT) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path

    return Path(root) / path


def load_config(path: str | Path, root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    config_path = resolve_path(path, root)

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    return config


def split_config(config: dict[str, Any], split: str) -> dict[str, Any]:
    splits = config["splits"]

    if split not in splits:
        available_splits = list(splits.keys())
        raise KeyError(
            f"Unknown split {split!r}; available splits: {available_splits}"
        )

    return splits[split]


def manifest_dir_from_config(
    config: dict[str, Any],
    root: str | Path = PROJECT_ROOT,
) -> Path:
    return resolve_path(config["output"]["manifest_dir"], root)


def output_root_from_config(
    config: dict[str, Any],
    root: str | Path = PROJECT_ROOT,
) -> Path:
    return resolve_path(config["output"]["root"], root)


def output_path_from_config(
    config: dict[str, Any],
    key: str,
    root: str | Path = PROJECT_ROOT,
    **format_values: str,
) -> Path:
    output = config["output"]

    if key not in output:
        raise KeyError(f"Missing output config key: {key!r}")

    raw_path = str(output[key])

    if format_values:
        raw_path = raw_path.format(**format_values)

    path = Path(raw_path)

    if path.is_absolute():
        return path

    return output_root_from_config(config, root) / path


def debug_dir_from_config(
    config: dict[str, Any],
    root: str | Path = PROJECT_ROOT,
) -> Path:
    return output_path_from_config(config, "debug_dir", root)


def manifest_path_from_config(
    config: dict[str, Any],
    split: str,
    root: str | Path = PROJECT_ROOT,
) -> Path:
    split_config(config, split)
    return manifest_dir_from_config(config, root) / f"{split}.jsonl"


def annotation_path_from_config(
    config: dict[str, Any],
    split: str,
    root: str | Path = PROJECT_ROOT,
) -> Path:
    return resolve_path(split_config(config, split)["annotation_file"], root)


def image_dir_from_config(
    config: dict[str, Any],
    split: str,
    root: str | Path = PROJECT_ROOT,
) -> Path:
    return resolve_path(split_config(config, split)["image_dir"], root)


def target_config_from_config(config: dict[str, Any]) -> dict[str, Any]:
    targets = config["targets"]
    corner = targets["corner"]
    center = targets["center"]
    offset = targets["offset"]

    return {
        "boundary_width": int(targets["boundary_width"]),
        "corner_radius": int(corner["radius"]),
        "corner_sigma": float(corner["sigma"]),
        "corner_source": str(corner["source"]),
        "corner_simplify_tolerance": float(corner["simplify_tolerance"]),
        "corner_cumulative_turn_angle_degrees": float(
            corner["cumulative_turn_angle_degrees"]
        ),
        "center_radius": int(center["radius"]),
        "center_sigma": float(center["sigma"]),
        "normalize_offset": bool(offset["normalize"]),
    }

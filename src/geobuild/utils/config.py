from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALLOWED_TARGET_NAMES = {"mask", "boundary", "corner", "center", "offset", "instance"}
DEFAULT_TARGET_CACHE_ROOT = "data/processed/target_cache"


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
    split_settings = split_config(config, split)

    if "manifest" in split_settings:
        return resolve_path(split_settings["manifest"], root)

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
        "active_targets": active_targets_from_config(config),
    }


def concrete_active_targets_from_config(config: dict[str, Any]) -> set[str]:
    active_targets = active_targets_from_config(config)

    if active_targets is None:
        return set(ALLOWED_TARGET_NAMES)

    return set(active_targets)


def target_cache_config_from_config(
    config: dict[str, Any],
    root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    targets = config.get("targets", {})
    cache = targets.get("cache")

    if not isinstance(cache, dict) or not bool(cache.get("enabled", False)):
        return {
            "enabled": False,
            "root": None,
            "targets": set(),
        }

    if "targets" not in cache:
        raise ValueError("targets.cache.targets must be an explicit list")

    if not isinstance(cache["targets"], list):
        raise TypeError("targets.cache.targets must be an explicit list")

    cache_targets = {str(name) for name in cache["targets"]}
    unknown = cache_targets - ALLOWED_TARGET_NAMES

    if unknown:
        raise ValueError(f"Unknown targets.cache.targets names: {sorted(unknown)}")

    active_targets = concrete_active_targets_from_config(config)
    inactive = cache_targets - active_targets

    if inactive:
        raise ValueError(
            "targets.cache.targets must be a subset of active targets; "
            f"inactive cache targets: {sorted(inactive)}, "
            f"active targets: {sorted(active_targets)}"
        )

    return {
        "enabled": True,
        "root": resolve_path(cache.get("root", DEFAULT_TARGET_CACHE_ROOT), root),
        "targets": cache_targets,
    }


def active_targets_from_config(config: dict[str, Any]) -> set[str] | None:
    targets = config.get("targets", {})
    active = targets.get("active", "auto")

    if active is None or str(active).lower() == "all":
        return None

    if isinstance(active, str):
        active_name = active.lower()

        if active_name == "auto":
            model_heads = config.get("model", {}).get("heads")

            if not isinstance(model_heads, dict):
                return None

            inferred = {
                str(name)
                for name, enabled in model_heads.items()
                if bool(enabled)
            }

            if not inferred:
                inferred = {"mask"}

            unknown = inferred - ALLOWED_TARGET_NAMES

            if unknown:
                raise ValueError(
                    f"Unknown target names in model.heads: {sorted(unknown)}"
                )

            return inferred

        if active_name in ALLOWED_TARGET_NAMES:
            return {active_name}

        raise ValueError(
            f"Unsupported targets.active value: {active!r}. "
            "Use 'auto', 'all', or a list of target names."
        )

    if isinstance(active, list):
        requested = {str(name) for name in active}
        unknown = requested - ALLOWED_TARGET_NAMES

        if unknown:
            raise ValueError(f"Unknown targets.active names: {sorted(unknown)}")

        if not requested:
            raise ValueError("targets.active list cannot be empty")

        return requested

    raise TypeError(
        "targets.active must be 'auto', 'all', a target name, or a list of names"
    )

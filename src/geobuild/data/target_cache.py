import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from geobuild.data.records import ImageRecord


TARGET_DTYPES = {
    "mask": np.dtype(np.uint8),
    "boundary": np.dtype(np.uint8),
    "corner": np.dtype(np.float32),
    "center": np.dtype(np.float32),
    "offset": np.dtype(np.float32),
    "instance": np.dtype(np.int32),
}


class TargetCache:
    def __init__(
        self,
        root: str | Path,
        targets: set[str],
        raster_config: dict[str, Any],
    ) -> None:
        self.root = Path(root)
        self.targets = set(targets)
        self.raster_config = dict(raster_config)

    def load(
        self,
        record: ImageRecord,
        active_targets: set[str],
    ) -> dict[str, np.ndarray]:
        cached = {}

        for target_name in sorted(self.targets & active_targets):
            array = self.load_target(record, target_name)

            if array is not None:
                cached[target_name] = array

        return cached

    def save(
        self,
        record: ImageRecord,
        targets: dict[str, np.ndarray],
    ) -> None:
        for target_name, array in targets.items():
            if target_name in self.targets:
                self.save_target(record, target_name, array)

    def load_target(
        self,
        record: ImageRecord,
        target_name: str,
    ) -> np.ndarray | None:
        cache_path = self.path_for(record, target_name)

        if not cache_path.exists():
            return None

        try:
            array = np.load(cache_path, allow_pickle=False)
        except (OSError, ValueError):
            return None

        if not self.is_valid(record, target_name, array):
            return None

        return np.ascontiguousarray(array)

    def save_target(
        self,
        record: ImageRecord,
        target_name: str,
        array: np.ndarray,
    ) -> None:
        if target_name not in TARGET_DTYPES:
            raise ValueError(f"Unknown target cache name: {target_name!r}")

        if not self.is_valid(record, target_name, array):
            raise ValueError(
                f"Cannot cache invalid {target_name!r} target: "
                f"shape={array.shape}, dtype={array.dtype}"
            )

        if self.load_target(record, target_name) is not None:
            return

        cache_path = self.path_for(record, target_name)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(
            f".{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )

        try:
            with tmp_path.open("wb") as f:
                np.save(f, np.ascontiguousarray(array), allow_pickle=False)

            existing = self.load_target(record, target_name)
            if existing is not None:
                return

            if cache_path.exists():
                tmp_path.replace(cache_path)
            else:
                try:
                    tmp_path.rename(cache_path)
                except FileExistsError:
                    if self.load_target(record, target_name) is None:
                        tmp_path.replace(cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def path_for(self, record: ImageRecord, target_name: str) -> Path:
        if target_name not in TARGET_DTYPES:
            raise ValueError(f"Unknown target cache name: {target_name!r}")

        return (
            self.root
            / target_name
            / self._hash_for(record, target_name)
            / f"{_safe_image_id(record.image_id)}.npy"
        )

    def is_valid(
        self,
        record: ImageRecord,
        target_name: str,
        array: np.ndarray,
    ) -> bool:
        return (
            tuple(array.shape) == _expected_shape(record, target_name)
            and array.dtype == TARGET_DTYPES[target_name]
        )

    def _hash_for(self, record: ImageRecord, target_name: str) -> str:
        payload = {
            "target": target_name,
            "params": _target_specific_params(target_name, self.raster_config),
            "record": {
                "image_id": record.image_id,
                "split": record.split,
                "width": int(record.width),
                "height": int(record.height),
                "annotations": _annotation_identity(record),
            },
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


def _expected_shape(record: ImageRecord, target_name: str) -> tuple[int, ...]:
    height = int(record.height)
    width = int(record.width)

    if target_name == "offset":
        return (2, height, width)

    return (height, width)


def _target_specific_params(
    target_name: str,
    raster_config: dict[str, Any],
) -> dict[str, Any]:
    if target_name == "boundary":
        return {"boundary_width": int(raster_config["boundary_width"])}

    if target_name == "corner":
        return {
            "corner_radius": int(raster_config["corner_radius"]),
            "corner_sigma": float(raster_config["corner_sigma"]),
            "corner_source": str(raster_config["corner_source"]),
            "corner_simplify_tolerance": float(
                raster_config["corner_simplify_tolerance"]
            ),
            "corner_cumulative_turn_angle_degrees": float(
                raster_config["corner_cumulative_turn_angle_degrees"]
            ),
        }

    if target_name == "center":
        return {
            "center_radius": int(raster_config["center_radius"]),
            "center_sigma": float(raster_config["center_sigma"]),
        }

    if target_name == "offset":
        return {"normalize_offset": bool(raster_config["normalize_offset"])}

    return {}


def _annotation_identity(record: ImageRecord) -> list[dict[str, Any]]:
    return [
        {
            "annotation_id": polygon.annotation_id,
            "category_id": polygon.category_id,
            "iscrowd": polygon.iscrowd,
            "area": polygon.area,
            "bbox": polygon.bbox,
            "exterior": polygon.exterior,
            "holes": polygon.holes,
        }
        for polygon in record.polygons
    ]


def _safe_image_id(image_id: int | str) -> str:
    raw = str(image_id)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]

    if not safe:
        safe = "image"

    if len(safe) > 80:
        safe = safe[:80].rstrip("._-")

    return f"{safe}_{digest}"

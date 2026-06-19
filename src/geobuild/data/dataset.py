import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from geobuild.data.rasterize import rasterize_record
from geobuild.data.records import ImageRecord
from geobuild.data.target_cache import TargetCache
from geobuild.data.transforms import EvalTransform, build_transform
from geobuild.utils.config import (
    concrete_active_targets_from_config,
    manifest_path_from_config,
    target_cache_config_from_config,
    target_config_from_config,
)


Sample = dict[str, Any]
Transform = Callable[[Sample], Sample]
TARGET_KEYS = ("mask", "boundary", "corner", "center", "offset", "instance")


class BuildingFootprintDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        target_config: dict[str, Any],
        transform: Transform | None = None,
        active_targets: set[str] | None = None,
        target_cache_config: dict[str, Any] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.target_config = dict(target_config)
        self.transform = transform if transform is not None else EvalTransform()
        self.active_targets = (
            set(active_targets)
            if active_targets is not None
            else _active_target_set(self.target_config.get("active_targets"))
        )
        self.target_cache = self._build_target_cache(target_cache_config)
        self._records = self._load_manifest(self.manifest_path)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Sample:
        record = self._records[index]

        with Image.open(record.image_path) as image:
            image_array = np.asarray(image.convert("RGB"))

        sample: Sample = {
            "image": image_array,
            "image_id": str(record.image_id),
        }
        sample.update(self._load_or_rasterize_targets(record))

        return self.transform(sample)

    def record_size(self, index: int) -> tuple[int, int]:
        record = self._records[index]
        return int(record.height), int(record.width)

    def _load_or_rasterize_targets(self, record: ImageRecord) -> dict[str, np.ndarray]:
        targets = {}

        if self.target_cache is not None:
            targets.update(self.target_cache.load(record, self.active_targets))

        missing_targets = self.active_targets - set(targets)

        if missing_targets:
            raster_config = {
                **self.target_config,
                "active_targets": missing_targets,
            }
            generated = rasterize_record(record, **raster_config).to_dict()
            generated = {
                name: value
                for name, value in generated.items()
                if name in missing_targets
            }
            targets.update(generated)

            if self.target_cache is not None:
                self.target_cache.save(record, generated)

        missing_after = self.active_targets - set(targets)

        if missing_after:
            raise RuntimeError(
                f"Missing active targets after rasterization: {sorted(missing_after)}"
            )

        return targets

    def _build_target_cache(
        self,
        cache_config: dict[str, Any] | None,
    ) -> TargetCache | None:
        if not cache_config or not bool(cache_config.get("enabled", False)):
            return None

        cache_targets = set(cache_config.get("targets", set()))

        if not cache_targets:
            return None

        return TargetCache(
            root=cache_config["root"],
            targets=cache_targets,
            raster_config=self.target_config,
        )

    @staticmethod
    def _load_manifest(path: Path) -> list[ImageRecord]:
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {path}")

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
                        f"Invalid manifest record in {path} at line {line_number}"
                    ) from exc

        return records


def _active_target_set(active_targets: Any) -> set[str]:
    if active_targets is None:
        return set(TARGET_KEYS)

    if isinstance(active_targets, str):
        if active_targets.lower() == "all":
            return set(TARGET_KEYS)
        return {active_targets}

    return {str(name) for name in active_targets}


def _pad_tensor(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    padded = torch.zeros(
        (tensor.shape[0], height, width),
        dtype=tensor.dtype,
    )
    padded[:, : tensor.shape[1], : tensor.shape[2]] = tensor
    return padded


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")

    return ((value + multiple - 1) // multiple) * multiple


def collate_samples(samples: list[Sample]) -> Sample:
    if not samples:
        raise ValueError("Cannot collate an empty batch")

    max_height = max(int(sample["image"].shape[-2]) for sample in samples)
    max_width = max(int(sample["image"].shape[-1]) for sample in samples)
    padded_height = _round_up_to_multiple(max_height, 32)
    padded_width = _round_up_to_multiple(max_width, 32)
    original_size = [
        (int(sample["image"].shape[-2]), int(sample["image"].shape[-1]))
        for sample in samples
    ]
    batch: Sample = {}
    target_keys = [
        key
        for key in TARGET_KEYS
        if key in samples[0]
    ]
    expected_keys = set(target_keys)

    for index, sample in enumerate(samples):
        actual_keys = {key for key in TARGET_KEYS if key in sample}

        if actual_keys != expected_keys:
            raise ValueError(
                "All samples in a batch must have the same target keys; "
                f"sample 0 has {sorted(expected_keys)}, "
                f"sample {index} has {sorted(actual_keys)}"
            )

    for key in ("image", *target_keys):
        batch[key] = torch.stack(
            [
                _pad_tensor(sample[key], padded_height, padded_width)
                for sample in samples
            ],
            dim=0,
        )

    valid_mask = torch.zeros(
        (len(samples), 1, padded_height, padded_width),
        dtype=torch.float32,
    )

    for index, (height, width) in enumerate(original_size):
        valid_mask[index, :, :height, :width] = 1.0

    batch["valid_mask"] = valid_mask
    batch["original_size"] = original_size
    batch["image_id"] = [str(sample["image_id"]) for sample in samples]

    return batch


def build_dataset(
    config: dict[str, Any],
    split: str,
    force_noaug: bool = False,
) -> BuildingFootprintDataset:
    return BuildingFootprintDataset(
        manifest_path=manifest_path_from_config(config, split),
        target_config=target_config_from_config(config),
        transform=build_transform(config, split, force_noaug=force_noaug),
        active_targets=concrete_active_targets_from_config(config),
        target_cache_config=target_cache_config_from_config(config),
    )


def build_dataloader(config: dict[str, Any], split: str) -> DataLoader:
    loader_config = config.get("loader", {})

    return DataLoader(
        build_dataset(config, split),
        batch_size=int(loader_config.get("batch_size", 4)),
        shuffle=bool(loader_config.get("shuffle", split == "train")),
        num_workers=int(loader_config.get("num_workers", 0)),
        pin_memory=bool(loader_config.get("pin_memory", True)),
        collate_fn=collate_samples,
    )

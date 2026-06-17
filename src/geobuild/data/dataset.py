import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from geobuild.data.rasterize import rasterize_record
from geobuild.data.records import ImageRecord
from geobuild.data.transforms import EvalTransform
from geobuild.utils.config import manifest_path_from_config, target_config_from_config


Sample = dict[str, Any]
Transform = Callable[[Sample], Sample]


class BuildingFootprintDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        target_config: dict[str, Any],
        transform: Transform | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.target_config = dict(target_config)
        self.transform = transform if transform is not None else EvalTransform()
        self.records = self._load_manifest(self.manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Sample:
        record = self.records[index]

        with Image.open(record.image_path) as image:
            image_array = np.asarray(image.convert("RGB"))

        targets = rasterize_record(record, **self.target_config)

        sample: Sample = {
            "image": image_array,
            "mask": targets.mask,
            "boundary": targets.boundary,
            "corner": targets.corner,
            "center": targets.center,
            "offset": targets.offset,
            "image_id": str(record.image_id),
        }

        return self.transform(sample)

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

    for key in ("image", "mask", "boundary", "corner", "center", "offset"):
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


def build_dataset(config: dict[str, Any], split: str) -> BuildingFootprintDataset:
    return BuildingFootprintDataset(
        manifest_path=manifest_path_from_config(config, split),
        target_config=target_config_from_config(config),
        transform=EvalTransform(),
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

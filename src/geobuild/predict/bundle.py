from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from geobuild.data.dataset import BuildingFootprintDataset, collate_samples
from geobuild.data.transforms import build_transform
from geobuild.models.factory import build_model
from geobuild.train.checkpoint import load_checkpoint
from geobuild.utils.config import (
    load_config,
    manifest_path_from_config,
    resolve_path,
    target_cache_config_from_config,
    target_config_from_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PredictionBundle:
    config: dict[str, Any]
    model: nn.Module
    loader: DataLoader
    checkpoint: dict[str, Any]
    checkpoint_path: Path
    split: str
    experiment_name: str
    device: torch.device


def resolve_device(device_arg: str | None = None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_eval_dataloader(config: dict[str, Any], split: str) -> DataLoader:
    loader_config = config.get("loader", {})
    dataset = BuildingFootprintDataset(
        manifest_path=manifest_path_from_config(config, split),
        target_config=target_config_from_config(config),
        transform=build_transform(config, split, force_noaug=True),
        active_targets=set(),
        target_cache_config=target_cache_config_from_config(config),
    )

    return DataLoader(
        dataset,
        batch_size=int(loader_config.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(loader_config.get("num_workers", 0)),
        pin_memory=bool(loader_config.get("pin_memory", True)),
        collate_fn=collate_samples,
    )


def _build_checkpoint_optimizer(
    model: nn.Module,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    train_config = config.get("train", {})
    optimizer_name = str(train_config.get("optimizer", "AdamW")).lower()

    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer: {train_config.get('optimizer')!r}")

    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("lr", 0.0001)),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )


def load_prediction_bundle(
    config_path: str | Path,
    checkpoint_path: str | Path,
    split: str,
    device_arg: str | None = None,
    root: str | Path = PROJECT_ROOT,
) -> PredictionBundle:
    config = load_config(config_path, root=root)
    resolved_checkpoint_path = resolve_path(checkpoint_path, root=root)
    device = resolve_device(device_arg)

    model = build_model(config).to(device)
    optimizer = _build_checkpoint_optimizer(model, config)
    _, _, checkpoint = load_checkpoint(
        resolved_checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=device,
    )
    model.eval()

    return PredictionBundle(
        config=config,
        model=model,
        loader=build_eval_dataloader(config, split),
        checkpoint=checkpoint,
        checkpoint_path=resolved_checkpoint_path,
        split=str(split),
        experiment_name=str(config.get("experiment", {}).get("name", "")),
        device=device,
    )

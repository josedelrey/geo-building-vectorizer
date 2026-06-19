import argparse
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]

from geobuild.data.dataset import build_dataset, collate_samples
from geobuild.losses.multitask import MultiTaskLoss
from geobuild.models.factory import build_model
from geobuild.train.loop import run_training
from geobuild.utils.config import load_config, output_path_from_config


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overfit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_loader(
    dataset: Any,
    config: dict[str, Any],
    shuffle: bool,
) -> DataLoader:
    loader_config = config.get("loader", {})

    return DataLoader(
        dataset,
        batch_size=int(loader_config.get("batch_size", 4)),
        shuffle=shuffle,
        num_workers=int(loader_config.get("num_workers", 0)),
        pin_memory=bool(loader_config.get("pin_memory", True)),
        collate_fn=collate_samples,
    )


def build_datasets(
    config: dict[str, Any],
    overfit: int | None,
) -> tuple[Any, Any]:
    if overfit is None:
        return build_dataset(config, "train"), build_dataset(config, "val")

    if overfit <= 0:
        raise ValueError(f"--overfit must be positive, got {overfit}")

    train_dataset = build_dataset(config, "train")
    val_dataset = build_dataset(config, "train", force_noaug=True)
    count = min(int(overfit), len(train_dataset))
    indices = list(range(count))
    return Subset(train_dataset, indices), Subset(val_dataset, indices)


def build_amp_scaler(config: dict[str, Any], device: torch.device) -> Any | None:
    amp_enabled = bool(config.get("train", {}).get("amp", False))

    if not amp_enabled or device.type != "cuda":
        return None

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=True)
        except TypeError:
            return torch.amp.GradScaler(enabled=True)

    return torch.cuda.amp.GradScaler(enabled=True)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> Any | None:
    scheduler_config = config.get("scheduler", {})
    scheduler_name = str(scheduler_config.get("name", "none")).lower()

    if scheduler_name == "none":
        return None

    if scheduler_name == "cosine":
        train_config = config.get("train", {})
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_config.get("epochs", 1)),
            eta_min=float(scheduler_config.get("min_lr", 0.0)),
        )

    raise ValueError(
        f"Unsupported scheduler: {scheduler_config.get('name')!r}. "
        "Supported schedulers: 'none', 'cosine'."
    )


def run_dir_from_config(config: dict[str, Any]) -> Path:
    output = config.get("output", {})

    if "run_dir" in output:
        return output_path_from_config(config, "run_dir", root=ROOT)

    experiment_name = config["experiment"]["name"]
    return output_path_from_config(
        {
            **config,
            "output": {
                **output,
                "run_dir": f"runs/{experiment_name}",
            },
        },
        "run_dir",
        root=ROOT,
    )


def configure_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
        force=True,
    )


def save_config(config: dict[str, Any], run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config, root=ROOT)

    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = int(args.epochs)

    set_seed(int(config.get("experiment", {}).get("seed", 42)))
    device = resolve_device(args.device)
    run_dir = run_dir_from_config(config)
    configure_logging(run_dir)
    save_config(config, run_dir)

    train_dataset, val_dataset = build_datasets(config, args.overfit)
    train_loader = build_loader(
        train_dataset,
        config,
        shuffle=bool(config.get("loader", {}).get("shuffle", True)),
    )
    val_loader = build_loader(val_dataset, config, shuffle=False)

    model = build_model(config)
    loss_fn = MultiTaskLoss(config)
    train_config = config.get("train", {})
    optimizer_name = str(train_config.get("optimizer", "AdamW")).lower()

    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer: {train_config.get('optimizer')!r}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("lr", 0.0001)),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    scheduler = build_scheduler(optimizer, config)
    scaler = build_amp_scaler(config, device)

    LOGGER.info("Run directory: %s", run_dir)
    LOGGER.info("Device: %s", device)
    LOGGER.info("Train samples: %d", len(train_dataset))
    LOGGER.info("Val samples: %d", len(val_dataset))

    history = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=config,
        run_dir=run_dir,
        device=device,
        loss_fn=loss_fn,
        scheduler=scheduler,
        scaler=scaler,
    )

    if history:
        LOGGER.info("Finished epoch %d", int(history[-1]["epoch"]))


if __name__ == "__main__":
    main()

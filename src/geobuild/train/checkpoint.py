import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


LOGGER = logging.getLogger(__name__)


def _state_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Expected object with state_dict() or dict, got {type(obj).__name__}")


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return

    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"].cpu())
        if "torch_cuda" in state and torch.cuda.is_available():
            cuda_rng_state = [rng_state.cpu() for rng_state in state["torch_cuda"]]
            torch.cuda.set_rng_state_all(cuda_rng_state)
    except Exception as exc:
        LOGGER.warning("Could not restore checkpoint RNG state: %s", exc)


def _load_torch_checkpoint(
    checkpoint_path: Path,
    map_location: torch.device | str | None,
) -> dict[str, Any]:
    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def _move_optimizer_state(optimizer: Optimizer, device: torch.device | str | None) -> None:
    if device is None:
        return

    target_device = torch.device(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(target_device)


def save_checkpoint(
    output_dir: str | Path,
    epoch: int,
    model: nn.Module | dict[str, Any],
    optimizer: Optimizer | dict[str, Any],
    best_val_iou: float,
    config: dict[str, Any],
    val_iou: float | None = None,
    scaler: Any | None = None,
    scheduler: Any | None = None,
    save_every: int | None = None,
) -> float:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    current_best_val_iou = float(best_val_iou)
    improved = val_iou is not None and float(val_iou) > current_best_val_iou

    if improved:
        current_best_val_iou = float(val_iou)

    checkpoint: dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": _state_dict(model),
        "optimizer_state_dict": _state_dict(optimizer),
        "best_val_iou": current_best_val_iou,
        "config": config,
        "rng_state": _rng_state(),
    }

    if scaler is not None:
        checkpoint["scaler_state_dict"] = _state_dict(scaler)

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = _state_dict(scheduler)

    torch.save(checkpoint, output_path / "last.pt")

    if improved:
        torch.save(checkpoint, output_path / "best.pt")

    if save_every is not None:
        save_every = int(save_every)
        if save_every <= 0:
            raise ValueError(f"save_every must be positive, got {save_every}")
        if int(epoch) % save_every == 0:
            torch.save(checkpoint, output_path / f"epoch_{int(epoch):03d}.pt")

    return current_best_val_iou


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    device: torch.device | str | None = None,
    map_location: torch.device | str | None = None,
) -> tuple[int, float, dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    if map_location is None:
        map_location = device

    checkpoint = _load_torch_checkpoint(path, map_location)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _move_optimizer_state(optimizer, device)

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    _restore_rng_state(checkpoint.get("rng_state"))

    checkpoint_epoch = int(checkpoint["epoch"])
    start_epoch = checkpoint_epoch + 1
    best_val_iou = float(checkpoint.get("best_val_iou", float("-inf")))

    return start_epoch, best_val_iou, checkpoint

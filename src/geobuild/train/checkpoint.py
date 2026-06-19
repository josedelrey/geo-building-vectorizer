from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def _state_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Expected object with state_dict() or dict, got {type(obj).__name__}")


def save_checkpoint(
    output_dir: str | Path,
    epoch: int,
    model: nn.Module | dict[str, Any],
    optimizer: Optimizer | dict[str, Any],
    best_val_iou: float,
    config: dict[str, Any],
    val_iou: float | None = None,
    scaler: Any | None = None,
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
    }

    if scaler is not None:
        checkpoint["scaler_state_dict"] = _state_dict(scaler)

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

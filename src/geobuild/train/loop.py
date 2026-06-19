from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..losses.multitask import MultiTaskLoss
from ..metrics.segmentation import SegmentationMetrics
from .checkpoint import save_checkpoint
from .logger import CSVLogger
from .preview import save_prediction_preview


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value

    return moved


def _autocast_context(device: torch.device, enabled: bool) -> Any:
    if not enabled:
        return nullcontext()

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)

    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)

    return nullcontext()


def _build_grad_scaler(device: torch.device, amp_enabled: bool) -> Any | None:
    enabled = bool(amp_enabled and device.type == "cuda")

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)

    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=enabled)

    return None


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _lr(optimizer: Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _training_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("train", config.get("training", {}))


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device | str,
    scaler: Any | None = None,
    amp_enabled: bool = False,
) -> dict[str, float]:
    device = torch.device(device)
    model.train()

    total_losses = []
    mask_losses = []

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        images = batch["image"]

        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, amp_enabled):
            outputs = model(images)
            loss_dict = loss_fn(outputs, batch)
            total_loss = loss_dict["total"]

        scaler_is_enabled = bool(getattr(scaler, "is_enabled", lambda: True)())

        if scaler is not None and scaler_is_enabled:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        total_losses.append(float(loss_dict["total"].detach().cpu()))
        mask_losses.append(float(loss_dict["mask"].detach().cpu()))

    return {
        "train_loss": _mean(total_losses),
        "train_mask_loss": _mean(mask_losses),
    }


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device | str,
    threshold: float = 0.5,
    amp_enabled: bool = False,
    preview_run_dir: str | Path | None = None,
    preview_epoch: int | None = None,
    preview_saved: bool = False,
) -> dict[str, float | bool]:
    device = torch.device(device)
    model.eval()

    total_losses = []
    mask_losses = []
    metrics = SegmentationMetrics(threshold=threshold)

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            images = batch["image"]

            with _autocast_context(device, amp_enabled):
                outputs = model(images)
                loss_dict = loss_fn(outputs, batch)

            total_losses.append(float(loss_dict["total"].detach().cpu()))
            mask_losses.append(float(loss_dict["mask"].detach().cpu()))
            metrics.update(outputs["mask"], batch["mask"], batch["valid_mask"])

            if (
                preview_run_dir is not None
                and preview_epoch is not None
                and not preview_saved
            ):
                cpu_batch = _move_batch_to_device(batch, torch.device("cpu"))
                cpu_outputs = {
                    key: value.detach().cpu()
                    for key, value in outputs.items()
                    if isinstance(value, torch.Tensor)
                }
                save_prediction_preview(
                    cpu_batch,
                    cpu_outputs,
                    preview_run_dir,
                    epoch=preview_epoch,
                    threshold=threshold,
                )
                preview_saved = True

    metric_values = metrics.compute()

    return {
        "val_loss": _mean(total_losses),
        "val_mask_loss": _mean(mask_losses),
        "val_iou": metric_values["iou"],
        "val_dice": metric_values["dice"],
        "val_precision": metric_values["precision"],
        "val_recall": metric_values["recall"],
        "preview_saved": preview_saved,
    }


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    config: dict[str, Any],
    run_dir: str | Path,
    device: torch.device | str,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> list[dict[str, float]]:
    device = torch.device(device)
    model.to(device)

    train_config = _training_config(config)
    epochs = int(train_config.get("epochs", 1))
    amp_enabled = bool(train_config.get("amp", False))
    threshold = float(config.get("metrics", {}).get("threshold", 0.5))
    preview_every = train_config.get("preview_every")
    save_every = train_config.get("save_every")
    if not save_every:
        save_every = None

    if scaler is None:
        scaler = _build_grad_scaler(device, amp_enabled)

    loss_fn = MultiTaskLoss(config)
    logger = CSVLogger(Path(run_dir) / "metrics.csv")
    best_val_iou = float("-inf")
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )

        should_preview = (
            preview_every is not None
            and int(preview_every) > 0
            and epoch % int(preview_every) == 0
        )
        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            threshold=threshold,
            amp_enabled=amp_enabled,
            preview_run_dir=run_dir if should_preview else None,
            preview_epoch=epoch if should_preview else None,
            preview_saved=False,
        )
        val_metrics.pop("preview_saved")

        if scheduler is not None:
            scheduler.step()

        row = {
            **train_metrics,
            **val_metrics,
            "epoch": epoch,
            "lr": _lr(optimizer),
        }
        logger.log(row)

        best_val_iou = save_checkpoint(
            output_dir=Path(run_dir) / "checkpoints",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_val_iou=best_val_iou,
            config=config,
            val_iou=float(val_metrics["val_iou"]),
            scaler=scaler,
            save_every=save_every,
        )
        history.append({key: float(value) for key, value in row.items()})

    return history

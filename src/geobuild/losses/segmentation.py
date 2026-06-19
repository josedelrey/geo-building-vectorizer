from typing import Any

import torch
import torch.nn.functional as F


def _check_binary_segmentation_shapes(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    expected_shape = tuple(logits.shape)

    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(
            f"logits must have shape [B, 1, H, W], got {tuple(logits.shape)}"
        )
    if tuple(target.shape) != expected_shape:
        raise ValueError(
            f"target shape must match logits shape {expected_shape}, "
            f"got {tuple(target.shape)}"
        )
    if tuple(valid_mask.shape) != expected_shape:
        raise ValueError(
            f"valid_mask shape must match logits shape {expected_shape}, "
            f"got {tuple(valid_mask.shape)}"
        )


def masked_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    _check_binary_segmentation_shapes(logits, target, valid_mask)

    bce_weight = float(config.get("bce_weight", 1.0))
    dice_weight = float(config.get("dice_weight", 1.0))
    eps = float(config.get("eps", 1e-6))

    valid_mask = valid_mask.to(dtype=logits.dtype)
    target = target.to(dtype=logits.dtype)

    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    bce = (bce * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

    probability = torch.sigmoid(logits)
    intersection = (probability * target * valid_mask).sum()
    denominator = (probability * valid_mask).sum() + (target * valid_mask).sum()
    dice = (2.0 * intersection + eps) / (denominator + eps)
    dice_loss = 1.0 - dice

    return bce_weight * bce + dice_weight * dice_loss

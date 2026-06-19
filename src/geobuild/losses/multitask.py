from typing import Any

import torch
from torch import nn

from .segmentation import masked_bce_dice_loss


class MultiTaskLoss(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if "mask" not in outputs:
            raise KeyError("MultiTaskLoss requires outputs['mask']")

        loss_config = self.config.get("loss", {})
        mask_config = loss_config.get("mask", {})
        mask_weight = float(loss_config.get("mask_weight", 1.0))

        mask_loss = masked_bce_dice_loss(
            logits=outputs["mask"],
            target=batch["mask"],
            valid_mask=batch["valid_mask"],
            config=mask_config,
        )
        total = mask_weight * mask_loss

        return {
            "mask": mask_loss,
            "total": total,
        }

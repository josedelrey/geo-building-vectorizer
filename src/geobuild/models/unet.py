from typing import Any

import segmentation_models_pytorch as smp
import torch
from torch import nn


class UNetMaskModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        encoder_name = str(config.get("encoder_name", "resnet34"))
        encoder_weights = config.get("encoder_weights", "imagenet")
        in_channels = int(config.get("in_channels", 3))
        classes = int(config.get("classes", 1))

        self.in_channels = in_channels
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
            1, 3, 1, 1
        )
        self.register_buffer("image_mean", mean)
        self.register_buffer("image_std", std)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.ndim != 4:
            raise ValueError(
                f"Expected image tensor shaped [B, C, H, W], got {tuple(image.shape)}"
            )
        if image.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {image.shape[1]}"
            )
        if image.shape[1] != self.image_mean.shape[1]:
            raise ValueError("ImageNet normalization currently requires 3 input channels")

        normalized = (image - self.image_mean) / self.image_std
        logits = self.model(normalized)
        return {"mask": logits}

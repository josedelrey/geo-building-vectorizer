from typing import Any

from torch import nn

from .unet import UNetMaskModel


def build_model(config: dict[str, Any]) -> nn.Module:
    model_config = config.get("model", {})
    model_name = str(model_config.get("name", "unet")).lower()

    if model_name == "unet":
        return UNetMaskModel(model_config)

    raise ValueError(
        f"Unknown model name: {model_name!r}. Supported model names: 'unet'."
    )

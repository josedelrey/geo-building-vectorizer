from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


def _image_id(batch: dict[str, Any], index: int) -> str:
    image_ids = batch.get("image_id")
    if image_ids is None:
        return f"sample_{index}"
    return str(image_ids[index])


def _crop_size(
    batch: dict[str, Any],
    index: int,
    height: int,
    width: int,
) -> tuple[int, int]:
    original_size = batch.get("original_size")
    if original_size is None:
        return height, width

    original_height, original_width = original_size[index]
    return min(int(original_height), height), min(int(original_width), width)


def _valid_array(valid_mask: torch.Tensor) -> np.ndarray:
    return valid_mask.detach().cpu().squeeze(0).bool().numpy()


def _image_panel(image: torch.Tensor, valid_mask: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    valid = _valid_array(valid_mask)
    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0).astype(np.uint8)
    array[~valid] = np.array([220, 220, 220], dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def _gray_panel(value: torch.Tensor, valid_mask: torch.Tensor) -> Image.Image:
    array = value.detach().cpu().squeeze(0).float().numpy()
    valid = _valid_array(valid_mask)
    gray = np.clip(array, 0.0, 1.0)
    gray = (gray * 255.0).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    rgb[~valid] = np.array([220, 220, 220], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _probability_panel(
    probability: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Image.Image:
    value = probability.detach().cpu().squeeze(0).float().numpy()
    valid = _valid_array(valid_mask)
    value = np.clip(value, 0.0, 1.0)
    red = (value * 255.0).astype(np.uint8)
    green = ((1.0 - np.abs(value - 0.5) * 2.0) * 255.0).astype(np.uint8)
    blue = ((1.0 - value) * 255.0).astype(np.uint8)
    rgb = np.stack([red, green, blue], axis=2)
    rgb[~valid] = np.array([220, 220, 220], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _error_panel(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Image.Image:
    prediction_bool = prediction.detach().cpu().squeeze(0).bool().numpy()
    target_bool = target.detach().cpu().squeeze(0).bool().numpy()
    valid = _valid_array(valid_mask)
    height, width = target_bool.shape

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[prediction_bool & target_bool] = np.array([255, 255, 255], dtype=np.uint8)
    rgb[prediction_bool & ~target_bool] = np.array([220, 40, 40], dtype=np.uint8)
    rgb[~prediction_bool & target_bool] = np.array([40, 120, 220], dtype=np.uint8)
    rgb[~valid] = np.array([220, 220, 220], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _titled_panel(title: str, image: Image.Image) -> Image.Image:
    title_height = 28
    panel = Image.new("RGB", (image.width, image.height + title_height), "white")
    panel.paste(image, (0, title_height))

    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), title, fill="black")
    return panel


def save_prediction_preview(
    batch: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    run_dir: str | Path,
    epoch: int,
    threshold: float = 0.5,
    index: int = 0,
) -> Path:
    if "mask" not in outputs:
        raise KeyError("save_prediction_preview requires outputs['mask']")

    image = batch["image"][index]
    target = batch["mask"][index]
    valid_mask = batch["valid_mask"][index]
    logits = outputs["mask"][index]

    height = int(image.shape[-2])
    width = int(image.shape[-1])
    crop_height, crop_width = _crop_size(batch, index, height, width)

    image = image[:, :crop_height, :crop_width]
    target = target[:, :crop_height, :crop_width]
    valid_mask = valid_mask[:, :crop_height, :crop_width]
    logits = logits[:, :crop_height, :crop_width]

    probability = torch.sigmoid(logits)
    prediction = (probability >= float(threshold)).to(dtype=torch.float32)

    panels = [
        _titled_panel("Input image", _image_panel(image, valid_mask)),
        _titled_panel("Ground-truth mask", _gray_panel(target, valid_mask)),
        _titled_panel(
            "Predicted probability",
            _probability_panel(probability, valid_mask),
        ),
        _titled_panel("Thresholded prediction", _gray_panel(prediction, valid_mask)),
        _titled_panel("Error map", _error_panel(prediction, target, valid_mask)),
    ]

    panel_width = max(panel.width for panel in panels)
    panel_height = max(panel.height for panel in panels)
    preview = Image.new("RGB", (panel_width * len(panels), panel_height), "white")

    for panel_index, panel in enumerate(panels):
        preview.paste(panel, (panel_index * panel_width, 0))

    preview_dir = Path(run_dir) / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        preview_dir / f"epoch_{int(epoch):03d}_{_image_id(batch, index)}.png"
    )
    preview.save(output_path)

    return output_path

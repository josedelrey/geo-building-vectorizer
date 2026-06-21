import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from geobuild.predict.bundle import PredictionBundle


SCALAR_DENSE_HEADS = ("mask", "boundary", "corner", "center")
OFFSET_HEAD = "offset"


@dataclass(frozen=True)
class PredictionExportResult:
    manifest_path: Path
    count: int


def _safe_filename_part(value: Any) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(value)
    )
    safe = safe.strip("._")
    return (safe or "image")[:128]


def _as_output_dict(outputs: Any) -> dict[str, torch.Tensor]:
    if isinstance(outputs, torch.Tensor):
        return {"mask": outputs}

    if not isinstance(outputs, dict):
        raise TypeError(
            "Model must return a tensor or a mapping of output heads to tensors, "
            f"got {type(outputs).__name__}"
        )

    tensor_outputs = {}

    for name, value in outputs.items():
        if isinstance(value, torch.Tensor):
            tensor_outputs[str(name)] = value

    return tensor_outputs


def _crop_output(
    output: torch.Tensor,
    batch_index: int,
    height: int,
    width: int,
    name: str,
) -> torch.Tensor:
    if output.ndim != 4:
        raise ValueError(
            f"Output {name!r} must have shape [B, C, H, W], got {tuple(output.shape)}"
        )

    sample = output[batch_index]

    if int(sample.shape[-2]) < height or int(sample.shape[-1]) < width:
        raise ValueError(
            f"Output {name!r} is smaller than original_size {(height, width)}: "
            f"{tuple(sample.shape)}"
        )

    return sample[:, :height, :width]


def _scalar_probability_array(
    output: torch.Tensor,
    batch_index: int,
    height: int,
    width: int,
    name: str,
) -> np.ndarray:
    cropped = _crop_output(
        torch.sigmoid(output),
        batch_index=batch_index,
        height=height,
        width=width,
        name=name,
    )

    if int(cropped.shape[0]) != 1:
        raise ValueError(
            f"Scalar dense output {name!r} must have one channel, "
            f"got {int(cropped.shape[0])}"
        )

    return cropped.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)


def _offset_array(
    output: torch.Tensor,
    batch_index: int,
    height: int,
    width: int,
) -> np.ndarray:
    cropped = _crop_output(
        output,
        batch_index=batch_index,
        height=height,
        width=width,
        name=OFFSET_HEAD,
    )

    if int(cropped.shape[0]) != 2:
        raise ValueError(f"Offset output must have two channels, got {cropped.shape[0]}")

    return cropped.detach().cpu().numpy().astype(np.float32, copy=False)


def _sample_arrays(
    outputs: dict[str, torch.Tensor],
    batch_index: int,
    height: int,
    width: int,
) -> dict[str, np.ndarray]:
    arrays = {}

    for name in SCALAR_DENSE_HEADS:
        if name not in outputs:
            continue

        arrays[name] = _scalar_probability_array(
            outputs[name],
            batch_index=batch_index,
            height=height,
            width=width,
            name=name,
        )

    if OFFSET_HEAD in outputs:
        arrays[OFFSET_HEAD] = _offset_array(
            outputs[OFFSET_HEAD],
            batch_index=batch_index,
            height=height,
            width=width,
        )

    if not arrays:
        raise ValueError(
            "Model did not return any supported prediction heads: "
            f"{[*SCALAR_DENSE_HEADS, OFFSET_HEAD]}"
        )

    return arrays


def export_predictions(
    bundle: PredictionBundle,
    out_dir: str | Path,
) -> PredictionExportResult:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "predictions.jsonl"
    count = 0

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for batch in tqdm(bundle.loader, desc=f"Predict {bundle.split}"):
            images = batch["image"].to(bundle.device, non_blocking=True)

            with torch.inference_mode():
                outputs = _as_output_dict(bundle.model(images))

            for batch_index, image_id in enumerate(batch["image_id"]):
                height, width = batch["original_size"][batch_index]
                height = int(height)
                width = int(width)
                arrays = _sample_arrays(
                    outputs,
                    batch_index=batch_index,
                    height=height,
                    width=width,
                )
                npz_path = (
                    output_dir
                    / f"{count:06d}_{_safe_filename_part(image_id)}.npz"
                )
                np.savez_compressed(npz_path, **arrays)

                record = {
                    "image_id": str(image_id),
                    "split": bundle.split,
                    "height": height,
                    "width": width,
                    "npz_path": str(npz_path),
                    "available_outputs": list(arrays.keys()),
                    "checkpoint_path": str(bundle.checkpoint_path),
                    "experiment_name": bundle.experiment_name,
                }
                manifest_file.write(json.dumps(record) + "\n")
                count += 1

    return PredictionExportResult(manifest_path=manifest_path, count=count)

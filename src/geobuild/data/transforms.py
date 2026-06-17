from typing import Any

import numpy as np
import torch


class EvalTransform:
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image = np.asarray(sample["image"], dtype=np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        transformed = {
            "image": torch.from_numpy(np.ascontiguousarray(image)).float(),
            "image_id": str(sample["image_id"]),
        }

        for key in ("mask", "boundary", "corner", "center"):
            target = np.asarray(sample[key], dtype=np.float32)
            target = target[None, :, :]
            transformed[key] = torch.from_numpy(np.ascontiguousarray(target)).float()

        offset = np.asarray(sample["offset"], dtype=np.float32)

        if offset.ndim != 3:
            raise ValueError(f"offset must have shape [2, H, W], got {offset.shape}")

        if offset.shape[0] != 2:
            raise ValueError(f"offset must have shape [2, H, W], got {offset.shape}")

        transformed["offset"] = torch.from_numpy(
            np.ascontiguousarray(offset)
        ).float()

        return transformed

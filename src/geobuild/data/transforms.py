from typing import Any

import numpy as np
import torch


TARGET_KEYS = ("mask", "boundary", "corner", "center")
SPATIAL_TARGET_KEYS = TARGET_KEYS + ("instance",)


def _validate_offset(offset: np.ndarray) -> None:
    if offset.ndim != 3:
        raise ValueError(f"offset must have shape [2, H, W], got {offset.shape}")

    if offset.shape[0] != 2:
        raise ValueError(f"offset must have shape [2, H, W], got {offset.shape}")


class EvalTransform:
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image = np.asarray(sample["image"], dtype=np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        transformed = {
            "image": torch.from_numpy(np.ascontiguousarray(image)).float(),
            "image_id": str(sample["image_id"]),
        }

        for key in SPATIAL_TARGET_KEYS:
            if key not in sample:
                continue

            target = np.asarray(sample[key], dtype=np.float32)
            target = target[None, :, :]
            transformed[key] = torch.from_numpy(np.ascontiguousarray(target)).float()

        if "offset" in sample:
            offset = np.asarray(sample["offset"], dtype=np.float32)
            _validate_offset(offset)

            transformed["offset"] = torch.from_numpy(
                np.ascontiguousarray(offset)
            ).float()

        return transformed


class SafeAugTransform:
    def __init__(
        self,
        hflip_p: float = 0.5,
        vflip_p: float = 0.5,
        rot90: bool = True,
    ) -> None:
        self.hflip_p = float(hflip_p)
        self.vflip_p = float(vflip_p)
        self.rot90 = bool(rot90)
        self.eval_transform = EvalTransform()

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        augmented = {
            "image": np.asarray(sample["image"]),
            "image_id": str(sample["image_id"]),
        }

        for key in SPATIAL_TARGET_KEYS:
            if key in sample:
                augmented[key] = np.asarray(sample[key])

        if "offset" in sample:
            offset = np.asarray(sample["offset"], dtype=np.float32)
            _validate_offset(offset)
            augmented["offset"] = offset

        if np.random.random() < self.hflip_p:
            augmented = self._horizontal_flip(augmented)

        if np.random.random() < self.vflip_p:
            augmented = self._vertical_flip(augmented)

        if self.rot90:
            augmented = self._rot90(augmented, int(np.random.randint(0, 4)))

        for key, value in augmented.items():
            if isinstance(value, np.ndarray):
                augmented[key] = np.ascontiguousarray(value)

        return self.eval_transform(augmented)

    @staticmethod
    def _horizontal_flip(sample: dict[str, Any]) -> dict[str, Any]:
        sample["image"] = np.flip(sample["image"], axis=1)

        for key in SPATIAL_TARGET_KEYS:
            if key in sample:
                sample[key] = np.flip(sample[key], axis=1)

        if "offset" in sample:
            offset = np.flip(sample["offset"], axis=2).copy()
            offset[0] *= -1.0
            sample["offset"] = offset
        return sample

    @staticmethod
    def _vertical_flip(sample: dict[str, Any]) -> dict[str, Any]:
        sample["image"] = np.flip(sample["image"], axis=0)

        for key in SPATIAL_TARGET_KEYS:
            if key in sample:
                sample[key] = np.flip(sample[key], axis=0)

        if "offset" in sample:
            offset = np.flip(sample["offset"], axis=1).copy()
            offset[1] *= -1.0
            sample["offset"] = offset
        return sample

    @staticmethod
    def _rot90(sample: dict[str, Any], k: int) -> dict[str, Any]:
        k = int(k) % 4

        if k == 0:
            return sample

        sample["image"] = np.rot90(sample["image"], k=k, axes=(0, 1))

        for key in SPATIAL_TARGET_KEYS:
            if key in sample:
                sample[key] = np.rot90(sample[key], k=k, axes=(0, 1))

        if "offset" in sample:
            offset = np.rot90(sample["offset"], k=k, axes=(1, 2)).copy()
            x = offset[0].copy()
            y = offset[1].copy()

            if k == 1:
                offset[0] = y
                offset[1] = -x
            elif k == 2:
                offset[0] = -x
                offset[1] = -y
            elif k == 3:
                offset[0] = -y
                offset[1] = x

            sample["offset"] = offset
        return sample


def build_transform(
    config: dict[str, Any],
    split: str,
    force_noaug: bool = False,
) -> EvalTransform | SafeAugTransform:
    if force_noaug or split != "train":
        return EvalTransform()

    augmentation_config = config.get("augmentation", {})
    augmentation_name = str(augmentation_config.get("name", "noaug")).lower()

    if augmentation_name == "noaug":
        return EvalTransform()

    if augmentation_name == "safeaug":
        return SafeAugTransform(
            hflip_p=float(augmentation_config.get("hflip_p", 0.5)),
            vflip_p=float(augmentation_config.get("vflip_p", 0.5)),
            rot90=bool(augmentation_config.get("rot90", True)),
        )

    raise ValueError(
        f"Unknown augmentation name: {augmentation_name!r}. "
        "Supported augmentation names: 'noaug', 'safeaug'."
    )

from typing import Any

import cv2
import numpy as np


def ensure_2d_array(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array)

    if value.ndim == 3 and int(value.shape[0]) == 1:
        value = value[0]

    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [H, W], got {value.shape}")

    return np.ascontiguousarray(value)


def binary_mask_from_probability(
    mask_prob: np.ndarray,
    threshold: float,
) -> np.ndarray:
    probability = ensure_2d_array(mask_prob, "mask_prob")
    return np.ascontiguousarray(probability >= float(threshold))


def connected_components(
    binary_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = ensure_2d_array(binary_mask, "binary_mask").astype(np.uint8, copy=False)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    return labels, stats


def component_mask(labels: np.ndarray, component_id: int) -> np.ndarray:
    return np.ascontiguousarray(labels == int(component_id))


def find_contours(binary_mask: np.ndarray) -> list[Any]:
    mask = ensure_2d_array(binary_mask, "binary_mask").astype(np.uint8, copy=False)
    result = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = result[-2]
    return list(contours)

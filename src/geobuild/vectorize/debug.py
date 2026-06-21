from pathlib import Path

import cv2
import numpy as np

from geobuild.vectorize.base import PredictedPolygon
from geobuild.vectorize.contours import ensure_2d_array


def save_debug_overlay(
    mask_prob: np.ndarray,
    binary_mask: np.ndarray,
    polygons: list[PredictedPolygon],
    output_path: str | Path,
) -> None:
    probability = ensure_2d_array(mask_prob, "mask_prob")
    binary = ensure_2d_array(binary_mask, "binary_mask")
    gray = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
    overlay = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)

    overlay[binary.astype(bool)] = (
        0.65 * overlay[binary.astype(bool)]
        + 0.35 * np.array([0, 255, 255], dtype=np.float32)
    ).astype(np.uint8)

    for polygon in polygons:
        exterior = np.asarray(polygon.polygon.exterior.coords, dtype=np.int32)

        if len(exterior) >= 2:
            cv2.polylines(
                overlay,
                [exterior.reshape(-1, 1, 2)],
                isClosed=True,
                color=(0, 0, 255),
                thickness=1,
            )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)

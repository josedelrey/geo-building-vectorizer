from typing import Any

import cv2
import numpy as np

from geobuild.vectorize.base import PredictionBundle, PredictedPolygon, Vectorizer
from geobuild.vectorize.contours import (
    binary_mask_from_probability,
    component_mask,
    connected_components,
    ensure_2d_array,
    find_contours,
)
from geobuild.vectorize.geometry import contour_to_polygon, simplify_polygon
from geobuild.vectorize.io import validate_required_outputs
from geobuild.vectorize.registry import register_vectorizer


class MaskCCVectorizer(Vectorizer):
    name = "mask_cc"
    required_outputs = {"mask_prob"}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = {} if config is None else dict(config)
        self.mask_threshold = float(config.get("mask_threshold", 0.5))
        self.min_area_px = int(config.get("min_area_px", 16))
        self.simplify_tolerance = float(config.get("simplify_tolerance", 1.0))

        if self.min_area_px < 0:
            raise ValueError(f"min_area_px must be non-negative, got {self.min_area_px}")

    def vectorize(self, prediction: PredictionBundle) -> list[PredictedPolygon]:
        validate_required_outputs(self, prediction)
        mask_prob = ensure_2d_array(prediction.arrays["mask_prob"], "mask_prob")

        if mask_prob.shape != (prediction.height, prediction.width):
            raise ValueError(
                "mask_prob shape must match prediction height/width: "
                f"shape={mask_prob.shape}, expected={(prediction.height, prediction.width)}"
            )

        binary_mask = binary_mask_from_probability(mask_prob, self.mask_threshold)
        labels, stats = connected_components(binary_mask)
        polygons: list[PredictedPolygon] = []

        for component_id in range(1, int(stats.shape[0])):
            component_area_px = int(stats[component_id, cv2.CC_STAT_AREA])

            if component_area_px < self.min_area_px:
                continue

            current_component_mask = component_mask(labels, component_id)
            component_score = float(mask_prob[current_component_mask].mean())
            contours = find_contours(current_component_mask)

            for contour_index, contour in enumerate(contours):
                polygon = contour_to_polygon(contour)

                if polygon is None:
                    continue

                polygon = simplify_polygon(polygon, self.simplify_tolerance)

                if polygon is None:
                    continue

                polygons.append(
                    PredictedPolygon(
                        image_id=prediction.image_id,
                        polygon=polygon,
                        score=component_score,
                        source=self.name,
                        source_id=component_id,
                        properties={
                            "component_id": component_id,
                            "component_area_px": component_area_px,
                            "contour_index": contour_index,
                            "polygon_area_px": float(polygon.area),
                            "mask_threshold": self.mask_threshold,
                        },
                    )
                )

        return polygons


def build_mask_cc_vectorizer(config: dict[str, Any]) -> MaskCCVectorizer:
    return MaskCCVectorizer(config)


register_vectorizer(MaskCCVectorizer.name, build_mask_cc_vectorizer)

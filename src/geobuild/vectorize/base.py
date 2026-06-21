from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import Polygon


@dataclass
class PredictionBundle:
    image_id: str
    height: int
    width: int
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictedPolygon:
    image_id: str
    polygon: Polygon
    score: float
    source: str
    source_id: int | str | None
    properties: dict[str, Any] = field(default_factory=dict)


class Vectorizer:
    name: str = ""
    required_outputs: set[str] = set()

    def vectorize(self, prediction: PredictionBundle) -> list[PredictedPolygon]:
        raise NotImplementedError

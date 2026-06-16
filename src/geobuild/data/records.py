from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PolygonInstance:
    exterior: list[list[float]]
    holes: list[list[list[float]]] = field(default_factory=list)
    category_id: int | None = None
    iscrowd: int = 0
    area: float | None = None
    bbox: list[float] | None = None
    annotation_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImageRecord:
    image_id: int | str
    image_path: str
    width: int
    height: int
    split: str
    polygons: list[PolygonInstance]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "split": self.split,
            "polygons": [polygon.to_dict() for polygon in self.polygons],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ImageRecord":
        return ImageRecord(
            image_id=data["image_id"],
            image_path=data["image_path"],
            width=int(data["width"]),
            height=int(data["height"]),
            split=data["split"],
            polygons=[
                PolygonInstance(
                    exterior=polygon["exterior"],
                    holes=polygon.get("holes", []),
                    category_id=polygon.get("category_id"),
                    iscrowd=polygon.get("iscrowd", 0),
                    area=polygon.get("area"),
                    bbox=polygon.get("bbox"),
                    annotation_id=polygon.get("annotation_id"),
                )
                for polygon in data["polygons"]
            ],
        )

    @property
    def path(self) -> Path:
        return Path(self.image_path)
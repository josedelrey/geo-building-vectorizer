import csv
from pathlib import Path
from typing import Any


class CSVLogger:
    columns = [
        "epoch",
        "train_loss",
        "train_mask_loss",
        "val_loss",
        "val_mask_loss",
        "val_iou",
        "val_dice",
        "val_precision",
        "val_recall",
        "lr",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict[str, Any]) -> None:
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        row = {column: metrics.get(column, "") for column in self.columns}

        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geobuild.data.records import ImageRecord


def load_record(manifest: Path, index: int) -> ImageRecord:
    with manifest.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return ImageRecord.from_dict(json.loads(line))

    raise IndexError(f"Index {index} not found in {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    manifest = ROOT / args.manifest
    record = load_record(manifest, args.index)

    image = Image.open(record.image_path).convert("RGB")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, aspect="equal")

    for polygon in record.polygons:
        xs = [point[0] for point in polygon.exterior]
        ys = [point[1] for point in polygon.exterior]

        if len(xs) == 0:
            continue

        xs.append(xs[0])
        ys.append(ys[0])

        ax.plot(xs, ys, linewidth=1)

    ax.set_title(
        f"{record.split} | image_id={record.image_id} | polygons={len(record.polygons)}"
    )
    ax.set_xlim(0, record.width)
    ax.set_ylim(record.height, 0)
    ax.axis("off")

    if args.out:
        output_path = ROOT / args.out
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

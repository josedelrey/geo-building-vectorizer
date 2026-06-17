import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geobuild.data.coco import build_image_records, validate_records
from geobuild.utils.config import (
    annotation_path_from_config,
    image_dir_from_config,
    load_config,
    manifest_path_from_config,
)


def write_jsonl(records, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    args = parser.parse_args()

    config = load_config(args.config, root=ROOT)
    splits = config["splits"]

    for split_name in splits:
        annotation_file = annotation_path_from_config(config, split_name, root=ROOT)
        image_dir = image_dir_from_config(config, split_name, root=ROOT)

        records = build_image_records(
            annotation_file=annotation_file,
            image_dir=image_dir,
            split=split_name,
        )

        stats = validate_records(records)
        output_path = manifest_path_from_config(config, split_name, root=ROOT)

        write_jsonl(records, output_path)

        print(f"\n[{split_name}]")
        print(f"annotation_file: {annotation_file}")
        print(f"image_dir:        {image_dir}")
        print(f"manifest:         {output_path}")

        for key, value in stats.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

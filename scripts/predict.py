import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from geobuild.predict.bundle import load_prediction_bundle
from geobuild.predict.export import export_predictions
from geobuild.utils.config import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_prediction_bundle(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        device_arg=args.device,
        root=ROOT,
    )
    result = export_predictions(
        bundle=bundle,
        out_dir=resolve_path(args.out, root=ROOT),
    )

    print(f"Saved {result.count} predictions to: {result.manifest_path}")


if __name__ == "__main__":
    main()

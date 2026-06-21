import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from geobuild.eval.tuning import tune_vectorizer
from geobuild.utils.config import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--vectorizer-config", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--selection-metric", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_config = load_config(args.config, root=ROOT)
    vectorizer_config = load_config(args.vectorizer_config, root=ROOT)
    result = tune_vectorizer(
        project_config=project_config,
        split=args.split,
        predictions_path=resolve_path(args.predictions, root=ROOT),
        vectorizer_config=vectorizer_config,
        output_dir=resolve_path(args.out, root=ROOT),
        selection_metric=args.selection_metric,
        root=ROOT,
    )

    print(f"Best trial: {result.best_trial.trial_index}")
    print(f"Best config: {resolve_path(args.out, root=ROOT) / 'best_config.yaml'}")


if __name__ == "__main__":
    main()

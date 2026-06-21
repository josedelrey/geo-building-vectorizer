import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from geobuild.eval.compare import load_summary_metrics, write_comparison_tables
from geobuild.utils.config import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "summaries",
        nargs="*",
        help="One or more summary_metrics.json paths.",
    )
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Additional summary_metrics.json path. Can be repeated.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/tables",
        help="Directory for comparison CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_paths = [*args.summaries, *args.summary]

    if not summary_paths:
        raise SystemExit("Provide at least one summary_metrics.json path")

    summaries = load_summary_metrics(
        [resolve_path(path, root=ROOT) for path in summary_paths]
    )
    written = write_comparison_tables(
        summaries,
        output_dir=resolve_path(args.out, root=ROOT),
    )

    for table_type, output_path in written.items():
        print(f"{table_type}: {output_path}")


if __name__ == "__main__":
    main()

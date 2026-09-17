"""Command-line interface for dataset-deduplicator."""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline import DatasetDeduplicator
from .resolve import keep_first, keep_longest_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataset-dedup",
        description=(
            "Remove exact and near-duplicate rows from a CSV or JSONL dataset. "
            "Writes the deduplicated file and a JSON audit report of every "
            "removed row and why it was removed."
        ),
    )
    parser.add_argument("input", help="Input .csv or .jsonl file")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output path for the deduplicated file (same format as input)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path for the JSON audit report (default: <output>.report.json)",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Column to scan for near-duplicate text (MinHash/LSH). "
        "If omitted, only exact + tabular detection run.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.8,
        help="Jaccard similarity >= threshold counts as a near-duplicate "
        "text (default: 0.8)",
    )
    parser.add_argument(
        "--tabular-max-distance",
        type=float,
        default=0.05,
        help="Normalized feature distance <= this counts as a near-duplicate "
        "row (default: 0.05). Use --no-tabular to disable.",
    )
    parser.add_argument(
        "--no-tabular",
        action="store_true",
        help="Disable tabular near-duplicate detection",
    )
    parser.add_argument(
        "--numeric-columns",
        default=None,
        help="Comma-separated numeric columns (default: auto-inferred)",
    )
    parser.add_argument(
        "--keep-longest-text",
        default=None,
        metavar="COLUMN",
        help="Keep the row with the longest COLUMN value per cluster "
        "instead of the first row",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10_000,
        help="Rows streamed per chunk (default: 10000)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Only print the final summary line"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.keep_longest_text:
        strategy = keep_longest_text(args.keep_longest_text)
    else:
        strategy = keep_first

    deduper = DatasetDeduplicator(
        text_column=args.text_column,
        text_threshold=args.text_threshold,
        tabular_max_distance=None if args.no_tabular else args.tabular_max_distance,
        numeric_columns=(
            [c.strip() for c in args.numeric_columns.split(",")]
            if args.numeric_columns
            else None
        ),
        keep_strategy=strategy,
    )

    report_path = args.report or f"{args.output}.report.json"
    report = deduper.deduplicate_file(
        args.input, args.output, report_path, chunksize=args.chunksize
    )

    summary = (
        f"rows: {report['total_rows']} -> {report['kept_rows']} kept, "
        f"{report['removed_rows']} removed "
        f"({report['duplicate_clusters']} clusters)"
    )
    if not args.quiet:
        print(summary)
        print(f"cleaned file: {args.output}")
        print(f"audit report: {report_path}")
        for reason, count in report["removed_by_reason"].items():
            print(f"  {count:4d} x {reason}")
    else:
        print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())

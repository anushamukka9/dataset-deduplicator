"""Runnable quickstart: deduplicate the sample reviews dataset.

Usage:
    python examples/quickstart.py
"""

from pathlib import Path

from dataset_deduplicator import DatasetDeduplicator
from dataset_deduplicator.io import read_all

HERE = Path(__file__).parent
SRC = HERE / "sample_data.csv"
CLEAN = HERE / "sample_data.clean.csv"
REPORT = HERE / "sample_data.report.json"


def main() -> None:
    rows = read_all(SRC)
    print(f"loaded {len(rows)} rows from {SRC.name}")

    deduper = DatasetDeduplicator(
        text_column="review", text_threshold=0.6, bands=32
    )
    kept, report = deduper.deduplicate(rows)

    print(f"kept {len(kept)} rows, removed {report['removed_rows']}")
    for entry in report["removed"]:
        print(f"  row {entry['row']}: {entry['reason']}")

    # same thing, but streaming through the chunked file pipeline
    file_report = deduper.deduplicate_file(SRC, CLEAN, REPORT, chunksize=3)
    print(f"\nfile pipeline: {file_report['kept_rows']} kept, "
          f"{file_report['removed_rows']} removed")
    print(f"cleaned file -> {CLEAN}")
    print(f"audit report -> {REPORT}")


if __name__ == "__main__":
    main()

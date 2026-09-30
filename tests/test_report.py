"""Markdown report rendering and dry-run mode."""

import csv
import json

from dataset_deduplicator import (
    DatasetDeduplicator,
    render_report_markdown,
)


def _sample_report():
    return {
        "total_rows": 5,
        "kept_rows": 3,
        "removed_rows": 2,
        "duplicate_clusters": 2,
        "removed_by_reason": {"exact duplicate": 1, "near-duplicate": 1},
        "removed": [
            {
                "row": 1,
                "kept_row": 0,
                "reason": "exact duplicate of row 0",
                "cluster": [0, 1],
            },
            {
                "row": 4,
                "kept_row": 3,
                "reason": "near-duplicate of row 3 (text similarity 0.912)",
                "cluster": [3, 4],
            },
        ],
        "params": {"text_column": "review", "num_perm": 128},
    }


def test_render_report_markdown_structure():
    md = render_report_markdown(_sample_report())
    assert md.startswith("# Dedup audit report")
    assert "- total rows: 5" in md
    assert "- removed rows: 2" in md
    assert "## Parameters" in md
    assert "| `text_column` | `review` |" in md
    assert "## Removals by reason" in md
    assert "| exact duplicate | 1 |" in md
    assert "## Removed rows" in md
    assert "| 1 | 0 | exact duplicate of row 0 | [0, 1] |" in md
    assert "| 4 | 3 | near-duplicate of row 3 (text similarity 0.912) | [3, 4] |" in md


def test_render_report_markdown_no_removals():
    report = _sample_report()
    report.update(
        removed=[], removed_by_reason={},
        total_rows=3, kept_rows=3, removed_rows=0, duplicate_clusters=0,
    )
    md = render_report_markdown(report)
    assert "## Removed rows" in md
    assert "- removed rows: 0" in md


def _write_csv(path):
    rows = [
        {"id": "0", "review": "The battery life on this laptop is amazing", "stars": "5"},
        {"id": "1", "review": "The battery life on this laptop is amazing", "stars": "5"},
        {"id": "2", "review": "Screen resolution is crisp and bright outdoors", "stars": "4"},
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "review", "stars"])
        writer.writeheader()
        writer.writerows(rows)


def test_deduplicate_file_dry_run_writes_no_cleaned_file(tmp_path):
    src = tmp_path / "in.csv"
    out = tmp_path / "clean.csv"
    rep = tmp_path / "audit.json"
    _write_csv(src)

    deduper = DatasetDeduplicator(text_column="review", text_threshold=0.6)
    report = deduper.deduplicate_file(
        str(src), str(out), str(rep), chunksize=2, dry_run=True
    )

    assert report["total_rows"] == 3
    assert report["removed_rows"] == 1
    assert report["params"]["dry_run"] is True
    assert rep.exists()  # the report is still written
    assert not out.exists()  # but the cleaned file is not
    json.loads(rep.read_text(encoding="utf-8"))  # still valid JSON


def test_report_params_include_tuning_knobs():
    rows = [
        {"id": "0", "review": "hello world foo bar"},
        {"id": "1", "review": "hello world foo bar"},
    ]
    deduper = DatasetDeduplicator(
        text_column="review", num_perm=64, bands=8, shingle_k=3, seed=7
    )
    _, report = deduper.deduplicate(rows)
    params = report["params"]
    assert params["num_perm"] == 64
    assert params["bands"] == 8
    assert params["shingle_k"] == 3
    assert params["seed"] == 7

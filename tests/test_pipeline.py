"""End-to-end pipeline tests: in-memory and chunked file dedup agree."""

import csv
import json

import pytest

from dataset_deduplicator import DatasetDeduplicator
from dataset_deduplicator.io import read_all


@pytest.fixture()
def toy_rows():
    return [
        {"id": "0", "review": "The battery life on this laptop is amazing", "stars": "5"},
        {"id": "1", "review": "Terrible keyboard, keys stick constantly", "stars": "1"},
        {"id": "0", "review": "The battery life on this laptop is amazing", "stars": "5"},  # exact dupe of row 0
        {"id": "3", "review": "The battery life of this laptop is amazing!", "stars": "5"},  # near-dupe of 0
        {"id": "4", "review": "Screen resolution is crisp and bright outdoors", "stars": "4"},
    ]


def test_in_memory_pipeline_removes_exact_and_near_dupes(toy_rows):
    deduper = DatasetDeduplicator(text_column="review", text_threshold=0.6, bands=32)
    kept, report = deduper.deduplicate(toy_rows)
    kept_ids = [r["id"] for r in kept]
    assert kept_ids == ["0", "1", "4"]
    assert report["total_rows"] == 5
    assert report["removed_rows"] == 2
    reasons = {e["row"]: e["reason"] for e in report["removed"]}
    assert "exact duplicate" in reasons[2]
    assert "near-duplicate" in reasons[3]


def test_in_memory_pipeline_keeps_all_when_clean(toy_rows):
    rows = [toy_rows[0], toy_rows[1], toy_rows[4]]
    deduper = DatasetDeduplicator(text_column="review", text_threshold=0.6, bands=32)
    kept, report = deduper.deduplicate(rows)
    assert len(kept) == 3
    assert report["removed_rows"] == 0


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "review", "stars"])
        writer.writeheader()
        writer.writerows(rows)


def test_chunked_file_pipeline_matches_in_memory(tmp_path, toy_rows):
    src = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    rep = tmp_path / "report.json"
    _write_csv(src, toy_rows)

    deduper = DatasetDeduplicator(text_column="review", text_threshold=0.6, bands=32)
    report = deduper.deduplicate_file(src, out, rep, chunksize=2)

    kept_mem, report_mem = deduper.deduplicate(toy_rows)
    assert report["removed_rows"] == report_mem["removed_rows"] == 2
    assert report["kept_rows"] == report_mem["kept_rows"] == 3

    out_rows = read_all(out)
    assert [r["id"] for r in out_rows] == ["0", "1", "4"]

    saved = json.loads(rep.read_text(encoding="utf-8"))
    assert saved["total_rows"] == 5
    assert {e["row"] for e in saved["removed"]} == {2, 3}


def test_chunked_jsonl_pipeline(tmp_path, toy_rows):
    src = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    with open(src, "w", encoding="utf-8") as fh:
        for row in toy_rows:
            fh.write(json.dumps(row) + "\n")

    deduper = DatasetDeduplicator(text_column="review", text_threshold=0.6, bands=32)
    report = deduper.deduplicate_file(src, out, chunksize=2)
    assert report["removed_rows"] == 2
    assert len(read_all(out)) == 3


def test_tabular_only_pipeline(tmp_path):
    rows = [
        {"city": "Austin", "temp": "72.0"},
        {"city": "Austin", "temp": "72.1"},  # near-dupe
        {"city": "Miami", "temp": "88.0"},
    ]
    deduper = DatasetDeduplicator(tabular_max_distance=0.05)
    kept, report = deduper.deduplicate(rows)
    assert [r["city"] for r in kept] == ["Austin", "Miami"]
    assert report["removed_rows"] == 1

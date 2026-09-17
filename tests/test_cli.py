"""CLI end-to-end test."""

import csv
import json
import subprocess
import sys

import pytest


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "review", "stars"])
        writer.writeheader()
        writer.writerows(rows)


def test_cli_dedups_csv_and_writes_report(tmp_path):
    rows = [
        {"id": "0", "review": "The battery life on this laptop is amazing", "stars": "5"},
        {"id": "1", "review": "The battery life on this laptop is amazing", "stars": "5"},
        {"id": "2", "review": "Screen resolution is crisp and bright outdoors", "stars": "4"},
    ]
    src = tmp_path / "in.csv"
    out = tmp_path / "clean.csv"
    rep = tmp_path / "audit.json"
    _write_csv(src, rows)

    proc = subprocess.run(
        [sys.executable, "-m", "dataset_deduplicator",
         str(src), "-o", str(out), "--report", str(rep),
         "--text-column", "review", "--text-threshold", "0.6",
         "--chunksize", "2"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "2 kept, 1 removed" in proc.stdout

    with open(out, newline="", encoding="utf-8") as fh:
        out_rows = list(csv.DictReader(fh))
    assert [r["id"] for r in out_rows] == ["0", "2"]

    report = json.loads(rep.read_text(encoding="utf-8"))
    assert report["removed_rows"] == 1
    assert report["removed"][0]["row"] == 1

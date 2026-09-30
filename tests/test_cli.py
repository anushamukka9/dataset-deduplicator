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


def test_cli_tuning_flags(tmp_path):
    """--shingle-k/--num-perm/--bands are accepted and honored."""
    rows = [
        {"id": "0", "review": "The battery life on this laptop is amazing", "stars": "5"},
        {"id": "1", "review": "The battery life on this laptop is amazing", "stars": "5"},
        {"id": "2", "review": "Screen resolution is crisp and bright outdoors", "stars": "4"},
    ]
    src = tmp_path / "in.csv"
    out = tmp_path / "clean.csv"
    _write_csv(src, rows)

    proc = subprocess.run(
        [sys.executable, "-m", "dataset_deduplicator",
         str(src), "-o", str(out),
         "--text-column", "review", "--text-threshold", "0.6",
         "--shingle-k", "4", "--num-perm", "64", "--bands", "8"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "2 kept, 1 removed" in proc.stdout
    # tuning knobs land in the report's params
    report = json.loads((tmp_path / "clean.csv.report.json").read_text(encoding="utf-8"))
    assert report["params"]["shingle_k"] == 4
    assert report["params"]["num_perm"] == 64
    assert report["params"]["bands"] == 8


def test_cli_report_format_markdown(tmp_path):
    rows = [
        {"id": "0", "review": "The battery life on this laptop is amazing", "stars": "5"},
        {"id": "1", "review": "The battery life on this laptop is amazing", "stars": "5"},
    ]
    src = tmp_path / "in.csv"
    out = tmp_path / "clean.csv"
    _write_csv(src, rows)

    proc = subprocess.run(
        [sys.executable, "-m", "dataset_deduplicator",
         str(src), "-o", str(out), "--report-format", "markdown",
         "--text-column", "review", "--text-threshold", "0.6"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    md_path = tmp_path / "clean.csv.report.md"
    assert md_path.exists()
    text = md_path.read_text(encoding="utf-8")
    assert text.startswith("# Dedup audit report")
    assert "near-duplicate of row 0 (text similarity 1.000)" in text


def test_cli_dry_run(tmp_path):
    rows = [
        {"id": "0", "review": "The battery life on this laptop is amazing", "stars": "5"},
        {"id": "1", "review": "The battery life on this laptop is amazing", "stars": "5"},
    ]
    src = tmp_path / "in.csv"
    out = tmp_path / "clean.csv"
    _write_csv(src, rows)

    proc = subprocess.run(
        [sys.executable, "-m", "dataset_deduplicator",
         str(src), "-o", str(out), "--dry-run",
         "--text-column", "review", "--text-threshold", "0.6"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "1 removed" in proc.stdout
    assert "dry run: cleaned file not written" in proc.stdout
    assert not out.exists()
    # the audit report is still produced
    report = json.loads((tmp_path / "clean.csv.report.json").read_text(encoding="utf-8"))
    assert report["removed_rows"] == 1
    assert report["params"]["dry_run"] is True

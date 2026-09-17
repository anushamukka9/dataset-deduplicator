"""Chunked CSV/JSONL reading and writing.

Files are streamed in chunks so datasets larger than RAM can be processed:
the pipeline keeps only hashes, MinHash signatures and LSH buckets in
memory — never the full row set.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping


def detect_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    raise ValueError(f"unsupported file type: {suffix!r} (expected .csv or .jsonl)")


def iter_chunks(
    path: str | Path, chunksize: int = 10_000
) -> Iterator[List[Dict[str, Any]]]:
    """Yield the file's rows as dicts, ``chunksize`` at a time."""
    fmt = detect_format(path)
    chunk: List[Dict[str, Any]] = []
    if fmt == "csv":
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                chunk.append(dict(row))
                if len(chunk) >= chunksize:
                    yield chunk
                    chunk = []
    else:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                chunk.append(json.loads(line))
                if len(chunk) >= chunksize:
                    yield chunk
                    chunk = []
    if chunk:
        yield chunk


def read_all(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for chunk in iter_chunks(path, chunksize=50_000):
        rows.extend(chunk)
    return rows


def write_rows(
    path: str | Path, rows: List[Mapping[str, Any]], fieldnames: List[str] | None = None
) -> None:
    """Write kept rows back out in the input's format."""
    fmt = detect_format(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        if fieldnames is None:
            fieldnames = list(rows[0].keys()) if rows else []
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
    else:
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: str | Path, report: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

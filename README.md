# dataset-deduplicator

Near-duplicate detection for training data. Before you train, make sure your dataset isn't training on the same row twice: find **exact duplicates** via row hashing, **near-duplicate text** via MinHash/LSH shingling, and **near-duplicate tabular rows** via normalized feature distance — then resolve each duplicate cluster to a single surviving row with a full audit report of what was removed and why.

Built by [Anusha Mukka](https://anushamukka.com).

## Why this exists

Duplicate and near-duplicate rows in training data cause silent problems: inflated eval scores (the same example in train and test), wasted compute, and models that memorize instead of generalize. Exact dedup is easy; the hard part is catching the *near* duplicates — the same product review with one word changed, the same record re-exported with rounding noise. That's what this tool does.

## Install

```bash
pip install dataset-deduplicator
```

Requires Python 3.9+ and `numpy` (the only dependency).

Or from source:

```bash
git clone https://github.com/anushamukka9/dataset-deduplicator
cd dataset-deduplicator
pip install -e .
```

## Quickstart

```bash
dataset-dedup reviews.csv -o reviews.clean.csv \
  --text-column review --text-threshold 0.8
```

Output:

```
rows: 51200 -> 49871 kept, 1329 removed (1184 clusters)
cleaned file: reviews.clean.csv
audit report: reviews.clean.csv.report.json
  1042 x exact duplicate
   287 x near-duplicate
```

Every removed row lands in the JSON report with the row number, the surviving row, and a human-readable reason such as `near-duplicate of row 12 (text similarity 0.912)`.

## Python API

```python
from dataset_deduplicator import DatasetDeduplicator

rows = [
    {"id": 1, "review": "The battery life is amazing", "stars": 5},
    {"id": 2, "review": "The battery life is amazing!", "stars": 5},
]

deduper = DatasetDeduplicator(text_column="review", text_threshold=0.8)
kept, report = deduper.deduplicate(rows)
print(report["removed_rows"], "rows removed")
for entry in report["removed"]:
    print(entry["row"], "->", entry["reason"])
```

For files too large for memory, stream in chunks (only hashes, MinHash signatures, and LSH buckets are kept in RAM — never the full row set):

```python
report = deduper.deduplicate_file(
    "big.csv", "big.clean.csv",
    report_path="audit.json", chunksize=10_000,
)
```

See [`examples/quickstart.py`](examples/quickstart.py) for a runnable end-to-end script, and [`docs/usage.md`](docs/usage.md) for the full guide.

## How it works

1. **Exact dedup** — each row is canonicalized (whitespace collapsed, key order normalized, missing values unified) and SHA-256 hashed. Identical rows share a digest: O(1) memory per distinct row, no pairwise comparisons.
2. **Text near-dup** — the text column is normalized and split into character 5-shingles; a 128-permutation MinHash signature sketches each shingle set; LSH banding proposes candidate pairs without O(n²) comparisons; candidates are verified with the true Jaccard similarity.
3. **Tabular near-dup** — numeric columns are min-max normalized (statistics gathered in a streaming pass), categoricals hashed deterministically; row distance is the mean per-feature distance. Discretized feature vectors go through the same MinHash/LSH machinery, then verified with the true distance.
4. **Resolution** — duplicate pairs are merged transitively (union-find) into clusters; exactly one row per cluster survives (default: first occurrence, or longest-text via `--keep-longest-text`); the audit report records every removal and why.

## CLI reference

```
dataset-dedup INPUT -o OUTPUT [--report REPORT.json]
  --text-column COL        column to scan for near-duplicate text
  --text-threshold F       Jaccard >= F counts as near-duplicate (default 0.8)
  --tabular-max-distance F normalized distance <= F counts as near-dupe (default 0.05)
  --no-tabular             disable tabular near-duplicate detection
  --numeric-columns A,B    numeric columns (default: auto-inferred)
  --keep-longest-text COL  keep the longest-text row per cluster
  --chunksize N            rows streamed per chunk (default 10000)
```

## License

MIT — Copyright 2026 Anusha Mukka. See [LICENSE](LICENSE).

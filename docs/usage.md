# Usage guide

## When to use each detection stage

| Stage | Catches | Cost |
|---|---|---|
| Exact (`hashing`) | byte-identical rows after normalization | one hash per row |
| Text (`text_column=...`) | paraphrases, rewordings, truncation | MinHash sketch per row |
| Tabular (`tabular_max_distance`) | rounding noise, typos, re-exports | two file passes (stats, then detect) |

Run exact-only for a fast first pass; add the text stage for review / document / prompt datasets; add the tabular stage for numeric/categorical record data. Disable stages you don't need — `--no-tabular`, or omit `--text-column`.

## Tuning the text stage

`--text-threshold` is the Jaccard similarity (0–1) at or above which two texts count as near-duplicates. Higher = stricter (fewer removals). 0.8 is a good default for reviews and documents.

Candidate generation uses LSH banding: a signature of `num_perm` values is split into `bands` bands. Pairs sharing a band become candidates. The banding has its own implicit similarity sweet spot, `t ≈ (1/bands)^(1/rows)` where `rows = num_perm/bands`:

- defaults: `num_perm=128`, `bands=16` → `t ≈ 0.71`. Good with the default threshold of 0.8.
- for lower thresholds (e.g. 0.5–0.6), raise `bands` (e.g. 32) so candidates aren't missed. Both are constructor arguments (`DatasetDeduplicator(..., num_perm=128, bands=32)`).

Short texts (a few words) have tiny shingle sets, so Jaccard moves in big steps — prefer a lower threshold or exact matching for very short fields.

## Tuning the tabular stage

`--tabular-max-distance` is the mean per-feature distance (0–1) at or below which two rows count as near-duplicates. Numeric columns are min-max normalized over the whole file, so a distance of 0.05 means "differs by ~5% of each column's range on average". Numeric columns are auto-inferred; override with `--numeric-columns temp,humidity`.

## Keep strategies

Default keeps the **first** row of each cluster (stable, order-preserving). To keep the most informative copy instead:

```python
from dataset_deduplicator import DatasetDeduplicator, keep_longest_text

deduper = DatasetDeduplicator(
    text_column="review",
    keep_strategy=keep_longest_text("review"),
)
```

or `--keep-longest-text review` on the CLI. Custom strategies are just callables taking `(members, rows, reasons)` and returning the surviving index.

## The audit report

`deduplicate_file` writes a JSON report (default `<output>.report.json`):

```json
{
  "total_rows": 51200,
  "kept_rows": 49871,
  "removed_rows": 1329,
  "duplicate_clusters": 1184,
  "removed_by_reason": {"exact duplicate": 1042, "near-duplicate": 287},
  "removed": [
    {
      "row": 871,
      "kept_row": 12,
      "reason": "near-duplicate of row 12 (text similarity 0.912)",
      "cluster": [12, 871]
    }
  ],
  "params": {"text_column": "review", "text_threshold": 0.8, ...}
}
```

Row numbers refer to input file order. Keep this file next to the cleaned dataset — it's the provenance record for what the model never saw.

## Large files

`deduplicate_file(..., chunksize=10_000)` streams the input. Memory holds: one hash per distinct row, one MinHash signature + shingle set per exact-unique row, and the LSH buckets — never the full row set. The tabular stage needs dataset-wide min/max, so it makes one extra streaming pass first; everything else is single-pass.

## Programmatic use

```python
from dataset_deduplicator import DatasetDeduplicator, hashing, text, tabular

# just the building blocks
hashing.find_exact_duplicates(rows)          # {digest: [indices]}
text.find_near_duplicate_texts(texts, threshold=0.8)  # [(i, j, sim)]
tabular.find_near_duplicate_rows(rows, max_distance=0.05)  # [(i, j, dist)]
```

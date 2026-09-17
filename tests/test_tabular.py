"""Tests for tabular near-duplicate detection."""

from dataset_deduplicator import tabular


def test_normalizer_scales_to_unit_interval():
    rows = [{"age": 20}, {"age": 40}, {"age": 60}]
    norm = tabular.FeatureNormalizer(["age"])
    norm.observe(rows)
    vecs = [norm.vectorize(r, []) for r in rows]
    assert vecs[0][0] == 0.0
    assert vecs[2][0] == 1.0
    assert vecs[1][0] == 0.5


def test_row_distance_zero_for_identical_vectors():
    import numpy as np

    v = [0.1, 0.9, 0.5]
    assert tabular.row_distance(
        __import__("numpy").asarray(v), __import__("numpy").asarray(v)
    ) == 0.0


def test_find_near_duplicate_rows_clusters_noisy_copies():
    rows = [
        {"city": "Austin", "temp": 72.0, "humidity": 40},
        {"city": "Denver", "temp": 55.0, "humidity": 20},
        {"city": "Austin", "temp": 72.05, "humidity": 40},  # noisy copy of 0
        {"city": "Miami", "temp": 88.0, "humidity": 90},
    ]
    pairs = tabular.find_near_duplicate_rows(rows, max_distance=0.08)
    hit_pairs = {(a, b) for a, b, _ in pairs}
    assert (0, 2) in hit_pairs
    assert not any(3 in (a, b) for a, b in hit_pairs)


def test_infer_numeric_columns_skips_text():
    rows = [
        {"name": "amy", "score": 0.9, "rank": 1},
        {"name": "bo", "score": 0.2, "rank": 5},
    ]
    numeric = tabular.infer_numeric_columns(rows)
    assert set(numeric) == {"score", "rank"}

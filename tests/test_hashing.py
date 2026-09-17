"""Tests for exact duplicate detection via row hashing."""

from dataset_deduplicator import hashing


def test_canonicalize_ignores_key_order_and_whitespace():
    a = {"name": "  Alice ", "age": 30}
    b = {"age": 30, "name": "Alice"}
    assert hashing.hash_row(a) == hashing.hash_row(b)


def test_canonicalize_unifies_missing_values():
    a = {"name": "Bob", "note": None}
    b = {"name": "Bob", "note": "   "}
    assert hashing.hash_row(a) == hashing.hash_row(b)


def test_different_rows_hash_differently():
    a = {"text": "hello world"}
    b = {"text": "hello world!"}
    assert hashing.hash_row(a) != hashing.hash_row(b)


def test_find_exact_duplicates_groups_indices():
    rows = [
        {"id": 1, "v": "x"},
        {"id": 2, "v": "y"},
        {"id": 1, "v": "x"},  # dup of row 0
        {"id": 3, "v": "z"},
        {"id": 2, "v": "y"},  # dup of row 1
    ]
    dupes = hashing.find_exact_duplicates(rows)
    groups = sorted(sorted(idxs) for idxs in dupes.values())
    assert groups == [[0, 2], [1, 4]]


def test_find_exact_duplicates_empty_when_unique():
    rows = [{"id": i} for i in range(5)]
    assert hashing.find_exact_duplicates(rows) == {}

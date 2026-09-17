"""Exact duplicate detection via canonical row hashing.

Two rows that are identical after canonicalization (whitespace collapsed,
key order normalized, missing values unified) hash to the same SHA-256
digest. This is the fast first pass of the pipeline: O(1) memory per
distinct row, no pairwise comparisons.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Sequence

_MISSING = "<missing>"


def normalize_value(value: Any) -> str:
    """Normalize a single cell value for comparison.

    - ``None`` and empty/whitespace-only strings become a single sentinel.
    - Strings are stripped and internal whitespace is collapsed.
    - Numbers and booleans are rendered in a canonical form.
    """
    if value is None:
        return _MISSING
    if isinstance(value, str):
        collapsed = " ".join(value.split())
        return collapsed if collapsed else _MISSING
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return " ".join(str(value).split())


def canonicalize(row: Mapping[str, Any] | Sequence[Any]) -> str:
    """Render a row as a canonical string, independent of key order."""
    if isinstance(row, Mapping):
        items = sorted((str(k), normalize_value(v)) for k, v in row.items())
        return json.dumps(items, separators=(",", ":"), ensure_ascii=True)
    return json.dumps(
        [normalize_value(v) for v in row], separators=(",", ":"), ensure_ascii=True
    )


def hash_row(row: Mapping[str, Any] | Sequence[Any], algorithm: str = "sha256") -> str:
    """Return the hex digest of a row's canonical form."""
    digest = hashlib.new(algorithm)
    digest.update(canonicalize(row).encode("utf-8"))
    return digest.hexdigest()


def find_exact_duplicates(
    rows: Iterable[Mapping[str, Any] | Sequence[Any]],
) -> Dict[str, List[int]]:
    """Map each content hash to the sorted row indices sharing it.

    Only hashes with more than one index are returned.
    """
    buckets: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        buckets.setdefault(hash_row(row), []).append(idx)
    return {h: idxs for h, idxs in buckets.items() if len(idxs) > 1}

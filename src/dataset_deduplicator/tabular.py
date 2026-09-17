"""Near-duplicate detection for tabular rows via normalized feature distance.

Numeric columns are min-max normalized to [0, 1] (statistics gathered in a
streaming pass, so large files never need to be fully materialized);
categorical columns contribute 0.0 when equal and 1.0 when different.
The row distance is the weighted mean of per-column distances, so rows
that differ only by small numeric noise or a typo land close together.

For large inputs the normalized feature vectors are discretized into
``"column:bin"`` tokens and fed through MinHash/LSH, giving sub-quadratic
candidate generation; candidates are then verified with the true distance.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .text import LSHIndex, minhash_signature


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def infer_numeric_columns(
    rows: Sequence[Mapping[str, Any]], sample: int = 1000
) -> List[str]:
    """Columns whose non-missing values are all numeric (over a sample)."""
    if not rows:
        return []
    columns = list(rows[0].keys())
    numeric: List[str] = []
    for col in columns:
        ok = True
        for row in rows[:sample]:
            v = row.get(col)
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            if not _is_number(v):
                try:
                    float(v)
                except (TypeError, ValueError):
                    ok = False
                    break
        if ok:
            numeric.append(col)
    return numeric


class FeatureNormalizer:
    """Streaming min-max statistics, then per-row normalization."""

    def __init__(self, numeric_columns: Sequence[str]):
        self.numeric_columns = list(numeric_columns)
        self.minima: Dict[str, float] = {c: math.inf for c in numeric_columns}
        self.maxima: Dict[str, float] = {c: -math.inf for c in numeric_columns}
        self.fitted = False

    def observe(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for row in rows:
            for col in self.numeric_columns:
                v = row.get(col)
                if v is None or (isinstance(v, str) and not v.strip()):
                    continue
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if f < self.minima[col]:
                    self.minima[col] = f
                if f > self.maxima[col]:
                    self.maxima[col] = f
        self.fitted = True

    def _scale(self, col: str, value: Any) -> float:
        lo, hi = self.minima[col], self.maxima[col]
        if not self.fitted or lo == math.inf or hi == -math.inf or hi == lo:
            return 0.5 if value is None else 0.0
        if value is None or (isinstance(value, str) and not value.strip()):
            return 0.5
        try:
            f = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, (f - lo) / (hi - lo)))

    def vectorize(
        self, row: Mapping[str, Any], categorical_columns: Sequence[str]
    ) -> np.ndarray:
        """[0,1]-valued feature vector: numerics scaled, categoricals hashed."""
        parts: List[float] = []
        for col in self.numeric_columns:
            parts.append(self._scale(col, row.get(col)))
        for col in categorical_columns:
            v = row.get(col)
            token = "" if v is None else " ".join(str(v).lower().split())
            # Deterministic hash of the token into [0, 1): equal values
            # collide exactly, and results don't depend on PYTHONHASHSEED.
            digest = hashlib.sha1(f"{col}\x00{token}".encode("utf-8")).digest()
            h = int.from_bytes(digest[:8], "big")
            parts.append(h / 2**64)
        return np.asarray(parts, dtype=float)


def row_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Mean absolute per-feature distance in [0, 1]."""
    if vec_a.shape != vec_b.shape or vec_a.size == 0:
        raise ValueError("vectors must be non-empty and the same length")
    return float(np.mean(np.abs(vec_a - vec_b)))


def _discretize(
    vector: np.ndarray, columns: Sequence[str], bins: int = 20
) -> List[str]:
    return [f"{col}:{int(v * bins)}" for col, v in zip(columns, vector)]


def find_near_duplicate_rows(
    rows: Sequence[Mapping[str, Any]],
    numeric_columns: Sequence[str] | None = None,
    max_distance: float = 0.05,
    num_perm: int = 128,
    bands: int = 16,
    bins: int = 20,
    seed: int = 42,
) -> List[Tuple[int, int, float]]:
    """Return ``(i, j, distance)`` for near-duplicate tabular row pairs.

    LSH over discretized feature vectors proposes candidates; each is
    verified with the true normalized distance and kept when
    ``distance <= max_distance``.
    """
    if not rows:
        return []
    if numeric_columns is None:
        numeric_columns = infer_numeric_columns(rows)
    categorical = [c for c in rows[0].keys() if c not in numeric_columns]
    normalizer = FeatureNormalizer(numeric_columns)
    normalizer.observe(rows)
    feature_cols = list(numeric_columns) + categorical

    vectors = [normalizer.vectorize(r, categorical) for r in rows]
    index = LSHIndex(num_perm=num_perm, bands=bands, seed=seed)
    pairs: List[Tuple[int, int, float]] = []
    for i, vec in enumerate(vectors):
        tokens = _discretize(vec, feature_cols, bins=bins)
        sig = minhash_signature(tokens, num_perm=num_perm, seed=seed)
        for j in sorted(index.query_candidates(sig)):
            dist = row_distance(vec, vectors[j])
            if dist <= max_distance:
                pairs.append((j, i, dist))
        index.add(sig)
    return pairs

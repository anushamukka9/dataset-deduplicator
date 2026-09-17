"""Near-duplicate text detection with MinHash and LSH banding.

Pipeline for a text field:
1. Normalize the text and extract character k-shingles.
2. Compute a MinHash signature (a fixed-length sketch whose collision
   probability approximates Jaccard similarity of the shingle sets).
3. Band the signature (locality-sensitive hashing): rows that share a
   band are candidate near-duplicates, avoiding O(n^2) comparisons.
4. Verify candidates with the estimated Jaccard similarity.

The LSH index supports incremental inserts, so large files can be
processed chunk by chunk with O(unique rows) memory.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

_MERSENNE_61 = (1 << 61) - 1  # large prime for the universal hash family


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace so trivial edits don't hide dupes."""
    return " ".join(str(text).lower().split())


def shingles(text: str, k: int = 5) -> Set[str]:
    """Character k-shingles of the normalized text.

    Falls back to the whole normalized string when it is shorter than k.
    """
    norm = normalize_text(text)
    if len(norm) <= k:
        return {norm} if norm else set()
    return {norm[i : i + k] for i in range(len(norm) - k + 1)}


def _shingle_hash(token: str) -> int:
    digest = hashlib.sha1(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % _MERSENNE_61


def _permutation_params(num_perm: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _MERSENNE_61, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, _MERSENNE_61, size=num_perm, dtype=np.uint64)
    return a, b


def minhash_signature(
    tokens: Iterable[str], num_perm: int = 128, seed: int = 42
) -> np.ndarray:
    """MinHash signature for a set of tokens.

    Deterministic for a fixed ``seed``: identical token sets always yield
    identical signatures.
    """
    token_list = list(tokens)
    a, b = _permutation_params(num_perm, seed)
    sig = np.full(num_perm, _MERSENNE_61, dtype=np.uint64)
    for token in token_list:
        x = _shingle_hash(token)
        hashed = (a * np.uint64(x) + b) % np.uint64(_MERSENNE_61)
        sig = np.minimum(sig, hashed)
    return sig


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if sig_a.shape != sig_b.shape or sig_a.size == 0:
        raise ValueError("signatures must be non-empty and the same length")
    return float(np.mean(sig_a == sig_b))


def exact_jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """True Jaccard similarity (used to verify LSH candidates)."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


class LSHIndex:
    """Incremental locality-sensitive hashing index over MinHash signatures.

    Each signature is split into ``bands`` bands of ``rows`` rows; rows that
    share at least one band land in the same bucket and become candidates.
    Choose bands/rows so that the similarity threshold ``t`` satisfies
    ``t ~= (1/bands) ** (1/rows)``.
    """

    def __init__(self, num_perm: int = 128, bands: int = 16, seed: int = 42):
        if num_perm % bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.seed = seed
        self._buckets: Dict[Tuple[int, bytes], List[int]] = {}
        self._signatures: List[np.ndarray] = []

    def _band_keys(self, signature: np.ndarray) -> List[Tuple[int, bytes]]:
        keys = []
        for band in range(self.bands):
            chunk = signature[band * self.rows : (band + 1) * self.rows]
            keys.append((band, chunk.tobytes()))
        return keys

    def query_candidates(self, signature: np.ndarray) -> Set[int]:
        """Row ids already indexed that share at least one band."""
        candidates: Set[int] = set()
        for key in self._band_keys(signature):
            candidates.update(self._buckets.get(key, ()))
        return candidates

    def add(self, signature: np.ndarray) -> int:
        """Insert a signature; returns its row id."""
        row_id = len(self._signatures)
        self._signatures.append(signature)
        for key in self._band_keys(signature):
            self._buckets.setdefault(key, []).append(row_id)
        return row_id

    def signature(self, row_id: int) -> np.ndarray:
        return self._signatures[row_id]

    def __len__(self) -> int:
        return len(self._signatures)


def find_near_duplicate_texts(
    texts: Sequence[str],
    threshold: float = 0.8,
    num_perm: int = 128,
    bands: int = 16,
    shingle_k: int = 5,
    seed: int = 42,
) -> List[Tuple[int, int, float]]:
    """Return ``(i, j, similarity)`` for near-duplicate text pairs.

    LSH proposes candidates; each candidate is verified with the exact
    shingle-set Jaccard similarity and kept only if ``>= threshold``.
    """
    index = LSHIndex(num_perm=num_perm, bands=bands, seed=seed)
    shingle_sets = [shingles(t, k=shingle_k) for t in texts]
    pairs: List[Tuple[int, int, float]] = []
    for i, tokens in enumerate(shingle_sets):
        sig = minhash_signature(tokens, num_perm=num_perm, seed=seed)
        for j in sorted(index.query_candidates(sig)):
            sim = exact_jaccard(tokens, shingle_sets[j])
            if sim >= threshold:
                pairs.append((j, i, sim))
        index.add(sig)
    return pairs

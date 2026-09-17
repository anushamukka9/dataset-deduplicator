"""Keep-one-per-cluster resolution with an audit report.

Each duplicate cluster keeps exactly one row (the survivor); every other
row is removed and recorded with the reason it was dropped, so the
deduplication decision is fully auditable — important when the output
feeds model training.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

KeepStrategy = Callable[[List[int], Sequence[Mapping[str, Any]], Dict[int, str]], int]


def keep_first(
    members: List[int],
    rows: Sequence[Mapping[str, Any]],
    reasons: Dict[int, str],
) -> int:
    """Keep the earliest row in the cluster."""
    return min(members)


def keep_longest_text(
    text_column: str,
) -> KeepStrategy:
    """Build a strategy that keeps the row with the longest text field."""

    def _strategy(
        members: List[int],
        rows: Sequence[Mapping[str, Any]],
        reasons: Dict[int, str],
    ) -> int:
        return max(members, key=lambda i: len(str(rows[i].get(text_column, ""))))

    _strategy.__name__ = f"keep_longest_text({text_column!r})"
    return _strategy


class Resolution:
    """Result of resolving duplicate clusters."""

    def __init__(
        self,
        kept: List[int],
        removed: List[Dict[str, Any]],
        clusters: List[List[int]],
    ):
        self.kept = kept          # surviving row indices, in original order
        self.removed = removed    # one dict per removed row (see report)
        self.clusters = clusters  # duplicate clusters that were resolved


def resolve(
    rows: Sequence[Mapping[str, Any]],
    clusters: Sequence[Sequence[int]],
    pair_scores: Dict[Tuple[int, int], Tuple[str, float]] | None = None,
    strategy: KeepStrategy = keep_first,
) -> Resolution:
    """Keep one row per cluster; report every removal with its reason.

    ``pair_scores`` maps ``(a, b)`` (a < b) to ``(kind, score)`` where kind
    is ``"exact"``, ``"text"`` or ``"tabular"`` and score is the similarity
    (or distance) that linked the pair. It feeds the human-readable reason
    strings in the report.
    """
    pair_scores = pair_scores or {}
    reasons: Dict[int, str] = {}
    removed: List[Dict[str, Any]] = []
    resolved_clusters: List[List[int]] = []

    for cluster in clusters:
        members = sorted(cluster)
        if len(members) < 2:
            continue
        survivor = strategy(members, rows, reasons)
        resolved_clusters.append(members)
        for idx in members:
            if idx == survivor:
                continue
            key = (min(idx, survivor), max(idx, survivor))
            kind, score = pair_scores.get(key, ("duplicate", 0.0))
            if kind == "exact":
                reason = f"exact duplicate of row {survivor}"
            elif kind == "text":
                reason = (
                    f"near-duplicate of row {survivor} "
                    f"(text similarity {score:.3f})"
                )
            elif kind == "tabular":
                reason = (
                    f"near-duplicate of row {survivor} "
                    f"(feature distance {score:.4f})"
                )
            else:
                reason = f"duplicate of row {survivor}"
            reasons[idx] = reason
            removed.append(
                {
                    "row": idx,
                    "kept_row": survivor,
                    "reason": reason,
                    "cluster": members,
                }
            )

    removed_idx = set(reasons)
    kept = [i for i in range(len(rows)) if i not in removed_idx]
    return Resolution(kept=kept, removed=removed, clusters=resolved_clusters)


def build_report(
    total_rows: int,
    resolution: Resolution,
    params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """JSON-serializable audit report of what was removed and why."""
    by_reason: Dict[str, int] = {}
    for entry in resolution.removed:
        kind = entry["reason"].split(" of row")[0]
        by_reason[kind] = by_reason.get(kind, 0) + 1
    return {
        "total_rows": total_rows,
        "kept_rows": len(resolution.kept),
        "removed_rows": len(resolution.removed),
        "duplicate_clusters": len(resolution.clusters),
        "removed_by_reason": by_reason,
        "removed": resolution.removed,
        "params": params or {},
    }

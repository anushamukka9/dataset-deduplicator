"""Cluster duplicate pairs into groups with union-find."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


class UnionFind:
    def __init__(self) -> None:
        self._parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        parent = self._parent.setdefault(x, x)
        if parent != x:
            self._parent[x] = self.find(parent)
        return self._parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)


def cluster_pairs(
    pairs: Iterable[Tuple[int, int, float]],
) -> List[List[int]]:
    """Group row indices linked (transitively) by duplicate pairs.

    Singleton rows are omitted; each cluster is a sorted list of indices.
    """
    uf = UnionFind()
    for a, b, _score in pairs:
        uf.union(a, b)
    groups: Dict[int, List[int]] = {}
    for idx in list(uf._parent.keys()):
        groups.setdefault(uf.find(idx), []).append(idx)
    return [sorted(members) for members in groups.values() if len(members) > 1]

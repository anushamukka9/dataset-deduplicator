"""End-to-end deduplication pipeline.

Stages, in order:
1. **Exact dedup** — SHA-256 over canonicalized rows (streaming).
2. **Text near-dup** — MinHash + LSH over a text column (incremental index).
3. **Tabular near-dup** — normalized feature distance + LSH (two passes:
   one to gather normalization statistics, one to detect).
4. **Resolution** — union-find clustering, keep-one-per-cluster, audit report.

``DatasetDeduplicator.deduplicate`` runs the full pipeline on an in-memory
list of row dicts. ``deduplicate_file`` streams CSV/JSONL through the same
stages chunk by chunk and writes the cleaned file plus a JSON report.
Memory stays at O(exact-unique rows): exact duplicates are dropped while
streaming, and only hashes, MinHash signatures and LSH buckets are kept
for rows already seen.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from . import hashing, tabular, text
from .cluster import cluster_pairs
from .io import detect_format, iter_chunks, write_report, write_rows
from .resolve import KeepStrategy, build_report, keep_first, resolve


class DatasetDeduplicator:
    def __init__(
        self,
        text_column: str | None = None,
        text_threshold: float = 0.8,
        tabular_max_distance: float = 0.05,
        numeric_columns: Sequence[str] | None = None,
        keep_strategy: KeepStrategy = keep_first,
        num_perm: int = 128,
        bands: int = 16,
        shingle_k: int = 5,
        seed: int = 42,
    ):
        self.text_column = text_column
        self.text_threshold = text_threshold
        self.tabular_max_distance = tabular_max_distance
        self.numeric_columns = list(numeric_columns) if numeric_columns else None
        self.keep_strategy = keep_strategy
        self.num_perm = num_perm
        self.bands = bands
        self.shingle_k = shingle_k
        self.seed = seed

    # ------------------------------------------------------------------ #
    # in-memory pipeline
    # ------------------------------------------------------------------ #
    def deduplicate(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> Tuple[List[Mapping[str, Any]], Dict[str, Any]]:
        """Deduplicate ``rows``; return ``(kept_rows, report)``."""
        rows = list(rows)
        pair_scores: Dict[Tuple[int, int], Tuple[str, float]] = {}
        all_pairs: List[Tuple[int, int, float]] = []

        # Stage 1: exact duplicates via row hashing.
        for digest, idxs in hashing.find_exact_duplicates(rows).items():
            first = idxs[0]
            for other in idxs[1:]:
                key = (first, other)
                all_pairs.append((first, other, 1.0))
                pair_scores[key] = ("exact", 1.0)

        # Stage 2: near-duplicate text.
        if self.text_column is not None:
            texts = [str(r.get(self.text_column, "")) for r in rows]
            for a, b, sim in text.find_near_duplicate_texts(
                texts,
                threshold=self.text_threshold,
                num_perm=self.num_perm,
                bands=self.bands,
                shingle_k=self.shingle_k,
                seed=self.seed,
            ):
                key = (a, b)
                all_pairs.append((a, b, sim))
                if key not in pair_scores:
                    pair_scores[key] = ("text", sim)

        # Stage 3: near-duplicate tabular rows.
        if self.tabular_max_distance is not None:
            for a, b, dist in tabular.find_near_duplicate_rows(
                rows,
                numeric_columns=self.numeric_columns,
                max_distance=self.tabular_max_distance,
                num_perm=self.num_perm,
                bands=self.bands,
                seed=self.seed,
            ):
                key = (a, b)
                all_pairs.append((a, b, dist))
                if key not in pair_scores:
                    pair_scores[key] = ("tabular", dist)

        # Stage 4: cluster + resolve.
        clusters = cluster_pairs(all_pairs)
        resolution = resolve(rows, clusters, pair_scores, self.keep_strategy)
        kept_rows = [rows[i] for i in resolution.kept]
        report = build_report(
            len(rows),
            resolution,
            params={
                "text_column": self.text_column,
                "text_threshold": self.text_threshold,
                "tabular_max_distance": self.tabular_max_distance,
                "numeric_columns": self.numeric_columns,
            },
        )
        return kept_rows, report

    # ------------------------------------------------------------------ #
    # chunked file pipeline
    # ------------------------------------------------------------------ #
    def deduplicate_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        report_path: str | Path | None = None,
        chunksize: int = 10_000,
    ) -> Dict[str, Any]:
        """Stream ``input_path`` through the pipeline; write cleaned rows.

        Exact duplicates are dropped while streaming (their removal is
        recorded). Near-duplicate pairs are detected against incremental
        LSH indexes and resolved with the same cluster + keep-one logic as
        the in-memory pipeline. Row numbers in the report refer to the
        input file's row order.
        """
        input_path, output_path = Path(input_path), Path(output_path)
        detect_format(input_path)

        # Pass 1 (only when tabular detection is on): normalization stats.
        normalizer: tabular.FeatureNormalizer | None = None
        cat_columns: List[str] = []
        feature_columns: List[str] = []
        fieldnames: List[str] = []
        if self.tabular_max_distance is not None:
            for chunk in iter_chunks(input_path, chunksize):
                if not fieldnames and chunk:
                    fieldnames = list(chunk[0].keys())
                    numeric = self.numeric_columns or tabular.infer_numeric_columns(
                        chunk
                    )
                    cat_columns = [c for c in fieldnames if c not in numeric]
                    normalizer = tabular.FeatureNormalizer(numeric)
                    feature_columns = list(numeric) + cat_columns
                assert normalizer is not None
                normalizer.observe(chunk)

        text_index = (
            text.LSHIndex(num_perm=self.num_perm, bands=self.bands, seed=self.seed)
            if self.text_column is not None
            else None
        )
        text_shingles: List[frozenset] = []
        tabular_index = (
            text.LSHIndex(num_perm=self.num_perm, bands=self.bands, seed=self.seed)
            if normalizer is not None
            else None
        )
        tabular_vectors: List[Any] = []

        # Pass 2: stream rows; drop exact dupes, index the rest, record pairs.
        # uid = dense id of an exact-unique row; gid = input file row number.
        seen_hashes: Dict[str, int] = {}  # digest -> uid of first row
        hash_members: Dict[str, List[int]] = {}  # digest -> gids sharing it
        uid_to_gid: List[int] = []
        rows_out: List[Dict[str, Any]] = []
        exact_removed: List[Dict[str, Any]] = []
        all_pairs: List[Tuple[int, int, float]] = []
        pair_scores: Dict[Tuple[int, int], Tuple[str, float]] = {}
        gid = 0

        for chunk in iter_chunks(input_path, chunksize):
            if not fieldnames and chunk:
                fieldnames = list(chunk[0].keys())
            for row in chunk:
                digest = hashing.hash_row(row)
                if digest in seen_hashes:
                    first_uid = seen_hashes[digest]
                    hash_members[digest].append(gid)
                    exact_removed.append(
                        {
                            "row": gid,
                            "kept_uid": first_uid,
                            "digest": digest,
                        }
                    )
                    gid += 1
                    continue
                seen_hashes[digest] = len(rows_out)
                hash_members[digest] = [gid]
                uid = len(rows_out)
                uid_to_gid.append(gid)

                if text_index is not None:
                    tokens = text.shingles(
                        str(row.get(self.text_column, "")), k=self.shingle_k
                    )
                    sig = text.minhash_signature(
                        tokens, num_perm=self.num_perm, seed=self.seed
                    )
                    for cand in sorted(text_index.query_candidates(sig)):
                        sim = text.exact_jaccard(tokens, text_shingles[cand])
                        if sim >= self.text_threshold:
                            all_pairs.append((cand, uid, sim))
                            pair_scores.setdefault((cand, uid), ("text", sim))
                    text_shingles.append(frozenset(tokens))
                    text_index.add(sig)

                if tabular_index is not None and normalizer is not None:
                    vec = normalizer.vectorize(row, cat_columns)
                    toks = tabular._discretize(vec, feature_columns, bins=20)
                    sig = text.minhash_signature(
                        toks, num_perm=self.num_perm, seed=self.seed
                    )
                    for cand in sorted(tabular_index.query_candidates(sig)):
                        dist = tabular.row_distance(vec, tabular_vectors[cand])
                        if dist <= self.tabular_max_distance:
                            all_pairs.append((cand, uid, dist))
                            pair_scores.setdefault((cand, uid), ("tabular", dist))
                    tabular_vectors.append(vec)
                    tabular_index.add(sig)

                rows_out.append(dict(row))
                gid += 1

        total_rows = gid

        # Resolve near-duplicate clusters over the exact-unique rows.
        clusters = cluster_pairs(all_pairs)
        resolution = resolve(rows_out, clusters, pair_scores, self.keep_strategy)

        # Map surviving/removal bookkeeping from uid space to input row numbers.
        removed_uid_to_kept_uid = {e["row"]: e["kept_row"] for e in resolution.removed}

        def ultimate_survivor(uid: int) -> int:
            seen = set()
            while uid in removed_uid_to_kept_uid and uid not in seen:
                seen.add(uid)
                uid = removed_uid_to_kept_uid[uid]
            return uid

        def gid_reason(reason: str, kept_uid: int) -> str:
            return re.sub(
                rf"\brow {kept_uid}\b",
                f"row {uid_to_gid[kept_uid]}",
                reason,
            )

        removed: List[Dict[str, Any]] = []
        for entry in resolution.removed:
            kept_gid = uid_to_gid[entry["kept_row"]]
            removed.append(
                {
                    "row": uid_to_gid[entry["row"]],
                    "kept_row": kept_gid,
                    "reason": gid_reason(entry["reason"], entry["kept_row"]),
                    "cluster": [uid_to_gid[u] for u in entry["cluster"]],
                }
            )
        for entry in exact_removed:
            survivor_uid = ultimate_survivor(entry["kept_uid"])
            survivor_gid = uid_to_gid[survivor_uid]
            removed.append(
                {
                    "row": entry["row"],
                    "kept_row": survivor_gid,
                    "reason": f"exact duplicate of row {survivor_gid}",
                    "cluster": sorted(hash_members[entry["digest"]]),
                }
            )
        removed.sort(key=lambda e: e["row"])

        resolution.removed = removed
        final_rows = [rows_out[i] for i in resolution.kept]
        write_rows(output_path, final_rows, fieldnames or None)
        report = build_report(
            total_rows,
            resolution,
            params={
                "text_column": self.text_column,
                "text_threshold": self.text_threshold,
                "tabular_max_distance": self.tabular_max_distance,
                "chunksize": chunksize,
            },
        )
        if report_path is not None:
            write_report(report_path, report)
        return report

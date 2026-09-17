"""Tests for clustering and keep-one-per-cluster resolution."""

from dataset_deduplicator import cluster, resolve


def test_cluster_pairs_merges_transitively():
    pairs = [(0, 1, 0.9), (1, 2, 0.85), (5, 6, 0.95)]
    clusters = cluster.cluster_pairs(pairs)
    assert sorted(map(sorted, clusters)) == [[0, 1, 2], [5, 6]]


def test_resolve_keeps_first_and_reports_reasons():
    rows = [{"t": "a"}, {"t": "a"}, {"t": "b"}]
    clusters = [[0, 1]]
    pair_scores = {(0, 1): ("exact", 1.0)}
    res = resolve.resolve(rows, clusters, pair_scores)
    assert res.kept == [0, 2]
    assert len(res.removed) == 1
    entry = res.removed[0]
    assert entry["row"] == 1
    assert entry["kept_row"] == 0
    assert "exact duplicate of row 0" in entry["reason"]


def test_resolve_keep_longest_text_strategy():
    rows = [
        {"body": "short"},
        {"body": "a much longer version of the same text"},
    ]
    clusters = [[0, 1]]
    pair_scores = {(0, 1): ("text", 0.8)}
    res = resolve.resolve(
        rows, clusters, pair_scores,
        strategy=resolve.keep_longest_text("body"),
    )
    assert res.kept == [1]
    assert res.removed[0]["row"] == 0


def test_build_report_counts():
    rows = [{"t": "a"}, {"t": "a"}, {"t": "b"}]
    res = resolve.resolve(rows, [[0, 1]], {(0, 1): ("exact", 1.0)})
    report = resolve.build_report(3, res, params={"k": "v"})
    assert report["total_rows"] == 3
    assert report["kept_rows"] == 2
    assert report["removed_rows"] == 1
    assert report["duplicate_clusters"] == 1
    assert report["removed_by_reason"] == {"exact duplicate": 1}

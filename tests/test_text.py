"""Tests for MinHash/LSH near-duplicate text detection."""

from dataset_deduplicator import text


def test_minhash_identical_texts_have_jaccard_one():
    tokens = text.shingles("The quick brown fox jumps over the lazy dog")
    sig_a = text.minhash_signature(tokens, seed=7)
    sig_b = text.minhash_signature(list(tokens), seed=7)
    assert text.estimate_jaccard(sig_a, sig_b) == 1.0


def test_minhash_is_deterministic_across_runs():
    tokens = text.shingles("deterministic sketching for duplicate detection")
    first = text.minhash_signature(tokens, seed=123)
    second = text.minhash_signature(tokens, seed=123)
    assert (first == second).all()


def test_find_near_duplicate_texts_finds_paraphrase():
    texts = [
        "How to train a neural network with PyTorch",
        "Completely unrelated: baking sourdough bread at home",
        "How to train a neural network with PyTorch!",  # near-dupe of 0
        "Quantum entanglement explained simply",
    ]
    # bands=32/rows=4 targets similarities well below the 0.7 threshold,
    # so candidate generation is (deterministically) reliable here
    pairs = text.find_near_duplicate_texts(texts, threshold=0.7, bands=32)
    hit_pairs = {(a, b) for a, b, _ in pairs}
    assert (0, 2) in hit_pairs
    assert not any(1 in (a, b) or 3 in (a, b) for a, b in hit_pairs)


def test_lsh_index_is_incremental():
    index = text.LSHIndex(num_perm=64, bands=8, seed=1)
    sig = text.minhash_signature(text.shingles("some example sentence here"), num_perm=64, seed=1)
    assert index.query_candidates(sig) == set()
    row_id = index.add(sig)
    assert index.query_candidates(sig) == {row_id}

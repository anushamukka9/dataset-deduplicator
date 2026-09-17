"""dataset-deduplicator: near-duplicate detection for training data.

Find exact and near-duplicate rows in datasets before they pollute model
training: exact dedup via row hashing, near-duplicate text detection via
MinHash/LSH shingling, near-duplicate tabular rows via normalized feature
distance, cluster + keep-one-per-cluster resolution with an audit report,
and memory-efficient chunked processing for large files.
"""

from .pipeline import DatasetDeduplicator
from .resolve import keep_first, keep_longest_text

__all__ = ["DatasetDeduplicator", "keep_first", "keep_longest_text"]
__version__ = "0.1.0"

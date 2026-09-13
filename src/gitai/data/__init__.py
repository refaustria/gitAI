"""Corpus acquisition, curation, and conversion to training-ready shards."""

from .acquire import REGISTRY, Manifest, Source, fetch, sha256_file, verify
from .curate import (
    CurationReport,
    QualityFilters,
    assign_split,
    check_leakage,
    corpus_stats,
    curate,
    minhash_signatures,
    near_duplicate_groups,
    read_documents,
    split_documents,
    write_parquet,
)
from .loader import BatchSampler
from .shards import ShardIndex, ShardInfo, SplitInfo, choose_dtype, tokenize_to_shards

__all__ = [
    "REGISTRY",
    "BatchSampler",
    "CurationReport",
    "Manifest",
    "QualityFilters",
    "ShardIndex",
    "ShardInfo",
    "Source",
    "SplitInfo",
    "assign_split",
    "check_leakage",
    "choose_dtype",
    "corpus_stats",
    "curate",
    "fetch",
    "minhash_signatures",
    "near_duplicate_groups",
    "read_documents",
    "sha256_file",
    "split_documents",
    "tokenize_to_shards",
    "verify",
    "write_parquet",
]

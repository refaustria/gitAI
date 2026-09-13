"""Shard writing and batch sampling."""

from __future__ import annotations

import numpy as np
import pytest

from gitai.data import BatchSampler, ShardIndex, choose_dtype, tokenize_to_shards
from gitai.tokenizer import ByteBPETokenizer, CharTokenizer

DOCS = [
    f"document number {i} with some repeated filler text about foxes and dogs. " * 3
    for i in range(40)
]


@pytest.fixture
def tokenizer() -> CharTokenizer:
    return CharTokenizer.train(DOCS)


@pytest.fixture
def corpus(tmp_path, tokenizer):
    tokenize_to_shards(DOCS[:32], tokenizer, tmp_path, split="train", shard_tokens=1500)
    tokenize_to_shards(DOCS[32:], tokenizer, tmp_path, split="val", shard_tokens=10**9)
    return tmp_path


# ----------------------------------------------------------------- dtype


def test_dtype_choice_follows_vocab_size():
    assert choose_dtype(256) == "uint16"
    assert choose_dtype(65536) == "uint16"
    assert choose_dtype(65537) == "uint32"


def test_uint16_is_used_for_a_realistic_vocab(corpus):
    """4k-8k vocab fits in 16 bits, halving the bytes and therefore the I/O."""
    assert ShardIndex.load(corpus).dtype == "uint16"


# ---------------------------------------------------------------- writing


def test_shards_round_trip_through_disk(tmp_path, tokenizer):
    tokenize_to_shards(DOCS[:4], tokenizer, tmp_path, split="train", shard_tokens=10**9)
    index = ShardIndex.load(tmp_path)
    raw = np.fromfile(tmp_path / index.split("train").shards[0].path, dtype=index.np_dtype)
    assert tokenizer.decode(raw.tolist()) == "".join(DOCS[:4])


def test_sharding_splits_at_the_token_budget(corpus):
    info = ShardIndex.load(corpus).split("train")
    assert len(info.shards) > 1
    assert all(s.tokens <= 1500 + 400 for s in info.shards[:-1])
    assert info.tokens == sum(s.tokens for s in info.shards)


def test_meta_records_utf8_bytes_for_bits_per_byte(corpus):
    """BPB's denominator. Recorded at tokenization time because that is the only
    moment the original byte count is still known."""
    info = ShardIndex.load(corpus).split("val")
    assert info.utf8_bytes == sum(len(d.encode("utf-8")) for d in DOCS[32:])
    assert info.utf8_bytes / info.tokens > 0


def test_document_separator_is_appended(tmp_path):
    tok = ByteBPETokenizer.train(DOCS, vocab_size=400, special_tokens=["<|endoftext|>"])
    eot = tok.special_tokens["<|endoftext|>"]
    tokenize_to_shards(DOCS[:3], tok, tmp_path, split="train", document_separator=eot)
    index = ShardIndex.load(tmp_path)
    raw = np.fromfile(tmp_path / index.split("train").shards[0].path, dtype=index.np_dtype)
    assert int((raw == eot).sum()) == 3


def test_appending_a_second_split_preserves_the_first(corpus, tokenizer):
    before = ShardIndex.load(corpus).split("train").tokens
    tokenize_to_shards(["extra text"], tokenizer, corpus, split="test")
    index = ShardIndex.load(corpus)
    assert index.split("train").tokens == before
    assert index.split("test").tokens > 0


def test_mixing_tokenizers_in_one_corpus_is_refused(corpus):
    """Silently meaningless data, otherwise: ids from two vocabularies in one
    array with nothing to distinguish them."""
    other = CharTokenizer.train(["completely different characters ЖЩЮ"])
    with pytest.raises(ValueError, match=r"Mixing tokenizers|vocab_size"):
        tokenize_to_shards(["x"], other, corpus, split="train")


# ------------------------------------------------------------ quarantine


def test_synthetic_shards_require_a_provenance_tag(tmp_path, tokenizer):
    """Enforces what safety.GeneratedDataQuarantined checks. Once generated and
    real data are mixed untagged, the collapse question is unanswerable."""
    with pytest.raises(ValueError, match="quarantine_tag"):
        tokenize_to_shards(DOCS[:2], tokenizer, tmp_path, split="train", synthetic=True)


def test_tagged_synthetic_shards_are_recorded(tmp_path, tokenizer):
    tokenize_to_shards(
        DOCS[:2], tokenizer, tmp_path, split="train", synthetic=True, quarantine_tag="iter7"
    )
    shard = ShardIndex.load(tmp_path).split("train").shards[0]
    assert shard.synthetic and shard.quarantine_tag == "iter7"


def test_shard_metadata_satisfies_the_safety_invariant(tmp_path, tokenizer):
    """End-to-end: shard metadata feeds straight into the promotion gate."""
    from gitai.safety import GeneratedDataQuarantined

    tokenize_to_shards(DOCS[:2], tokenizer, tmp_path, split="train")
    tokenize_to_shards(
        DOCS[2:4], tokenizer, tmp_path, split="train", synthetic=True, quarantine_tag="iter7"
    )
    shards = [
        {"path": s.path, "synthetic": s.synthetic, "quarantine_tag": s.quarantine_tag}
        for s in ShardIndex.load(tmp_path).split("train").shards
    ]
    assert GeneratedDataQuarantined().check({"training_shards": shards}).passed


# ------------------------------------------------------------- sampling


def test_batch_shapes_and_dtype(corpus):
    sampler = BatchSampler(corpus, "train", seq_len=16)
    x, y = sampler.batch(8, np.random.default_rng(0))
    assert x.shape == y.shape == (8, 16)
    assert x.dtype == np.int64


def test_targets_are_inputs_shifted_by_one(corpus):
    """The entire causal-LM objective, and an easy off-by-one to get wrong."""
    sampler = BatchSampler(corpus, "train", seq_len=16)
    x, y = sampler.batch(4, np.random.default_rng(0))
    assert np.array_equal(x[:, 1:], y[:, :-1])


def test_ids_are_within_the_vocabulary(corpus):
    sampler = BatchSampler(corpus, "train", seq_len=16)
    x, y = sampler.batch(32, np.random.default_rng(0))
    assert x.min() >= 0 and x.max() < sampler.vocab_size
    assert y.min() >= 0 and y.max() < sampler.vocab_size


def test_sampling_is_deterministic_given_a_seed(corpus):
    """Without this, two runs see different data and no ablation means anything."""
    sampler = BatchSampler(corpus, "train", seq_len=16)
    first = list(sampler.batches(4, steps=5, seed=123))
    second = list(sampler.batches(4, steps=5, seed=123))
    for (xa, ya), (xb, yb) in zip(first, second, strict=True):
        assert np.array_equal(xa, xb) and np.array_equal(ya, yb)


def test_different_seeds_give_different_batches(corpus):
    sampler = BatchSampler(corpus, "train", seq_len=16)
    a = next(iter(sampler.batches(4, steps=1, seed=1)))[0]
    b = next(iter(sampler.batches(4, steps=1, seed=2)))[0]
    assert not np.array_equal(a, b)


def test_sequential_covers_every_token_once(corpus):
    """Evaluation must not score some tokens twice and skip others."""
    sampler = BatchSampler(corpus, "val", seq_len=8)
    seen = sum(x.size for x, _ in sampler.sequential(batch_size=4))
    assert seen <= sampler.total_tokens
    assert seen >= sampler.total_tokens - 8 * 4 - 1


def test_sequential_is_reproducible(corpus):
    sampler = BatchSampler(corpus, "val", seq_len=8)
    a = [x.tolist() for x, _ in sampler.sequential(4)]
    b = [x.tolist() for x, _ in sampler.sequential(4)]
    assert a == b


def test_seq_len_longer_than_every_shard_is_rejected(corpus):
    with pytest.raises(ValueError, match="Reduce seq_len"):
        BatchSampler(corpus, "train", seq_len=10**6)


def test_unknown_split_is_rejected(corpus):
    with pytest.raises(KeyError, match="no split"):
        BatchSampler(corpus, "nonexistent", seq_len=8)


def test_throughput_reports_a_number(corpus):
    stats = BatchSampler(corpus, "train", seq_len=16).throughput(batch_size=4, steps=5)
    assert stats["tokens_per_sec"] > 0

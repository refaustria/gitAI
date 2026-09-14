"""The self-training arms, corpus generation, and the shared training loop."""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch

from gitai.data import BatchSampler, tokenize_to_shards
from gitai.model import RUNGS, ModelConfig, Transformer
from gitai.selftrain import ARMS, Accumulate, Control, Replace, diversity, generate_corpus
from gitai.tokenizer import ByteBPETokenizer
from gitai.training import TrainConfig, evaluate_bpb, lr_at, train

DOCS = [
    f"story {i}: the fox ran across the field and into the trees at dusk. " * 3 for i in range(60)
]


@pytest.fixture(scope="module")
def tokenizer() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(DOCS, vocab_size=320, special_tokens=["<|endoftext|>"])


@pytest.fixture
def model(tokenizer) -> Transformer:
    torch.manual_seed(0)
    m = Transformer(
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            seq_len=32,
            d_model=32,
            n_layer=2,
            n_head=4,
            **RUNGS["v6_modern"],
        )
    )
    m.eval()
    return m


# ---------------------------------------------------------------- diversity


def test_diversity_of_a_perfectly_repetitive_sequence_is_low():
    """The signature of collapse: the same token forever scores ~0 on every
    distinct-n measure."""
    scores = diversity([7] * 100, vocab_size=50)
    assert scores["distinct_1"] == pytest.approx(0.01, abs=1e-3)
    assert scores["distinct_2"] == pytest.approx(1 / 99, abs=1e-3)
    assert scores["vocabulary_used"] == 1


def test_diversity_of_a_fully_varied_sequence_is_high():
    scores = diversity(list(range(100)), vocab_size=100)
    assert scores["distinct_1"] == 1.0
    assert scores["distinct_3"] == 1.0
    assert scores["vocabulary_fraction"] == 1.0


def test_diversity_handles_sequences_shorter_than_n():
    assert diversity([1], vocab_size=10)["distinct_3"] == 0.0


# ------------------------------------------------------------ corpus generation


def test_generate_corpus_produces_the_requested_token_count(model, tokenizer):
    documents, stats = generate_corpus(
        model, tokenizer, target_tokens=400, batch_size=8, temperature=1.0, seed=0
    )
    assert stats.tokens == 400
    assert documents
    assert stats.tokens_per_sec > 0
    assert 0.0 < stats.distinct_1 <= 1.0


def test_generate_corpus_is_reproducible(model, tokenizer):
    a, _ = generate_corpus(model, tokenizer, 200, batch_size=8, seed=3)
    b, _ = generate_corpus(model, tokenizer, 200, batch_size=8, seed=3)
    assert a == b


def test_generate_corpus_differs_with_the_seed(model, tokenizer):
    a, _ = generate_corpus(model, tokenizer, 200, batch_size=8, seed=1)
    b, _ = generate_corpus(model, tokenizer, 200, batch_size=8, seed=2)
    assert a != b


def test_generate_corpus_requires_a_separator(model):
    """Without one the generated stream cannot be split into documents."""

    class NoSpecials:
        special_tokens: ClassVar[dict] = {}

        def decode(self, ids):
            return ""

    with pytest.raises(ValueError, match="no document separator"):
        generate_corpus(model, NoSpecials(), 100)


def test_generated_documents_contain_no_separator(model, tokenizer):
    documents, _ = generate_corpus(model, tokenizer, 300, batch_size=8, seed=0)
    assert all("<|endoftext|>" not in d for d in documents)


# --------------------------------------------------------------------- arms


def test_control_never_includes_synthetic_data():
    """The baseline's whole job. If synthetic data reaches it, the experiment has
    no control and any decline is uninterpretable."""
    control = Control()
    assert not control.needs_generation
    assert control.corpus(["real"], [["fake"], ["faker"]]) == ["real"]


def test_replace_uses_only_the_most_recent_generation():
    replace = Replace()
    assert replace.corpus(["real"], [["gen0"], ["gen1"]]) == ["gen1"]


def test_replace_falls_back_to_real_data_with_no_parent():
    """Generation 0 has nothing to learn from yet."""
    assert Replace().corpus(["real"], []) == ["real"]


def test_accumulate_keeps_everything():
    combined = Accumulate().corpus(["real"], [["gen0"], ["gen1"]])
    assert combined == ["real", "gen0", "gen1"]


def test_every_arm_is_registered():
    assert set(ARMS) == {"control", "replace", "accumulate"}
    for name, cls in ARMS.items():
        assert cls().name == name


def test_arms_do_not_share_synthetic_history():
    """Each lineage must be independent, or the comparison is not between
    lineages at all."""
    history_a: list[list[str]] = [["a"]]
    history_b: list[list[str]] = [["b"]]
    assert Replace().corpus([], history_a) != Replace().corpus([], history_b)


# ----------------------------------------------------------------- training


@pytest.fixture
def corpus(tmp_path, tokenizer):
    tokenize_to_shards(DOCS[:48], tokenizer, tmp_path, split="train")
    tokenize_to_shards(DOCS[48:], tokenizer, tmp_path, split="val")
    return tmp_path


def test_lr_schedule_warms_up_then_decays():
    peak, total, warmup = 1e-3, 100, 10
    assert lr_at(0, total, peak, warmup) < peak
    assert lr_at(warmup - 1, total, peak, warmup) == pytest.approx(peak)
    assert lr_at(total - 1, total, peak, warmup) < peak
    assert lr_at(total - 1, total, peak, warmup) > 0


def test_lr_schedule_handles_zero_warmup():
    assert lr_at(0, 100, 1e-3, 0) > 0


def test_train_reduces_loss(corpus, tokenizer):
    torch.manual_seed(0)
    model = Transformer(
        ModelConfig(vocab_size=tokenizer.vocab_size, seq_len=32, d_model=32, n_layer=2, n_head=4)
    )
    train_sampler = BatchSampler(corpus, "train", 32)
    val_sampler = BatchSampler(corpus, "val", 32)
    first = evaluate_bpb(model, val_sampler, 4, max_batches=4)["val_bpb"]
    result = train(
        model, train_sampler, val_sampler, TrainConfig(steps=60, eval_every=30, log_every=1000)
    )
    assert result.best_bpb < first
    assert result.final_train_loss > 0
    assert result.history


def test_train_restores_the_best_checkpoint(corpus, tokenizer):
    """The caller must receive the weights the reported metric describes, not
    whatever the final step happened to produce."""
    torch.manual_seed(0)
    model = Transformer(
        ModelConfig(vocab_size=tokenizer.vocab_size, seq_len=32, d_model=32, n_layer=2, n_head=4)
    )
    train_sampler = BatchSampler(corpus, "train", 32)
    val_sampler = BatchSampler(corpus, "val", 32)
    result = train(
        model, train_sampler, val_sampler, TrainConfig(steps=40, eval_every=20, log_every=1000)
    )
    after = evaluate_bpb(model, val_sampler, 4, max_batches=50)["val_bpb"]
    assert after == pytest.approx(result.best_bpb, abs=1e-6)


def test_training_is_deterministic(corpus, tokenizer):
    def run() -> float:
        torch.manual_seed(0)
        model = Transformer(
            ModelConfig(
                vocab_size=tokenizer.vocab_size, seq_len=32, d_model=32, n_layer=2, n_head=4
            )
        )
        return train(
            model,
            BatchSampler(corpus, "train", 32),
            BatchSampler(corpus, "val", 32),
            TrainConfig(steps=30, eval_every=15, log_every=1000, seed=7),
        ).best_bpb

    assert run() == run()


def test_evaluate_bpb_rejects_scoring_zero_tokens(corpus, tokenizer):
    """max_batches=0 means nothing was scored, so there is no BPB to report.
    Returning 0.0 or NaN here would propagate a meaningless number into the
    results table."""
    torch.manual_seed(0)
    model = Transformer(
        ModelConfig(vocab_size=tokenizer.vocab_size, seq_len=32, d_model=32, n_layer=2, n_head=4)
    )
    sampler = BatchSampler(corpus, "val", 32)
    with pytest.raises(ValueError, match="no evaluable tokens"):
        evaluate_bpb(model, sampler, 4, max_batches=0)

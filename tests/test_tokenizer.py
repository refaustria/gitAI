"""Tokenizer correctness.

The round-trip tests are the important ones. A tokenizer that mangles rare
characters corrupts training data silently — no exception, no warning, the model
just never learns those characters and nobody ever finds out why.
"""

from __future__ import annotations

import json

import pytest

from gitai.tokenizer import ByteBPETokenizer, CharTokenizer

CORPUS = [
    "the quick brown fox jumps over the lazy dog. " * 20,
    "the rain in spain falls mainly on the plain. " * 20,
    "she sells sea shells by the sea shore, she does. " * 20,
]


def _rich_corpus() -> list[str]:
    """A corpus with enough distinct byte sequences to support thousands of
    merges. CORPUS above is deliberately tiny and exhausts its merge budget
    after ~64 merges, which is correct behaviour but cannot test a vocab target."""
    import random

    rng = random.Random(7)
    syllables = ["ka", "lo", "mi", "ter", "sun", "bre", "nal", "oth", "ist", "ver", "pra", "dom"]
    words = ["".join(rng.choice(syllables) for _ in range(rng.randint(2, 4))) for _ in range(1500)]
    return [" ".join(rng.choice(words) for _ in range(400)) for _ in range(20)]


RICH = _rich_corpus()

TRICKY = [
    "hello world",
    "",
    " ",
    "\n\n\t  \n",
    "café naïve résumé",
    "日本語のテキストです",
    "emoji: 🤖🔥👩‍👩‍👧‍👦 and a flag 🇦🇹",
    "math: \u2200x\u2208\u211d, x\u00b2 \u2265 0",  # ambiguous-looking glyphs are the point
    "zero\x00width​joiner",
    "'twas the night; don't, won't, we'll, I'm, they've, he'd",
    "MiXeD CaSe 12345 !@#$%^&*()",
    "a" * 500,
    "\U0001f600" * 50,
]


@pytest.fixture(scope="module")
def bpe() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(CORPUS, vocab_size=400)


# --------------------------------------------------------------- char tokenizer


def test_char_round_trip():
    tok = CharTokenizer.train(CORPUS)
    for text in CORPUS:
        assert tok.decode(tok.encode(text)) == text


def test_char_unknown_maps_to_replacement():
    tok = CharTokenizer.train(["abc"])
    assert tok.decode(tok.encode("axc")) == "a�c"


def test_char_save_load(tmp_path):
    tok = CharTokenizer.train(CORPUS)
    tok.save(tmp_path / "char.json")
    reloaded = CharTokenizer.load(tmp_path / "char.json")
    assert reloaded.encode(CORPUS[0]) == tok.encode(CORPUS[0])
    assert reloaded.fingerprint() == tok.fingerprint()


# ------------------------------------------------------------------- BPE core


@pytest.mark.parametrize("text", TRICKY)
def test_bpe_round_trip_is_exact(bpe, text):
    """decode(encode(s)) == s for every string, including ones the tokenizer has
    never seen. Byte-level BPE has no out-of-vocabulary case, so this must hold
    for emoji, CJK and control characters alike."""
    assert bpe.decode(bpe.encode(text)) == text


def test_bpe_round_trip_on_random_unicode():
    """Fuzz across the whole BMP, excluding surrogates (which are not valid
    UTF-8 and cannot appear in a real corpus)."""
    import random

    rng = random.Random(0)
    tok = ByteBPETokenizer.train(CORPUS, vocab_size=300)
    for _ in range(200):
        codepoints = [
            rng.choice(
                [rng.randint(0x20, 0x7E), rng.randint(0xA0, 0xD7FF), rng.randint(0xE000, 0xFFFF)]
            )
            for _ in range(rng.randint(1, 40))
        ]
        text = "".join(chr(c) for c in codepoints)
        assert tok.decode(tok.encode(text)) == text


def test_untrained_tokenizer_is_still_lossless():
    """With zero merges it is just the identity on bytes — still total."""
    tok = ByteBPETokenizer()
    assert tok.vocab_size == 256
    for text in TRICKY:
        assert tok.decode(tok.encode(text)) == text
        assert tok.encode(text) == list(text.encode("utf-8"))


def test_bpe_vocab_size_is_respected():
    """Given a corpus rich enough to support them, the merge budget is used in full."""
    for size in (300, 512, 1024):
        assert ByteBPETokenizer.train(RICH, vocab_size=size).vocab_size == size


def test_vocab_size_accounts_for_special_tokens():
    tok = ByteBPETokenizer.train(RICH, vocab_size=512, special_tokens=["<|endoftext|>", "<|pad|>"])
    assert tok.vocab_size == 512
    assert len(tok.merges) == 512 - 256 - 2


def test_bpe_rejects_impossible_vocab_size():
    with pytest.raises(ValueError, match="leaves no room"):
        ByteBPETokenizer.train(CORPUS, vocab_size=100)


def test_bpe_stops_early_when_no_pair_is_frequent_enough():
    """A tiny corpus cannot support 1000 merges. It must stop, not invent them."""
    tok = ByteBPETokenizer.train(["ab"], vocab_size=1256, min_frequency=2)
    assert tok.vocab_size < 1256


def test_bpe_actually_compresses(bpe):
    text = CORPUS[0]
    assert len(bpe.encode(text)) < len(text.encode("utf-8"))
    assert bpe.compression_ratio(text) > 1.5


def test_larger_vocab_compresses_at_least_as_well():
    """Monotonic by construction: more merges can only shorten a sequence.
    This is the measurement behind the Decision 4 vocab trade-off."""
    text = " ".join(RICH[0].split()[:2000])
    ratios = [
        ByteBPETokenizer.train(RICH, vocab_size=v).compression_ratio(text)
        for v in (300, 512, 1024, 2048)
    ]
    assert ratios == sorted(ratios)
    assert ratios[-1] > ratios[0]  # and it is a strict gain, not a plateau


# ------------------------------------------------------------- determinism


def test_training_is_deterministic():
    """Same corpus, byte-identical merges. Tie-breaking is by an explicit rule,
    not by dict ordering — without that, two runs can silently disagree and
    every downstream comparison becomes invalid."""
    a = ByteBPETokenizer.train(CORPUS, vocab_size=400)
    b = ByteBPETokenizer.train(CORPUS, vocab_size=400)
    assert a.merges == b.merges
    assert a.fingerprint() == b.fingerprint()


def test_different_vocab_sizes_have_different_fingerprints():
    a = ByteBPETokenizer.train(CORPUS, vocab_size=300)
    b = ByteBPETokenizer.train(CORPUS, vocab_size=400)
    assert a.fingerprint() != b.fingerprint()


def test_encoding_is_stable_across_calls(bpe):
    """The chunk cache must not change results."""
    text = CORPUS[0]
    assert bpe.encode(text) == bpe.encode(text) == bpe.encode(text)


# --------------------------------------------------------- pre-tokenization


def test_merges_never_cross_a_word_boundary():
    """Pre-tokenization prevents 'dog.' and ' the' merging into one token.
    Every learned token must decode to text within a single regex chunk."""
    import regex

    from gitai.tokenizer import GPT2_SPLIT_PATTERN

    tok = ByteBPETokenizer.train(RICH + CORPUS, vocab_size=1024)
    compiled = regex.compile(GPT2_SPLIT_PATTERN)
    for idx in range(256, 256 + len(tok.merges)):
        piece = tok.vocab[idx].decode("utf-8", errors="replace")
        assert len(compiled.findall(piece)) <= 1, f"token {idx!r} spans chunks: {piece!r}"


def test_leading_space_is_part_of_the_token():
    """' the' and 'the' are different tokens. This is why GPT-style models are
    sensitive to trailing spaces in prompts."""
    tok = ByteBPETokenizer.train(CORPUS, vocab_size=400)
    assert tok.encode(" the") != tok.encode("the")


# ------------------------------------------------------------ special tokens


def test_special_tokens_get_ids_above_the_merges():
    tok = ByteBPETokenizer.train(CORPUS, vocab_size=400, special_tokens=["<|endoftext|>"])
    eot = tok.special_tokens["<|endoftext|>"]
    assert eot == tok.vocab_size - 1
    assert tok.encode("<|endoftext|>") == [eot]
    assert tok.decode([eot]) == "<|endoftext|>"


def test_special_tokens_split_surrounding_text():
    tok = ByteBPETokenizer.train(CORPUS, vocab_size=400, special_tokens=["<|endoftext|>"])
    ids = tok.encode("the dog<|endoftext|>the fox")
    assert tok.special_tokens["<|endoftext|>"] in ids
    assert tok.decode(ids) == "the dog<|endoftext|>the fox"


def test_special_tokens_can_be_disabled_for_untrusted_input():
    """Text containing the literal '<|endoftext|>' must not be able to inject a
    document boundary into the training stream."""
    tok = ByteBPETokenizer.train(CORPUS, vocab_size=400, special_tokens=["<|endoftext|>"])
    ids = tok.encode("<|endoftext|>", allow_special=False)
    assert tok.special_tokens["<|endoftext|>"] not in ids
    assert tok.decode(ids) == "<|endoftext|>"


# ---------------------------------------------------------- serialisation


def test_bpe_save_load_round_trip(tmp_path, bpe):
    path = tmp_path / "bpe.json"
    bpe.save(path)
    reloaded = ByteBPETokenizer.load(path)
    assert reloaded.merges == bpe.merges
    assert reloaded.fingerprint() == bpe.fingerprint()
    for text in TRICKY:
        assert reloaded.encode(text) == bpe.encode(text)


def test_saved_merges_preserve_order(tmp_path, bpe):
    """Merge order IS the tokenizer. Serialising to a JSON object would lose it,
    which is why merges are stored as a list."""
    path = tmp_path / "bpe.json"
    bpe.save(path)
    merges = json.loads(path.read_text())["merges"]
    assert [m[2] for m in merges] == sorted(m[2] for m in merges)


def test_load_rejects_the_wrong_kind(tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"kind": "char", "characters": ["a"]}))
    with pytest.raises(ValueError, match="not a byte-BPE"):
        ByteBPETokenizer.load(path)


def test_decode_rejects_an_out_of_range_id(bpe):
    with pytest.raises(ValueError, match="not in a vocabulary"):
        bpe.decode([999_999])


# ------------------------------------------------- validation against references


def _naive_train(texts, vocab_size, pattern=None, min_frequency=2):
    """An obviously-correct BPE trainer: recount every pair from scratch after
    every merge. O(merges x corpus) and far too slow for real use, but it has
    nowhere to hide a bug.

    The real trainer keeps incremental pair counts and an index from pair to
    containing words, which is where a subtle bookkeeping error would live. If
    the two disagree on a single merge, the fast one is wrong.
    """
    from itertools import pairwise

    import regex

    from gitai.tokenizer import GPT2_SPLIT_PATTERN
    from gitai.tokenizer.bpe import _merge_word

    compiled = regex.compile(pattern or GPT2_SPLIT_PATTERN)
    freqs: dict[bytes, int] = {}
    for text in texts:
        for chunk in compiled.findall(text):
            key = chunk.encode("utf-8")
            freqs[key] = freqs.get(key, 0) + 1

    words = [list(w) for w in freqs]
    weights = list(freqs.values())

    merges: dict[tuple[int, int], int] = {}
    for step in range(vocab_size - 256):
        counts: dict[tuple[int, int], int] = {}
        for word, weight in zip(words, weights, strict=True):
            for pair in pairwise(word):
                counts[pair] = counts.get(pair, 0) + weight
        if not counts:
            break
        best = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
        if counts[best] < min_frequency:
            break
        new_id = 256 + step
        merges[best] = new_id
        words = [_merge_word(w, best, new_id) for w in words]
    return merges


def test_incremental_trainer_matches_the_naive_one():
    """The headline correctness test for training. Byte-identical merges, in
    byte-identical order, between the fast implementation and the slow one."""
    corpus = RICH[:4] + CORPUS
    fast = ByteBPETokenizer.train(corpus, vocab_size=700).merges
    slow = _naive_train(corpus, vocab_size=700)
    assert list(fast.items()) == list(slow.items())


def _naive_encode_chunk(tok, data: bytes) -> list[int]:
    """Obviously-correct encoder: scan the whole merge table in learned order,
    applying each merge everywhere it fits, one merge at a time."""
    from gitai.tokenizer.bpe import _merge_word

    ids = list(data)
    for pair, new_id in tok.merges.items():
        ids = _merge_word(ids, pair, new_id)
    return ids


@pytest.mark.parametrize("text", [*TRICKY, *CORPUS])
def test_fast_encoder_matches_the_naive_encoder(text):
    """The optimised encoder picks the lowest-rank applicable merge each round;
    the naive one replays the whole merge table in order. They must agree on
    every input, or encoding does not reproduce what training decided."""
    import regex

    from gitai.tokenizer import GPT2_SPLIT_PATTERN

    tok = ByteBPETokenizer.train(RICH[:4] + CORPUS, vocab_size=600)
    compiled = regex.compile(GPT2_SPLIT_PATTERN)
    expected: list[int] = []
    for chunk in compiled.findall(text):
        expected.extend(_naive_encode_chunk(tok, chunk.encode("utf-8")))
    assert tok.encode(text) == expected


def test_compression_is_comparable_to_huggingface():
    """Sanity check against a production implementation.

    Not token-for-token: HF's trainer breaks frequency ties differently and
    applies its own alphabet initialisation, so identical merge tables are not
    expected. What must hold is that ours is not materially worse at the job —
    a bug in merge selection shows up here as visibly poorer compression.
    """
    hf = pytest.importorskip("tokenizers")

    corpus = RICH[:8]
    ours = ByteBPETokenizer.train(corpus, vocab_size=1024)

    ref = hf.Tokenizer(hf.models.BPE())
    ref.pre_tokenizer = hf.pre_tokenizers.ByteLevel(add_prefix_space=False)
    ref.decoder = hf.decoders.ByteLevel()
    ref.train_from_iterator(corpus, hf.trainers.BpeTrainer(vocab_size=1024, show_progress=False))

    sample = RICH[9]
    our_ratio = len(sample.encode("utf-8")) / len(ours.encode(sample))
    ref_ratio = len(sample.encode("utf-8")) / len(ref.encode(sample).ids)
    assert our_ratio > 0.85 * ref_ratio, f"ours {our_ratio:.3f} vs huggingface {ref_ratio:.3f}"

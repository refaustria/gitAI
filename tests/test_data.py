"""Curation and acquisition.

Curation is the highest-leverage work in the project — at this scale a better
corpus beats a better model — so these are treated as seriously as the model
tests."""

from __future__ import annotations

import json

import pytest

from gitai.data import (
    REGISTRY,
    Manifest,
    QualityFilters,
    assign_split,
    check_leakage,
    corpus_stats,
    curate,
    minhash_signatures,
    near_duplicate_groups,
    sha256_file,
    split_documents,
    write_parquet,
)
from gitai.data.acquire import Source, fetch
from gitai.data.curate import document_hash

GOOD = "The fox ran across the field and into the trees beyond the river bank."


def _docs(n: int) -> list[str]:
    return [f"Document {i}. {GOOD} It carried on for a while after that." for i in range(n)]


# ------------------------------------------------------------------- splitting


def test_split_assignment_is_deterministic():
    assert assign_split(GOOD) == assign_split(GOOD) == assign_split(GOOD)


def test_identical_documents_always_share_a_split():
    """The whole reason for hashing content instead of shuffling: a duplicate
    cannot straddle the train/val boundary and leak."""
    for doc in _docs(200):
        assert assign_split(doc) == assign_split(doc + "")


def test_split_proportions_are_roughly_as_configured():
    counts = {"train": 0, "val": 0, "test": 0}
    for doc in _docs(5000):
        counts[assign_split(doc)] += 1
    assert 0.95 < counts["train"] / 5000 < 1.0
    assert counts["val"] > 0 and counts["test"] > 0


def test_no_leakage_between_splits():
    splits = split_documents(_docs(2000) + _docs(500))  # deliberate duplicates
    assert all(count == 0 for count in check_leakage(splits).values())


def test_adding_data_does_not_reshuffle_existing_documents():
    """A random split re-run after adding data silently invalidates every earlier
    result. Hash-based splitting is stable under growth."""
    first = {d: assign_split(d) for d in _docs(100)}
    grown = split_documents(_docs(300))
    for doc, split in first.items():
        assert doc in grown[split]


# --------------------------------------------------------------------- filters


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("hi", "too_short"),
        ("aaaaaaaaaaaaaaaaaaaaaa", "too_few_words"),
        ("!@# $%^ &*( )_+ =-~ ;:/ " * 8, "mostly_non_alphabetic"),
        ("line\nline\nline\nline\nline\nline of text here", "repetitive_lines"),
        ("valid text here but with\x07a bell character inside it", "control_characters"),
    ],
)
def test_filters_reject_and_name_the_reason(text, reason):
    assert QualityFilters().check(text) == reason


def test_good_text_passes():
    assert QualityFilters().check(GOOD) is None


def test_tabs_and_newlines_are_not_control_characters():
    assert QualityFilters().check("a line of real text\n\tand an indented one here") is None


def test_max_chars_is_optional_but_enforced_when_set():
    assert QualityFilters(max_chars=10).check(GOOD) == "too_long"
    assert QualityFilters(max_chars=None).check(GOOD) is None


# ---------------------------------------------------------------- near-dupes


def test_minhash_signatures_have_the_expected_shape():
    assert minhash_signatures(_docs(5), num_perm=64).shape == (5, 64)


def test_minhash_detects_a_near_duplicate():
    """A document with one word changed must be caught; an unrelated one must not."""
    base = GOOD * 8
    near = (GOOD * 8).replace("fox", "cat", 1)
    far = "Mountains rise above the valley where snow gathers in winter months. " * 8
    groups = near_duplicate_groups(minhash_signatures([base, near, far]), threshold=0.7)
    assert len(groups) == 1
    assert groups[0] == {0, 1}


def test_minhash_leaves_distinct_documents_alone():
    texts = [
        f"An entirely separate topic number {i} about {w}." * 5
        for i, w in enumerate(["rivers", "engines", "pastry", "geology", "violins"])
    ]
    assert near_duplicate_groups(minhash_signatures(texts), threshold=0.8) == []


def test_identical_documents_form_one_group():
    groups = near_duplicate_groups(minhash_signatures([GOOD * 5] * 3), threshold=0.9)
    assert len(groups) == 1 and groups[0] == {0, 1, 2}


def test_near_duplicate_groups_handles_a_single_document():
    assert near_duplicate_groups(minhash_signatures([GOOD])) == []


def test_bands_must_divide_the_signature():
    with pytest.raises(ValueError, match="divide evenly"):
        near_duplicate_groups(minhash_signatures(_docs(3), num_perm=64), bands=7)


# -------------------------------------------------------------------- pipeline


def test_curate_reports_every_stage():
    docs = [*_docs(10), *_docs(3), "short", "x" * 5]
    kept, report = curate(docs)
    assert report.input_documents == 15
    assert report.output_documents == len(kept)
    assert [s.stage for s in report.stages] == [
        "exact_duplicates",
        "quality_filters",
        "near_duplicates",
    ]
    assert report.stages[0].removed == 3
    assert report.rejections["too_short"] == 2


def test_curate_is_deterministic():
    docs = [*_docs(30), *_docs(5)]
    a, ra = curate(docs)
    b, rb = curate(docs)
    assert a == b
    assert str(ra) == str(rb)


def test_curate_removes_near_duplicates_when_enabled():
    base = GOOD * 8
    docs = [base, base.replace("fox", "cat", 1), "A distinct document about engines. " * 8]
    kept, _ = curate(docs, near_duplicate_threshold=0.7)
    assert len(kept) == 2


def test_curate_can_skip_near_duplicate_detection():
    base = GOOD * 8
    docs = [base, base.replace("fox", "cat", 1)]
    kept, report = curate(docs, near_duplicate_threshold=None)
    assert len(kept) == 2
    assert "near_duplicates" not in [s.stage for s in report.stages]


def test_report_renders_without_error():
    _, report = curate(_docs(20))
    assert "input:" in str(report) and "output:" in str(report)


# --------------------------------------------------------------------- parquet


def test_parquet_round_trip(tmp_path):
    from gitai.data import read_documents

    docs = _docs(25)
    path = tmp_path / "corpus.parquet"
    assert write_parquet(docs, path, source="test") == 25
    assert sorted(read_documents(path)) == sorted(docs)


def test_parquet_split_filter(tmp_path):
    from gitai.data import read_documents

    docs = _docs(500)
    path = tmp_path / "corpus.parquet"
    write_parquet(docs, path)
    train = read_documents(path, split="train")
    assert 0 < len(train) <= len(docs)
    assert all(assign_split(d) == "train" for d in train)


def test_corpus_stats_via_duckdb(tmp_path):
    """DuckDB answering analytical questions over Parquet with no server — the
    whole justification for choosing it in Decision 7."""
    docs = [*_docs(40), *_docs(5)]
    path = tmp_path / "corpus.parquet"
    write_parquet(docs, path)
    stats = corpus_stats(path)
    assert stats["documents"] == 45
    assert stats["distinct_documents"] == 40
    assert stats["exact_duplicate_documents"] == 5
    assert stats["chars_max"] >= stats["chars_median"] >= stats["chars_min"]
    assert sum(stats["splits"].values()) == 45


# ----------------------------------------------------------------- acquisition


def test_registry_entries_are_complete():
    for name, source in REGISTRY.items():
        assert source.name == name
        assert source.url.startswith("https://")
        assert source.licence and source.description


def test_tinystories_hash_is_absent_rather_than_invented():
    """A wrong checksum in a registry is worse than an absent one: it fails
    verification on correct data and teaches you to skip the check."""
    assert REGISTRY["tinystories"].sha256 is None
    assert REGISTRY["tinyshakespeare"].sha256 is not None


def test_sha256_file(tmp_path):
    import hashlib

    path = tmp_path / "f.txt"
    path.write_bytes(b"hello")
    assert sha256_file(path) == hashlib.sha256(b"hello").hexdigest()


def test_manifest_round_trip(tmp_path):
    path = tmp_path / "MANIFEST.json"
    source = Source(name="x", url="https://e/x", licence="MIT", description="d")
    manifest = Manifest(path)
    manifest.record(source, tmp_path / "x.txt", "abc123", 42)

    reloaded = Manifest(path)
    assert reloaded.digest_for("x") == "abc123"
    entry = json.loads(path.read_text())["sources"]["x"]
    assert entry["licence"] == "MIT" and entry["actual_bytes"] == 42
    assert "retrieved_at" in entry


def test_fetch_refuses_a_checksum_mismatch(tmp_path, monkeypatch):
    """A changed upstream corpus must be an error, never a silent overwrite:
    every result computed from the old bytes needs re-examining."""
    monkeypatch.setitem(
        REGISTRY,
        "fake",
        Source(name="fake", url="https://example/x", sha256="0" * 64, licence="x", description="x"),
    )
    (tmp_path / "fake.txt").write_text("wrong content")
    with pytest.raises(RuntimeError, match="does not match the recorded checksum"):
        fetch("fake", dest_dir=tmp_path)


def test_fetch_rejects_an_unknown_source(tmp_path):
    with pytest.raises(KeyError, match="unknown source"):
        fetch("not-a-corpus", dest_dir=tmp_path)


def test_existing_file_with_a_matching_checksum_is_not_refetched(tmp_path, monkeypatch):
    import hashlib

    content = b"known bytes"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setitem(
        REGISTRY,
        "fake",
        Source(name="fake", url="https://example/x", sha256=digest, licence="x", description="x"),
    )
    (tmp_path / "fake.txt").write_bytes(content)

    def explode(*args, **kwargs):
        raise AssertionError("should not have downloaded")

    monkeypatch.setattr("urllib.request.urlopen", explode)
    assert fetch("fake", dest_dir=tmp_path).read_bytes() == content


def test_document_hash_is_stable():
    assert document_hash("abc") == document_hash("abc")
    assert document_hash("abc") != document_hash("abd")

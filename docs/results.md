# Results

Measurements, as they are made. Every entry states what was measured, on what
data, and what it implies. Negative and null results are recorded with the same
care as positive ones — see [evaluation.md](evaluation.md) for why.

---

## R1 — Vocabulary size vs compression (TinyShakespeare)

**Date:** 2026-09-13 · **Corpus:** TinyShakespeare, 7,018 documents / 1.1 MB after
curation · **Tokenizer:** own byte-level BPE · **Reproduce:**
`make data-sweep`

| Vocab | Merges | Bytes/token | Tokens in sample | Train time |
|------:|-------:|------------:|-----------------:|-----------:|
| 256 | 0 | 1.000 | 289,047 | 0.3 s |
| 512 | 256 | 1.963 | 147,255 | 0.9 s |
| 1,024 | 768 | 2.477 | 116,715 | 2.0 s |
| 2,048 | 1,792 | 2.949 | 98,002 | 5.2 s |
| 4,096 | 3,840 | 3.327 | 86,881 | 12.7 s |
| 8,192 | 7,936 | 3.607 | 80,145 | 26.9 s |

### What it says

Compression rises monotonically, as it must — more merges can only shorten a
sequence — but with sharply diminishing returns. The question for
[Decision 4](decisions.md#4-tokenizer) is not "does a bigger vocab compress
better" (always yes) but **what each doubling costs in parameters, and what it
buys in sequence length.**

At `d_model = 384`, each vocabulary entry costs 384 embedding parameters:

| Step | Compression gain | Embedding cost | Verdict |
|---|---|---|---|
| 1,024 → 2,048 | +19.0% | +393 K params | Clearly worth it |
| 2,048 → 4,096 | +12.8% | +786 K params | Worth it |
| 4,096 → 8,192 | **+8.4%** | **+1.57 M params** | **Bad trade** |

On a 10M-parameter model, that last step spends ~15% of the entire budget to
process 8% fewer tokens — parameters moved out of computation and into a lookup
table. This is the concrete form of the warning in Decision 4 against copying
GPT-2's 50,257: at `d_model=384` that vocabulary would be 19.3M embedding
parameters, several times the size of the model it was meant to serve.

**Provisional conclusion: 2,048–4,096 for TinyShakespeare-scale work.**

### Caveats — do not over-read this

- **TinyShakespeare is 1.1 MB.** A 1MB corpus cannot support a large vocabulary
  well; rare merges are learned from a handful of occurrences. The curve on
  TinyStories (~2 GB) will sit higher and bend later, and the decision should be
  re-made there. This table is a method demonstration, not the final answer.
- **Compression is a proxy, not the objective.** Fewer tokens means more text per
  context window and fewer steps per epoch, but it also means a larger softmax
  and rarer tokens seen less often each. The metric that actually decides is
  held-out **bits-per-byte at matched parameter count** — which is exactly why
  BPB, and not perplexity, is the primary metric ([evaluation.md](evaluation.md)).
- **Single measurement, no seeds.** BPE training is deterministic, so there is no
  seed variance *here* — but the downstream model comparison this feeds into will
  need ≥3 seeds before any of it means anything.

### Follow-up

Re-run on TinyStories, then train matched-parameter models at 2k/4k/8k vocab and
compare held-out BPB. That is the real version of this experiment and it belongs
in Phase 5.

---

## R2 — Pipeline throughput (this container, not a target machine)

**Date:** 2026-09-13 · **Caveat:** measured in a cloud container, **not** on the
project's laptop. Recorded only as a sanity check that the loader is not
pathologically slow. The number that matters comes from `make bench` on the real
hardware.

| Stage | Measurement |
|---|---|
| Curation (7,018 docs, incl. MinHash) | 2.3 s |
| BPE training, vocab 1,024 | 2.0 s |
| Tokenize + shard 442,719 tokens | < 1 s |
| **Loader throughput** (batch 8 × seq 256) | **18.6 M tokens/sec**, 0.11 ms/batch |

The loader figure is the useful one. A CPU transformer at this scale will manage
somewhere in the thousands of tokens/sec, so the loader has roughly three orders
of magnitude of headroom — it will not be the bottleneck at this corpus size.

Worth re-checking on the real corpus: 18.6M tokens/sec is measured against a
442K-token shard that fits entirely in page cache. A 2 GB TinyStories shard on a
laptop with less RAM will behave differently, and that is the case where the
[warning in data-and-storage.md](data-and-storage.md#3-training-ready-tokens--memory-mapped-binary-shards)
about the loader becoming the bottleneck actually applies.

---

## R3 — Curation on TinyShakespeare

| Stage | Removed | Remaining |
|---|---:|---:|
| Input | — | 7,148 |
| Exact duplicates | 0 | 7,148 |
| Quality filters | 130 (1.8%) | 7,018 |
| Near-duplicates (MinHash, ≥0.85) | 0 | 7,018 |

Rejections: `too_short` 108, `too_few_words` 22.

Split: train 6,869 / val 84 / test 65. **Leakage between every split pair: 0**,
as it must be — splitting on a content hash makes a duplicate straddling the
boundary structurally impossible.

Nothing surprising here, which is the point: a clean, curated corpus should show
a small, explicable loss. The number to be suspicious of is a large one. When
this pipeline runs on TinyStories or web text, expect exact- and near-duplicate
removal to do real work, and read the rejection table before trusting the output.

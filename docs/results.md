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

## R6 — Model collapse: does self-training degrade a small LM?  ⭐ the headline result

**Date:** 2026-09-14 · **Pre-registered:** [lab-notebook E3](lab-notebook.md#e3--does-a-self-training-loop-collapse-and-does-accumulation-prevent-it)
· **Reproduce:** `make collapse` · 30 training runs, 15 corpus generations, 34.5 min

Three arms differing **only** in what generation *n* trains on. Each generation
is a freshly initialised model; what is inherited is the data, not the weights.
Synthetic corpora are generated unconditionally at temperature 1.0 with no
truncation, sized to match the real corpus. Every arm is scored on the same
held-out **real** validation set.

### Held-out BPB on real data (mean ± std, 3 seeds)

| gen | control | accumulate | replace |
|---:|---:|---:|---:|
| 1 | 2.0396 ±0.0092 | 2.1187 ±0.0086 | 2.1995 ±0.0144 |
| 2 | 2.0450 ±0.0081 | 2.1543 ±0.0153 | 2.2955 ±0.0182 |
| 3 | 2.0461 ±0.0118 | 2.1871 ±0.0054 | 2.3673 ±0.0158 |
| **drift g1→g3** | **+0.0065** | **+0.0684** | **+0.1678** |

Gap versus control at generation 3: `accumulate` **+0.1410** (35× noise floor),
`replace` **+0.3212** (80× noise floor).

### The headline

> **Self-training on your own output degrades a small language model,
> substantially and reproducibly. Accumulating real data alongside synthetic
> slows the degradation to ~41% of its rate but does not stop it.**

Three independent seeds; `replace` lands at 2.3514 / 2.3676 / 2.3829 at
generation 3. The effect replicates in magnitude and in per-generation
increment, not merely in sign.

### But the mechanism is not the one I predicted

This is the part worth reading. **Generated-corpus diversity does not fall — it
rises.**

| corpus | distinct_1 | distinct_2 | distinct_3 | vocabulary |
|---|---:|---:|---:|---:|
| generation 0 | 0.0048 | 0.2918 | 0.7076 | 86.6% |
| replace gen 1 | 0.0048 | 0.3194 | 0.7166 | 87.4% |
| replace gen 2 | 0.0048 | **0.3358** | 0.7180 | 87.8% |

Classic model collapse is the distribution **narrowing** — tails disappear,
diversity drops, the model converges on a confident, wrong mode. None of that is
happening here. Vocabulary coverage is flat at ~87%, distinct-3 is flat, and
distinct-2 climbs steadily.

`replace`'s **training loss is higher** than control's (4.35 → 4.61 versus
3.86 → 4.03), not lower. It is not fitting a narrower distribution more snugly;
it is struggling with harder data.

**The reading:** at temperature 1.0 with no truncation, a small parent model
produces text that is a *noisier, higher-entropy* approximation of the real
distribution — not a narrower one. Each generation trains on a worse model of
reality and adds its own error. This is **error accumulation**, not mode
collapse. Both degrade the model; they are different failure modes with
different signatures.

That suggests a concrete, testable follow-up, and it is the most interesting
thing this run produced:

> **Sampling temperature likely selects which failure mode you get.** Low
> temperature or top-k truncation cuts the tails at generation time and should
> produce the narrowing signature the literature describes; temperature 1.0
> preserves and compounds entropy and produces noise accumulation. The diversity
> metrics distinguish the two directly.

### Prediction scorecard — 3 of 5 wrong

| # | Prediction | Outcome |
|---|---|---|
| 1 | control flat within 0.0040 | **Marginal.** Drift +0.0065 over 3 generations (1.6× floor). Per-generation ~0.003, inside the floor; cumulative slightly outside |
| 2 | replace degrades monotonically, +0.05–0.30 by gen 3 | **Confirmed**, magnitude slightly exceeded (+0.3212). Monotonic in all three seeds |
| 3 | accumulate within ~0.02 of control | **Wrong.** +0.079 at gen 1 rising to +0.141. Off by ~7× |
| 4 | diversity falls before loss rises | **Falsified.** Diversity is flat or rising throughout |
| 5 | training loss falls while held-out rises | **Falsified.** Replace's training loss is *higher* than control's and rising |

Predictions 4 and 5 were both downstream of assuming the narrowing mechanism.
Getting them wrong is what located the actual mechanism — which is the argument
for pre-registration in one table.

### Limitations, stated plainly

- **Underpowered by design.** Three seeds per arm means a permutation test
  enumerates C(6,3)=20 splits and cannot produce p < 0.10. The conclusions rest
  on effect sizes of 35–80× the measured noise floor, not on p-values. Five
  seeds per arm is the minimum for significance; this correction is now in
  `Comparison.underpowered`.
- **`accumulate` dilutes the real data.** "Real + all synthetic" means the real
  share falls 50% → 33% → 25% as the pool grows. So this arm tests a *shrinking
  fraction* of real data, not a fixed one — a weaker claim than the accumulation
  hypothesis proper. A fourth arm holding the real fraction constant is the
  honest test and was **not** run.
- **One corpus, one model size, one temperature, three generations.** Nothing
  here establishes where the trajectory goes at generation 10, or whether the
  same holds on TinyStories or at 10M parameters.
- **The parent is weak.** A 500-step, 1M-parameter model is a poor generator. A
  stronger parent would produce a better corpus and plausibly a different
  mechanism — quite possibly the narrowing one.

### Follow-ups, in priority order

1. **Temperature sweep** (0.5 / 0.8 / 1.0, and top-k 40) — does low temperature
   produce the narrowing signature? This is the mechanism question and it is
   cheap.
2. **Fixed-real-fraction arm** — separates "real data helps" from "less
   repetition helps".
3. **Five seeds** so the statistics can certify what the effect sizes already
   show.
4. **More generations** — does `replace` plateau at a degraded equilibrium or
   keep falling? Increments were +0.16, +0.10, +0.07: decelerating, not flat.

---

## R5 — Seed noise floor  ⭐ the number every later comparison depends on

**Date:** 2026-09-13 · **Reproduce:** `make noise-floor` · **Pre-registered:**
[lab-notebook E1](lab-notebook.md#e1--what-is-the-seed-noise-floor)

One configuration (`v6_modern`, 1,049,216 params, 1000 steps), five seeds,
nothing else varied. The seed controls both weight initialisation and batch
order — which is the right thing to measure, because that is what differs
between any two runs you would actually compare.

| seed | val BPB | tokens/sec |
|---:|---:|---:|
| 0 | 1.8892 | 1,764 |
| 1 | 1.8910 | 19,160 |
| 2 | 1.8908 | 19,506 |
| 3 | 1.8991 | 19,665 |
| 4 | 1.8950 | 19,623 |

**mean 1.8930 · std 0.0040 · spread 0.0099**

### What it says

> **0.0040 BPB is the significance threshold for this project.** A configuration
> change that moves held-out BPB by less than that has not been shown to do
> anything, no matter how clean the plot looks or how small the p-value is.

`compare_groups` enforces this: an effect below the noise floor is reported as
"below the noise floor — no effect" even when the permutation test says p < 0.01.
A statistically significant difference smaller than seed variance is an artefact
of too small a sample, not a finding.

This also sets the `NoRegression` tolerance for the Phase 8 improvement loop.
The default 0.01 is 2.5x the floor — deliberately conservative, because a gate
tighter than the noise floor fires on chance and then gets disabled, which is
worse than one that occasionally passes a small regression.

### The prediction was wrong

E1 predicted std 0.005–0.02; the measured 0.0040 is below that range. Recorded
rather than rounded into "as expected". The model is more seed-stable than I
expected, most likely because 6.6 epochs over a small corpus means every run sees
nearly the same data and a 1M-parameter model is far from capacity-limited on it.

### Seed 0 ran 11x slower and still scored best

It was competing with two test processes for four cores. Its BPB was
nevertheless the lowest of the five. **CPU contention changes wall-clock, not
results** — the determinism guarantees hold regardless of what else the machine
is doing.

### Caveats

- **Measured at one configuration.** The noise floor is not a universal
  constant; a larger model, a different lr, or a bigger corpus will have its own.
  Re-measure when the configuration changes materially, especially on TinyStories.
- **Five seeds is not many.** The std itself has meaningful uncertainty. Treat
  0.0040 as an order-of-magnitude guide, not a precise constant.
- **The bigram baseline has n=1.** With an effect of 1.35 BPB (337x the noise
  floor) that is academic, but it should get more seeds before appearing in any
  write-up as a measured comparison.

---

## R4 — First trained model (TinyShakespeare)

**Date:** 2026-09-13 · **Reproduce:** `make data && make train`

| | |
|---|---|
| Architecture | `v6_modern` — 4L x 128d x 4h, RMSNorm / RoPE / SwiGLU / no bias / tied |
| Parameters | **1,049,216** (787,072 non-embedding, 262,144 embedding) |
| Data | 372,993 train tokens, vocab 2,048 |
| Training | 1,200 steps x 16 x 128 = 6.6 epochs |
| Throughput | ~19,200 tokens/sec (4-core cloud CPU — **not** a target-machine number) |
| Wall clock | ~2 minutes |
| **Held-out BPB** | **1.8732** |
| Val loss | 3.6710 |

Sample at temperature 0.8, top-k 40:

```
First Citizen:
This is the house I do attend you to them?
First Senator:
If he become, thou cross,
To see what thou hast made it to wide?
ESCALUS:
I am of them in the rest.
```

At one million parameters and two minutes of CPU, the model has learned the
corpus *format* essentially perfectly — speaker names, the colon, the newline,
document separators in the right places — plus grammatical local English and
character names specific to the source text. It has not learned meaning, and at
this scale it will not.

The comparison class is small models on similar data, never a frontier model.
This is ~0.001% the size of one; comparing them is a category error that would
obscure whether 1.87 BPB is actually good here.

### Still missing, and needed before this number means much

- **No seed variance.** One run, one seed. Per [evaluation.md](evaluation.md),
  nothing should be concluded from a single seed — the noise floor has not been
  measured yet, so the significance threshold is unknown. That is a Phase 4 task
  and it blocks every comparison.
- **No bigram baseline.** `BigramModel` exists but has not been run on this
  corpus, so the floor that turns "is 1.87 good?" into a measurement is missing.

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

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

## R10 — The loop matches a compute-matched control, and rediscovers its own constraints

**Date:** 2026-09-14 · **Reproduce:** `python scripts/loop_vs_control.py --run --seeds 0 1 2`
· 3 seeds × (15 iterations + two controls)

**This result previously claimed the loop beat a compute-matched control by 4.7×
the noise floor. That claim was wrong**, for three independent reasons, all
found after it was written. What follows is the corrected version; the original
is in git history and the failures are written up in
[evaluation.md](evaluation.md) because each one is more transferable than the
result itself.

### The loop does not beat a step-matched control

Every number below is measured on the **`test` split, which neither the loop nor
the controls ever read** — not the `val` split the loop gates on. All three seeds
stopped on their iteration count, not the wall clock.

| seed | loop *reports* | loop (test) | acc control (test) | tot control (test) |
|---:|---:|---:|---:|---:|
| 0 | 1.8460 | 1.8992 | 1.9089 | 1.9438 |
| 1 | 1.8612 | 1.9028 | 1.9130 | 1.9429 |
| 2 | 1.8824 | 1.9200 | **1.9011** | 1.9178 |

Against the **accumulated-steps control** — plain training for exactly the steps
the loop kept — the loop is **indistinguishable**:

> acc 1.9077 ± 0.0061 · loop 1.9074 ± 0.0111 · **difference −0.0003, below the
> 0.0040 noise floor — no effect.**

It loses outright on seed 2. Against the **total-spent control** — plain training
for every step the loop burned, rejected candidates included — the loop is ahead
by 0.0275 (6.9× noise), but that comparison flatters it: the total-spent control
trains 3150 steps on 373k tokens, which is **17.3 epochs**, and it is visibly
overtraining. Beating an overtrained baseline is a statement about the step
budget, not about search.

With 3 seeds per arm a permutation test cannot go below p=0.10, so neither
comparison is significant; see the underpowered-test trap in
[TODO.md](../TODO.md).

**The honest summary: the loop's selection is worth roughly what it costs — it
finds where to stop, and nothing more than that.** Its advantage over the naive
long run is avoiding overtraining, which a learning-rate schedule and early
stopping also achieve, far more cheaply.

### The loop's own reported number is optimistic by 11× the noise floor

| | mean |
|---|---:|
| loop's reported "best held-out BPB" (val) | 1.8632 |
| the same models on `test` | 1.9073 |
| **gap** | **+0.0441 (per seed +0.0532, +0.0416, +0.0376)** |

The loop gates promotion on `val_bpb` and then reports the *minimum* `val_bpb`
over its promotions. That split is held out from training but not from
selection, and the minimum of a selected set is biased by exactly what the
selection was worth. **+0.0441 is eleven times the noise floor, and larger than
any effect this result reports.**

This is the project's own safety finding
([R5](#r5--lower-training-loss-predicted-worse-held-out-performance)) one level
up. Gating on data the loop cannot influence is necessary and was done. It is
not sufficient: the number you *report* must also be one nothing selected on.

### The loop rediscovered R6–R9 on its own — replicated

Pooled over 45 iterations across 3 seeds:

| synthetic fraction | rejected |
|---:|---:|
| 0% | 2/16 (12%) |
| 25% | 3/13 (23%) |
| **50%** | **10/16 (62%)** |

Monotonic, and cleaner than the single-run version this replaces.
**Nobody told the loop that synthetic data hurts.** It found out by having its
candidates refused, and the pattern reproduces the dose-response relationship
[R9](#r9--a-little-real-data-does-almost-all-the-work) measured directly.

All 15 rejections across all three seeds were `no_regression` — the candidate
was worse than the incumbent and did not get in. Promotion rate 67%.

This remains the clearest vindication of a decision made in
[self-improvement.md](self-improvement.md): *rejections are recorded, not
discarded, because they are the research data.* A loop that logged only its
promotions would have produced the same models and none of this table.

### Induction never appeared — replicated

All **45** iterations scored between **−0.756 and −0.058 bits**. Never once
positive, on any seed. A negative score means the second copy of a repeated
random sequence is *harder* to predict than the first.

A 1M-parameter model trained on 373k tokens of Shakespeare appears to form **no
induction circuit at all**. That is a concrete data point for
[Decision 12's question D](decisions.md#12-research-question), and held-out loss
alone would never have surfaced it — which was the argument for capability
probes in the first place.

### What this result got wrong the first time

Recorded because the failures generalise and the result does not:

1. **It reported a number selected on the gating split** (+0.0441 of bias),
   having carefully arranged for the gate itself to be uninfluenceable.
2. **It was not reproducible.** `scripts/run_loop.py` built its incumbent before
   seeding anything, so identical seeds gave identical *proposals* and different
   *measurements* — enough at one iteration to flip a promotion into a
   rejection. The single-seed original cannot be regenerated.
3. **One seed was truncated by a wall-clock budget** that bound before the
   iteration count, making machine load an experimental variable.
4. **The driver was never committed**, so none of it could be audited. It is
   `scripts/loop_vs_control.py` now.

The first version of this result was a single unreproducible run of an
unreviewable script, reporting a contaminated metric, against one control that
was arguably the wrong one. It reached a conclusion that was directionally
wrong. Three of the four defects were found by re-running it, which is the
argument for re-running things.

### Limitations

- **3 seeds per arm cannot reach p<0.05.** 5 would.
- **The incumbent is deliberately under-trained** so the loop has headroom.
  Starting from a converged model would look very different and probably duller.
- **Wall-clock is unfavourable and the step-matched framing hides it.** The loop
  spends generation and evaluation time the controls do not.
- **Random search over a small space**, sampled roughly 2.5× over 15 iterations,
  so late iterations largely re-test.
- **The `test` split is 3.5k tokens**, small enough that its own sampling error
  is not negligible at this precision.

---

## R9 — A little real data does almost all the work

**Date:** 2026-09-14 · **Pre-registered:** [lab-notebook E6](lab-notebook.md#e6--is-it-the-fraction-of-real-data-or-the-amount)
· 18 runs completing the dose-response curve at temperature 0.5

A fixed-pool `anchor` arm holds the total corpus at the real corpus size and
**discards** real data to make room for synthetic — unlike `accumulate`, which
keeps all of it and lets the fraction fall. That separates two claims R8 could
not.

### The dose-response curve, at fixed pool size

| real fraction | real tokens | arm | BPB (gen 3) | damage recovered |
|---:|---:|---|---:|---:|
| 0% | 0 | `replace` | 4.5339 | — |
| **25%** | 93,248 | `anchor0.25` | **2.3216** | **89%** |
| 50% | 186,496 | `anchor0.5` | 2.1660 | 95% |
| 100% | 372,993 | `control` | 2.0461 | 100% |

"Damage recovered" is the share of `replace`'s 2.488 BPB excess that is closed.

### The finding

> **The first slice of real data does almost all the work.** Going from none to
> a quarter recovers **89%** of the damage. The remaining three quarters of the
> corpus buys the last 11%.

This is a steeply diminishing return, and it is the practically useful shape:
for a self-training loop, what matters is that the real fraction is *non-zero*,
far more than that it is *large*. A lineage does not need a balanced mix. It
needs to not be starved.

### Both amount and fraction matter — neither alone explains it

The decisive comparison is `accumulate` against `anchor0.25`, which share a
nominal 25% real fraction but differ fourfold in real tokens:

| | real fraction | real tokens | BPB |
|---|---:|---:|---:|
| `anchor0.25` | 25% | 93k | 2.3216 |
| `accumulate` | 25% | **373k** | **2.2099** |

**0.112 BPB apart, 28× the noise floor** — at the same fraction. So the absolute
quantity of real data matters, as predicted.

But it is not only quantity:

| | real fraction | real tokens | BPB |
|---|---:|---:|---:|
| `accumulate` | 25% | 373k | 2.2099 |
| `anchor0.5` | **50%** | **187k** | **2.1660** |

`anchor0.5` holds **half** the real tokens and still scores better — doubling the
fraction outweighed halving the amount, in this range. **Both levers are real,
and neither dominates.** My E6 mechanism story — that coverage is purely an
absolute property — is therefore incomplete.

### Prediction scorecard

| Prediction | Outcome |
|---|---|
| Amount matters, not only fraction (~65% confidence) | **Correct in direction** — 28× the noise floor at matched fraction |
| `anchor` @ 25% in 2.5–3.0 | **Wrong.** Actual 2.3216, below the range |
| `anchor` @ 50% in 2.25–2.45 | **Wrong.** Actual 2.1660, below the range |
| Mechanism: coverage is absolute, so fraction is incidental | **Incomplete.** Fraction has its own effect, and a strong one |

**Third experiment running in a row where my magnitude was wrong in the
favourable direction** (R8: predicted 2.3–2.8, got 2.21; here: predicted
2.5–3.0, got 2.32). That is a consistent bias worth naming: **I have been
systematically over-estimating how much damage self-training does once any real
data is present.** Directions have held up; sizes have not. Future predictions
should widen their lower bound.

### It qualifies the Phase 8 search space

`SearchSpace.min_real_fraction` defaults to 0.25, justified by R8. R9 sharpens
what that buys: at 25% real in a fixed pool the lineage still sits **+0.28 BPB
above control — 70× the noise floor.** So the floor prevents *catastrophe*
(4.53), not *degradation*.

That is the correct division of labour and worth stating explicitly: the search
space rules out the unrecoverable regime, and `NoRegression` catches the
recoverable one. Neither is doing the other's job, and the floor should not be
mistaken for a safety guarantee.

### Limitations

- **Three seeds**; effect sizes carry the argument, not p-values.
- **One temperature (0.5).** The curve's shape at T1.0, where damage is mild,
  is untested and could differ.
- **The 89% figure is a single point on a coarse grid.** Fractions between 0 and
  25% are where the curve is steepest and are entirely unsampled — the knee
  could be at 5% or at 20%.
- **Fixed pool means less total data**, so `anchor` arms see fewer unique tokens
  than `accumulate` in absolute terms, which is the effect being measured but
  also a confound with corpus size per se.

---

## R8 — Retaining real data makes a lineage regime-proof  ⭐ the practical result

**Date:** 2026-09-14 · **Pre-registered:** [lab-notebook E5](lab-notebook.md#e5--does-accumulating-real-data-rescue-a-collapsing-lineage)
· One new cell (9 runs, ~13 min) completing the 2×2

### The completed factorial — held-out BPB at generation 3

| | T1.0 | T0.5 | temperature effect |
|---|---:|---:|---:|
| **replace** | 2.3673 (+0.321) | **4.5339 (+2.488)** | **+2.167** |
| **accumulate** | 2.1871 (+0.141) | **2.2099 (+0.164)** | **+0.023** |
| *control* | *2.0461* | | |

Excess over control retained by accumulation: **43.9%** at T1.0,
**6.6%** at T0.5.

### The finding

> **Keeping real data in the mix makes the lineage almost immune to the sampling
> regime.** `replace` swings 2.17 BPB across temperatures; `accumulate` swings
> 0.023 — about six times the noise floor, and two orders of magnitude smaller.

This is strongly **non-additive**. Accumulation is not a fixed-percentage
discount on the damage; it is a qualitatively better regime whose benefit grows
with the severity of what it is protecting against. It rescues the catastrophic
case far more effectively than the mild one.

### The mechanism, confirmed directly

The `accumulate` lineage at T0.5 produced corpora **just as degenerate** as
`replace` did:

| generation | vocabulary coverage | distinct_3 |
|---|---:|---:|
| 1 | 34.9% | 0.0335 |
| 2 | 10.9% | 0.0048 |

Its own output collapsed completely — and its *model* finished at 2.21 BPB
against `replace`'s 4.53. The real data did not prevent the corpus from
degenerating; it prevented the **model** from following it down. That is
tail anchoring, observed directly rather than inferred: the retained real data
supplies distribution coverage the synthetic data structurally cannot, and 25%
real was enough.

### Prediction scorecard

| Prediction | Outcome |
|---|---|
| Tail anchoring over proportional scaling (~55% confidence) | **Correct** — 6.6% retained, nowhere near the 44% proportional scaling predicted |
| Generation-3 BPB in 2.3–2.8 | **Wrong, in the favourable direction.** Actual 2.2099 — *below* my range. The effect is stronger than I predicted |
| Seed variance comparable to replace's | Correct — spread 0.022, in line |

**Honouring my own pre-registered scepticism.** E5 stated that a result ≤ 2.2
"would need a fourth seed set before I believed it." The measured 2.2099 sits
just above that line — close enough that the caution applies in spirit. The
direction and mechanism are solid; the *magnitude* deserves five seeds before it
goes in any write-up as a headline number.

### It broke the gate I had just shipped

[R7](#r7--sampling-temperature-selects-the-collapse-failure-mode-) established
that corpus vocabulary coverage predicts collapse a generation early, and that
finding was wired into `CorpusDiversityFloor` as a promotion gate. **R8 shows
that gate would have refused this healthy lineage** — 34.9% coverage is well
below its 50% floor, yet the model was fine.

The signal is real but conditional: **corpus degeneracy predicts model collapse
only when real data is absent from the training mix.** The gate now takes
`real_data_fraction` and treats a narrow corpus as informational when at least
20% of the mix is real (the threshold R8 actually measured), while still
enforcing the floor when the fraction is low *or unknown* — not knowing the mix
means you cannot claim the anchor.

A useful reminder that an early-warning signal validated in one regime is not
automatically valid in another, and that shipping a gate one experiment after
discovering its basis is fast enough to get caught out.

### What this means for the Phase 8 loop

The design implication is concrete and now evidence-backed: **the loop must
never train a generation on synthetic data alone.** Retaining real data costs
essentially nothing at T1.0 (+0.02 BPB against `replace`… in fact it *helps*)
and is the difference between 2.21 and 4.53 at T0.5. It converts the sampling
temperature from a parameter that must be chosen carefully into one that barely
matters.

### Limitations

- **Three seeds**, so effect sizes rather than p-values carry the argument.
- **Three generations.** `accumulate` at T0.5 is still drifting upward
  (2.106 → 2.170 → 2.210); whether it plateaus or eventually follows is unknown,
  and its real-data share is falling 50% → 33% → 25% as it goes.
- **The dilution confound persists** from [R6](#r6--model-collapse-does-self-training-degrade-a-small-lm--the-headline-result):
  a fixed-real-fraction arm would separate "real data anchors" from "a larger,
  more varied pool helps".
- **One corpus, one model size.**

---

## R7 — Sampling temperature selects the collapse failure mode  ⭐

**Date:** 2026-09-14 · **Pre-registered:** [lab-notebook E4](lab-notebook.md#e4--does-sampling-temperature-select-the-collapse-failure-mode)
· **Reproduce:** `make collapse-all` · 27 runs

> **Re-run 2026-09-14** after the seeding fix; see
> [R6–R9 re-run](#r6r9-re-run--what-moved-and-what-did-not).

The `replace` arm only, at four sampling regimes, 3 generations × 3 seeds each.
Everything else identical to [R6](#r6--model-collapse-does-self-training-degrade-a-small-lm--the-headline-result),
so the sampling regime is the single independent variable. `control` never
generates, so it is regime-independent and reused (2.035 at generation 3).

### Held-out BPB on real data

| regime | gen 1 | gen 2 | gen 3 | drift | vs control | own sd (g3) |
|---|---:|---:|---:|---:|---:|---:|
| **T0.5** | 2.6742 | 3.7377 | **4.2887** | +1.615 | **+2.25** | 0.151 |
| T1.0 + top-k 40 | 2.6784 | 3.0013 | 3.1816 | +0.503 | +1.15 | 0.068 |
| T0.8 | 2.2207 | 2.4784 | 2.8397 | +0.619 | +0.80 | 0.007 |
| T1.0 | 2.1862 | 2.2814 | 2.3552 | +0.169 | +0.32 | 0.015 |

The last column replaces the "×noise floor" multipliers this table used to
carry. The 0.0040 floor was measured on a *healthy* config; a collapsed arm's
own seed spread is up to 38× wider, so quoting its excess in floor-units
("622× noise") overstated the precision badly. Every row is still many of its
own standard deviations above control — T0.5 is 15σ — but that is the honest
scale.

### The finding

> **It is not temperature. It is how much of the distribution's tail survives
> sampling.**

Top-k 40 at temperature 1.0 degrades the model *more* than temperature 0.8 does
(+1.15 vs +0.80 against control), despite leaving the temperature untouched.
Truncation and temperature act through the same channel — both discard the tails
at generation time — and the resulting collapse tracks that, not the temperature
parameter itself.

### The mechanism, measured

| regime | distinct_3 (g1) | vocabulary (g1) | training loss (g2) | BPB (g3) |
|---|---:|---:|---:|---:|
| T0.5 | 0.0413 | **37.4%** | **1.10** | 4.2887 |
| T1.0 + top-k 40 | 0.3529 | 47.6% | 2.90 | 3.1816 |
| T0.8 | 0.3846 | 80.7% | 2.88 | 2.8397 |
| T1.0 | 0.7184 | **87.4%** | **4.41** | 2.3552 |

**Vocabulary coverage of the generated corpus at generation 1 predicts
generation-3 BPB monotonically.** That is an early-warning signal: it is
measurable one full generation before the damage is visible in held-out loss,
and it is cheap — no evaluation set required, just a token count over the corpus
the loop already produced.

At T0.5 the model emits **13.0% of its vocabulary** by generation 2 and scores
4.29 BPB — **worse than the bigram baseline of 3.24**. A 1M-parameter
transformer, trained on its own output for three rounds, ends up worse than a
context-free lookup table. Even top-k 40 lands at 3.18, essentially *at* the
bigram floor.

### Lower training loss predicts worse real performance

| regime | training loss (g2) | held-out BPB (g3) |
|---|---:|---:|
| T0.5 | 1.10 | 4.29 |
| T1.0 | 4.41 | 2.36 |

The regime that fits its training data **four times better** ends up **twice as
bad** on real text. Across the extremes the relationship is a clean inversion.
(It is not a perfect ranking: T0.8 and T1.0+top-k sit at 2.88 and 2.91 training
loss but 2.86 and 3.23 BPB, so training loss does not resolve the middle.)

**The operational consequence** for the Phase 8 loop is direct: a promotion gate
reading training loss would rank these regimes **exactly backwards** and would
select the catastrophic one. The held-out gate that
[constitution.md](constitution.md) requires is not a nicety — it is the only
thing between the loop and confidently optimising itself into uselessness.

### The counterintuitive practical lesson

The instinct when building a self-training loop is to generate with low
temperature or top-k, because that produces "higher quality" text — more
grammatical, more on-distribution, better to read. **That instinct is exactly
backwards.** The cleaner the samples look, the faster the model dies. Sampling
that preserves the tails produces worse-looking text and a far healthier
lineage.

### Prediction scorecard — 3 of 3 correct

| # | Prediction | Outcome |
|---|---|---|
| 1 | diversity falls with temperature (high confidence) | **Confirmed**, dramatically: vocabulary 87% → 37% → 11% |
| 2 | low temperature degrades faster (60% confidence) | **Confirmed**, by far more than expected (+2.49 vs +0.32) |
| 3 | training loss lower at T0.5 | **Confirmed**: 1.04 vs 4.44 |

A marked contrast with [R6](#r6--model-collapse-does-self-training-degrade-a-small-lm--the-headline-result),
where three of five were wrong. The predictions improved because R6's data
informed them — which is what iteration is for. Worth not over-reading: the
mechanism was already half-identified before these predictions were written, so
this is less impressive than 3/3 sounds.

### Amendment to R6

R6 recorded E3's prediction 5 — "training loss falls while held-out BPB rises" —
as **falsified**. That was correct for the regime tested and **too general as
stated**. The accurate version: it is *false at temperature 1.0 and emphatically
true at temperature 0.5*. The prediction was not wrong, it was **regime-
dependent**, and R6's scorecard has been amended to say so.

### Limitations

- **Three seeds**, so the permutation test cannot certify these (p floor 0.10).
  The effects are 80–622× the noise floor, so the conclusions rest on effect
  size — but the caveat stands.
- **One weak parent.** A 500-step 1M-parameter model. A stronger generator would
  produce a better corpus at every temperature and might move the whole curve.
- **Three generations.** T0.5 is still falling steeply at generation 3; where it
  settles is unknown.
- **`replace` only.** Whether accumulation rescues a low-temperature lineage is
  untested and is the obvious next question.

### Follow-ups

1. **Does accumulation rescue T0.5?** If keeping real data prevents the
   catastrophic case, that is the practically important result.
2. **Vocabulary coverage as a live gate** — wire it into `NoRegression` as an
   early-warning invariant, since it fires a generation before BPB does.
3. Five seeds; more generations; a stronger parent.

---

## R6 — Model collapse: does self-training degrade a small LM?  ⭐ the headline result

**Date:** 2026-09-14 · **Pre-registered:** [lab-notebook E3](lab-notebook.md#e3--does-a-self-training-loop-collapse-and-does-accumulation-prevent-it)
· **Reproduce:** `make collapse-all` · 30 training runs, 15 corpus generations

> **Re-run 2026-09-14** after a seeding bug was found in
> `collapse_experiment.py` (see [R6-R9 re-run](#r6r9-re-run--what-moved-and-what-did-not)).
> Numbers below are from the reproducible run; the pre-fix figures are archived
> in `runs/archive-unseeded-collapse/`.

Three arms differing **only** in what generation *n* trains on. Each generation
is a freshly initialised model; what is inherited is the data, not the weights.
Synthetic corpora are generated unconditionally at temperature 1.0 with no
truncation, sized to match the real corpus. Every arm is scored on the same
held-out **real** validation set.

### Held-out BPB on real data (mean ± std, 3 seeds)

| gen | control | accumulate | replace |
|---:|---:|---:|---:|
| 1 | 2.0396 ±0.0092 | 2.0980 ±0.0103 | 2.1862 ±0.0079 |
| 2 | 2.0417 ±0.0018 | 2.1394 ±0.0057 | 2.2814 ±0.0072 |
| 3 | 2.0350 ±0.0171 | 2.1680 ±0.0011 | 2.3552 ±0.0149 |
| **drift g1→g3** | **−0.0045** | **+0.0700** | **+0.1690** |

Gap versus control at generation 3: `accumulate` **+0.1329** (33× noise floor),
`replace` **+0.3202** (80× noise floor).

### The headline

> **Self-training on your own output degrades a small language model,
> substantially and reproducibly. Accumulating real data alongside synthetic
> slows the degradation to ~42% of its rate but does not stop it.**

Three independent seeds; `replace` lands at 2.3460 / 2.3472 / 2.3724 at
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
| 5 | training loss falls while held-out rises | **Falsified at this temperature** — replace's training loss is *higher* than control's. **Amended by [R7](#r7--sampling-temperature-selects-the-collapse-failure-mode-):** the prediction is regime-dependent, and is emphatically true at temperature 0.5 (training loss 1.04, BPB 4.53) |

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

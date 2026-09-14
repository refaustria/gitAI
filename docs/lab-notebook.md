# Lab Notebook

One entry per experiment, **written before the runs finish**. Question,
prediction, and what result would change your mind.

This is a discipline, not bureaucracy. It is the mechanism that stops you
retrofitting a story onto whatever came out — the most common failure mode in
solo research, precisely because no reviewer is there to catch it. Writing down
"I expect X" and then getting Y is how you learn something; deciding afterwards
that Y is what you expected teaches you nothing.

Entries are append-only. When a prediction turns out wrong, **leave it** and add
the outcome underneath. The wrong predictions are the valuable part.

---

## E1 — What is the seed noise floor?

**Date:** 2026-09-13 · **Status:** prediction recorded before results

### Question

Train one configuration five times, changing nothing but the seed. How much does
final held-out BPB vary?

This number is the significance threshold for every later comparison in the
project. Until it exists, no claim that one configuration beats another means
anything.

### Setup

`v6_modern`, 4L × 128d × 4h, 1,049,216 parameters, 1000 steps, batch 16 × seq
128, lr 3e-3, TinyShakespeare (372,993 train tokens, vocab 2,048). Seeds 0–4.
The seed controls weight initialisation *and* batch order, so this measures the
combined effect — which is the right thing, since that is what differs between
any two runs you would actually compare.

### Prediction (recorded before the sweep completed)

- Standard deviation in val BPB: **0.005 – 0.02**.
- Spread (max − min) roughly 2–3× the std, so **0.01 – 0.06**.
- Reasoning: 6.6 epochs over a small corpus means each run sees essentially the
  same data, so the variation should come mostly from initialisation and batch
  order rather than from which data was seen. The model is also small enough to
  be far from capacity-limited on this corpus.

### What would change my mind

- **std > 0.05** would mean training is unstable at this lr/size, and the lr
  schedule or warmup needs revisiting before any experiment is worth running.
- **std < 0.001** would be suspicious rather than reassuring — it would suggest
  the seed is not actually varying the run, and I would go check that batch
  order really is seed-dependent.

### Result

| seed | val BPB |
|---:|---:|
| 0 | 1.8892 |
| 1 | 1.8910 |
| 2 | 1.8908 |
| 3 | 1.8991 |
| 4 | 1.8950 |

**mean 1.8930 · std 0.0040 · spread 0.0099**

### Verdict on the prediction: **wrong, narrowly**

I predicted std **0.005–0.02**. The measured value is **0.0040** — below the
bottom of my range. Same for spread: predicted 0.01–0.06, measured 0.0099.

The miss is small but it is a miss, and it is recorded rather than quietly
rounded into "as expected". The model is *more* stable across seeds than I
expected. The likely reason is the one I half-identified in the reasoning and
then under-weighted: at 6.6 epochs over a 373k-token corpus every run sees
essentially the same data in a different order, and at 1M parameters the model is
nowhere near capacity-limited, so there is little room for seeds to diverge.

Neither falsification criterion fired: std was not above 0.05 (training is
stable at this lr), and not below 0.001 (the seed is genuinely varying the run).

### Unplanned observation

Seed 0 ran at **1,764 tokens/sec** while the other four ran at ~19,500 — it was
competing with two test processes for four cores. Its BPB (1.8892) was
nevertheless the *best* of the five. Worth stating plainly: **CPU contention
changes wall-clock, not results.** The run is reproducible regardless of what
else the machine was doing, which is exactly what the determinism work was for.

### What it means for the project

**0.0040 BPB is now the significance threshold for this project.** Any
configuration change producing less than that is indistinguishable from chance,
and `compare_groups` refuses to call it an effect regardless of p-value.

It also sets the `NoRegression` tolerance for the Phase 8 loop. The current
default of 0.01 is 2.5x the noise floor — conservative, which is the right side
to err on: a gate tighter than the noise floor fires on chance and gets disabled,
which is worse than one that occasionally lets a small regression through.

---

## E2 — Does the transformer beat the context-free floor? *(answered)*

**Date:** 2026-09-13

### Question

A bigram model is a pure lookup table: token *t* predicts token *t+1* with no
context at all. Its loss is the best any context-free model can do on this
corpus. Does the transformer beat it, and by how much?

Without this the headline number is uninterpretable — "1.87 BPB" is an opinion
until something anchors it.

### Prediction

The transformer should beat it clearly. A bigram on a 2,048-token vocabulary has
4.2M parameters — *four times* the transformer's 1.05M — and still cannot
condition on more than one token, so this is a test of whether context is worth
anything at all. If the transformer failed to beat it decisively, something would
be badly wrong.

### Result

| model | parameters | val BPB |
|---|---:|---:|
| Bigram (context-free floor) | 4,194,304 | **3.2406** |
| `v6_modern` transformer | 1,049,216 | **1.8732** |

**42% lower BPB with a quarter of the parameters.** Context is worth a great
deal, as expected. The transformer is not merely memorising unigram/bigram
statistics.

Incidentally: the bigram was trained twice by accident and produced
`3.240565592697779` both times — bit-identical. An unplanned confirmation that
the determinism guarantee holds end to end.

### Follow-up

A stronger floor would be an n-gram model with backoff, which would tell us how
much of the transformer's advantage is just longer-context statistics versus
something structural. Cheap to build; worth doing before claiming the model has
learned anything interesting.

---

## E3 — Does a self-training loop collapse, and does accumulation prevent it?

**Date:** 2026-09-14 · **Status:** prediction recorded before the experiment ran

### Question

The project's headline question ([Decision 12](decisions.md#12-research-question),
question F). Train a model, have it generate a corpus, train the next generation
on that corpus, repeat. Does the model degrade? And does *accumulating* real and
synthetic data, rather than *replacing* real with synthetic, change the answer?

### Setup

Three arms, differing only in the training corpus for generation *n*:

| arm | corpus |
|---|---|
| `control` | real data only, every generation |
| `replace` | only generation *n-1*'s output |
| `accumulate` | real + every synthetic corpus so far |

3 generations × 3 seeds × 3 arms, plus a shared generation-0 model per seed.
Each generation is a **freshly initialised** model, so what is inherited is the
data, not the weights — that isolates the data effect, which is the question.
Synthetic corpora are generated unconditionally from the document separator, so
no real text leaks in through prompts, and sized to match the real corpus so the
arms differ in data *composition*, not volume.

Every arm is scored on the same held-out **real** validation set. This is the
whole point: a collapsing model gets better at predicting its own output while
getting worse at predicting reality, and only real held-out data separates those.

### Predictions

1. **`control` stays flat.** Variation across generations within the measured
   noise floor of 0.0040 BPB. If it drifts more than that, the experiment is
   measuring something other than what I think.
2. **`replace` degrades monotonically**, and by more than the noise floor by
   generation 1 already. Magnitude by generation 3: **+0.05 to +0.30 BPB** worse
   than control. This is the prediction I hold most confidently — it is what the
   collapse literature reports and the mechanism is clear.
3. **`accumulate` stays close to control** — within ~0.02 BPB — and possibly
   slightly *better*, since it sees strictly more tokens.
4. **Diversity falls before loss rises.** `distinct_3` on the generated corpus
   drops in `replace` at generation 1, before held-out BPB has moved much.
5. **Training loss will mislead.** `replace` models should reach *lower* training
   loss each generation while their held-out BPB rises — the failure mode that
   makes naive self-improvement loops dangerous, because their own metrics
   applaud.

### What would change my mind

- **`replace` not degrading beyond 0.0040 by generation 3** would mean either
  three generations is too few at this scale, or the synthetic corpus is closer
  to the real distribution than expected. Either way the conclusion would be
  "no collapse observed at this scale", stated plainly, not explained away.
- **`accumulate` degrading as fast as `replace`** would contradict the
  accumulation hypothesis and be the more interesting result.
- **`control` drifting** would invalidate everything and send me back to the
  harness.

### Known confound, stated up front

`accumulate` draws from a larger corpus each generation. **Corrected after
looking at the harness:** training is a fixed 500 steps for every arm, so all
arms process the *same number of tokens* — compute is matched. What differs is
the size and diversity of the pool those tokens are drawn from: `accumulate`
repeats itself less within a run.

So the confound is narrower than "more data", but it is real: some of any
advantage `accumulate` shows could be reduced repetition rather than the
presence of real data. The clean follow-up is to subsample the accumulated
corpus back to the real corpus size, holding pool size fixed and varying only
composition. That variant is **not** run here.

### Result

Full write-up: [results.md R6](results.md#r6--model-collapse-does-self-training-degrade-a-small-lm--the-headline-result).

**Headline:** self-training degrades the model substantially and reproducibly
(`replace` +0.3212 BPB versus control by generation 3, 80× the noise floor).
Accumulating real data slows it to ~41% of that rate but does not stop it
(+0.1410, 35× the noise floor).

**Scorecard: 3 of 5 predictions wrong.**

| # | Prediction | Outcome |
|---|---|---|
| 1 | control flat within noise | Marginal — +0.0065 cumulative, 1.6× floor |
| 2 | replace degrades +0.05–0.30 | **Confirmed**, magnitude slightly exceeded |
| 3 | accumulate within 0.02 of control | **Wrong** — off by ~7× |
| 4 | diversity falls before loss rises | **Falsified** — diversity *rises* |
| 5 | training loss falls as held-out rises | **Falsified** — it rises |

**What the wrong predictions bought.** Predictions 4 and 5 were both downstream
of assuming the classic narrowing mechanism. Their failure is what located the
actual one: at temperature 1.0 with no truncation, a weak parent produces a
*noisier* approximation of the real distribution rather than a narrower one, so
the failure mode is error accumulation rather than mode collapse. Had I not
written those predictions down, I would very likely have reported "collapse
confirmed" and never looked at the diversity numbers at all.

That is the argument for pre-registration, in one experiment.

**Next question, now well-posed:** does sampling temperature select the failure
mode? Low temperature and top-k truncate the tails at generation time and should
produce narrowing; temperature 1.0 compounds entropy and produces noise. The
diversity metrics distinguish them directly, and the experiment is cheap.

---

## E4 — Does sampling temperature select the collapse failure mode?

**Date:** 2026-09-14 · **Status:** prediction recorded before the experiment ran

### Question

[R6](results.md) found that self-training degrades a small LM, but **not** by
the mechanism the literature describes. Diversity rose rather than fell, and the
degrading models had *higher* training loss, not lower. The reading was that at
temperature 1.0 a weak parent produces a noisier, higher-entropy approximation
of the real distribution — error accumulation, not mode collapse.

That reading makes a falsifiable claim: **the failure mode should depend on the
sampling temperature.** Low temperature and top-k truncation cut the tails at
generation time, which is precisely the narrowing that classic collapse
describes. Temperature 1.0 preserves and compounds entropy instead.

### Setup

The `replace` arm only (the one that degrades fastest), at temperatures **0.5**,
**0.8** and **1.0**, plus **top-k 40 at temperature 1.0**. 3 generations × 3
seeds each. Everything else identical to R6, so temperature is the single
independent variable. The `control` arm never generates, so it is
temperature-independent and reused from R6.

### Predictions

**High confidence — diversity falls with temperature.** Generated-corpus
`distinct_2` and `distinct_3` should drop monotonically as temperature drops.
This is close to definitional: sharpening a distribution reduces the entropy of
samples drawn from it. If this does *not* hold, the diversity metrics are
measuring something other than what I think and every conclusion in R6 that
rests on them needs revisiting.

**Genuinely uncertain — which direction BPB moves.** The two hypotheses make
**opposite** predictions, which is what makes this worth running:

| hypothesis | at low temperature | reasoning |
|---|---|---|
| **narrowing** | degrades *faster* | tails are cut at generation time, so the child never sees them and is badly surprised by real tail tokens. Cross-entropy punishes confident wrongness hard |
| **structure** | degrades *slower* | low-temperature text is more grammatical and more Shakespeare-like on the surface, so it teaches local structure better |

**My call: faster, at roughly 60% confidence.** I expect the tail-loss effect to
dominate, because bits-per-byte is an average over *all* real tokens including
rare ones, and a model trained on tail-free data assigns them near-zero
probability — which is enormously expensive in log loss. But I hold this
loosely; the structure argument is not silly, and low-temperature text really is
closer to Shakespeare to read.

**Secondary:** if narrowing is the mechanism at low temperature, `replace`'s
*training* loss at temp 0.5 should be **lower** than at temp 1.0 — the child
fits a sharper, easier distribution. That would be prediction 5 from E3 finally
coming true, just at a temperature I did not test.

### What would change my mind

- **Diversity flat across temperatures** — the metrics are broken, and R6's
  mechanism claim collapses with them.
- **BPB flat across temperatures** — temperature does not select the failure
  mode; something else explains R6, and I have no candidate for what.
- **Non-monotonic BPB** — would suggest two competing effects of comparable
  size, which would be the most interesting outcome and the hardest to write up
  honestly.

### Result

Full write-up: [results.md R7](results.md#r7--sampling-temperature-selects-the-collapse-failure-mode-).

**All three predictions correct**, and the central one by a much larger margin
than expected.

| regime | drift over 3 gens | vocabulary (g1) | training loss (g2) |
|---|---:|---:|---:|
| T0.5 | **+1.849** | 36.9% | 1.04 |
| T1.0 + top-k 40 | +0.552 | 46.7% | 2.91 |
| T0.8 | +0.630 | 80.6% | 2.88 |
| T1.0 | +0.168 | 87.4% | 4.44 |

**The refinement I did not predict, and the better finding:** it is not
temperature. Top-k 40 *at temperature 1.0* degrades the model more than
temperature 0.8 does, without touching the temperature. Truncation and
temperature act through one channel — how much of the distribution's tail
survives sampling — and collapse tracks that, not the temperature parameter.

**Practically useful:** generated-corpus vocabulary coverage at generation 1
predicts generation-3 BPB monotonically. It is an early-warning signal available
a full generation before the damage shows in held-out loss, and it costs a token
count over a corpus the loop already has.

**Amends E3.** Prediction 5 there ("training loss falls while held-out rises")
was recorded as falsified. It was *regime-dependent*, not false — true at T0.5,
false at T1.0. R6's scorecard now says so.

**Honest note on the 3/3.** The mechanism was already half-identified from R6
before these predictions were written, so this is a weaker test of foresight
than the score suggests. E3's 2/5 on genuinely open questions was the more
informative experiment.

**Next:** does accumulation rescue a low-temperature lineage? If keeping real
data prevents the catastrophic case, that is the practically important result.

---

## E5 — Does accumulating real data rescue a collapsing lineage?

**Date:** 2026-09-14 · **Status:** prediction recorded before the experiment ran

### Question

[R6](results.md) showed accumulation slows collapse to ~44% of `replace`'s
excess at temperature 1.0, where the damage is mild. [R7](results.md) showed
that at temperature 0.5 the damage is *catastrophic* — 4.53 BPB, worse than the
bigram baseline, with the corpus down to 11% of the vocabulary.

The practically important question is whether accumulation still helps when
things are actually going wrong. A brake that works only on gentle slopes is not
a brake.

### Setup

One missing cell: **`accumulate` at temperature 0.5**, 3 generations × 3 seeds.
The other three cells already exist, giving a clean 2×2:

| | T1.0 | T0.5 |
|---|---|---|
| `replace` | 2.367 (R6) | 4.534 (R7) |
| `accumulate` | 2.187 (R6) | **this run** |

`control` is regime-independent at 2.046.

### The two hypotheses

**Proportional scaling.** Accumulation reduces the excess over control by a
fixed fraction regardless of severity. At 44%, that predicts
`2.046 + 0.44 × 2.488 = ` **~3.14** — still worse than the bigram baseline of
3.24, i.e. barely a rescue at all.

**Tail anchoring.** Accumulation should help *more* at low temperature, not the
same amount. The mechanism: at T0.5 the synthetic corpus covers only ~37% of the
vocabulary, so the retained real data is the only source of tail coverage and
contributes something the synthetic data structurally cannot. At T1.0 the
synthetic corpus already covers 87%, so real data adds far less unique signal.
If this is right the excess should shrink to well below 44%.

### Prediction

**Tail anchoring, ~55% confidence.** Generation-3 BPB in **2.3–2.8**, i.e. the
excess reduced to roughly 15–30% of `replace`'s rather than 44%.

Held loosely. The counter-argument is real: degenerate, highly repetitive text
may dominate the gradient out of proportion to its share of the corpus, in which
case dilution does not save you and the result lands at 3.1 or worse.

There is also a structural drag I already flagged in E3: `accumulate` dilutes
the real data from 50% to 25% across generations, so its real-data anchor weakens
exactly as it is needed most.

### What would change my mind

- **Result ≥ 3.1** — proportional scaling wins, accumulation is not a rescue but
  a discount, and the honest headline is that nothing tested so far prevents
  catastrophic collapse.
- **Result ≤ 2.2** — accumulation nearly eliminates the effect even at T0.5,
  which would be a stronger result than the accumulation literature claims and
  would need a fourth seed set before I believed it.
- **Higher variance across seeds than `replace` showed** — would suggest the
  outcome depends on which degenerate mode the parent fell into, making the mean
  much less meaningful than the spread.

### Result

*(filled in when the experiment completes — see [results.md](results.md) R8)*

---

## Template

```markdown
## E<n> — <question in one line>

**Date:** · **Status:** prediction recorded / complete

### Question
What do you want to know, and why does it matter for the project?

### Setup
Config, data, seeds, what varies and what is held fixed. One independent
variable.

### Prediction
What you expect, and roughly how strongly.

### What would change my mind
Concrete results that would falsify the prediction or invalidate the setup.

### Result
Numbers, with seed spread. Compare against the noise floor before concluding.

### What it means
Including "nothing — the effect was below the noise floor", which is a result.
```

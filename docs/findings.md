# Findings

What this project set out to do, what it found, what it got wrong, and what none
of it establishes.

The individual experiments are in [results.md](results.md) with their
pre-registrations in [lab-notebook.md](lab-notebook.md). This is the synthesis.

---

## The project

Train a language model from scratch — own tokenizer, own transformer, own
training loop — on a laptop with no GPU, and use it to answer a research
question. Then wrap it in a bounded, unattended self-improvement loop.

The enabling constraint was corpus design. A general web corpus makes this
impossible without a datacentre; a narrow one makes it a weekend project. Every
architectural decision follows from targeting **1–50M parameters on a narrow
distribution** rather than a general-purpose model.

**Scale reached:** 1,049,216 parameters, 1.87 bits-per-byte on held-out
TinyShakespeare, about two minutes of CPU training. The bigram floor is 3.24, so
context is worth 42% with a quarter of the parameters.

---

## What it found

### 1. Self-training degrades a model, reproducibly

Training a model on its own output, three generations deep, costs **+0.32 BPB**
against a control — 80× the measured noise floor, monotonic across three
independent seeds and matching in magnitude, not just direction
([R6](results.md)).

### 2. But *how* it degrades depends entirely on the sampling regime

This is the finding I did not expect and consider the most interesting.

| regime | vocabulary (gen 1) | held-out BPB (gen 3) | mechanism |
|---|---:|---:|---|
| T1.0 | 87.4% | 2.37 | error accumulation |
| T0.8 | 80.6% | 2.86 | — |
| T1.0 + top-k 40 | 46.7% | 3.23 | narrowing |
| T0.5 | 36.9% | **4.53** | narrowing (severe) |

At temperature 1.0 the generated corpus gets **more** diverse while the model
degrades — the classic "model collapse" narrowing is absent, and what happens
instead is error accumulation. At temperature 0.5 the corpus collapses to 11% of
its vocabulary and the model lands **worse than a bigram lookup table**
([R7](results.md)).

And it is not temperature as such: **top-k 40 at temperature 1.0 does more damage
than temperature 0.8**, without touching the temperature. The variable is how
much of the distribution's tail survives sampling.

> **The practical lesson runs against instinct.** Generating with low temperature
> or top-k produces cleaner, more grammatical, better-looking synthetic text —
> and kills the lineage faster. Sampling that preserves the tails looks worse and
> survives.

### 3. Real data anchors a lineage, and a little goes a long way

| real fraction | BPB (gen 3) | damage recovered |
|---:|---:|---:|
| 0% | 4.53 | — |
| **25%** | 2.32 | **89%** |
| 50% | 2.17 | 95% |
| 100% | 2.05 | 100% |

The first quarter of real data recovers **89%** of the damage; the remaining
three quarters buy the last 11% ([R9](results.md)). A lineage retaining real data
is also nearly **immune to the sampling regime** — `replace` swings 2.17 BPB
across temperatures, `accumulate` swings 0.023 ([R8](results.md)).

Both the *amount* and the *fraction* matter, and neither alone explains the
results.

### 4. Lower training loss predicted *worse* real-world performance

Across sampling regimes, the arm that fit its training data four times better
ended up twice as bad on held-out real text ([R7](results.md)).

**This is the safety-critical finding.** A self-improvement loop gating on
training loss would rank these regimes exactly backwards and select the
catastrophic one. The held-out gate is not a nicety; it is the only thing between
such a loop and confidently optimising itself into uselessness.

### 5. A bounded loop works, and rediscovers its own constraints

25 iterations improved an under-trained model from 2.17 to 1.86 BPB, promoting
10 candidates and refusing 15 ([R10](results.md)).

The result worth reporting is not the improvement. It is that **rejections
clustered monotonically in synthetic fraction** — 40% rejected at 0% synthetic,
57% at 25%, 88% at 50%. Nobody told the loop that synthetic data hurts. It found
out by having candidates refused, reproducing the dose-response relationship R9
had measured directly.

That vindicates a design decision made before any of the experiments existed:
*rejections are recorded, not discarded, because they are the research data.* A
loop logging only its promotions would have produced the identical model and none
of this.

### 6. No induction circuit appeared, at any point

Across 25 loop iterations the induction score stayed between −0.52 and −0.87
bits — negative throughout, meaning the second copy of a repeated random sequence
was *harder* to predict than the first.

A 1M-parameter model on 373k tokens of Shakespeare appears to form no in-context
copying circuit at all. Held-out loss would never have surfaced this, which is
precisely the argument for capability probes.

---

## What I got wrong

Every experiment was pre-registered with predictions and falsification criteria
before it ran. The record:

| experiment | score | what the misses taught |
|---|---|---|
| E1 (noise floor) | marginal | The model was *more* seed-stable than expected |
| E3 (collapse) | **2 of 5** | Both failures assumed the narrowing mechanism — and their failure is what located the actual one |
| E4 (temperature) | 3 of 3 | Weak test: the mechanism was half-known from E3 before these were written |
| E5 (accumulation) | mechanism right, **magnitude wrong** | Predicted 2.3–2.8, measured 2.21 |
| E6 (dose-response) | direction right, **mechanism incomplete, magnitude wrong** | Predicted 2.5–3.0, measured 2.32 |
| R6–R9 re-run | conclusions right, **magnitude wrong** | Predicted all cells move < 0.0040; six of nine moved 2.8–61× that |
| Benchmark correction | direction right, **magnitude wrong** | Measured the architecture ratio on one machine, predicted 2–3× on another; it was 1.3–1.5× |

**A systematic bias, named rather than buried:** five occasions running, my
magnitude predictions were wrong in the *same* direction — I under-estimate how
much a number can move while its meaning stays put. In E5 and E6 that showed up
as **over-estimating how much damage self-training does once any real data is
present**; in the re-run it showed up as over-estimating the stability of
individual cells. Directions and mechanisms have held every time. Sizes have
not held once.

The benchmark correction is the clearest instance of the mechanism. I measured
a real effect — the benchmark timed the wrong architecture — on a Linux
container, then extrapolated the *ratio* to an Intel Mac running a different
torch, and stated a conclusion ("an evening, not an overnight") off that
extrapolation. I had already written down the right instruction: re-run the
benchmark on the target machine. Then I quoted a number anyway. **The failure
is not the estimate; it is quoting an estimate when the measurement was two
minutes away and already scheduled.**

The R6–R9 re-run is the sharpest instance of self-contradiction, because I
falsified it myself within the same breath. Having predicted "magnitudes move less than the noise floor", I
immediately added a caveat — that arms had not shared an initialisation, so some
archived variance was init noise — which directly contradicted the prediction I
had just made. The caveat was right. **The lesson is not to predict better; it
is that a stated prediction plus an honest caveat beats a confident prediction,
and the caveat should have replaced the claim rather than trailing it.**

**The strongest argument for pre-registration in this project** is E3. Predictions
4 and 5 were both downstream of assuming classic model collapse. Had I not
written them down beforehand, I would have reported "collapse confirmed", never
examined the diversity numbers, and missed that the mechanism was something else
entirely.

---

## Methodology that earned its place

**The noise floor.** Five seeds of one configuration: **0.0040 BPB**
([R5](results.md)). Every comparison in this project is judged against it, and
`compare_groups` refuses to call anything below it an effect.

**Underpowered ≠ null.** A permutation test with three seeds per arm enumerates
C(6,3)=20 splits, so its smallest possible p-value is 0.10 — it *cannot* reach
significance. The code reports that as UNDERPOWERED rather than "not
significant", after a 75×-noise-floor effect was being described as the latter.

**Compute-matched, not step-matched.** The first draft of R10 had no control at
all, which would have made "the loop improved the model" indistinguishable from
"the model trained longer".

---

## Limitations

Stated plainly, because several are load-bearing.

- **One corpus, 1.1 MB.** Everything rests on TinyShakespeare. It is a smoke
  test, not a destination. TinyStories (~2 GB) is the corpus the project was
  designed around and has not been run.
- **One model size**, ~1M parameters, and one architecture.
- **Three seeds per arm**, so conclusions rest on effect sizes (17–622× the noise
  floor) rather than on p-values, which cannot be computed at that sample size.
- **Three generations** in the collapse experiments. Where the trajectories
  settle is unknown.
- **The parent models are weak** — 500-step, 1M-parameter generators. A stronger
  generator would produce better synthetic data and might well show a different
  mechanism.
- **The 0–25% region of the dose-response curve is unsampled**, and it is where
  the curve is steepest. The knee could be at 5%.
- **The loop ran once**, from a deliberately under-trained incumbent, with random
  search over an 18-candidate space sampled 1.4 times over.
- **Wall-clock favours plain training.** The loop buys its advantage with search,
  and search costs.

---

## Reproducing this

```bash
git clone <repo> && cd gitAI
make quickstart     # setup + data + test + train
```

Verified from a clean clone: setup, data, the full test suite, and the
improvement loop all run end to end.

| | command | time |
|---|---|---|
| Train a model | `make train` | ~2 min |
| Look inside it | `make inspect` | seconds |
| Noise floor | `make noise-floor` | ~10 min |
| Collapse experiment | `make collapse` | ~35 min |
| Temperature sweep | `make collapse-temps` | ~35 min |
| The loop | `python scripts/run_loop.py --iterations 25` | ~11 min |

Every run writes a manifest with its git SHA, data hash, tokenizer fingerprint
and seeds. The corpus is not in the repository — it is fetched and rebuilt
locally against a committed checksum.

---

## What I would do next

1. **TinyStories.** Every number above is provisional until it is re-measured on
   a corpus that is not 1.1 MB.
2. **Five seeds**, so the statistics can certify what the effect sizes already
   show.
3. **The 0–25% dose-response region**, where the curve is steepest.
4. **A stronger parent model** for the collapse experiments — the mechanism may
   be an artefact of a weak generator.
5. **Does induction ever appear?** More layers, more data, a longer context. The
   answer is a real contribution either way.

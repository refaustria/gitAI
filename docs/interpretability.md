# Looking Inside the Model

> *"Once the model runs, can I see its reasoning to understand the thinking approach?"*

Short answer: **not a reasoning trace — something better, and the small scale is
why.**

---

## Why there is no reasoning trace

A ~1M parameter model trained on TinyShakespeare does not deliberate. It
predicts next tokens. Chain-of-thought is a behaviour of very large
instruction-tuned models, produced by *generating more tokens* rather than by
exposing an internal process. Prompting this model to "think step by step" would
produce more blank verse.

There is a sharper point underneath. Even where a model *does* emit a written
reasoning trace, that trace is not a guaranteed account of the computation that
produced the answer — a model can give reasoning that does not reflect the
actual causes of its output. So "show me its reasoning in words" is a **weaker**
form of understanding than the one available here.

## What you get instead

At this scale every intermediate is observable, exhaustively, on a laptop. This
is exactly why mechanistic interpretability research uses small models: frontier
models are too large to inspect properly. The project's binding constraint is
the field's preferred experimental condition.

Because we wrote the model ourselves, instrumentation needs no hooks or
monkey-patching — the forward pass simply hands over what it already has:

```python
logits, loss, cache = model.run_with_cache(ids)
```

When no cache is passed the cost is a handful of `is not None` checks.

---

## The four levels

| Level | Module | Question | Causal? |
|---|---|---|---|
| Prediction | `interpret.surprisal` | Where was it confident, where surprised? | — |
| Forming | `interpret.lens` | How did the prediction assemble across depth? | No |
| Attribution | `interpret.attribution` | Which component contributed this logit? | No |
| Intervention | `interpret.patching` | Which component **caused** it? | **Yes** |

Only the last supports claims about what mattered. The first three report what
the model computed; components can cancel, and a component that contributes may
still be replaceable.

Run all of it on a trained checkpoint with:

```bash
make inspect
```

---

## 1. Surprisal — the loss, decomposed

Surprisal is `-log2 p(actual token)`: 0 bits is certainty, 1 bit is a coin flip,
10 bits is being blindsided. Mean surprisal **is** the cross-entropy loss in a
different unit, and dividing by bytes-per-token gives bits-per-byte. Three views
of one number — which is why this is the right thing to look at first. It shows
*where* the loss was incurred, not merely how much.

From the trained 1M-parameter model on `"First Citizen:\nBefore we proceed any
further, hear me"`:

| pos | context | actual | bits | rank | top predictions |
|---:|---|---|---:|---:|---|
| 1 | `·Citizen` | `:` | 0.17 | 1 | `':'`:0.89 |
| 2 | `:` | `⏎` | **0.02** | 1 | `'⏎'`:0.99 |
| 7 | `·pro` | `ceed` | 1.72 | 1 | `'ceed'`:0.30 |
| 8 | `ceed` | `·any` | **12.19** | 291 | `','`:0.23 `':'`:0.06 |
| 10 | `·fur` | `ther` | **0.14** | 1 | `'ther'`:0.91 |

Three things are legible immediately. The model has *completely* learned the
corpus format — a colon after a speaker name is followed by a newline with 99%
confidence. It has learned subword completion: given `·fur` it says `ther` at
91%. And position 8 is the interesting failure: after "proceed" it confidently
expected punctuation, and was wrong by 12 bits. Not ignorance — a learned rule
that did not apply here. `TokenPrediction.confident_and_wrong` flags exactly
this case.

## 2. Logit lens — watching the prediction form

The residual stream is the model's working memory: every block *adds* to it and
none overwrite. So the state at layer 2 is a partial answer expressed in the
same space as the final one, and decoding it with the output head shows what the
model would have predicted had it stopped there.

At position 8, where the model got it wrong:

| layer | p(`·any`) | rank | top predictions |
|---|---:|---:|---|
| embed | 0.0000 | 1568 | noise |
| block 0 | 0.0000 | 762 | `'ings'`:0.36 `'ed'`:0.14 `'ing'`:0.13 |
| block 1 | 0.0001 | 617 | `'ings'`:0.22 `'ing'`:0.22 |
| block 2 | 0.0000 | 469 | `'ed'`:0.20 `'ings'`:0.09 |
| block 3 | 0.0002 | 291 | `','`:0.23 `':'`:0.06 |

A visible change of strategy. Blocks 0–2 treat `ceed` as a word *fragment* and
try to continue it morphologically — `ings`, `ed`, `ing`. Only at block 3 does
the model recognise "proceed" as a complete word and switch to predicting
punctuation. The correct answer climbs steadily the whole way (rank 1568 → 291)
without ever getting close.

That is a real, mechanical account of how this prediction was built. No
narration involved.

**The caveat that must travel with it:** `norm_f` was trained on final-layer
activations, so applying it to intermediate layers is an approximation. Early
rows are often noise. Treat a clean trajectory as evidence and a messy one as
uninformative — and confirm anything important with patching.

## 3. Direct logit attribution — exact, not approximate

Because the residual stream is additive and the final norm is affine once its
per-position scale is taken from the real forward pass, one logit decomposes
**exactly** into one number per component:

```
component        logit
L3.attn          0.969   ██████████████████████
L0.mlp          -0.749  ◀     █████████████████
L1.mlp          -0.284  ◀                ██████
embed            0.130   ██
...
sum             -0.279
actual logit    -0.279     ✓
```

Exactly, not approximately: `test_attribution_sums_to_the_logit` asserts the
parts reconstruct the whole, for both norm types. **An attribution method whose
parts do not add back to what it explains is telling you a story.**

`attribute_heads` splits this per attention head, because attention output is
`concat(heads) @ W_O` and that matmul is a sum over head-sized slices. This is
the tool for finding induction heads (Decision 12, question D).

## 4. Interventions — the causal level

**Head ablation.** Zero each head, measure the loss increase. On the trained
model, all 16 heads matter, led by L2H1 at +0.156. A caveat: ablating one head
at a time misses redundancy — two heads computing the same thing both look
useless.

**Activation patching.** The sharpest tool. Run clean, run corrupted, splice one
clean activation into the corrupted run, and see whether the output recovers.

Corrupting position 4 and measuring at position 8:

| layer | pos 4 | 5 | 6 | 7 | 8 | recovered |
|---|---:|---:|---:|---:|---:|---:|
| L0 | -5.70 | -5.39 | -5.27 | -5.22 | **-5.56** | 84% |
| L1 | -5.77 | -5.48 | -5.24 | -5.22 | **-5.36** | 84% |
| L2 | -5.31 | -5.36 | -5.34 | -5.39 | -5.68 | 33% |
| L3 | -5.37 | -5.37 | -5.37 | -5.37 | -5.55 | 0% |

The information travels through the residual stream at the final position in the
early layers; by L3 patching recovers nothing, because the computation has
already happened.

> A trap worth naming: corrupting the *same* position you measure makes patching
> trivially recover 100% at every layer and shows nothing. Corrupt upstream, so
> the information has to travel. The first version of `inspect_model.py` got
> this wrong.

---

## Attention: read with care

`cache.attention(layer)` gives the `(batch, head, query, key)` weights, and
`attention_heatmap` renders them. In the trained model, layer 0 head 0 attends
almost entirely to itself while head 1 attends back to the `First Citizen:`
prefix.

But **attention weights are suggestive, not explanatory.** A head attending to a
token does not prove that token caused the output. This is a well-known trap.
When the claim matters, patch.

---

## The loop's reasoning is a separate question

If the question is instead *why the improvement loop made the choices it made* —
that is fully auditable by design and already built. The hash-chained lineage
records every promotion **and every rejection**, with the evidence and the
invariant results that permitted or blocked it. "Why is this the current best
model?" has an answer you can verify rather than believe. See
[constitution.md](constitution.md) and [self-improvement.md](self-improvement.md).

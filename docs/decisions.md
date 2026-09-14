# Fundamental Decisions

Every decision below is stated as: **options → trade-offs → recommendation →
status**. Nothing here is binding until its status is `Decided` and an ADR
exists in [`adr/`](adr/).

Status values: `Proposed` (my recommendation, awaiting your call) ·
`Decided` (settled, ADR written) · `Deferred` (deliberately not deciding yet) ·
`Open` (needs your input, no default)

---

## Summary table

| # | Decision | Recommendation | Status |
|---|---|---|---|
| 1 | Language & framework | Python 3.12 + PyTorch 2.x, preceded by a NumPy autograd warm-up | Proposed |
| 2 | Dependency management | `uv` | Proposed |
| 3 | Model architecture | Llama-style decoder-only, built incrementally | Proposed |
| 4 | Tokenizer | Own byte-level BPE, small vocab (4k–8k) | Proposed |
| 5 | Configuration | Dataclass schema + YAML + CLI override; no Hydra | Proposed |
| 6 | Experiment tracking | Local JSONL + run manifest as source of truth; W&B optional on top | Proposed |
| 7 | Storage / "database" | Four stores for four jobs — see [data-and-storage.md](data-and-storage.md) | Proposed |
| 8 | Compute strategy | Laptop-first, cloud-burst, device-agnostic code | Proposed |
| 9 | Code quality & testing | ruff + pytest + pyright + pre-commit + CPU-only CI | Proposed |
| 10 | Primary metric | Bits-per-byte, not perplexity | Proposed |
| 11 | Corpus | TinyShakespeare → TinyStories → FineWeb-Edu sample | Proposed |
| 12 | **Research question** | **F — when does a self-improvement loop help vs. collapse?** | **Decided** |
| 13 | Licence | MIT for code; per-dataset tracking for data | Proposed |
| 14 | Self-improvement scope | The *loop* improves, not the model; weights/data/config only | **Decided** |
| 15 | Code immutability | The loop may never modify its own source. Absolute | **Decided** |

---

## 1. Language & core framework

**Options**

| Option | For | Against |
|---|---|---|
| **PyTorch 2.x** | One codebase runs CPU / Apple MPS / CUDA; every reference implementation you'll learn from (nanoGPT, litGPT, Hugging Face) is PyTorch; best debugger and error messages; `torch.compile` gives real CPU speedups | Slightly less elegant than JAX for functional transforms |
| JAX + Flax | Beautiful `vmap`/`grad` composition, excellent for research | Payoff is on TPU/XLA, which you don't have; tracer errors are brutal to debug; smaller ecosystem for LLM reference code |
| NumPy only | Deepest possible understanding — you write backprop yourself | Far too slow for anything past a toy; no GPU path when you rent one |
| tinygrad / micrograd | Tiny, readable, educational | Not built for reproducible research; you'd fight the framework |
| Rust / C++ | Fast, no Python overhead | Enormous time cost, tiny ecosystem, wrong problem to solve first |

**Recommendation: PyTorch 2.x on Python 3.12** — *with one deliberate exception.*

Phase 0 includes writing a ~200-line reverse-mode autograd engine in NumPy and
training an MLP with it. This is not busywork: it is the only reliable way to
stop treating `.backward()` as magic, and it takes a weekend. After that,
everything real is PyTorch.

The deciding factor for PyTorch over JAX is **device portability**. You are on a
laptop now but will rent a GPU later (Decision 8). PyTorch code moves between
CPU, MPS and CUDA with a device string; that must be true from commit one.

> ⚠️ **CPU performance caveat to verify in Phase 0.** On x86, PyTorch CPU speed
> depends heavily on the BLAS/oneDNN backend; on Apple Silicon the MPS backend
> is dramatically faster than CPU for matmul but has gaps in operator coverage.
> Which laptop you have materially changes throughput. Phase 0 has a benchmark
> task precisely so every later estimate is measured, not guessed.

**Status: Proposed**

---

## 2. Dependency & environment management

**Options:** `uv` · Poetry · pip + venv + requirements.txt · conda/mamba · pixi

**Recommendation: `uv`.** Resolution is 10–100× faster than Poetry, it manages
Python versions itself, produces a real lockfile, and is a single static binary.
`uv.lock` committed to the repo is your reproducibility floor.

**Escape hatch:** if you hit BLAS/MKL pain on x86 CPU, conda-forge sometimes ships
better-linked numerical builds. Revisit only if the Phase 0 benchmark is
disappointing — don't pre-optimise.

**Status: Proposed**

---

## 3. Model architecture

**Options:** GPT-2 vanilla (LayerNorm, learned positions, GELU, biases) ·
**Llama-style** (RMSNorm, RoPE, SwiGLU, no biases) · state-space (Mamba) ·
encoder-decoder.

**Recommendation: Llama-style decoder-only, built incrementally.**

The modern components cost the same effort to implement and are strictly better
at small scale:

| Component | Instead of | Why |
|---|---|---|
| RMSNorm | LayerNorm | Fewer ops, no mean subtraction, no measurable quality loss |
| RoPE | Learned positional embeddings | Extrapolates past training context; saves parameters that a small model badly needs |
| SwiGLU | GELU MLP | Better loss at equal parameter count (use 8/3× hidden ratio to compensate for the third matrix) |
| No biases | Biases everywhere | Free parameter savings, no quality cost |
| Weight tying (embed ↔ unembed) | Separate matrices | **Critical at this scale** — see vocab note in Decision 4 |
| Pre-norm | Post-norm | Trains stably without warmup gymnastics |

**Build it as a ladder, tagging each rung in git.** The ladder *is* the learning,
and each rung is independently debuggable:

`bigram` → `single-head attention` → `multi-head + MLP (GPT-2)` → `RMSNorm` →
`RoPE` → `SwiGLU` → `KV-cache inference`

Skip GQA and MoE — at 20M parameters they solve problems you don't have.

**Status: Proposed**

---

## 4. Tokenizer

**Options:** character-level · byte-level · **own BPE** · Hugging Face
`tokenizers` · SentencePiece/Unigram.

**Recommendation: write your own byte-level BPE, validated against Hugging Face
`tokenizers`.** It is ~250 lines, it is squarely in scope for "from scratch", and
tokenizer bugs are a classic silent killer. Use char-level in Phase 1 as a smoke
test so the training loop can be debugged before the tokenizer exists.

> 🔑 **The non-obvious decision: vocabulary size.** Instinct says copy GPT-2's
> 50,257. That is wrong here. At `d_model = 384`, a 32k vocab costs
> `32,000 × 384 ≈ 12.3M` parameters in the embedding table alone — which would be
> most of a 20M parameter model, spent on embeddings rather than computation.
>
> Target **4k–8k vocab**, and *tie* the input and output embeddings. Vocab size
> is not a detail here; it is one of the more interesting research variables you
> have (see Decision 12).

**Status: Proposed**

---

## 5. Configuration & experiment definition

**Options:** Hydra · OmegaConf · dataclasses + `tyro`/`simple-parsing` · plain
argparse · bare YAML.

**Recommendation: Python dataclasses as the schema, YAML files for experiment
definitions, CLI flags for overrides.** Avoid Hydra initially — it rewrites your
working directory, hides resolution behind magic, and makes stack traces hard to
read. You can adopt it later if sweep ergonomics genuinely hurt.

**Hard requirement regardless of choice:** every run writes its *fully resolved*
config into its own run directory. A config that lives only in your shell history
is a lost experiment.

**Status: Proposed**

---

## 6. Experiment tracking

**Options:** Weights & Biases · MLflow · TensorBoard · Aim · plain JSONL.

**Recommendation: local-first, cloud-optional.**

Every run writes, unconditionally, to `runs/<timestamp>-<name>/`:

- `config.yaml` — fully resolved
- `manifest.json` — git SHA, dirty-tree flag, dataset hash, tokenizer hash,
  hardware, library versions, wall-clock, random seeds
- `metrics.jsonl` — one JSON object per logged step
- `checkpoints/` — safetensors (**never pickle**)

W&B or TensorBoard then sits *on top* as a viewer. The reason for this ordering
is blunt: research that only exists inside a SaaS account is one quota change or
outage away from being unreproducible. The local directory is the record.

`manifest.json` is the highest-value file in the repo. Six months from now,
"which data and which code produced this curve?" is the only question that
matters, and it is unanswerable without it.

**Status: Proposed**

---

## 7. Storage — the "database" question

Short answer: **you do not want a database for training data**, and you *do* want
one for three other things. There are four separate storage problems here and
they have four different right answers.

Full reasoning in **[data-and-storage.md](data-and-storage.md)**. Summary:

| Concern | Choice |
|---|---|
| Raw corpora | Files on disk + checksummed manifest |
| Curation, dedup, filtering, stats | **DuckDB over Parquet** |
| Training-ready tokens | **Memory-mapped flat `uint16` shards** |
| Run metadata & eval results | **SQLite** + JSONL |

Postgres, pgvector and vector databases are **deferred** — they belong to a
product with retrieval, which this is not.

**Status: Proposed**

---

## 8. Compute strategy & infrastructure

**Recommendation: laptop-first, cloud-burst.**

- **Laptop** carries all development, debugging, unit tests, tokenizer training,
  data curation, and every small ablation. This is ~90% of the wall-clock work.
- **Rented GPU** (RunPod / Vast.ai / Lambda, hourly) is used only for
  confirmation runs and the largest scaling points. Renting beats buying at this
  usage level by a wide margin — verify current hourly rates when you get there,
  they move.
- **Hard rule:** no code path may assume a device. `device` is config, never a
  literal. Every test runs on CPU in CI.

**Phase 0 produces a benchmark script** reporting tokens/sec across model sizes on
*your* hardware. Everything downstream — how big a model, how many tokens, how
long a run — is then derived from measurement. Do not let me or anyone else hand
you throughput numbers for hardware we haven't measured.

**Status: Proposed**

---

## 9. Code quality, testing, CI

**Recommendation:** `ruff` (lint + format, replaces black/isort/flake8) ·
`pytest` · `pyright` in basic mode · `pre-commit` · GitHub Actions running the
CPU-only fast suite on every push.

The generic advice is not the valuable part. **These ML-specific tests are:**

| Test | Catches |
|---|---|
| **Overfit one batch** — 32 examples to ≈0 loss in <500 steps | Roughly 90% of all model bugs, in 30 seconds |
| **Causality** — perturbing token *t* must not change logits at positions < *t* | Off-by-one in the attention mask. This bug still trains to a plausible-looking loss curve, which is why it survives for weeks |
| **Tokenizer round-trip** — `decode(encode(s)) == s` over random Unicode | Byte-boundary and normalisation bugs |
| **Determinism** — same seed ⇒ same loss for N steps, to a tolerance seven orders of magnitude tighter than a real bug and seven looser than BLAS noise | Hidden nondeterminism that makes ablations meaningless |
| **Checkpoint resume** — save, reload, continue ⇒ identical trajectory | Optimiser state and RNG state not being saved |
| **Grad-accumulation equivalence** — `accum=4, bs=8` ≈ `accum=1, bs=32` | Loss-scaling errors |
| **Finite-difference gradient check** on the NumPy engine | Backprop maths errors |

**Status: Proposed**

---

## 10. Primary metric

**Recommendation: bits-per-byte (BPB) as the headline metric. Not perplexity.**

Perplexity is measured per *token*, so it is **not comparable across different
tokenizers**. Since vocabulary size is an explicit research variable here
(Decision 4), reporting perplexity would make your own experiments
incomparable with each other — a subtle, project-ruining mistake.

BPB normalises by raw bytes and is tokenizer-independent. Report perplexity too
if you like, but never compare across vocab settings with it.

Details and the rest of the eval suite: **[evaluation.md](evaluation.md)**.

**Status: Proposed**

---

## 11. Corpus

Since no domain is fixed, choose for tractability. A ladder, not a single pick:

| Stage | Corpus | Size | Role |
|---|---|---|---|
| 0 | TinyShakespeare | ~1 MB | Smoke test. Trains in minutes. Proves the loop works |
| 1 | **TinyStories** | ~2 GB | **Primary.** Purpose-built so tiny models produce coherent text |
| 2 | FineWeb-Edu sample / WikiText-103 | 1–10 GB | "Real" distribution. Harder, for data-quality experiments |
| 3 | Synthetic formal languages (modular arithmetic, Dyck words, copying) | generated | Clean mechanistic experiments where you control ground truth exactly |

Stage 3 deserves more attention than it usually gets: when you generate the data,
you know the true generating process, so you can ask whether the model learned
*the algorithm* rather than surface statistics. That is real research and it costs
almost no compute.

Record every dataset's licence and provenance in `data/LICENSES.md` before
downloading it.

**Status: Proposed**

---

## 12. Research question

"Train an LLM" is a project, not a research question. Research needs a question
whose answer you do not already know. Since compute is the binding constraint,
the good questions are the ones where *small scale is a feature*.

**Decided: question F**, which follows directly from the self-improvement
requirement rather than being chosen arbitrarily:

> **Under what conditions does a bounded self-improvement loop improve a small
> language model, and when does it collapse? Does accumulating real and
> synthetic data, rather than replacing real with synthetic, change the answer?**

This is a genuinely open question with a live literature and a real chance of a
negative result. Training a model on its own output degrades it — and crucially,
**training loss keeps falling while the model gets worse**, so a naive loop walks
into collapse while its own metrics applaud. There is a competing result holding
that data *accumulation* rather than *replacement* avoids collapse. That is
directly testable at laptop scale with the infrastructure this project builds
anyway.

Full design in [self-improvement.md](self-improvement.md).

The original shortlist is retained below, because each remains a good question
and several make excellent sub-experiments once the loop exists — **A** in
particular is the natural first thing for the loop to search over.

| # | Question | Effort |
|---|---|---|
| A | Depth vs width at a fixed parameter budget | Low |
| B | Vocabulary size vs model capacity at a fixed budget | Low |
| C | Do Chinchilla scaling laws hold at 1–50M params? | Medium |
| D | When do induction heads emerge? | Medium |
| E | Data quality vs quantity | Medium |

**Status: Decided**

---

## 13. Licensing

**Recommendation:** MIT for code. Data licences tracked per-dataset in
`data/LICENSES.md` — corpora carry their own terms and some forbid
redistribution, so never commit corpus content to git regardless.

**Status: Proposed**

---

## 14. Self-improvement scope

**What improves is the loop, not the model.** A 1–50M parameter model cannot
improve itself in the recursive sense — it cannot read its own source, reason
about its architecture, or write a correct patch. Building machinery for
recursive self-modification would mean elaborate plumbing around a component
incapable of using it, and the plumbing is what would then carry the bugs.

The loop is a bounded search procedure over **weights, data, and configuration**.
It proposes candidates, trains, evaluates on held-out data it cannot see, and
promotes only through a fail-closed invariant gate. That is self-improvement in
the only sense available at this scale, and — worth noting — in the sense that
has produced essentially every real advance in the field.

Rejected candidates are recorded, not discarded: the collapse boundary is
defined by where rejections cluster, so a loop that logs only its wins answers
no question at all.

**Status: Decided** — see [self-improvement.md](self-improvement.md)

---

## 15. Code immutability

**The loop may never modify its own source code.** Absolute, not a default.

- **May change:** model weights, data mixtures, generated data, hyperparameters,
  architecture configuration within a pre-declared search space.
- **May never change:** its own source, the invariants, the halt switch, the
  budget, the lineage log, or anything under `src/gitai/safety/`.

Code changes go through git, a diff, and a human. Every loop version that ever
ran is pinned by commit SHA in the lineage.

This is both the safety boundary and the honesty boundary. Recursive code
self-modification is where a self-improving system becomes genuinely hazardous,
and — far more relevantly here — it is not a capability a TinyStories-scale
model has. Pretending otherwise is cargo-culting, and the pretence is what would
let a real bug through.

**Status: Decided** — enforced per [constitution.md](constitution.md)

---

## Deliberately deferred

Not decided now, and that is on purpose. Revisit only when the named trigger
fires.

| Deferred | Trigger to revisit |
|---|---|
| Serving / inference API | A model good enough that someone wants to call it |
| Postgres / pgvector / vector DB | A retrieval product exists |
| Docker / Kubernetes | A second machine or collaborator |
| Distributed training (FSDP/DeepSpeed) | A single GPU is genuinely the bottleneck |
| Hydra, sweep orchestration | Manual sweeps become the slow part |
| Quantisation, distillation | A trained model needs to be smaller |
| Instruction tuning / RLHF | A base model worth aligning exists |
| The improvement loop itself | Phase 7 complete — a model worth improving exists. Its *constraints* are already built (Phase 0) |

Adding any of these before its trigger is the most common way a solo research
project dies: infrastructure grows faster than results.

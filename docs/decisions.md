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
| 12 | **Research question** | Needs your choice — shortlist below | **Open** |
| 13 | Licence | MIT for code; per-dataset tracking for data | Proposed |

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
| **Determinism** — same seed ⇒ bit-identical loss for N steps | Hidden nondeterminism that makes ablations meaningless |
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

## 12. Research question ⚠️ OPEN — needs your decision

"Train an LLM" is a project, not a research question. Research needs a question
with an answer that is not already known to you. Since compute is the binding
constraint, the good questions are the ones where *small scale is a feature*.

Shortlist, all genuinely tractable at 1–50M parameters on a laptop:

| # | Question | Why it works here | Effort |
|---|---|---|---|
| **A** | **Depth vs width at a fixed parameter budget.** Given exactly 10M params, is 4 layers × 512 dim better than 12 layers × 288? | Clean, cheap, a real open question at small scale, ~15 runs | Low |
| **B** | **Vocabulary size vs model capacity.** At a fixed total parameter budget, how should you split between embedding table and transformer blocks? | Directly follows from Decision 4; under-studied at small scale; you'd learn tokenizers deeply | Low |
| **C** | **Do Chinchilla scaling laws hold at 1–50M params?** Fit the compute-optimal token/parameter ratio on a laptop | Beautiful result if it holds, more interesting if it doesn't. Needs a careful grid | Medium |
| **D** | **When do induction heads emerge?** At what depth/width/data volume does in-context copying appear? | Mechanistic interpretability, needs small models to be tractable, visually compelling results | Medium |
| **E** | **Data quality vs quantity.** TinyStories vs filtered web at matched token counts | Practically important; the curation work is reusable | Medium |

**My recommendation: A or D.**

- **A** if you want a guaranteed publishable-quality result and a smooth
  on-ramp — it is the cheapest, the least likely to fail, and the
  infrastructure it forces you to build (sweep harness, seed variance, scaling
  plots) is exactly what every later question needs.
- **D** if you want the more intellectually exciting project. Induction heads are
  the clearest known example of a concrete, interpretable circuit appearing in a
  transformer, and watching one form in a model you wrote yourself is a genuinely
  rare experience. Higher variance, higher payoff.

You can start Phases 0–3 without answering this — the infrastructure is identical
either way. It must be answered before Phase 5.

**Status: OPEN**

---

## 13. Licensing

**Recommendation:** MIT for code. Data licences tracked per-dataset in
`data/LICENSES.md` — corpora carry their own terms and some forbid
redistribution, so never commit corpus content to git regardless.

**Status: Proposed**

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

Adding any of these before its trigger is the most common way a solo research
project dies: infrastructure grows faster than results.

# Roadmap

Phased plan for building a small language model from scratch and using it to do
research. Rationale for the choices below lives in
[docs/decisions.md](docs/decisions.md).

**Rules of engagement**

- Phases are sequential; tasks inside a phase often aren't.
- Each phase has an **exit criterion**. Don't advance until it's met — every one
  of them exists because skipping it costs more later.
- Time estimates assume part-time work and **no GPU**. They are estimates;
  Phase 0's benchmark replaces guesswork with measurement.

| Phase | Theme | Estimate |
|---|---|---|
| [0](#phase-0--foundations) | Foundations & understanding | 1–2 weeks |
| [1](#phase-1--data-pipeline) | Data pipeline | 1–2 weeks |
| [2](#phase-2--the-model) | The model | 2–3 weeks |
| [3](#phase-3--training-loop) | Training loop | 1–2 weeks |
| [4](#phase-4--evaluation-harness) | Evaluation harness | 1–2 weeks |
| [5](#phase-5--research) | Research | ongoing |
| [6](#phase-6--scale-up) | Scale-up (optional) | 1–2 weeks |
| [7](#phase-7--inference--write-up) | Inference & write-up | 1–2 weeks |

---

## Phase 0 — Foundations

*Goal: a working environment, and a genuine understanding of backpropagation.*

### Decisions to close

- [ ] Confirm or override the recommendations in [docs/decisions.md](docs/decisions.md) (1–11, 13)
- [ ] Write ADRs for each confirmed decision in `docs/adr/`
- [ ] Note which laptop this is (x86 vs Apple Silicon) — it changes throughput materially

### Repository & tooling

- [ ] `uv init`, Python 3.12, `src/` layout, package `gitai`
- [ ] Dependencies: `torch`, `numpy`, `duckdb`, `pyarrow`, `safetensors`, `pytest`, `ruff`, `pyright`
- [ ] `ruff` config (lint + format), `pyright` basic mode
- [ ] `pre-commit` hooks: ruff, ruff-format, trailing whitespace, `nbstripout`
- [ ] GitHub Actions: lint + typecheck + fast CPU test suite on push
- [ ] `Makefile` or `justfile`: `setup`, `test`, `lint`, `train`, `eval`
- [ ] Commit `uv.lock`

### Understand backprop (do not skip)

- [ ] Implement a reverse-mode autograd engine in NumPy (~200 lines): scalar/tensor `Value`, `+ * @ tanh relu`, topological-sort `backward()`
- [ ] Finite-difference gradient check against it
- [ ] Train an MLP on XOR, then MNIST, with your own engine
- [ ] Implement SGD, then Adam, from the paper — not from memory

> This is the one deliberately "inefficient" task in the plan. After it,
> `.backward()` stops being magic, and debugging a real model becomes tractable
> instead of mystical. It takes a weekend and pays for itself in Phase 2.

### Benchmark your hardware ⚠️

- [ ] `scripts/benchmark.py`: tokens/sec for several `(d_model, n_layer, seq_len)` on **your** machine
- [ ] Measure CPU vs MPS (Apple Silicon) or CPU vs CPU+oneDNN (x86)
- [ ] Measure with and without `torch.compile`
- [ ] Record results in `docs/hardware-baseline.md`
- [ ] Derive: largest model trainable in ~4h, in ~24h

> Every later scoping decision depends on this number. Do not accept throughput
> figures for hardware nobody has measured — including mine.

**Exit criterion:** your own autograd engine trains MNIST; `docs/hardware-baseline.md` contains measured numbers; CI is green.

---

## Phase 1 — Data pipeline

*Goal: raw text → memory-mapped token shards, reproducibly.*
*See [docs/data-and-storage.md](docs/data-and-storage.md).*

### Acquisition

- [ ] `data/` layout: `raw/`, `interim/`, `processed/` — all gitignored
- [ ] Download TinyShakespeare (Stage 0) and TinyStories (Stage 1)
- [ ] `data/raw/MANIFEST.json`: url, sha256, size, licence, retrieval date
- [ ] `data/LICENSES.md` — before redistributing anything

### Curation (DuckDB + Parquet)

- [ ] Convert raw → Parquet
- [ ] Corpus statistics notebook: length distribution, vocab coverage, character sets
- [ ] Exact-duplicate removal; near-duplicate detection (MinHash) — report how much there was
- [ ] Quality filters (length bounds, encoding errors, boilerplate); log rejection counts per filter
- [ ] Deterministic train/val/test split by document hash — **never random, never by line**
- [ ] Verify no leakage across the split (duplicate documents landing on both sides)

### Tokenizer

- [ ] Character-level tokenizer first — unblocks Phase 2/3 immediately
- [ ] Byte-level BPE from scratch: train, encode, decode, save/load
- [ ] Round-trip test over random Unicode, including emoji and CJK
- [ ] Validate token-for-token against Hugging Face `tokenizers` on a fixed corpus
- [ ] Train vocabs at 2k / 4k / 8k / 16k; record compression ratio (bytes per token) for each
- [ ] Decide the default vocab — remember the embedding-budget maths in [Decision 4](docs/decisions.md#4-tokenizer)

### Training shards

- [ ] Tokenize corpus → `uint16` `.bin` shards + `meta.json` (vocab size, dtype, token counts, tokenizer hash)
- [ ] `np.memmap` DataLoader: random offsets, configurable `seq_len`/`batch_size`
- [ ] **Profile the loader** — on CPU it will likely be the bottleneck, not the matmuls
- [ ] Assert reproducibility: same seed ⇒ identical batch sequence

**Exit criterion:** one command turns a fresh checkout into training shards; token counts and hashes are recorded; the loader's throughput is measured.

---

## Phase 2 — The model

*Goal: a correct decoder-only transformer, built one rung at a time.*
*Each rung gets a git tag — the ladder is the learning.*

### The ladder

- [ ] **v0 — bigram.** Lookup table. Establishes the loss baseline everything else must beat
- [ ] **v1 — single-head self-attention.** Write the causal mask by hand
- [ ] **v2 — multi-head + MLP + residual + LayerNorm.** This is GPT-2
- [ ] **v3 — RMSNorm** replaces LayerNorm
- [ ] **v4 — RoPE** replaces learned positional embeddings
- [ ] **v5 — SwiGLU** replaces the GELU MLP (use 8/3× hidden ratio)
- [ ] **v6 — remove biases, tie embeddings**

### Correctness tests (the highest-value code in the repo)

- [ ] **Overfit one batch**: 32 examples → ≈0 loss in <500 steps. Catches most bugs in 30 seconds
- [ ] **Causality**: perturb token *t*, assert logits at positions < *t* are bit-identical
- [ ] Shape and dtype contracts at every layer boundary
- [ ] Determinism: same seed ⇒ identical loss for N steps
- [ ] Parameter count matches a hand-derived formula (catches silent architecture errors)
- [ ] Compare v2 numerically against a reference GPT-2 implementation on identical weights

> The causality test deserves special attention. A mask off-by-one **still trains
> to a plausible loss curve** — the model just cheats. Without this test the bug
> can survive for weeks and quietly invalidate every result.

### Initialisation & numerics

- [ ] Scaled init (residual projections scaled by `1/sqrt(2 * n_layer)`)
- [ ] Verify activation and gradient statistics don't explode or vanish with depth
- [ ] Parameter-count and FLOPs calculators, embedding vs non-embedding split out

**Exit criterion:** v6 passes every test above and overfits a single batch reliably.

---

## Phase 3 — Training loop

*Goal: runs that are resumable, reproducible, and fully recorded.*

### Core loop

- [ ] AdamW with correct weight-decay grouping (no decay on norms, biases, embeddings)
- [ ] Cosine LR schedule with linear warmup
- [ ] Gradient clipping
- [ ] Gradient accumulation — **test equivalence**: `accum=4, bs=8` ≈ `accum=1, bs=32`
- [ ] Mixed precision where the device supports it (bf16 on MPS/CUDA; measure on CPU before assuming a win)
- [ ] Periodic eval on held-out data, with sample generation

### Run infrastructure

- [ ] `runs/<timestamp>-<name>/` created per run
- [ ] Resolved `config.yaml` written at start
- [ ] `manifest.json`: git SHA, dirty flag, data hash, tokenizer hash, hardware, versions, seeds
- [ ] `metrics.jsonl` streamed per log step
- [ ] Checkpoints in **safetensors**, never pickle
- [ ] Checkpoint rotation — a full laptop disk mid-run is a real and infuriating failure
- [ ] **Resume**: reload model, optimiser state, LR schedule position, RNG state, data position
- [ ] Test resume: interrupt at step N, resume, assert trajectory matches an uninterrupted run

### Ergonomics

- [ ] Dataclass config schema + YAML + CLI override
- [ ] Sensible progress output: loss, LR, tokens/sec, ETA, grad norm
- [ ] Optional W&B/TensorBoard viewer layered on top of the local JSONL

**Exit criterion:** a multi-hour TinyStories run completes, survives a deliberate interruption and resume, and generates recognisably English text.

---

## Phase 4 — Evaluation harness

*Goal: turn training runs into comparable measurements.*
*See [docs/evaluation.md](docs/evaluation.md).*

- [ ] **Bits-per-byte** implementation (primary metric) + unit test against a known value
- [ ] Perplexity and token accuracy (clearly labelled tokenizer-dependent)
- [ ] Fixed-prompt, fixed-seed generation dumped to file at every eval
- [ ] Synthetic probes: copying, sorting, modular arithmetic, Dyck
- [ ] Optional LLM-as-judge scoring for grammar / consistency / creativity
- [ ] `runs/` → `runs/index.db` ingest script (SQLite, rebuildable from scratch)
- [ ] `report.py`: comparison tables and plots **generated from the database**, never by hand
- [ ] Eval-suite version stamped into every result

### Methodology setup ⚠️

- [ ] **Measure the seed noise floor**: 5 seeds, one config, record the spread in final val loss
- [ ] Write that number into `docs/hardware-baseline.md` and treat it as the significance threshold from then on
- [ ] Start `docs/lab-notebook.md` — question, prediction, and falsification criterion, written *before* each experiment

**Exit criterion:** one command produces a seed-averaged comparison table across N runs; the noise floor is a known number.

---

## Phase 5 — Research

*Goal: answer the question. This is the point of everything above.*

### Blocking

- [ ] **Choose the research question** — [Decision 12](docs/decisions.md#12-research-question--open--needs-your-decision) is still open (my recommendation: A or D)

### Then

- [ ] Write the experimental design: variables, controls, grid, seeds, success criteria
- [ ] Pre-register predictions in `docs/lab-notebook.md`
- [ ] Sweep runner (a for-loop over configs is fine; resist building a framework)
- [ ] Run the grid, ≥3 seeds per point
- [ ] Analyse against the noise floor — discard every effect smaller than it
- [ ] Plot results; write them up in `docs/results.md`
- [ ] **Record negative results with equal care**
- [ ] Iterate: the first grid usually reveals the question was slightly wrong

**Exit criterion:** `docs/results.md` answers the question with seed-averaged evidence, or states clearly why the question was unanswerable at this scale.

---

## Phase 6 — Scale-up (optional)

*Goal: confirm laptop-scale findings hold one order of magnitude up.*

- [ ] `torch.compile`, fused optimisers, attention kernel improvements — measure each, keep only what wins
- [ ] Verify the code runs unchanged on CUDA (device-agnostic from day one, so this should be a config change)
- [ ] Cost out a rented GPU run; check current hourly prices, they move
- [ ] Run the largest scaling point on rented hardware
- [ ] Confirm the Phase 5 conclusion survives the scale increase — **or report that it doesn't, which is a more interesting result**

**Exit criterion:** the headline finding is tested at ~10× scale.

---

## Phase 7 — Inference & write-up

- [ ] KV-cache inference (and a test that cached and uncached generation agree exactly)
- [ ] Sampling: temperature, top-k, top-p, with seeded reproducibility
- [ ] A small CLI or notebook for interacting with a trained model
- [ ] Export final weights + tokenizer + model card
- [ ] Write-up: method, results, negative results, limitations, reproduction instructions
- [ ] Verify reproducibility end-to-end from a clean checkout
- [ ] Consider publishing weights to Hugging Face

---

## Things that will bite you

Collected failure modes, written down now so they're recognisable later.

| Trap | Mitigation |
|---|---|
| **Attention mask off-by-one** — still trains to a plausible loss; model cheats | The causality test, Phase 2 |
| **Data loader is the CPU bottleneck**, not the matmuls — opposite of GPU intuition | Profile it in Phase 1 |
| **Seed noise exceeds your effect size** — false discoveries that look clean | Measure the noise floor before experimenting, Phase 4 |
| **Two variables changed in one run** — zero information about either | One independent variable per experiment, always |
| **Perplexity compared across vocab sizes** — silently meaningless | Bits-per-byte as the primary metric |
| **Checkpoints fill the laptop disk** mid-run | Rotation from the start, Phase 3 |
| **`pickle` checkpoints** — brittle across versions, unsafe to share | safetensors |
| **Notebooks as source of truth** — unreviewable, unrunnable, undiffable | Notebooks explore; `src/` is truth; `nbstripout` in pre-commit |
| **Comparing against ChatGPT** — a category error that hides real results | Compare against small models on similar data |
| **Infrastructure outgrows results** — the classic solo-project death | The deferred list in decisions.md; respect the triggers |
| **Retrofitting a story onto whatever happened** | Pre-register predictions before each run |
| **Vocab copied from GPT-2 (50k)** — embeddings eat the parameter budget | 4k–8k, tied embeddings, Decision 4 |

---

## Open questions for you

1. **[Decision 12](docs/decisions.md#12-research-question--open--needs-your-decision) — the research question.** Blocks Phase 5, nothing earlier. My recommendation: **A** (depth vs width at fixed budget) for the safe, guaranteed-result path; **D** (induction-head emergence) for the more exciting one.
2. **Which laptop?** x86 vs Apple Silicon changes achievable scale substantially.
3. **Time budget per week?** Determines whether the phase estimates are realistic.
4. **Any appetite for spending money on rented GPU?** Changes whether Phase 6 is real or theoretical.

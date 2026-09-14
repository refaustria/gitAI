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
| [1](#phase-1--data-pipeline) | Data pipeline | ✅ done |
| [2](#phase-2--the-model) | The model | ✅ done |
| [3](#phase-3--training-loop) | Training loop | done |
| [4](#phase-4--evaluation-harness) | Evaluation harness | ✅ done |
| [5](#phase-5--research) | Research | first result |
| [6](#phase-6--scale-up) | Scale-up (optional) | 1–2 weeks |
| [7](#phase-7--inference--write-up) | Inference & write-up | 1–2 weeks |
| [8](#phase-8--the-self-improvement-loop) | The self-improvement loop | demonstrated |

---

## Phase 0 — Foundations ✅

*Goal: a working environment, and a genuine understanding of backpropagation.*

**Status: complete.** 96 tests green, ruff clean, CI running on 3.11 and 3.12.
The one task still outstanding is the hardware benchmark, which can only be run
on your own laptop — see below.

### Decisions to close

- [ ] Confirm or override the recommendations in [docs/decisions.md](docs/decisions.md) (1–11, 13)
- [ ] Write ADRs for each confirmed decision in `docs/adr/`
- [ ] Note which laptop this is (x86 vs Apple Silicon) — it changes throughput materially

### Repository & tooling

- [x] `uv init`, Python 3.11+, `src/` layout, package `gitai`
- [x] Dependency groups: base (numpy only), `[train]`, `[data]`, `[dev]` — Phase 0 deliberately cannot import torch
- [x] `ruff` config (lint + format), `pyright` basic mode
- [x] `pre-commit` hooks: ruff, ruff-format, whitespace, `nbstripout`, large-file block
- [x] GitHub Actions: lint + format check + CPU test suite, Python 3.11 and 3.12
- [x] `Makefile`: `setup`, `test`, `lint`, `fmt`, `typecheck`, `bench`, `demo`, `halt`
- [x] Commit `uv.lock`

### Understand backprop (do not skip)

- [x] Reverse-mode autograd engine in NumPy — `src/gitai/autograd/tensor.py`. Broadcasting-correct backward, iterative topo sort (a 5000-deep graph would blow the recursion limit otherwise)
- [x] Finite-difference gradcheck — `gradcheck.py`, applied to every primitive
- [x] MLP solves XOR with the hand-written engine (`make demo`)
- [ ] Extend the demo to MNIST (optional; XOR already proves the gradients)
- [x] SGD (+momentum/Nesterov), Adam, AdamW from the papers, with tests that catch a missing bias correction and prove AdamW's decoupling differs from Adam's

> This is the one deliberately "inefficient" task in the plan. After it,
> `.backward()` stops being magic, and debugging a real model becomes tractable
> instead of mystical. It takes a weekend and pays for itself in Phase 2.

### Benchmark your hardware ⚠️

- [x] `scripts/benchmark.py` written — four configs, tokens/sec, hours-per-100M-tokens
- [ ] **Run it on your laptop** (`make bench`) — measure CPU vs MPS, or CPU vs oneDNN on x86
- [ ] Measure with and without `torch.compile`
- [ ] Record results in `docs/hardware-baseline.md`
- [ ] Derive: largest model trainable in ~4h, in ~24h

> Every later scoping decision depends on this number. Do not accept throughput
> figures for hardware nobody has measured — including mine.

Also delivered, ahead of schedule and on purpose — the constraint layer for Phase 8:

- [x] `src/gitai/safety/` — halt switch, budget, path guard, hash-chained lineage, fail-closed invariant gate
- [x] `tests/test_safety.py` — adversarial: symlink escape, `..` traversal, log tampering, truncation, swallowing a halt in a broad `except`
- [x] `scripts/halt.py` — operator stop button
- [x] `docs/constitution.md`, `docs/self-improvement.md`

Brakes before engine: a stop button retrofitted to a running loop is a stop
button nobody tested.

**Exit criterion:** ✅ engine trains XOR, suite green, CI passing. ⬜ `docs/hardware-baseline.md` still needs numbers from your machine.

---

## Phase 1 — Data pipeline ✅

*Goal: raw text → memory-mapped token shards, reproducibly.*

**Status: complete.** The full pipeline runs end-to-end on TinyShakespeare:
`make data`. First measurements are in [docs/results.md](docs/results.md).
*See [docs/data-and-storage.md](docs/data-and-storage.md).*

### Acquisition

- [x] `data/` layout: `raw/`, `interim/`, `processed/` — gitignored via `dir/*` so the manifest negation still works
- [x] `REGISTRY` + `fetch()` with checksum verification; TinyShakespeare fetched
- [ ] Fetch TinyStories (Stage 1) — ~2GB, run on your own machine
- [x] `data/raw/MANIFEST.json`: url, sha256, size, licence, retrieval date — written on cache hits too
- [x] `data/LICENSES.md` — written before downloading anything

### Curation (DuckDB + Parquet)

- [x] Convert raw → Parquet (zstd, typed)
- [x] `corpus_stats()` — DuckDB over Parquet: counts, byte totals, length quantiles, duplicate counts
- [x] Exact dedup + MinHash near-duplicate detection with banded LSH and union-find grouping
- [x] Quality filters with per-filter rejection accounting in `CurationReport`
- [x] Split by content hash — stable under corpus growth, duplicates cannot straddle
- [x] `check_leakage()`; asserted in the pipeline and in tests

### Tokenizer

- [x] `CharTokenizer` — unblocks Phase 2/3 immediately
- [x] `ByteBPETokenizer` from scratch — incremental pair counts, deterministic tie-breaking, GPT-2 pre-tokenization
- [x] Round-trip exact over emoji, CJK, control chars, and 200 random-Unicode fuzz cases
- [x] Validated against a naive reference trainer **and** encoder (byte-identical), plus a compression comparison against Hugging Face
- [x] Vocab sweep 256→8192 with bytes/token recorded — [R1 in results.md](docs/results.md)
- [x] Provisional: **2,048–4,096**. 4k→8k buys 8.4% compression for 1.57M embedding params — a bad trade at 10M total. Re-decide on TinyStories

### Training shards

- [x] `uint16` shards + `meta.json` with vocab, dtype, counts, tokenizer fingerprint, **and utf8_bytes for BPB**
- [x] `BatchSampler`: length-weighted shard choice, random offsets, plus `sequential()` for evaluation
- [x] `throughput()` — 19.4M tokens/sec in this container; **re-measure on your laptop against a 2GB shard**
- [x] Same seed ⇒ identical batches, asserted in tests

Also delivered:

- [x] Synthetic shards **require** a provenance tag at write time — wired directly into `safety.GeneratedDataQuarantined`
- [x] Mixing two tokenizers in one corpus directory is refused rather than silently producing meaningless ids
- [x] `docs/results.md` started, with the vocab sweep and curation report

**Exit criterion:** ✅ `make data` turns a fresh checkout into training shards; counts, hashes and throughput all recorded.

---

## Phase 2 — The model ✅

*Goal: a correct decoder-only transformer, built one rung at a time.*

**Status: complete.** The ladder is expressed as configuration (`model/config.py`
`RUNGS`) rather than seven classes, so each rung is one flag from the last and
"did this help?" is a controlled experiment. A 1M-parameter model trains to
1.87 BPB on TinyShakespeare in ~2 minutes — [R4](docs/results.md).

Instrumentation was built in from the first rung rather than retrofitted; see
[docs/interpretability.md](docs/interpretability.md).

### The ladder

- [x] **v0 — bigram.** `BigramModel` — starts at exactly ln(V), no non-embedding params
- [x] Bigram floor measured: **3.2406 BPB** (lab-notebook E2). The transformer's 1.87 beats it by 42% with a quarter of the parameters
- [x] **v1 — single-head self-attention**, causal mask written by hand
- [x] **v2 — multi-head + MLP + residual + LayerNorm** (GPT-2)
- [x] **v3 — RMSNorm** replaces LayerNorm
- [x] **v4 — RoPE** replaces learned positions, with a test of the relative-position property
- [x] **v5 — SwiGLU** at 8/3 width, with a test that it matches GELU's parameter count
- [x] **v6 — no biases, tied embeddings**
- [ ] Tag each rung in git

### Correctness tests (the highest-value code in the repo)

- [x] **Overfit one batch** — every rung, 8 sequences to <0.1 loss
- [x] **Causality** — every rung, both the fused and instrumented attention paths
- [x] Shape and dtype contracts; cache completeness
- [x] Determinism: same seed ⇒ identical weights and identical forward (to 1e-9 relative — not bitwise reproducible on every platform; see [evaluation.md](docs/evaluation.md#reproducibility-is-not-bitwise-everywhere))
- [x] Parameter count matches a hand-derived formula, every rung, embedding split out
- [x] Fused vs instrumented attention agree numerically (the dual-implementation check)
- [x] Initial loss = ln(vocab); tied embeddings' copy-prior pinned down as a test

> The causality test deserves special attention. A mask off-by-one **still trains
> to a plausible loss curve** — the model just cheats. Without this test the bug
> can survive for weeks and quietly invalidate every result.

### Initialisation & numerics

- [x] Scaled init — residual projections by `1/sqrt(2 * n_layer)`
- [x] Gradients reach every parameter (no dead weights)
- [ ] Activation/gradient statistics vs depth — still to plot
- [x] Parameter-count calculator with the embedding split
- [ ] FLOPs calculator (needed for compute-matched comparison in Phase 5)

Also delivered — instrumentation and a working trainer, both brought forward:

- [x] `ActivationCache` — every intermediate optionally captured, free when unused
- [x] `src/gitai/interpret/` — surprisal, logit lens, exact logit attribution, head ablation, activation patching, terminal rendering
- [x] `scripts/inspect_model.py` (`make inspect`) and `scripts/train.py` (`make train`)
- [x] [docs/interpretability.md](docs/interpretability.md)

**Exit criterion:** ✅ every rung passes causality, overfit-one-batch and the parameter formula; a trained model reaches 1.87 BPB and can be inspected end to end.

---

## Phase 3 — Training loop (done)

*Goal: runs that are resumable, reproducible, and fully recorded.*

**Status: complete.** The core loop came forward into Phase 2; this phase added
the parts that make a run survive interruption — which an unattended Phase 8
loop cannot do without.

Resume is exact: `test_resume_reproduces_an_uninterrupted_run` trains 20 steps
with checkpoints, deletes the last one, resumes from step 9 in a fresh model,
and asserts bit-identical weights and loss trajectory. Drop any one of the five
pieces of state — weights, optimiser moments, step number, torch RNG, numpy RNG
— and it fails while everything still looks healthy.

### Core loop

- [x] AdamW with correct weight-decay grouping (matrices only)
- [x] Cosine LR schedule with linear warmup
- [x] Gradient clipping, norm logged
- [x] Gradient accumulation, with equivalence asserted at the gradient level on identical data
- [ ] Mixed precision — deferred until there is a device that benefits; bf16 on CPU is not obviously a win and should be measured, not assumed
- [x] Periodic eval (sequential, not sampled) reporting loss and BPB, plus sample generation

### Run infrastructure

- [x] `runs/<timestamp>-<name>/` created per run
- [x] Resolved `config.yaml` written before the first step
- [x] `manifest.json`: git SHA, dirty flag, tokenizer fingerprint, hardware, versions, seeds
- [x] `metrics.jsonl` streamed per log step
- [x] Checkpoints in **safetensors** — via `save_model`, because tied weights alias storage and `save_file` refuses to write it
- [x] Checkpoint rotation, sorting numerically so step-100 outranks step-99
- [x] **Resume**: weights, optimiser moments, step number, torch RNG and numpy RNG — all five
- [x] Resume test: interrupt at step 9, resume, assert bit-identical weights and trajectory

### Ergonomics

- [ ] YAML config files — `TrainConfig` is already a dataclass with CLI override; file loading is the remaining piece
- [x] Progress output: loss, LR, grad norm, tokens/sec, ETA
- [ ] Optional W&B/TensorBoard viewer layered on top of the local JSONL

**Exit criterion:** resume is proven exact by test. Still to do on your machine: a multi-hour TinyStories run that survives a real interruption end to end.

---

## Phase 4 — Evaluation harness ✅

*Goal: turn training runs into comparable measurements.*

**Status: complete.** `make eval` scores a checkpoint, `make report` rebuilds the
index and prints seed-aggregated comparisons, `make noise-floor` measures the
significance threshold. Results in [docs/results.md](docs/results.md); method in
[docs/lab-notebook.md](docs/lab-notebook.md).
*See [docs/evaluation.md](docs/evaluation.md).*

- [x] **Bits-per-byte** + a test that an untrained model scores exactly log2(vocab)
- [x] Perplexity, bits/token and top-1/top-5 accuracy, all labelled tokenizer-dependent in the output
- [x] Fixed-prompt, fixed-seed samples written into every `EvalResult`
- [x] Synthetic tasks: copy, sort, modular arithmetic, Dyck — with generators, exact-match scoring, and tests that each task's own answers are correct
- [x] **Induction probe** — works on any trained model, with a positive control that a model trained on repeats develops it
- [ ] Optional LLM-as-judge scoring — deferred; the fixed-prompt samples cover the qualitative need for now
- [x] `runs/` → `runs/index.db` (SQLite), rebuilt from scratch on every invocation
- [x] `report.py`: leaderboard and seed-aggregated tables, all queries
- [ ] Plots (tables only so far)
- [x] `EVAL_SUITE_VERSION` stamped into every result

### Methodology setup ⚠️

- [x] **Seed noise floor measured** — 5 seeds, one config; see results.md R5
- [x] Recorded as the significance threshold; `compare_groups` refuses any effect below it
- [x] `docs/lab-notebook.md` started, with E1's prediction recorded before the sweep finished

Also delivered:

- [x] Exact **permutation test** for significance — dependency-free, no normality assumption, and honest about how little power 5 seeds buys
- [x] `compare_groups` vetoes any effect below the measured noise floor, even when statistically significant
- [x] **Bigram baseline run** — the context-free floor that makes the headline number interpretable

**Exit criterion:** ✅ `make report` produces a seed-averaged comparison table; the noise floor is a measured number.

---

## Phase 5 — Research (first result in)

*Goal: answer the question. This is the point of everything above.*

**Status: first experiment complete.** `make collapse` runs it.
Self-training degrades a small LM by +0.32 BPB over three generations (80x the
noise floor); accumulating real data slows this to ~41% of that rate but does
not stop it. Three of five pre-registered predictions were wrong, and their
failure located the actual mechanism -- see
[results.md R6](docs/results.md) and [E3](docs/lab-notebook.md).

Question **F** is now decided ([Decision 12](docs/decisions.md#12-research-question)):
*when does a bounded self-improvement loop help, and when does it collapse?*
Phase 5 runs the manual, single-variable version of that; Phase 8 automates it.

- [x] Experimental design: 3 arms x 3 generations x 3 seeds, one independent variable
- [x] Pre-registered five predictions with falsification criteria (E3) before running
- [x] `scripts/collapse_experiment.py` -- incremental JSONL so partial runs survive
- [x] 30 training runs, 15 corpus generations, 34.5 min
- [ ] **Re-run at 5 seeds** -- 3 seeds cannot reach p<0.05 by permutation test (floor p=0.10)
- [x] `scripts/analyse_collapse.py` -- all effects 17-80x the measured noise floor
- [x] Written up as R6
- [ ] Plots (tables only)
- [x] Three falsified predictions recorded in full, including the one that mattered most
- [x] It did -- the mechanism is error accumulation, not distribution narrowing
- [x] **Temperature sweep** -- answered: yes, and it is tail truncation rather than temperature per se (R7)
- [x] **Does accumulation rescue a low-temperature lineage?** Yes, almost entirely -- retains 6.6% of replace's excess at T0.5 (R8)
- [x] **Fixed-real-fraction arm** — done (R9): both amount and fraction matter, and the first 25% of real data recovers 89% of the damage
- [ ] **Sample the 0-25% range** — the curve is steepest there and entirely unsampled; the knee could be at 5% or 20%
- [x] **Vocabulary coverage wired into the promotion gate** as `CorpusDiversityFloor`, conditioned on retained real data after R8 showed it false-positives without that
- [ ] **Five seeds for the R8 magnitude** -- direction is solid, the size wants more evidence
- [ ] Matched-token `accumulate` variant — subsample the accumulated pool back to the real corpus size
- [ ] More generations -- does `replace` plateau or keep falling?

**Exit criterion:** met. R6 answers the question with seed-averaged evidence, states the mechanism it found instead of the one predicted, and lists what it cannot conclude.

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

- [x] KV-cache inference — brought forward in Phase 5; cached and uncached generation are bit-identical on every rung
- [x] Sampling: temperature, top-k, greedy, seeded and reproducible
- [ ] top-p (nucleus) sampling — not yet
- [ ] A small CLI or notebook for interacting with a trained model
- [ ] Export final weights + tokenizer + model card
- [ ] Write-up: method, results, negative results, limitations, reproduction instructions
- [ ] Verify reproducibility end-to-end from a clean checkout
- [ ] Consider publishing weights to Hugging Face

---

## Phase 8 — The self-improvement loop

*Goal: the loop from [self-improvement.md](docs/self-improvement.md), running
unattended and bounded, answering question F.*

**Status: built, tested, and demonstrated** — see [R10](docs/results.md).
25 iterations improved an under-trained incumbent from ~2.17 to 1.8581 BPB and
beat a compute-matched control by 4.7x the noise floor. Its rejection pattern
reproduced R6-R9 without being told.
Its central design question was answered first, by R6-R8 — which is what Phase 5
was for. The strongest expression of that: the loop *cannot* propose a
synthetic-only corpus, because `SearchSpace` refuses to contain the regime R8
measured collapsing to 4.53 BPB.

### Already built (Phase 0)

- [x] Halt switch, finite budget, path guard, hash-chained lineage, invariant gate
- [x] Adversarial tests for all of the above

### The loop

- [x] `SearchSpace` — declared, bounded, and **it cannot contain the collapse regime**: R8 encoded as a construction-time refusal rather than something the gate must catch
- [x] `SearchSpace.sample()` — random search over the declared grid, validated on the way *in* as well as out
- [x] `generate()` — incumbent's own output, tagged at birth
- [x] `filter()` — quality, dedup, near-duplicate removal on the generated half; rejects retained (capped sample, complete counts)
- [x] **Real data in every corpus** — enforced by the search space's `min_real_fraction`, not by convention
- [x] `real_data_fraction` and `corpus_stats` passed into every gate call to judge correctly
- [x] `train()` — warm-started from the incumbent, bounded steps
- [x] `evaluate()` — held-out BPB, seeds configurable
- [x] Induction probe measured every iteration and recorded in the lineage
- [x] `gate()` — `InvariantSuite`, fail-closed
- [x] `record()` — both outcomes, with the invariant that refused
- [x] Halt and budget checked at every boundary, before and after each phase
- [x] Whole loop wrapped in `deny_network()`
- [x] Promoted models are never overwritten or deleted — asserted by test

### The experiment

- [x] **Compute-matched control** — plain training at the loop's accumulated step count. The loop wins by 0.019 BPB (4.7x noise floor), though at ~3x the wall-clock
- [ ] Arm 1: **replace** real data with synthetic each round
- [ ] Arm 2: **accumulate** real + synthetic each round
- [x] Rejections cluster monotonically in synthetic fraction (40% / 57% / 88%) — the loop rediscovered R6-R9 unprompted
- [x] Confirmed in R7, at low sampling temperature — the loop's own space excludes that regime by design
- [ ] Multiple seeds for the loop-vs-control comparison — n=1 each cannot certify 0.019 BPB
- [ ] Vary synthetic fraction, filter aggressiveness, generation temperature
- [x] Written up as [R10](docs/results.md), including that induction never appeared and the wall-clock comparison is unfavourable

### Operational

- [x] [docs/runbook.md](docs/runbook.md) — start, stop, the morning check, and a symptom table
- [ ] OS-level isolation for long runs — container, unprivileged user, cgroup limits
      (the Python guards are not a security boundary; see constitution.md)
- [ ] Pin thread counts per subprocess — concurrent torch runs each claim every
      core and thrash; measured in this project as a 2-minute run stretching past
      20 while three processes competed for 4 cores
- [x] `scripts/report_loop.py` — the morning digest, lineage integrity checked before any number is interpreted

**Exit criterion:** the loop runs unattended for its full budget, stops cleanly
on exhaustion, never promotes a regression, and produces a lineage that verifies.
Then: an answer to question F.

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
| **Model collapse** — self-generated data degrades the model *while training loss falls* | Held-out gate the loop cannot influence; `NoRegression` |
| **Provenance loss** — synthetic data mixed into the corpus untagged, permanently | `GeneratedDataQuarantined`; tag at generation time |
| **A loop that logs only its wins** — the rejections were the dataset | Lineage records rejections with reasons |
| **`except Exception` swallowing a stop request** | `HaltRequested` derives from `BaseException` |
| **A little real data does almost all the work** — 25% recovers 89% of self-training damage, the remaining 75% buys 11% | For a loop, non-zero matters far more than large (R9) |
| **`min_real_fraction` prevents catastrophe, not degradation** — at 25% a lineage is still 70x the noise floor above control | The search space rules out the unrecoverable regime; `NoRegression` catches the recoverable one |
| **An early-warning signal validated in one regime may not hold in another** — `CorpusDiversityFloor` was built from R7 and immediately false-positived on R8's healthy lineage | Condition the gate on what the evidence actually covers; fail closed only when provenance is unknown |
| **Never train a generation on synthetic data alone** — retaining real data costs nothing and is the difference between 2.21 and 4.53 BPB | Phase 8 loop must always mix real data into every generation (R8) |
| **Low-temperature / top-k sampling accelerates collapse** — the "higher quality" instinct is exactly backwards | Preserve the tails when generating training data; watch generated-corpus vocabulary coverage (R7) |
| **Training loss inversely predicts held-out quality across sampling regimes** — a loop gating on it selects the catastrophic regime | Gate on held-out data the loop cannot influence, never on training loss |
| **A permutation test with 3 seeds per arm cannot reach p<0.05** — floor is 2/C(6,3)=0.10, so a 10x effect reads "not significant" | `Comparison.underpowered`; use 5 seeds per arm when p-values must mean something |
| **Changing an error message breaks tests that match on it** — and running only the new test file misses it | Run the whole suite before committing, not the files you touched |
| **A resume that restores four of five pieces of state looks completely healthy and silently diverges** | Assert bit-identical weights *and* loss trajectory against an uninterrupted run |
| **"Same seed ⇒ bit-identical" does not hold on every platform** — identical NumPy code diverged by 1 ULP (3.6e-16 relative) on an Intel Mac and not on Linux; a real determinism bug is 9.7e-02, fourteen orders larger | Assert determinism to a tolerance inside that gap, plus a negative control proving the loosened assertion still fails on a seed change. Do not assert exact equality on any float that came out of a matmul |
| **Naming a mechanism is not diagnosing one** — I attributed the above to Apple Accelerate before checking that the Intel-Mac numpy pin ships OpenBLAS | Write down what was measured and what was inferred, separately; leave the cause open until someone runs `numpy.show_config()` on the affected machine |
| **Seeding inside the training function is too late** — the model was already constructed, so the loop's incumbent came from an unseeded generator and no lineage could be regenerated; the controls were fine because their script seeded first | Seed before constructing anything; record a fingerprint of the starting weights so a divergent start is visible in the artifacts rather than inferred from drift |
| **Determinism inherited from a sibling function's side effect** — `collapse_experiment.py` built models from the global generator, which `train()` happened to reseed on entry, so every model after the first was incidentally reproducible and the first was not | Seed at the point of construction and pass the seed in. A property that holds only because of call order cannot be asserted and will not survive a refactor |
| **A reproducibility check that compares the plan instead of the measurements** — proposals matched identically all through the bug, because the search space has its own generator | Assert on measured BPB and promotion decisions, the things that actually diverged |
| **A wall-clock budget that binds turns machine load into an experimental variable** — seed 0 of the loop stopped on the 1800s budget at 2150 steps, seed 1 completed its iterations at 2350; the arms were not the same procedure | For experiments set `max_wall_seconds` so it never binds and let `max_iterations` stop the run; assert every arm's recorded halt reason is the intended one before pooling seeds |
| **Concurrent torch processes fight over cores** — each grabs all of them, so N runs on N cores is ~N× slower than serial, not equal | `OMP_NUM_THREADS` / `torch.set_num_threads` per process; matters most for the Phase 8 loop, which must not overlap its own evaluations |

---

## Open questions for you

1. ~~**Which laptop?**~~ **Answered** — Intel Mac, 4 threads, CPU only. Measured in [hardware-baseline.md](docs/hardware-baseline.md): `tiny` (1.3M) for experiments at 0.4 days per 3-seed comparison, `small` (5.8M) as the overnight showcase model, `large` out of reach at 109 days per comparison.
2. **Do you accept the two changes to the axioms?** Corrigibility added above self-preservation, and "never hurt itself" read as *never destroy your own auditability* rather than *never cease to exist*. Reasoning in [constitution.md](docs/constitution.md); push back if you disagree.
3. **Any appetite for spending money on rented GPU?** Changes whether Phase 6 is real or theoretical.

# Data & Storage

This document answers the "what database do we use?" question. The short version
is that it's the wrong question — there are **four different storage problems**
here, they have different access patterns, and using one system for all of them
would be actively harmful.

---

## The four storage problems

| # | Concern | Access pattern | Choice | Size |
|---|---|---|---|---|
| 1 | Raw corpora | Write once, read once, never mutate | Files on disk + checksummed manifest | GB–TB |
| 2 | Curation & analysis | Complex analytical queries, out-of-core | **DuckDB over Parquet** | GB |
| 3 | Training-ready tokens | Random-offset reads at GB/s, millions/sec | **Memory-mapped flat `uint16` shards** | GB |
| 4 | Run metadata & results | Small, relational, queried ad-hoc forever | **SQLite** + JSONL | MB |

---

## 1. Raw corpora — just files

Downloaded corpora are immutable blobs. A database adds nothing but overhead.

```
data/raw/tinystories/…          # never committed to git
data/raw/MANIFEST.json          # committed: url, sha256, size, licence, date
data/LICENSES.md                # committed
```

The manifest is what makes this reproducible: it lets anyone verify they have
byte-identical inputs to yours. Corpus content itself never enters git — it
would bloat the repository permanently and may violate the licence.

If versioned data ever becomes a real need, **DVC** is the tool. Not yet — one
person, one machine, checksums are sufficient.

---

## 2. Curation & analysis — DuckDB over Parquet

This is the stage where a database earns its place. Before tokenizing, you need
to answer questions like:

- What's the document length distribution?
- How much near-duplicate content is there? (MinHash / n-gram overlap)
- What fraction is non-English, or boilerplate, or truncated?
- What does the corpus look like after applying filter *X*?

These are analytical queries over more data than fits in RAM. That is precisely
DuckDB's purpose:

```sql
-- runs on a laptop, out-of-core, no server process
SELECT length(text)/1000 AS kb, count(*)
FROM 'data/interim/*.parquet'
GROUP BY 1 ORDER BY 1;
```

**Why DuckDB and not the alternatives:**

| Option | Verdict |
|---|---|
| **DuckDB** | Zero server, single file, SQL, out-of-core, reads Parquet directly, trivially fast. Purpose-built for exactly this |
| Pandas | Falls over on out-of-RAM data; you'd spend the project fighting memory |
| Polars | Genuinely excellent and a reasonable substitute — pick it if you prefer a DataFrame API to SQL. DuckDB wins on ad-hoc exploration |
| Postgres | Server to run, slow bulk ingest, wrong tool for columnar analytics |
| Spark | Absurd overkill for one laptop |

**Format: Parquet.** Columnar, compressed, typed, readable by everything. Not
JSONL (bloated, untyped), not CSV (no types, quoting hell).

> **Data curation is the highest-leverage work in this project.** The TinyStories
> result is a *data* result, not an architecture result. A better corpus will beat
> a better architecture at this scale, essentially every time. Budget accordingly
> — Phase 1 is not a chore to rush through.

---

## 3. Training-ready tokens — memory-mapped binary shards

Once curated and tokenized, the training data becomes a flat array of token IDs:

```
data/processed/tinystories/train_000.bin   # uint16, raw token IDs, nothing else
data/processed/tinystories/val_000.bin
data/processed/tinystories/meta.json       # vocab size, dtype, counts, tokenizer hash
```

The training loop reads it via `np.memmap` and samples random offsets.

**Why no database here — and this one is not a close call.** The training loop
needs millions of random reads per second, sequential in memory, with zero
deserialisation. A flat `uint16` array memory-mapped by the OS *is* the optimal
data structure: the page cache does the work, there is no query planner, no
serialisation, no index. Any database is strictly slower at this, by orders of
magnitude.

`uint16` because vocab ≤ 65,535 (Decision 4 targets 4k–8k, so this holds with
room to spare) — half the bytes, half the I/O, of `uint32`.

> ⚠️ **On CPU, the data loader becomes the bottleneck before the matmuls do.**
> This is the opposite of the GPU intuition everyone brings to the problem. If
> your Phase 0 benchmark shows low utilisation, profile the loader first. Budget
> real time here; it is not premature optimisation on this hardware.

---

## 4. Run metadata & results — SQLite + JSONL

Per-run files (`metrics.jsonl`, `manifest.json`, `config.yaml`) are the source of
truth — see Decision 6. But once there are 200 runs, "show me every run with
`d_model=384`, grouped by depth, with mean and spread across seeds" is a query,
and grepping JSON files is not the way to answer it.

So: a small ingest script walks `runs/` and loads summaries into a single SQLite
file, `runs/index.db`. SQLite because it is one file, zero configuration, present
everywhere, and handles millions of rows without complaint. The database is a
*derived index* — delete it any time and rebuild from the run directories. That
property is what keeps the real data safe.

(If you'd rather not write this yourself, MLflow's local SQLite backend does the
same job. The trade-off is a heavier dependency and less control over the schema.)

---

## Explicitly deferred

| Technology | Why not now | Revisit when |
|---|---|---|
| **Postgres** | Nothing here is transactional, concurrent, or served | A web service exists |
| **pgvector / Qdrant / Chroma** | You are training a model, not retrieving over embeddings | A RAG product exists |
| **MongoDB** | Solves no problem present here | — |
| **DVC / lakeFS** | Checksummed manifests are enough for one person | A collaborator, or datasets that mutate |
| **Feature stores** | Not that kind of ML | — |

---

## Corpus ladder

Climb it; don't jump to the top.

### Stage 0 — TinyShakespeare (~1 MB)

The smoke test. Trains in minutes on CPU. Use it for every "does the loop work"
check and every CI run. Loss should reach ~1.5 with char-level tokenization; if it
doesn't, you have a bug, and finding it here costs minutes instead of hours.

### Stage 1 — TinyStories (~2 GB) ← primary

The workhorse. Synthetic short stories using only vocabulary a 3–4 year old
knows. A 1–35M parameter model trained on it writes grammatical, coherent,
internally consistent English. This is what makes a no-GPU from-scratch LLM a
real project rather than a toy.

The narrow distribution is the point: a small model can actually *cover* it, so
you see real capability emerge rather than uniform mush.

### Stage 2 — FineWeb-Edu sample / WikiText-103 (1–10 GB)

Real, messy, heavy-tailed text. Your model will be visibly worse here, and that
contrast is itself a result. This is where data-quality experiments live
(Decision 12, question E).

### Stage 3 — Synthetic formal languages (generated on demand)

Underrated, and nearly free. Generate data from a process you define exactly:
modular arithmetic, Dyck-language bracket matching, sequence copying, sorting.

Because you know the true generating function, you can ask whether the model
learned *the algorithm* or merely surface statistics — a question that is
unanswerable on natural text. Induction-head research (Decision 12, question D)
depends on exactly this kind of controlled data.

---

## Pipeline

```
  raw files ──▶ Parquet ──▶ [DuckDB: filter, dedup, analyse] ──▶ curated Parquet
                                                                       │
                                                          [train BPE tokenizer]
                                                                       │
                                                                       ▼
                                                    uint16 .bin shards + meta.json
                                                                       │
                                                          [np.memmap DataLoader]
                                                                       │
                                                                       ▼
                                                                 training loop
                                                                       │
                                       runs/<id>/{config,manifest,metrics,ckpts}
                                                                       │
                                                         [ingest] ──▶ runs/index.db
```

Every arrow is a script that is re-runnable and records a hash of its output. If
a stage can't be re-run from the stage before it, the pipeline is broken.

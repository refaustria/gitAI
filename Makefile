.PHONY: help setup test test-fast lint fmt typecheck bench demo data data-tinystories data-sweep train train-tinystories train-tinystories-smoke resume inspect eval report noise-floor collapse collapse-temps collapse-doses collapse-all halt clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## create the venv and install everything needed to train and run the loop
	uv venv --python 3.11
	uv pip install -e ".[dev,train,data]"
	.venv/bin/pre-commit install
	@echo
	@echo "Next: make data   (downloads TinyShakespeare, builds tokenizer and shards)"
	@echo "      make test   (should be all green before any long run)"

quickstart:  ## fresh clone -> a trained model you can inspect, in one command
	$(MAKE) setup
	$(MAKE) data
	$(MAKE) test
	$(MAKE) train

test:  ## run the full suite
	.venv/bin/pytest -q

test-fast:  ## skip anything marked slow
	.venv/bin/pytest -q -m "not slow"

lint:  ## check style
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

fmt:  ## fix style in place
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .

typecheck:  ## static types
	.venv/bin/pyright

data:  ## raw corpus -> curated parquet -> tokenizer -> shards
	.venv/bin/python scripts/prepare_data.py --corpus tinyshakespeare --vocab-size 2048

data-tinystories:  ## fetch and prepare TinyStories (~2GB download, the real corpus)
	.venv/bin/python scripts/prepare_data.py --corpus tinystories --vocab-size 4096

data-sweep:  ## compare vocabulary sizes (the Decision 4 measurement)
	.venv/bin/python scripts/prepare_data.py --corpus tinyshakespeare --sweep

train:  ## train a small model on the prepared shards
	.venv/bin/python scripts/train.py --steps 1200

# The `small` rung from scripts/benchmark.py: 5.77M params, 6 layers, d_model 256.
# 28,000 steps x 16 x 256 = 114.7M tokens, ~20 tokens per parameter, and about a
# quarter of one pass over TinyStories -- so nothing is seen twice.
TINYSTORIES_SMALL = --data data/processed/tinystories \
	--d-model 256 --n-layer 6 --n-head 8 --seq-len 256 --batch-size 16

train-tinystories-smoke:  ## 200 steps on TinyStories: does the pipeline work at all? (~2 min)
	.venv/bin/python scripts/train.py $(TINYSTORIES_SMALL) \
		--steps 200 --lr 1e-3 --warmup 50 --eval-every 100 --name tinystories-smoke

train-tinystories:  ## the showcase run: 5.8M params, 115M tokens, resumable
	.venv/bin/python scripts/train.py $(TINYSTORIES_SMALL) \
		--steps 28000 --lr 1e-3 --warmup 500 --eval-every 500 \
		--checkpoint-every 500 --keep-last 3 --name tinystories-small

# A laptop will sleep, and a run this long will be interrupted. Resume is exact:
# weights, optimiser moments, step, and both RNG streams. Point it at the run dir.
resume:  ## continue an interrupted run: make resume RUN=runs/<dir>
	@test -n "$(RUN)" || { echo "usage: make resume RUN=runs/<dir>"; exit 1; }
	.venv/bin/python scripts/train.py $(TINYSTORIES_SMALL) \
		--steps 28000 --lr 1e-3 --warmup 500 --eval-every 500 \
		--checkpoint-every 500 --keep-last 3 --resume $(RUN)

inspect:  ## look inside the most recently trained model
	.venv/bin/python scripts/inspect_model.py

eval:  ## full evaluation suite on the latest run, with probes
	.venv/bin/python scripts/evaluate.py --probes

report:  ## rebuild the runs index and print seed-aggregated comparisons
	.venv/bin/python scripts/report.py

noise-floor:  ## train 5 seeds of one config to measure the significance threshold
	for s in 0 1 2 3 4; do .venv/bin/python scripts/train.py --steps 1000 --seed $$s --name noisefloor-s$$s; done
	.venv/bin/python scripts/report.py

collapse:  ## run the model-collapse experiment (Phase 5, ~35 min)
	.venv/bin/python scripts/collapse_experiment.py --generations 3 --seeds 3 --steps 500
	.venv/bin/python scripts/analyse_collapse.py

collapse-temps:  ## temperature sweep: does the sampling regime select the failure mode?
	for t in 0.5 0.8; do .venv/bin/python scripts/collapse_experiment.py --generations 3 --seeds 3 --steps 500 --arms replace --temperature $$t; done
	.venv/bin/python scripts/collapse_experiment.py --generations 3 --seeds 3 --steps 500 --arms replace --temperature 1.0 --top-k 40
	.venv/bin/python scripts/analyse_collapse.py

collapse-doses:  ## fixed-pool anchor arm: the real-data dose-response curve (R9)
	for f in 0.25 0.5; do .venv/bin/python scripts/collapse_experiment.py --generations 3 --seeds 3 --steps 500 --real-fraction $$f --temperature 0.5; done
	.venv/bin/python scripts/analyse_collapse.py

collapse-all:  ## every cell behind R6-R9, from a clean results file (~2h, serial)
	@test ! -e runs/collapse || { echo "runs/collapse exists -- move it aside first, results.jsonl appends"; exit 1; }
	.venv/bin/python scripts/collapse_experiment.py --generations 3 --seeds 3 --steps 500
	.venv/bin/python scripts/collapse_experiment.py --generations 3 --seeds 3 --steps 500 --arms accumulate,replace --temperature 0.5
	.venv/bin/python scripts/collapse_experiment.py --generations 3 --seeds 3 --steps 500 --arms replace --temperature 0.8
	.venv/bin/python scripts/collapse_experiment.py --generations 3 --seeds 3 --steps 500 --arms replace --temperature 1.0 --top-k 40
	for f in 0.25 0.5; do .venv/bin/python scripts/collapse_experiment.py --generations 3 --seeds 3 --steps 500 --real-fraction $$f --temperature 0.5; done
	.venv/bin/python scripts/analyse_collapse.py

bench:  ## measure this machine (needs the train extra)
	.venv/bin/python scripts/benchmark.py --json docs/hardware-baseline.json

demo:  ## train XOR with the hand-written engine
	.venv/bin/python scripts/demo_autograd.py

halt:  ## stop a running improvement loop
	.venv/bin/python scripts/halt.py stop

clean:
	rm -rf .pytest_cache .ruff_cache .venv **/__pycache__ src/*.egg-info

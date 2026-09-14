.PHONY: help setup test test-fast lint fmt typecheck bench demo data data-tinystories data-sweep train inspect eval report noise-floor collapse collapse-temps halt clean

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

bench:  ## measure this machine (needs the train extra)
	.venv/bin/python scripts/benchmark.py --json docs/hardware-baseline.json

demo:  ## train XOR with the hand-written engine
	.venv/bin/python scripts/demo_autograd.py

halt:  ## stop a running improvement loop
	.venv/bin/python scripts/halt.py stop

clean:
	rm -rf .pytest_cache .ruff_cache .venv **/__pycache__ src/*.egg-info

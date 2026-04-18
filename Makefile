PYTHON := .venv/bin/python

.PHONY: help install test run run-fast ci-check precommit

help:
	@echo "Available targets:"
	@echo "  install    Install dependencies into .venv"
	@echo "  test       Run unit tests"
	@echo "  run        Run full Topic E pipeline"
	@echo "  run-fast   Run pipeline with faster settings"
	@echo "  ci-check   Run checks used in CI"
	@echo "  precommit  Run pre-commit hooks on all files"

install:
	$(PYTHON) -m pip install --index-url https://pypi.org/simple -r requirements.txt

test:
	$(PYTHON) -m pytest -q

run:
	$(PYTHON) src/run_experiment.py \
		--data-dir data \
		--max-rows 80000 \
		--sample-size 30000 \
		--rf-n-estimators 80 \
		--rf-max-depth 16 \
		--output-dir artifacts \
		--log-level INFO

run-fast:
	$(PYTHON) src/run_experiment.py \
		--data-dir data \
		--max-rows 40000 \
		--sample-size 12000 \
		--rf-n-estimators 60 \
		--rf-max-depth 14 \
		--output-dir artifacts \
		--log-level INFO

ci-check: test

precommit:
	pre-commit run --all-files

# WMoNbZrTiTa diffusion pipeline - workflow shortcuts.
#
# Every target is a thin wrapper around run_pipeline.py or the module CLIs.
# Set TEST=1 to force the fast validation configuration, e.g.:
#     make generate TEST=1

PYTHON   ?= python3
PIPELINE := $(PYTHON) run_pipeline.py
TEST_FLAG := $(if $(TEST),--test,)

.DEFAULT_GOAL := help
.PHONY: help setup generate tm-inputs submit-tm tm-patch diffusion-inputs submit analyze test lint clean

help:
	@echo "WMoNbZrTiTa diffusion pipeline"
	@echo ""
	@echo "  make setup             Install pinned dependencies (editable install)"
	@echo "  make generate          Stages 0-2: validate, compositions, constants"
	@echo "  make tm-inputs         Stage 3b: write GRACE coexistence inputs"
	@echo "  make submit-tm         Submit slurm/submit_Tm_array.sh"
	@echo "  make tm-patch          Stage 3c: patch real Tm into constants"
	@echo "  make diffusion-inputs  Stages 3-4: write ADP inputs and submit script"
	@echo "  make submit            Submit slurm/submit_diffusion.sh"
	@echo "  make analyze           Stages 5-10: MSD, Dv, Cv, D*, SRO, postprocess"
	@echo "  make test              Run the pytest suite"
	@echo "  make lint              Byte-compile every module"
	@echo "  make clean             Remove caches and build artifacts"
	@echo ""
	@echo "  Fast validation run:   make generate TEST=1"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

generate:
	$(PIPELINE) --from_stage 0 --to_stage 2 $(TEST_FLAG)

tm-inputs:
	$(PIPELINE) --only_stage 31 $(TEST_FLAG)

submit-tm:
	sbatch slurm/submit_Tm_array.sh

tm-patch:
	$(PIPELINE) --only_stage 32 $(TEST_FLAG)

diffusion-inputs:
	$(PIPELINE) --from_stage 3 --to_stage 4 $(TEST_FLAG)

submit:
	sbatch slurm/submit_diffusion.sh

analyze:
	$(PIPELINE) --from_stage 5 $(TEST_FLAG)

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m compileall -q config.py run_pipeline.py logging_config.py reporting.py pipeline analysis

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache build dist *.egg-info
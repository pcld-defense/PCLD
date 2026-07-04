PYTHON ?= python3.10
VENV   := .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

.PHONY: setup setup-cuda smoke clean-venv

## Create .venv and install pinned dependencies (CPU torch from PyPI).
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .
	@echo ""
	@echo "Done. Activate with:  source $(VENV)/bin/activate"

## Same, but install the CUDA 12.1 torch build first.
setup-cuda:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
	$(PIP) install -e .
	@echo ""
	@echo "Done. Activate with:  source $(VENV)/bin/activate"

## Run the end-to-end smoke-test experiment.
smoke:
	$(PY) scripts/run.py experiment=smoke_test

clean-venv:
	rm -rf $(VENV)

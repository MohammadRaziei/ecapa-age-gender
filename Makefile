.PHONY: setup download prepare train evaluate export clean distclean all

PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PY := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
CONFIG ?= training/config.yaml
CHECKPOINT ?= training/checkpoints/best_model.pt
EXPORT_DIR ?= export

# All `python -m ...` calls below are run from the repo root so that both the
# `ecapa_age_gender` (inference) and `training` (training-only) packages resolve as
# top-level imports - this is what keeps the two cleanly separated (see idea.md).

## Create a virtualenv and install the package + training-only dependencies
setup:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r training/requirements.txt

## Download Common Voice splits (requires HF_TOKEN, see training/download_data.py)
download:
	$(VENV_PY) -m training.download_data --config $(CONFIG)

## Filter valid rows and build manifest CSVs
prepare:
	$(VENV_PY) -m training.prepare_data --config $(CONFIG)

## Train the multitask model
train:
	$(VENV_PY) -m training.train --config $(CONFIG)

## Evaluate the best checkpoint on the test split
evaluate:
	$(VENV_PY) -m training.evaluate --config $(CONFIG)

## Export the best checkpoint to a self-contained .pt + .onnx + label_map bundle
## Push to the Hub too by passing e.g. `make export HUB_REPO=you/ecapa-age-gender`
export:
	$(VENV_PY) -m training.export_and_push --checkpoint $(CHECKPOINT) --out_dir $(EXPORT_DIR) \
		$(if $(HUB_REPO),--push_to_hub $(HUB_REPO),)

## Full pipeline: setup -> download -> prepare -> train -> evaluate
all: setup download prepare train evaluate

## Remove venv, caches, and checkpoints (keeps raw downloaded data)
clean:
	rm -rf $(VENV_DIR) training/checkpoints/*.pt training/checkpoints/*.json \
		**/__pycache__ pretrained_models $(EXPORT_DIR)

## Also wipe downloaded/prepared data (forces a fresh download)
distclean: clean
	rm -rf training/data/raw training/data/manifests

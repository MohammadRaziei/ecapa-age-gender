.PHONY: setup download prepare train evaluate export clean distclean all

PYTHON ?= python
PIP ?= pip
CONFIG ?= config.yaml
CHECKPOINT ?= checkpoints/best_model.pt
EXPORT_DIR ?= export

# This Makefile is meant to be run with training/ as the working directory
# (`cd training && make ...`, or `make -C training ...`).
#
# `ecapa_age_gender` (the inference package) and `training` (this directory, as a
# package) both need to resolve as top-level imports for `python -m training.xxx` to
# work, so the repo root (one level up) is put on PYTHONPATH for every recipe below.
export PYTHONPATH := ..:$(PYTHONPATH)

## Install the ecapa_age_gender package (editable) + training-only dependencies.
## No virtualenv - installs straight into whatever `python3`/`pip3` you already have on
## PATH. Use your own venv/conda env beforehand if you want isolation.
setup:
	$(PIP) install -r requirements.txt

## Download Common Voice splits (requires HF_TOKEN, see download_data.py)
download:
	$(PYTHON) -m training.download_data --config $(CONFIG)

## Filter valid rows and build manifest CSVs
prepare:
	$(PYTHON) -m training.prepare_data --config $(CONFIG)

## Train the multitask model
train:
	$(PYTHON) -m training.train --config $(CONFIG)

## Evaluate the best checkpoint on the test split
evaluate:
	$(PYTHON) -m training.evaluate --config $(CONFIG)

## Export the best checkpoint to a self-contained .pt + .onnx + label_map bundle.
## Push to the Hub too by passing e.g. `make export HUB_REPO=you/ecapa-age-gender`
export:
	$(PYTHON) -m training.export_and_push --checkpoint $(CHECKPOINT) --out_dir $(EXPORT_DIR) \
		$(if $(HUB_REPO),--push_to_hub $(HUB_REPO),)

## Full pipeline: setup -> download -> prepare -> train -> evaluate
all: setup download prepare train evaluate

## Remove caches, checkpoints, and exported artifacts (keeps downloaded data)
clean:
	rm -rf checkpoints/*.pt checkpoints/*.json __pycache__ ../pretrained_models $(EXPORT_DIR)

## Also wipe downloaded/prepared data (forces a fresh download)
distclean: clean
	rm -rf data/raw data/manifests

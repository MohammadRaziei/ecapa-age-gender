.PHONY: setup download prepare train evaluate export clean distclean all

PYTHON ?= python
PIP ?= pip
CONFIG ?= config.yaml
CHECKPOINT ?= checkpoints/best_model.pt
EXPORT_DIR ?= export

# This folder is meant to be run as plain scripts, from inside training/ itself
# (`cd training && make ...`, or `make -C training ...`). No package/venv setup - it's
# a working folder for training and further development, not something meant to be
# `pip install`-ed by a normal user of the eventual model.

## Install training dependencies straight into whatever python3/pip3 you already have on
## PATH (use your own venv/conda env beforehand if you want isolation)
setup:
	$(PIP) install -r requirements.txt

## Download Common Voice splits (requires HF_TOKEN, see download_data.py)
download:
	$(PYTHON) download_data.py --config $(CONFIG)

## Filter valid rows and build manifest CSVs
prepare:
	$(PYTHON) prepare_data.py --config $(CONFIG)

## Train the multitask model
train:
	$(PYTHON) train.py --config $(CONFIG)

## Evaluate the best checkpoint on the test split
evaluate:
	$(PYTHON) evaluate.py --config $(CONFIG)

## Export the best checkpoint to a self-contained .pt + .onnx + label_map bundle.
## Push to the Hub too by passing e.g. `make export HUB_REPO=you/ecapa-age-gender`
export:
	$(PYTHON) export_and_push.py --checkpoint $(CHECKPOINT) --out_dir $(EXPORT_DIR) \
		$(if $(HUB_REPO),--push_to_hub $(HUB_REPO),)

## Full pipeline: setup -> download -> prepare -> train -> evaluate
all: setup download prepare train evaluate

## Remove caches, checkpoints, and exported artifacts (keeps downloaded data)
clean:
	rm -rf checkpoints/*.pt checkpoints/*.json __pycache__ ../pretrained_models $(EXPORT_DIR)

## Also wipe downloaded/prepared data (forces a fresh download)
distclean: clean
	rm -rf data/raw data/manifests

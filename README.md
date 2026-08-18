# ecapa-age-gender

Joint age-bucket + gender estimation from speech, built on a pretrained **ECAPA-TDNN**
speaker embedder (not a transformer — fast, small, and, because it's a speaker-ID model,
already rich in the acoustic cues that correlate with age/gender). See
**[idea.md](idea.md)** for the full design rationale, architecture, and references.

## Current status: training only

Right now this repo is just `training/` — everything needed to reproduce the model: data
download, manifest building, the training loop, evaluation, and exporting a trained
checkpoint to `.pt` / `.onnx`. This folder isn't meant for normal end users, only for
training and further development.

A small, `pip install`-able inference package (load a checkpoint, call `.predict(...)`, no
`datasets`/`pandas`/training deps required) is planned once the model itself is in good
shape — see [idea.md, section 5](idea.md#5-roadmap--next-steps).

```
ecapa-age-gender/
├── idea.md              <- design doc / rationale
├── README.md             <- you are here
└── training/
    ├── Makefile             <- setup / download / prepare / train / evaluate / export
    ├── requirements.txt
    ├── config.yaml            <- all hyperparameters
    ├── download_data.py         <- pulls Common Voice via `datasets`
    ├── prepare_data.py           <- filters valid rows, builds manifest CSVs
    ├── dataset.py                 <- PyTorch Dataset + collate_fn
    ├── model.py                    <- MultiTaskECAPA (backbone + freezing + heads)
    ├── train.py                      <- training loop
    ├── evaluate.py                     <- gender acc / age acc / age MAE-in-buckets
    ├── export_and_push.py               <- checkpoint -> model.pt + model.onnx + Hub push
    ├── data/                              <- (gitignored) downloaded/prepared data
    └── checkpoints/                        <- (gitignored) training checkpoints
```

## Quickstart

Everything runs as plain scripts from inside `training/`:

```bash
cd training

# 1. Install dependencies (no venv created for you - use your own if you want one)
make setup

# 2. Get a Hugging Face token and accept the Common Voice dataset terms:
#    https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
export HF_TOKEN=hf_xxx...

# 3. Download Common Voice (train/validation/test splits, English by default)
make download

# 4. Filter rows with valid age+gender labels and build manifests
make prepare

# 5. Train (frozen early ECAPA blocks + fine-tuned later blocks + gender/age heads)
make train

# 6. Evaluate the best checkpoint on the held-out test split
make evaluate

# 7. Export to model.pt + model.onnx + label_map.json, and (optionally) push to the Hub
make export HUB_REPO=your-username/ecapa-age-gender
```

Or run steps 1-6 in one shot: `make all`. Everything is configured from a single file:
**[training/config.yaml](training/config.yaml)**.

Each script also works directly, e.g. `python3 train.py --config config.yaml`, if you'd
rather not go through `make`.

## Requirements

- Python 3.10+
- A CUDA GPU is strongly recommended for `make train` (CPU works but will be slow)
- A free Hugging Face account + access token (for downloading Common Voice, and again if
  you want to push a trained model to the Hub)

## Status

First runnable scaffold: data pipeline, model, training loop, evaluation, and checkpoint
export are all in place. Reported literature numbers to aim for on Common Voice / TIMIT:
~99.6% gender accuracy, ~5-year age MAE (Kwasny & Hemmerling, 2021) — see
[idea.md](idea.md#6-references).

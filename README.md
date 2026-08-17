# ecapa-age-gender

Joint age-bucket + gender estimation from speech, built on a pretrained **ECAPA-TDNN**
speaker embedder (not a transformer — fast, small, and, because it's a speaker-ID model,
already rich in the acoustic cues that correlate with age/gender). See
**[idea.md](idea.md)** for the full design rationale, architecture, and references.

The repo is split the way SpeechBrain-style projects usually are:

- **`ecapa_age_gender/`** — the installable, dependency-light *inference* package. This is
  all an end user needs: `pip install` it, load a checkpoint, call `.predict_file(...)`.
- **`training/`** — everything needed to reproduce or extend training: data download,
  manifest building, the training loop, evaluation, and exporting/publishing a trained
  checkpoint to the Hugging Face Hub. Has its own (heavier) dependencies and its own
  `Makefile` — training is a self-contained unit of the repo.

```
ecapa-age-gender/
├── idea.md                       <- design doc / rationale
├── README.md                     <- you are here
├── pyproject.toml                <- installable package metadata (ecapa-age-gender)
│
├── ecapa_age_gender/              <- INSTALLABLE INFERENCE PACKAGE
│   ├── __init__.py
│   ├── model.py                    <- MultiTaskECAPA (backbone + freezing + heads)
│   └── inference.py                 <- AgeGenderClassifier.from_pretrained(...) / .predict(...)
│
├── training/                      <- TRAINING-ONLY CODE (not shipped in the pip package)
│   ├── Makefile                     <- setup / download / prepare / train / evaluate / export
│   ├── requirements.txt
│   ├── config.yaml                  <- all hyperparameters
│   ├── download_data.py               <- pulls Common Voice via `datasets`
│   ├── prepare_data.py                <- filters valid rows, builds manifest CSVs
│   ├── dataset.py                      <- PyTorch Dataset + collate_fn
│   ├── train.py                         <- training loop
│   ├── evaluate.py                       <- gender acc / age acc / age MAE-in-buckets
│   ├── export_and_push.py                 <- checkpoint -> model.pt + model.onnx + Hub push
│   ├── data/                              <- (gitignored) downloaded/prepared data
│   └── checkpoints/                        <- (gitignored) training checkpoints
│
├── examples/
│   └── predict_example.py         <- minimal end-to-end usage of the installed package
└── tests/
    └── test_model.py              <- shape / freezing smoke tests (pytest)
```

## Quickstart: training

All training commands run from inside `training/` (that's where its `Makefile` lives):

```bash
cd training

# 1. Install the ecapa_age_gender package + training-only dependencies
#    (no virtualenv is created for you - use your own venv/conda env first if you want one)
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

## Quickstart: using a trained model

Once a model has been exported (step 7 above) and pushed to the Hub, using it needs none
of the training dependencies:

```bash
pip install -e .          # from the repo root; installs only ecapa_age_gender + torch/torchaudio/speechbrain
```

```python
from ecapa_age_gender import AgeGenderClassifier

clf = AgeGenderClassifier.from_pretrained("your-username/ecapa-age-gender")
result = clf.predict_file("someone_talking.wav")
print(result.as_dict())
# {'gender': 'female', 'gender_probs': {...}, 'age_bucket': 'thirties', 'age_probs': {...}}
```

`from_pretrained(...)` also accepts a local directory (e.g. `training/export/` from step 7),
so you can try a checkpoint before publishing it anywhere.

## Requirements

- Python 3.10+
- A CUDA GPU is strongly recommended for `make train` (CPU works but will be slow); the
  exported model runs comfortably on CPU for inference
- A free Hugging Face account + access token (for downloading Common Voice, and again if
  you want to push your trained model to the Hub)

## Status

First runnable scaffold: data pipeline, model, training loop, evaluation, and Hub export
are all in place. Reported literature numbers to aim for on Common Voice / TIMIT: ~99.6%
gender accuracy, ~5-year age MAE (Kwasny & Hemmerling, 2021) — see
[idea.md](idea.md#7-references).

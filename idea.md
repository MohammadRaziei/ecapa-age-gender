# AgeGenderNet — Joint Age-Bucket & Gender Estimation from an ECAPA-TDNN Speaker Embedder

## 1. Motivation

There are open-source models that predict **age or gender from speech** (e.g. `audeering/wav2vec2-large-robust-24-ft-age-gender`
on Hugging Face), but they are all built on top of **wav2vec2**, a self-supervised model pretrained mostly to preserve
*linguistic/phonetic content* (it is an ASR-oriented representation). For a demographic-attribute task, the content of what
is said matters far less than *how* it is said — pitch range, formant structure, vocal-tract length, speaking rate — exactly
the cues that **speaker-verification embeddings (x-vector / ECAPA-TDNN)** are trained to preserve, because a speaker
embedder's entire job is to separate individuals, and age/gender are two of the strongest, most stable components of "who
is speaking."

Empirically this holds up: on a gender-classification benchmark, x-vector embeddings reached **99.2%** accuracy vs.
wav2vec2's **76.3%** (Table 2, arxiv.org/pdf/2406.00022). Academically, Kwasny & Hemmerling (ICASSP 2021,
arXiv:2012.01551 / Sensors 2021) built exactly this idea — an x-vector-style embedder extended with a multitask head for
joint age + gender — and reached **99.6% gender accuracy** and **~5.1 year MAE** on age, state-of-the-art at the time.
A related face-voice-association paper trains a **small ECAPA-TDNN** for age/gender using the *pre-final* embedding layer
rather than the last (speaker-ID-specific) layer.

**No open-source, ready-to-run implementation of this idea exists.** This project fills that gap: a small, fast,
transformer-free, ECAPA-TDNN-based joint age-bucket + gender model, released with training/eval code and (eventually)
weights.

## 2. Why ECAPA-TDNN over wav2vec2

| Property | wav2vec2 | x-vector / ECAPA-TDNN |
|---|---|---|
| Pretraining objective | Self-supervised, ASR-oriented → preserves *content* | Supervised speaker discrimination → preserves *identity* (incl. age/gender cues) |
| Architecture | 24-layer Transformer, large (~300M params) | TDNN + channel attention, no self-attention over time, much smaller (~6-20M params) |
| Inference speed | Slow (quadratic-ish attention, large model) | Fast (convolutional, streams well) |
| Gender accuracy (linear probe) | ~76% | ~99% |
| Good fit for age/gender | Indirect — needs the model to "learn to ignore" content | Direct — identity-relevant acoustic cues are already the whole point |

This matches the intuition: **a representation this rich, distinguishing thousands of individual speakers, must already
encode age and gender as major axes of variation** — we're not asking the network to learn something new, we're asking it
to *decode* something it already represents.

## 3. Architecture

```
                      ┌─────────────────────────────┐
   16kHz waveform ──► │  Mel-filterbank frontend     │  (frozen)
                      └──────────────┬──────────────┘
                                     ▼
                      ┌─────────────────────────────┐
                      │  ECAPA-TDNN backbone         │
                      │  (pretrained on VoxCeleb1+2) │
                      │                              │
                      │  SE-Res2Blocks[0..k-1]  ── FROZEN
                      │  SE-Res2Blocks[k..N-1]  ── fine-tuned
                      │  Multi-layer feature aggregation
                      │  Attentive Statistics Pooling ── fine-tuned
                      └──────────────┬──────────────┘
                                     ▼
                       192-dim pooled embedding
                       (the pretrained speaker-ID linear
                        classifier layer is DISCARDED —
                        it's over-specialized to the
                        VoxCeleb training identities and
                        not useful for us)
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                       ▼
    ┌──────────────────┐  ┌──────────────────┐   ┌───────────────────────┐
    │  Gender head      │  │  Age-bucket head  │   │  Embedding-consistency │
    │  MLP → 2/3 classes│  │  MLP → K classes  │   │  aux. loss (optional)  │
    └──────────────────┘  └──────────────────┘   └───────────────────────┘
```

Key design decisions (matching the discussion that motivated this project):

1. **Freeze the early SE-Res2Blocks.** Low/mid-level acoustic feature extraction (formants, pitch contour, spectral
   envelope) is largely task-agnostic and already well learned from ~2,000+ hours of VoxCeleb. Freezing these saves
   compute and reduces overfitting.
2. **Fine-tune the later blocks + attentive pooling.** These layers combine low-level cues into an
   identity-discriminative summary — we want this recombined slightly toward "age/gender-discriminative" rather than
   purely "individual-discriminative."
3. **Throw away the pretrained AAM-softmax classifier head.** It was trained to separate ~7,000 VoxCeleb identities; its
   decision boundary is not useful for a 2-output / K-output demographic task. We only reuse the embedding trunk.
4. **Two lightweight task heads on top of the shared 192-dim embedding** (small MLPs, cheap to train, easy to swap).
5. **Age as buckets (~10-year ranges), not regression.** Precise age from speech is intrinsically noisy — even humans
   struggle. Common Voice conveniently already labels age in decade buckets (`teens`, `twenties`, ... `nineties`), so
   we treat this as an 8-class classification problem. Classification also converges more easily than regression and is
   naturally robust to label noise/outliers. (An ordinal-aware loss, e.g. CORAL, is a natural follow-up if plain CE
   underperforms.)
6. **Optional auxiliary consistency loss.** To avoid catastrophically forgetting the rich speaker representation while
   fine-tuning, we add an optional cosine-similarity penalty between the fine-tuned embedding and the frozen pretrained
   embedding (computed once, cached). This is a light regularizer, not a hard requirement — controlled by a config flag.

## 4. Data

We use **Mozilla Common Voice** (`mozilla-foundation/common_voice_17_0`, English by default, configurable), because it is
the largest open speech corpus with **self-reported age and gender** labels already provided per-clip, and age is already
bucketed into decades — a perfect match for the design above.

- `age`: `teens, twenties, thirties, forties, fifties, sixties, seventies, eighties, nineties` → mapped to class indices `0..7`
- `gender`: `male, female` (optionally `other`, config-gated; kept out of the default 2-class setup because samples are
  very scarce and would mostly add noise to a first baseline)
- Rows with missing/blank age or gender are dropped

Common Voice on Hugging Face is a **gated dataset** — you must accept its terms on the dataset page once, then use a
Hugging Face access token. See `README` "Setup" section below.

## 5. Repo layout

The repo follows the split common in SpeechBrain-style projects: a small, dependency-light
**installable inference package**, kept separate from the (heavier, optional) **training
code**. A user who just wants predictions never needs `datasets`, `pandas`, `scikit-learn`,
or the Common Voice download machinery at all.

```
ecapa-age-gender/
├── idea.md                       <- this file
├── README.md                     <- quickstart / usage
├── pyproject.toml                <- installable package metadata (ecapa-age-gender)
│
├── ecapa_age_gender/              <- INSTALLABLE INFERENCE PACKAGE (pip install-able)
│   ├── __init__.py
│   ├── model.py                    <- MultiTaskECAPA (backbone + freezing logic + heads)
│   └── inference.py                 <- AgeGenderClassifier.from_pretrained(...) / .predict(...)
│
├── training/                      <- TRAINING-ONLY CODE, not shipped in the pip package
│   ├── Makefile                     <- setup / download / prepare / train / evaluate / export
│   │                                    (no venv - run inside your own env if you want one)
│   ├── requirements.txt             <- installs ecapa_age_gender + training extras
│   ├── config.yaml                   <- all hyperparameters
│   ├── download_data.py                <- pulls Common Voice splits via `datasets`
│   ├── prepare_data.py                  <- filters valid rows, builds manifests + label maps
│   ├── dataset.py                        <- PyTorch Dataset + collate_fn
│   ├── train.py                           <- training loop, checkpointing, logging
│   ├── evaluate.py                         <- gender acc / age-bucket acc / age MAE-in-buckets
│   ├── export_and_push.py                   <- checkpoint -> model.pt + model.onnx + Hub push
│   ├── data/                                 <- (gitignored) downloaded/prepared data
│   └── checkpoints/                           <- (gitignored) saved models
│
├── examples/
│   └── predict_example.py         <- minimal end-to-end usage of the installed package
└── tests/
    └── test_model.py              <- shape / freezing smoke tests (pytest)
```

All `training/Makefile` targets run from inside `training/` (`cd training && make ...`), and
put the repo root on `PYTHONPATH` for the duration of each command so that both
`ecapa_age_gender` and `training` resolve as top-level Python packages — no venv is created
for you; use your own virtualenv/conda env first if you want isolation.

The final `model.pt` / `model.onnx` / `label_map.json` bundle produced by
`training/export_and_push.py` is what gets pushed to the Hugging Face Hub — from that point
on, `ecapa_age_gender.AgeGenderClassifier.from_pretrained("user/ecapa-age-gender")` is all
downstream users ever need to touch.

## 6. Roadmap / next steps

1. **Baseline** — fully frozen ECAPA + linear heads → sanity-check the pipeline and get a fast reference number.
2. **Partial fine-tune** — unfreeze last `k` blocks + pooling, as designed above → main experiment.
3. **Compare against literature** — Kwasny & Hemmerling's TIMIT/Common-Voice numbers (gender 99.6%, age MAE ~5 years) and
   audEERING's wav2vec2 baseline, on the same Common Voice test split, for an apples-to-apples comparison.
4. **Ordinal loss for age** (CORAL / soft-ordinal CE) if plain classification underperforms on adjacent-bucket confusions.
5. **Release** — publish weights + card on Hugging Face once results are competitive, so this stops being a gap in the
   open-source ecosystem.

## 7. References

- Kwasny, D. & Hemmerling, D. (2021). *Joint gender and age estimation based on speech signals using x-vectors and
  transfer learning.* ICASSP 2021 / arXiv:2012.01551.
- Kwasny, D. & Hemmerling, D. (2021). *Gender and Age Estimation Methods Based on Speech Using Deep Neural Networks.*
  Sensors, 21(14), 4785.
- Desplanques, B., Thienpondt, J., & Demuynck, K. (2020). *ECAPA-TDNN: Emphasized Channel Attention, Propagation and
  Aggregation in TDNN Based Speaker Verification.* Interspeech 2020.
- SpeechBrain `spkrec-ecapa-voxceleb` pretrained model (Hugging Face).
- Face-voice association work using a small ECAPA-TDNN trained on Common Voice for age/gender, taking the pre-final
  embedding layer (arXiv:2512.04814).
- Mozilla Common Voice dataset (`mozilla-foundation/common_voice_17_0`).

"""High-level, dependency-light inference API — the kind of thing users of the package
actually import, in the same spirit as SpeechBrain's `EncoderClassifier.from_hparams(...)`.

Example
-------
    from ecapa_age_gender import AgeGenderClassifier

    clf = AgeGenderClassifier.from_pretrained("your-hf-username/ecapa-age-gender")
    result = clf.predict_file("someone_talking.wav")
    print(result)
    # {'gender': 'female', 'gender_probs': {...}, 'age_bucket': 'thirties', 'age_probs': {...}}
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import torchaudio

from ecapa_age_gender.model import MultiTaskECAPA, MultiTaskECAPAConfig

DEFAULT_GENDERS = ["male", "female"]
DEFAULT_AGE_BUCKETS = [
    "teens", "twenties", "thirties", "forties",
    "fifties", "sixties", "seventies", "eighties", "nineties",
]


@dataclass
class PredictionResult:
    gender: str
    gender_probs: Dict[str, float]
    age_bucket: str
    age_probs: Dict[str, float]
    embedding: torch.Tensor

    def as_dict(self) -> dict:
        return {
            "gender": self.gender,
            "gender_probs": self.gender_probs,
            "age_bucket": self.age_bucket,
            "age_probs": self.age_probs,
        }


class AgeGenderClassifier:
    """Thin wrapper around `MultiTaskECAPA` for easy loading + single-call inference."""

    def __init__(
        self,
        model: MultiTaskECAPA,
        genders: List[str],
        age_buckets: List[str],
        sample_rate: int = 16000,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device).eval()
        self.genders = genders
        self.age_buckets = age_buckets
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str,
        checkpoint_filename: str = "model.pt",
        label_map_filename: str = "label_map.json",
        device: Optional[Union[str, torch.device]] = None,
    ) -> "AgeGenderClassifier":
        """Load either from a local directory or a Hugging Face Hub repo id.

        The repo/directory is expected to contain:
          - `model.pt`        (a dict with "model_state_dict" and "model_config")
          - `label_map.json`  (a dict with "genders" and "age_buckets" lists)
        (this is exactly what `training/export_and_push.py` produces.)
        """
        if os.path.isdir(repo_id_or_path):
            ckpt_path = os.path.join(repo_id_or_path, checkpoint_filename)
            label_map_path = os.path.join(repo_id_or_path, label_map_filename)
        else:
            from huggingface_hub import hf_hub_download

            ckpt_path = hf_hub_download(repo_id_or_path, checkpoint_filename)
            label_map_path = hf_hub_download(repo_id_or_path, label_map_filename)

        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        genders = label_map.get("genders", DEFAULT_GENDERS)
        age_buckets = label_map.get("age_buckets", DEFAULT_AGE_BUCKETS)

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model_cfg = MultiTaskECAPAConfig(**checkpoint["model_config"])
        model = MultiTaskECAPA(model_cfg)
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model=model, genders=genders, age_buckets=age_buckets, device=device)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, waveform: torch.Tensor, sample_rate: Optional[int] = None) -> PredictionResult:
        """`waveform`: 1-D float tensor, mono. Resampled automatically if `sample_rate`
        differs from the model's expected sample rate."""
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)  # downmix to mono if needed

        if sample_rate is not None and sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        waveform = waveform.unsqueeze(0).to(self.device)  # (1, T)
        rel_length = torch.ones(1, device=self.device)

        out = self.model(waveform, rel_length)
        gender_probs = F.softmax(out["gender_logits"], dim=-1).squeeze(0).cpu()
        age_probs = F.softmax(out["age_logits"], dim=-1).squeeze(0).cpu()

        gender_idx = int(gender_probs.argmax())
        age_idx = int(age_probs.argmax())

        return PredictionResult(
            gender=self.genders[gender_idx],
            gender_probs={g: float(p) for g, p in zip(self.genders, gender_probs)},
            age_bucket=self.age_buckets[age_idx],
            age_probs={a: float(p) for a, p in zip(self.age_buckets, age_probs)},
            embedding=out["embedding"].squeeze(0).cpu(),
        )

    def predict_file(self, audio_path: str) -> PredictionResult:
        waveform, sr = torchaudio.load(audio_path)
        return self.predict(waveform.squeeze(0), sample_rate=sr)

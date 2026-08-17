"""PyTorch Dataset that reads manifest CSVs (built by prepare_data.py) and lazily pulls
decoded audio out of the cached Hugging Face Arrow dataset (built by download_data.py).
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset


class CommonVoiceAgeGenderDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        manifest_path: str,
        split: str,
        sample_rate: int = 16000,
        max_audio_seconds: float = 6.0,
        min_audio_seconds: float = 1.0,
    ) -> None:
        self.hf_dataset = load_from_disk(os.path.join(raw_dir, split))
        self.manifest = pd.read_csv(manifest_path)
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_seconds * sample_rate)
        self.min_samples = int(min_audio_seconds * sample_rate)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.manifest.iloc[idx]
        hf_row = self.hf_dataset[int(row["row_index"])]

        audio = np.asarray(hf_row["audio"]["array"], dtype=np.float32)

        # Skip-too-short clips are handled by padding; skip-too-long by random cropping,
        # so every batch element ends up at a bounded, but not identical, length -
        # the collate_fn pads the batch to its own max length.
        if len(audio) > self.max_samples:
            start = np.random.randint(0, len(audio) - self.max_samples + 1)
            audio = audio[start : start + self.max_samples]
        elif len(audio) < self.min_samples:
            # pad short clips up to the minimum so the backbone gets a usable window
            pad = self.min_samples - len(audio)
            audio = np.pad(audio, (0, pad))

        waveform = torch.from_numpy(audio)
        return {
            "waveform": waveform,
            "length": torch.tensor(waveform.shape[0], dtype=torch.long),
            "gender_label": torch.tensor(int(row["gender_label"]), dtype=torch.long),
            "age_label": torch.tensor(int(row["age_label"]), dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Right-pads waveforms to the longest one in the batch and returns a relative-length
    tensor (0-1) the way SpeechBrain/ECAPA expects for its internal masking."""
    max_len = max(item["waveform"].shape[0] for item in batch)
    padded = torch.zeros(len(batch), max_len)
    rel_lengths = torch.zeros(len(batch))
    gender_labels = torch.zeros(len(batch), dtype=torch.long)
    age_labels = torch.zeros(len(batch), dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["waveform"].shape[0]
        padded[i, :n] = item["waveform"]
        rel_lengths[i] = n / max_len
        gender_labels[i] = item["gender_label"]
        age_labels[i] = item["age_label"]

    return {
        "waveforms": padded,
        "rel_lengths": rel_lengths,
        "gender_labels": gender_labels,
        "age_labels": age_labels,
    }

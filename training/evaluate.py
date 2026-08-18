"""Evaluate a trained checkpoint on the test manifest.

Reports:
  - gender accuracy
  - age-bucket accuracy
  - age MAE-in-buckets (mean absolute distance between predicted and true bucket index,
    a proxy for "how many decades off" the model typically is)

Usage:
    python3 evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pt
"""
from __future__ import annotations

import argparse
import os

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from model import build_model

from dataset import CommonVoiceAgeGenderDataset, collate_fn
from utils import load_config, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Overrides eval.checkpoint_path in config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["train"]["device"])
    data_cfg = cfg["data"]

    checkpoint_path = args.checkpoint or cfg["eval"]["checkpoint_path"]
    print(f"[evaluate] loading checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(
        cfg["model"],
        num_genders=len(data_cfg["genders"]),
        num_age_buckets=len(data_cfg["age_buckets"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_ds = CommonVoiceAgeGenderDataset(
        raw_dir=data_cfg["raw_dir"],
        manifest_path=os.path.join(data_cfg["manifest_dir"], "test.csv"),
        split="test",
        sample_rate=data_cfg["sample_rate"],
        max_audio_seconds=data_cfg["max_audio_seconds"],
        min_audio_seconds=data_cfg["min_audio_seconds"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
    )

    all_gender_true, all_gender_pred = [], []
    all_age_true, all_age_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            waveforms = batch["waveforms"].to(device)
            rel_lengths = batch["rel_lengths"].to(device)
            preds = model.predict(waveforms, rel_lengths)

            all_gender_true.extend(batch["gender_labels"].tolist())
            all_gender_pred.extend(preds["gender_pred"].cpu().tolist())
            all_age_true.extend(batch["age_labels"].tolist())
            all_age_pred.extend(preds["age_pred"].cpu().tolist())

    gender_names = data_cfg["genders"]
    age_names = data_cfg["age_buckets"]

    print("\n=== Gender ===")
    print(classification_report(all_gender_true, all_gender_pred, target_names=gender_names))
    print("Confusion matrix:")
    print(confusion_matrix(all_gender_true, all_gender_pred))

    print("\n=== Age bucket ===")
    print(classification_report(all_age_true, all_age_pred, target_names=age_names))
    print("Confusion matrix:")
    print(confusion_matrix(all_age_true, all_age_pred))

    mae_buckets = sum(abs(t - p) for t, p in zip(all_age_true, all_age_pred)) / len(all_age_true)
    print(f"\nAge MAE (in decade-buckets): {mae_buckets:.3f}")


if __name__ == "__main__":
    main()

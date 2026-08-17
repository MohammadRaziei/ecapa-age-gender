"""Train MultiTaskECAPA on the prepared Common Voice manifests.

Usage:
    python -m src.train --config config.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ecapa_age_gender.model import build_model

from training.dataset import CommonVoiceAgeGenderDataset, collate_fn
from training.utils import ensure_dir, load_config, resolve_device, save_json, set_seed


def run_epoch(model, loader, device, optimizer, cfg, train: bool):
    model.train(mode=train)
    ce_gender = nn.CrossEntropyLoss()
    ce_age = nn.CrossEntropyLoss()

    total_loss, total_gender_correct, total_age_correct, total_n = 0.0, 0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc="train" if train else "val", leave=False)
        for step, batch in enumerate(pbar):
            waveforms = batch["waveforms"].to(device)
            rel_lengths = batch["rel_lengths"].to(device)
            gender_labels = batch["gender_labels"].to(device)
            age_labels = batch["age_labels"].to(device)

            out = model(waveforms, rel_lengths)
            loss_gender = ce_gender(out["gender_logits"], gender_labels)
            loss_age = ce_age(out["age_logits"], age_labels)
            loss = (
                cfg["train"]["gender_loss_weight"] * loss_gender
                + cfg["train"]["age_loss_weight"] * loss_age
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg["train"]["grad_clip_norm"],
                )
                optimizer.step()

            bs = waveforms.shape[0]
            total_loss += loss.item() * bs
            total_gender_correct += (out["gender_logits"].argmax(-1) == gender_labels).sum().item()
            total_age_correct += (out["age_logits"].argmax(-1) == age_labels).sum().item()
            total_n += bs

            if train and step % cfg["train"]["log_every_n_steps"] == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}")

    return {
        "loss": total_loss / max(total_n, 1),
        "gender_acc": total_gender_correct / max(total_n, 1),
        "age_acc": total_age_correct / max(total_n, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = resolve_device(cfg["train"]["device"])
    print(f"[train] device={device}")

    data_cfg = cfg["data"]
    train_ds = CommonVoiceAgeGenderDataset(
        raw_dir=data_cfg["raw_dir"],
        manifest_path=os.path.join(data_cfg["manifest_dir"], "train.csv"),
        split="train",
        sample_rate=data_cfg["sample_rate"],
        max_audio_seconds=data_cfg["max_audio_seconds"],
        min_audio_seconds=data_cfg["min_audio_seconds"],
    )
    val_ds = CommonVoiceAgeGenderDataset(
        raw_dir=data_cfg["raw_dir"],
        manifest_path=os.path.join(data_cfg["manifest_dir"], "validation.csv"),
        split="validation",
        sample_rate=data_cfg["sample_rate"],
        max_audio_seconds=data_cfg["max_audio_seconds"],
        min_audio_seconds=data_cfg["min_audio_seconds"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
    )

    model = build_model(
        cfg["model"],
        num_genders=len(data_cfg["genders"]),
        num_age_buckets=len(data_cfg["age_buckets"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.trainable_parameter_groups(
            base_lr=cfg["train"]["lr"],
            head_lr_multiplier=cfg["train"]["head_lr_multiplier"],
        ),
        weight_decay=cfg["train"]["weight_decay"],
    )

    ensure_dir(cfg["train"]["checkpoint_dir"])
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, device, optimizer, cfg, train=True)
        val_metrics = run_epoch(model, val_loader, device, optimizer, cfg, train=False)
        dt = time.time() - t0

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_gender_acc={train_metrics['gender_acc']:.3f} "
            f"train_age_acc={train_metrics['age_acc']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_gender_acc={val_metrics['gender_acc']:.3f} "
            f"val_age_acc={val_metrics['age_acc']:.3f} "
            f"({dt:.1f}s)"
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], "best_model.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    # dataclasses.asdict(model.cfg) is exactly what
                    # MultiTaskECAPAConfig(**model_config) expects on reload, so this
                    # checkpoint is self-describing and can be loaded (or exported via
                    # `training/export_and_push.py`) without also needing config.yaml.
                    "model_config": dataclasses.asdict(model.cfg),
                    "full_config": cfg,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                ckpt_path,
            )
            print(f"[train] new best model -> {ckpt_path}")

    # Also drop a label_map.json next to the checkpoints: this is the other file
    # `AgeGenderClassifier.from_pretrained(...)` / `export_and_push.py` expect.
    save_json(
        {"genders": data_cfg["genders"], "age_buckets": data_cfg["age_buckets"]},
        os.path.join(cfg["train"]["checkpoint_dir"], "label_map.json"),
    )
    save_json(history, os.path.join(cfg["train"]["checkpoint_dir"], "history.json"))
    print("[train] done.")


if __name__ == "__main__":
    main()

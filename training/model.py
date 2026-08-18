"""MultiTaskECAPA: a pretrained SpeechBrain ECAPA-TDNN speaker embedder, partially
frozen, with its original (VoxCeleb-identity-specific) classifier head discarded and
replaced by two small task heads: gender and age-bucket.

Backbone source: speechbrain/spkrec-ecapa-voxceleb (downloaded automatically on first
run and cached under `pretrained_models/` by SpeechBrain).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from speechbrain.inference.speaker import EncoderClassifier


@dataclass
class MultiTaskECAPAConfig:
    pretrained_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    freeze_num_blocks: int = 2
    freeze_frontend: bool = True
    embedding_dim: int = 192
    head_hidden_dim: int = 128
    head_dropout: float = 0.2
    num_genders: int = 2
    num_age_buckets: int = 9
    use_embedding_consistency_loss: bool = False


def _make_head(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class MultiTaskECAPA(nn.Module):
    def __init__(self, cfg: MultiTaskECAPAConfig, savedir: str = "pretrained_models/ecapa") -> None:
        super().__init__()
        self.cfg = cfg

        pretrained = EncoderClassifier.from_hparams(
            source=cfg.pretrained_source,
            savedir=savedir,
        )
        # Pull out the three sub-modules that make up the forward pass:
        #   waveform -> Fbank features -> mean/var norm -> ECAPA_TDNN embedder
        # We deliberately do NOT keep `pretrained.mods.classifier`: that's the
        # AAM-softmax head trained to separate ~7000 VoxCeleb identities, which is not
        # useful for a demographic-attribute task (see idea.md, section 3).
        self.compute_features = pretrained.mods.compute_features
        self.mean_var_norm = pretrained.mods.mean_var_norm
        self.embedding_model = pretrained.mods.embedding_model

        self._apply_freezing()

        self.gender_head = _make_head(
            cfg.embedding_dim, cfg.head_hidden_dim, cfg.num_genders, cfg.head_dropout
        )
        self.age_head = _make_head(
            cfg.embedding_dim, cfg.head_hidden_dim, cfg.num_age_buckets, cfg.head_dropout
        )

    def _apply_freezing(self) -> None:
        if self.cfg.freeze_frontend:
            for p in self.compute_features.parameters():
                p.requires_grad = False
            for p in self.mean_var_norm.parameters():
                p.requires_grad = False

        # embedding_model.blocks is a ModuleList: [TDNNBlock, SERes2NetBlock, SERes2NetBlock, ...]
        # Freeze the first `freeze_num_blocks` of them (low/mid-level acoustic features);
        # leave the rest of blocks + the MFA conv + attentive stat pooling + fc trainable.
        blocks = getattr(self.embedding_model, "blocks", None)
        if blocks is None:
            raise AttributeError(
                "embedding_model has no `.blocks` attribute - the SpeechBrain ECAPA_TDNN "
                "implementation may have changed; inspect `self.embedding_model` and update "
                "_apply_freezing() accordingly."
            )
        n_freeze = min(self.cfg.freeze_num_blocks, len(blocks))
        for block in blocks[:n_freeze]:
            for p in block.parameters():
                p.requires_grad = False

    def trainable_parameter_groups(self, base_lr: float, head_lr_multiplier: float):
        """Two param groups: fine-tuned backbone layers at base_lr, task heads at a
        higher lr (they start from scratch and should move faster)."""
        backbone_params = [p for p in self.embedding_model.parameters() if p.requires_grad]
        head_params = list(self.gender_head.parameters()) + list(self.age_head.parameters())
        return [
            {"params": backbone_params, "lr": base_lr},
            {"params": head_params, "lr": base_lr * head_lr_multiplier},
        ]

    def embed(self, waveforms: torch.Tensor, rel_lengths: torch.Tensor) -> torch.Tensor:
        feats = self.compute_features(waveforms)
        feats = self.mean_var_norm(feats, rel_lengths)
        embedding = self.embedding_model(feats, rel_lengths)  # (B, 1, embedding_dim)
        return embedding.squeeze(1)

    def forward(self, waveforms: torch.Tensor, rel_lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.embed(waveforms, rel_lengths)
        return {
            "embedding": embedding,
            "gender_logits": self.gender_head(embedding),
            "age_logits": self.age_head(embedding),
        }

    @torch.no_grad()
    def predict(self, waveforms: torch.Tensor, rel_lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        out = self.forward(waveforms, rel_lengths)
        return {
            "gender_pred": out["gender_logits"].argmax(dim=-1),
            "age_pred": out["age_logits"].argmax(dim=-1),
            "embedding": out["embedding"],
        }


def build_model(cfg_dict: dict, num_genders: int, num_age_buckets: int) -> MultiTaskECAPA:
    model_cfg = MultiTaskECAPAConfig(
        pretrained_source=cfg_dict["pretrained_source"],
        freeze_num_blocks=cfg_dict["freeze_num_blocks"],
        freeze_frontend=cfg_dict["freeze_frontend"],
        embedding_dim=cfg_dict["embedding_dim"],
        head_hidden_dim=cfg_dict["head_hidden_dim"],
        head_dropout=cfg_dict["head_dropout"],
        num_genders=num_genders,
        num_age_buckets=num_age_buckets,
        use_embedding_consistency_loss=cfg_dict["use_embedding_consistency_loss"],
    )
    return MultiTaskECAPA(model_cfg)

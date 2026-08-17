"""Lightweight tests that don't require downloading data - just the pretrained ECAPA
checkpoint (downloaded automatically by SpeechBrain on first run, then cached).

Run with:
    pytest tests/
"""
from __future__ import annotations

import torch

from ecapa_age_gender.model import MultiTaskECAPA, MultiTaskECAPAConfig


def _tiny_model() -> MultiTaskECAPA:
    cfg = MultiTaskECAPAConfig(
        freeze_num_blocks=2,
        num_genders=2,
        num_age_buckets=9,
    )
    return MultiTaskECAPA(cfg)


def test_forward_shapes():
    model = _tiny_model()
    waveforms = torch.randn(2, 16000 * 2)  # 2 utterances, 2 seconds @ 16kHz
    rel_lengths = torch.tensor([1.0, 0.75])

    out = model(waveforms, rel_lengths)

    assert out["embedding"].shape == (2, model.cfg.embedding_dim)
    assert out["gender_logits"].shape == (2, 2)
    assert out["age_logits"].shape == (2, 9)


def test_frozen_blocks_have_no_grad():
    model = _tiny_model()
    blocks = model.embedding_model.blocks
    for block in blocks[: model.cfg.freeze_num_blocks]:
        for p in block.parameters():
            assert not p.requires_grad
    for block in blocks[model.cfg.freeze_num_blocks :]:
        assert any(p.requires_grad for p in block.parameters())


def test_predict_returns_class_indices():
    model = _tiny_model()
    waveforms = torch.randn(1, 16000 * 2)
    rel_lengths = torch.ones(1)

    preds = model.predict(waveforms, rel_lengths)
    assert preds["gender_pred"].item() in (0, 1)
    assert 0 <= preds["age_pred"].item() < 9

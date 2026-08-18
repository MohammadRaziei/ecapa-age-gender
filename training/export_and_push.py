"""Package a training checkpoint into a self-contained model.pt + model.onnx + label_map
bundle, and (optionally) push it to the Hugging Face Hub.

NOTE: there is currently no separate `ecapa_age_gender` inference package in this repo -
this script only produces the raw artifacts. Loading them back is a few lines of plain
PyTorch (see the model card template below). Packaging that into a proper installable
`pip install`-able inference library is a later step (see idea.md, section 5 "Roadmap").

Produces, in --out_dir:
  - model.pt          torch checkpoint: {"model_state_dict", "model_config"}
  - model.onnx         same model exported to ONNX (fixed input names: waveform, length)
  - label_map.json     {"genders": [...], "age_buckets": [...]}
  - README.md            minimal Hugging Face model card (edit before publishing!)

Usage:
    # local export only
    python3 export_and_push.py --checkpoint checkpoints/best_model.pt --out_dir export/

    # export + push to the Hub (requires `huggingface-cli login` or HF_TOKEN)
    python3 export_and_push.py --checkpoint checkpoints/best_model.pt \
        --out_dir export/ --push_to_hub your-username/ecapa-age-gender
"""
from __future__ import annotations

import argparse
import os
import shutil

import torch

from model import MultiTaskECAPA, MultiTaskECAPAConfig
from utils import ensure_dir, save_json

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
tags:
  - speech
  - ecapa-tdnn
  - age-estimation
  - gender-classification
  - audio-classification
---

# ecapa-age-gender

Joint age-bucket + gender classification from speech, built on a partially fine-tuned
`speechbrain/spkrec-ecapa-voxceleb` backbone (ECAPA-TDNN — no transformer, fast CPU
inference). See the project's `idea.md` for the full design rationale.

## Usage

```python
import json
import torch
from model import MultiTaskECAPA, MultiTaskECAPAConfig  # see this repo's training/model.py

ckpt = torch.load("model.pt", map_location="cpu")
model = MultiTaskECAPA(MultiTaskECAPAConfig(**ckpt["model_config"]))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

labels = json.load(open("label_map.json"))
waveform, rel_length = ...  # 16kHz mono waveform tensor (1, T), rel_length = torch.ones(1)
out = model(waveform, rel_length)
gender = labels["genders"][out["gender_logits"].argmax(-1).item()]
age_bucket = labels["age_buckets"][out["age_logits"].argmax(-1).item()]
```

A proper `pip install`-able wrapper package (with a one-line `from_pretrained(...)` API)
is planned for later - see this repo's `idea.md`.

## Labels

- Gender: see `label_map.json` -> "genders"
- Age: decade buckets, see `label_map.json` -> "age_buckets"
"""


class _ONNXWrapper(torch.nn.Module):
    """torch.onnx.export wants a plain tuple of tensors out, not a dict."""

    def __init__(self, model: MultiTaskECAPA) -> None:
        super().__init__()
        self.model = model

    def forward(self, waveform: torch.Tensor, length: torch.Tensor):
        out = self.model(waveform, length)
        return out["embedding"], out["gender_logits"], out["age_logits"]


def export_onnx(model: MultiTaskECAPA, out_path: str, sample_rate: int = 16000) -> None:
    model.eval()
    wrapper = _ONNXWrapper(model)
    dummy_waveform = torch.randn(1, sample_rate * 3)  # 3 seconds, batch size 1
    dummy_length = torch.ones(1)

    torch.onnx.export(
        wrapper,
        (dummy_waveform, dummy_length),
        out_path,
        input_names=["waveform", "length"],
        output_names=["embedding", "gender_logits", "age_logits"],
        dynamic_axes={
            "waveform": {0: "batch", 1: "time"},
            "length": {0: "batch"},
            "embedding": {0: "batch"},
            "gender_logits": {0: "batch"},
            "age_logits": {0: "batch"},
        },
        opset_version=17,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint saved by training/train.py")
    parser.add_argument("--label_map", default=None, help="Defaults to label_map.json next to --checkpoint")
    parser.add_argument("--out_dir", default="export")
    parser.add_argument("--skip_onnx", action="store_true", help="Skip ONNX export (torch checkpoint only)")
    parser.add_argument(
        "--push_to_hub",
        default=None,
        metavar="REPO_ID",
        help="e.g. your-username/ecapa-age-gender - if set, uploads --out_dir to this HF Hub model repo",
    )
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = MultiTaskECAPAConfig(**checkpoint["model_config"])
    model = MultiTaskECAPA(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 1) clean torch checkpoint (state dict + the dataclass config needed to rebuild it)
    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": checkpoint["model_config"]},
        os.path.join(args.out_dir, "model.pt"),
    )
    print(f"[export] wrote {args.out_dir}/model.pt")

    # 2) label map
    label_map_path = args.label_map or os.path.join(os.path.dirname(args.checkpoint), "label_map.json")
    if os.path.isfile(label_map_path):
        shutil.copy(label_map_path, os.path.join(args.out_dir, "label_map.json"))
    else:
        print(f"[export] WARNING: no label_map.json found at {label_map_path}; writing a default 2/9-class map")
        save_json(
            {
                "genders": ["male", "female"],
                "age_buckets": [
                    "teens", "twenties", "thirties", "forties", "fifties",
                    "sixties", "seventies", "eighties", "nineties",
                ],
            },
            os.path.join(args.out_dir, "label_map.json"),
        )
    print(f"[export] wrote {args.out_dir}/label_map.json")

    # 3) ONNX (optional)
    if not args.skip_onnx:
        onnx_path = os.path.join(args.out_dir, "model.onnx")
        export_onnx(model, onnx_path)
        print(f"[export] wrote {onnx_path}")

    # 4) minimal model card
    repo_id = args.push_to_hub or "your-username/ecapa-age-gender"
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))
    print(f"[export] wrote {args.out_dir}/README.md (edit before publishing)")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=args.out_dir, repo_id=args.push_to_hub)
        print(f"[export] pushed {args.out_dir} -> https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()

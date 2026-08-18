"""Download Common Voice splits and cache them locally as Hugging Face Arrow datasets.

Common Voice is a *gated* dataset on the Hugging Face Hub: you must accept its terms once
at https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 (or whichever
version/config you set in config.yaml) while logged in, then create an access token at
https://huggingface.co/settings/tokens and export it:

    export HF_TOKEN=hf_xxx...

Usage:
    python3 download_data.py --config config.yaml
    python3 download_data.py --config config.yaml --splits train validation test
"""
from __future__ import annotations

import argparse
import os

from datasets import Audio, load_dataset

from utils import ensure_dir, load_config


def download_split(hf_dataset: str, language: str, split: str, sample_rate: int,
                    out_dir: str, token: str | None) -> None:
    print(f"[download_data] fetching split='{split}' of {hf_dataset} ({language}) ...")
    ds = load_dataset(hf_dataset, language, split=split, token=token, trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))

    split_dir = os.path.join(out_dir, split)
    ensure_dir(split_dir)
    ds.save_to_disk(split_dir)
    print(f"[download_data] saved {len(ds)} rows -> {split_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Which Common Voice splits to download.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    token = os.environ.get("HF_TOKEN")
    if token is None:
        raise SystemExit(
            "HF_TOKEN environment variable not set.\n"
            f"1. Accept the dataset terms at https://huggingface.co/datasets/{data_cfg['hf_dataset']}\n"
            "2. Create a token at https://huggingface.co/settings/tokens\n"
            "3. export HF_TOKEN=hf_xxx...\n"
            "then re-run this script."
        )

    ensure_dir(data_cfg["raw_dir"])
    for split in args.splits:
        download_split(
            hf_dataset=data_cfg["hf_dataset"],
            language=data_cfg["language"],
            split=split,
            sample_rate=data_cfg["sample_rate"],
            out_dir=data_cfg["raw_dir"],
            token=token,
        )

    print("[download_data] done.")


if __name__ == "__main__":
    main()

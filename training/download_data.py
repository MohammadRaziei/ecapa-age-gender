"""Download Common Voice splits and cache them locally as Hugging Face Arrow datasets.

NOTE: as of October 2025, Mozilla pulled the official `mozilla-foundation/common_voice_*`
repos off the Hugging Face Hub (they're now distributed exclusively through Mozilla Data
Collective: https://datacollective.mozillafoundation.org). `load_dataset(...)` on those
repo ids will fail with `EmptyDatasetError` regardless of any token - the repos are
literally empty now, not gated. Some community re-uploads (e.g. fsicoli/*) still ship the
old loading-script format, which now fails too with "Dataset scripts are no longer
supported" on current `datasets` versions. `config.yaml` defaults to a plain-parquet
mirror (`fixie-ai/common_voice_17_0`, no script) that keeps `age`/`gender` columns -
note its gender values are `male_masculine`/`female_feminine` rather than plain
`male`/`female`; `prepare_data.py` normalizes that automatically.

If the mirror happens to be gated or rate-limited for you, log in once with
`huggingface-cli login` (or `hf auth login`, depending on your `huggingface_hub` version)
and this script will pick up that cached token automatically - no HF_TOKEN env var
required. Setting HF_TOKEN still works too, and overrides the cached login if both are
present.

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
    ds = load_dataset(hf_dataset, language, split=split, token=token)
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

    # `token=None` is not "no token" - huggingface_hub resolves it to whatever you're
    # already logged in as via `huggingface-cli login` / `hf auth login`. HF_TOKEN, if
    # set, takes priority. Neither is required for the default public mirror.
    token = os.environ.get("HF_TOKEN")

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

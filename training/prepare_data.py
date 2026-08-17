"""Filter the downloaded Common Voice splits to rows with valid age+gender labels,
and write lightweight CSV manifests (row index into the cached HF dataset + integer
class labels) that `dataset.py` reads at train time.

We deliberately do NOT copy/re-encode the audio here: the cached HF Arrow dataset from
`download_data.py` already stores it efficiently, and `dataset.py` loads samples from it
lazily by index. The manifest only carries the (split, row_index, gender_label, age_label)
tuples plus clip duration, so filtering/inspection is fast without touching audio at all.

Usage:
    python -m src.prepare_data --config config.yaml
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
from datasets import load_from_disk

from training.utils import ensure_dir, load_config, save_json


def build_manifest_for_split(raw_dir: str, split: str, genders: list[str],
                              age_buckets: list[str]) -> pd.DataFrame:
    split_dir = os.path.join(raw_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"{split_dir} not found. Run `python -m src.download_data` first."
        )

    ds = load_from_disk(split_dir)
    gender_to_idx = {g: i for i, g in enumerate(genders)}
    age_to_idx = {a: i for i, a in enumerate(age_buckets)}

    rows = []
    for i, (age, gender) in enumerate(zip(ds["age"], ds["gender"])):
        age = (age or "").strip().lower()
        gender = (gender or "").strip().lower()
        if age not in age_to_idx or gender not in gender_to_idx:
            continue
        rows.append(
            {
                "row_index": i,
                "gender_label": gender_to_idx[gender],
                "age_label": age_to_idx[age],
                "gender": gender,
                "age": age,
            }
        )

    df = pd.DataFrame(rows)
    print(
        f"[prepare_data] split={split}: kept {len(df)}/{len(ds)} rows "
        f"({100 * len(df) / max(len(ds), 1):.1f}%) with valid age+gender labels"
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    ensure_dir(data_cfg["manifest_dir"])

    for split in args.splits:
        df = build_manifest_for_split(
            raw_dir=data_cfg["raw_dir"],
            split=split,
            genders=data_cfg["genders"],
            age_buckets=data_cfg["age_buckets"],
        )
        out_path = os.path.join(data_cfg["manifest_dir"], f"{split}.csv")
        df.to_csv(out_path, index=False)
        print(f"[prepare_data] wrote {out_path}")

    save_json(
        {
            "genders": data_cfg["genders"],
            "age_buckets": data_cfg["age_buckets"],
        },
        data_cfg["label_map_path"],
    )
    print(f"[prepare_data] wrote {data_cfg['label_map_path']}")


if __name__ == "__main__":
    main()

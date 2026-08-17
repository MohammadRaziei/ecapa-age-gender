"""Minimal example of using the published package for inference.

Once a model has been trained and exported (see training/export_and_push.py), anyone can
do this without ever touching the training/ code or its dependencies:

    pip install ecapa-age-gender   # or: pip install -e . (from repo root, for now)
    python examples/predict_example.py path/to/audio.wav your-username/ecapa-age-gender
"""
from __future__ import annotations

import sys

from ecapa_age_gender import AgeGenderClassifier


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <audio_file> <hf_repo_id_or_local_export_dir>")
        raise SystemExit(1)

    audio_path, repo_id_or_path = sys.argv[1], sys.argv[2]

    clf = AgeGenderClassifier.from_pretrained(repo_id_or_path)
    result = clf.predict_file(audio_path)

    print(f"gender:     {result.gender}  {result.gender_probs}")
    print(f"age bucket: {result.age_bucket}  {result.age_probs}")


if __name__ == "__main__":
    main()

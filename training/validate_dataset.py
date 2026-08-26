from __future__ import annotations

import argparse

from training.data import load_sft_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="data/sft/amitai_sft_v0.jsonl")
    args = parser.parse_args()

    dataset = load_sft_dataset(args.path)
    print(f"OK: {len(dataset)} valid SFT conversations in {args.path}")


if __name__ == "__main__":
    main()

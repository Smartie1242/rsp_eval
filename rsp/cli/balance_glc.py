"""CLI for deterministic GLC balancing."""

from __future__ import annotations

import argparse

from ..glc import balance_glc


def build_parser():
    parser = argparse.ArgumentParser(description="Balance a GLC language dataset by split.")
    parser.add_argument("--input", default="data/GLC", help="Input GLC directory.")
    parser.add_argument("--output", default="data/GLC_balanced", help="Output balanced GLC directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated split names to balance. Default: train,val,test.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    manifest = balance_glc(args.input, args.output, seed=args.seed, splits=splits)

    print(f"Balanced GLC written to: {manifest['output_dir']}")
    print(f"Seed: {manifest['seed']}")
    for split, data in manifest["splits"].items():
        print(f"{split}: target={data['target_count']} counts={data['output_counts']}")


if __name__ == "__main__":
    main()

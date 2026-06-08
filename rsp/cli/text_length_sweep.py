"""CLI for text-length detector performance sweeps."""

from __future__ import annotations

import argparse
import os

from ..datasets import load_csv, resolve_dataset_paths
from ..metrics import filter_evaluation_rows, filter_wili_eval_overlap_frames
from ..routing import HYBRID_CUTOFF, LENGTH_BIN_LABELS, add_length_features, run_length_sweep


def has_url_metadata(dataset_name: str) -> bool:
    return str(dataset_name).startswith("OWI_slice")


def process_dataset_frame(name, df, output_dir, hybrid_cutoff):
    print(f"\nProcessing dataset: {name}")
    df = add_length_features(df)

    out_dir = os.path.join(output_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    for i, length_bin in enumerate(LENGTH_BIN_LABELS):
        print(f"  Evaluating text-length bin {length_bin} ({i + 1}/{len(LENGTH_BIN_LABELS)})")
    results_df = run_length_sweep(df, hybrid_cutoff=hybrid_cutoff)
    if not has_url_metadata(name):
        results_df = results_df[~results_df["model"].astype(str).str.contains("URL", regex=False)].copy()
    results_df.to_csv(os.path.join(out_dir, "length_sweep.csv"), index=False)
    print(f"  Saved length sweep data for {name}")


def build_parser():
    parser = argparse.ArgumentParser(description="Text-length detector performance sweep")
    parser.add_argument("--output_dir", type=str, default="results/length_sweep", help="Directory for outputs.")
    parser.add_argument(
        "--hybrid-cutoff",
        type=float,
        default=HYBRID_CUTOFF,
        help="OOP cutoff for RP composite fallback models. Default: 1200.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="Override or add a dataset CSV path. Can be repeated.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    dataset_paths = resolve_dataset_paths(args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    print("Starting length sweep...\n")

    frames = {}
    for name, path in dataset_paths.items():
        if not os.path.exists(path):
            print(f"Skipping {name} (file not found)")
            continue
        frames[name] = filter_evaluation_rows(load_csv(path), name)

    frames, label_universe = filter_wili_eval_overlap_frames(frames)
    if label_universe:
        print(f"[filter] evaluation universe: {len(label_universe):,} non-WiLI label(s)")

    for name, df in frames.items():
        process_dataset_frame(name, df, args.output_dir, args.hybrid_cutoff)

    print("\nDone.")


if __name__ == "__main__":
    main()

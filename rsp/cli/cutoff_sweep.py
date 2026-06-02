"""CLI for score cutoff routing sweeps."""

from __future__ import annotations

import argparse
import os

from ..datasets import load_csv, resolve_dataset_paths
from ..plots import plot_cutoff_comparison
from ..routing import SCORE_CUTOFFS, evaluate_score_cutoff


def process_dataset(dataset_name, csv_path, output_dir):
    print(f"\nProcessing dataset: {dataset_name}")
    df = load_csv(csv_path)
    dataset_output = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output, exist_ok=True)

    results = []
    for i, cutoff in enumerate(SCORE_CUTOFFS):
        print(f"  Evaluating cutoff {cutoff} ({i + 1}/{len(SCORE_CUTOFFS)})")
        results.append(evaluate_score_cutoff(df, cutoff))

    import pandas as pd

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(dataset_output, "cutoff_sweep.csv"), index=False)
    print(f"  Saved results for {dataset_name}")
    plot_cutoff_comparison(results_df, dataset_name, dataset_output)


def build_parser():
    parser = argparse.ArgumentParser(description="Fallback methods performance")
    parser.add_argument("--output_dir", type=str, default="results/cutoff_sweep", help="Directory for outputs.")
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
    print("Starting cutoff sweep...\n")

    for dataset_name, path in dataset_paths.items():
        if not os.path.exists(path):
            print(f"Skipping {dataset_name} (file not found)")
            continue
        process_dataset(dataset_name, path, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()


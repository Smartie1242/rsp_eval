"""CLI for score-gap analysis."""

from __future__ import annotations

import argparse
import os

from ..datasets import load_csv, resolve_dataset_paths
from ..metrics import add_k_score_features, add_k_score_features_ft, compute_summary, filter_evaluation_rows
from ..plots import (
    plot_gap_vs_lang_distance,
    plot_histogram,
    plot_rank_correctness_share,
    plot_simple_gap_summary,
)


def process_dataset(dataset_name, csv_path, output_dir, plot_type="all"):
    print(f"Processing {dataset_name}")
    df = filter_evaluation_rows(load_csv(csv_path), dataset_name)
    dfs = [("rp", add_k_score_features(df)), ("ft", add_k_score_features_ft(df))]
    dataset_output = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output, exist_ok=True)

    for name, data in dfs:
        model_output = os.path.join(dataset_output, name)
        os.makedirs(model_output, exist_ok=True)

        summary = compute_summary(data)
        summary.to_csv(os.path.join(dataset_output, "score_gap_summary.csv"))

        gap_columns = [col for col in data.columns if col.startswith("gap_1_")]
        if plot_type in {"hist", "all", "violin", "box"}:
            for gap in gap_columns:
                plot_histogram(data, f"{dataset_name}_{name}", model_output, gap)

        plot_simple_gap_summary(data, f"{dataset_name}_{name}", model_output, "oop")
        plot_rank_correctness_share(data, f"{dataset_name}_{name}", model_output)
        plot_gap_vs_lang_distance(data, f"{dataset_name}_{name}", model_output, "oop")


def build_parser():
    parser = argparse.ArgumentParser(description="Score-gap analysis for Resiliparse outputs")
    parser.add_argument(
        "--plot",
        type=str,
        default="all",
        choices=["violin", "box", "hist", "all"],
        help="Plot type to generate. Legacy values are accepted; current output is histogram plus summaries.",
    )
    parser.add_argument("--output_dir", type=str, default="results/score_gap", help="Directory for outputs.")
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

    for dataset_name, path in dataset_paths.items():
        if not os.path.exists(path):
            print(f"Skipping {dataset_name}, file not found: {path}")
            continue
        process_dataset(dataset_name, path, args.output_dir, args.plot)

    print("\nDone.")


if __name__ == "__main__":
    main()

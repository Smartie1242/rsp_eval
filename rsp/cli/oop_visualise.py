"""CLI for PCA visualisation of OOP score spaces."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..datasets import ensure_dir
from ..plots import compute_pca_coordinates, plot_full_pca, plot_pairwise_pca


def parse_score_cols(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def run_oop_visualisation(input_path, output_dir, score_cols, label_col):
    import pandas as pd

    df = pd.read_csv(input_path)
    missing = [col for col in [*score_cols, label_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {', '.join(missing)}")

    output_dir = ensure_dir(Path(output_dir))
    plot_df = compute_pca_coordinates(df, score_cols)
    plot_pairwise_pca(plot_df, label_col, output_dir)
    plot_full_pca(plot_df, label_col, output_dir)
    print(f"Wrote PCA plots to {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="Visualise OOP scores with PCA plots")
    parser.add_argument("--input", default="oop_scores_balanced.csv", help="Input CSV containing score columns.")
    parser.add_argument("--output-dir", default=".", help="Directory for PCA plot PNG files.")
    parser.add_argument(
        "--score-cols",
        default="score_nl,score_de,score_fy,score_en",
        help="Comma-separated score columns to use for PCA.",
    )
    parser.add_argument("--label-col", default="true_lang", help="Column containing true language labels.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_oop_visualisation(
        args.input,
        args.output_dir,
        parse_score_cols(args.score_cols),
        args.label_col,
    )


if __name__ == "__main__":
    main()

"""CLI for comparing two Label Studio annotation exports."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..annotations import compare_annotation_files, save_disagreements


def build_parser():
    parser = argparse.ArgumentParser(description="Compare two Label Studio annotation JSON exports.")
    parser.add_argument("file1", help="First annotation JSON export.")
    parser.add_argument("file2", help="Second annotation JSON export.")
    parser.add_argument("--out", help="Optional JSON path for disagreement records.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = compare_annotation_files(Path(args.file1), Path(args.file2))

    print(f"Total samples: {summary.total}")
    print(f"Agreement: {summary.agree}")
    print(f"Disagreement: {summary.disagreement_count}")
    print(f"Agreement %: {summary.agreement_rate:.2%}")

    if args.out:
        output_path = Path(args.out)
        save_disagreements(summary.disagreements, output_path)
        print(f"\nSaved disagreements to {output_path}")


if __name__ == "__main__":
    main()

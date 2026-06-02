"""CLI for preparing OWI data for Label Studio annotation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..datasets import ensure_dir
from ..preannotation import process_inputs
from .common import print_header


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare raw OWI JSONL datasets for Label Studio annotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m rsp.cli.prepare_datasets --input data/OWI_slice/frisian/raw.jsonl
  python -m rsp.cli.prepare_datasets --input data/OWI_slice/frisian/raw.jsonl --input data/OWI_slice/dutch/raw.jsonl
""",
    )
    parser.add_argument("--input", action="append", required=True, help="Raw JSONL file to process. Can be repeated.")
    parser.add_argument("--output", default="data/OWI_slice", help="Output root for per-slice cleaned and Label Studio JSON files.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    print_header("Phase 1: Pre-Annotation Dataset Preparation")
    output_dir = Path(args.output)
    ensure_dir(output_dir)
    total_count = process_inputs(args.input, output_dir)
    print_header("Complete")
    print(f"Total documents processed: {total_count}")
    print(f"Output directory: {output_dir}")
    print("\nNext step: Import these JSON files into Label Studio for annotation")
    print("After annotation, enrich corrected slices and generate model outputs.")


if __name__ == "__main__":
    main()

"""CLI for generating detector outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..datasets import get_dataset_tag, load_json_file, load_jsonl_file, load_local_dataset
from ..extractors import get_extractor
from ..outputs import generate_outputs


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate Resiliparse/FastText/URL language detection outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m rsp.cli.resiliparse_outputs --n-results 5 --split test
  python -m rsp.cli.resiliparse_outputs --extractor owi --dataset data/OWI_slice --local
  python -m rsp.cli.resiliparse_outputs --extractor wili --dataset data/WiLI_2018 --local
  python -m rsp.cli.resiliparse_outputs --extractor wiki --dataset data/GLC_balanced --local
  python -m rsp.cli.resiliparse_outputs --extractor simple --dataset my_dataset.jsonl --local
""",
    )
    parser.add_argument("--n-results", type=int, default=5, help="Number of top results per sample.")
    parser.add_argument("--output", type=str, default="resiliparse_outputs.csv", help="Output CSV filename.")
    parser.add_argument("--dataset", type=str, default="commoncrawl/CommonLID", help="HuggingFace dataset or local dataset path.")
    parser.add_argument(
        "--extractor",
        type=str,
        choices=["commonlid", "owi", "wili", "wiki", "simple", "custom"],
        default="commonlid",
        help="Dataset extractor type.",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument("--local", action="store_true", help="Treat --dataset as a local file/directory.")
    return parser


def output_path_for(args, dataset_name):
    if args.output == "resiliparse_outputs.csv":
        if dataset_name.startswith("OWI_slice_"):
            return Path("extracted") / "OWI_slice" / dataset_name.removeprefix("OWI_slice_") / "rp_outputs.csv"
        return Path("extracted") / dataset_name / "rp_outputs.csv"
    return Path(args.output)


def dataset_name_for(extractor_type: str, dataset_path: str) -> str:
    """Return the canonical result dataset name for an extractor/path pair."""
    if extractor_type == "wiki":
        return "GLC"
    if extractor_type == "wili":
        return "WiLI_2018"
    return extractor_type + "_" + get_dataset_tag(dataset_path)


def process_owi_directory(args, extractor, dataset_path: Path):
    print(f"Loading OWI slice directory: {dataset_path}")
    for slice_dir in sorted(dataset_path.iterdir()):
        if not slice_dir.is_dir():
            continue

        dataset_file = slice_dir / "enriched.json"
        if not dataset_file.exists():
            dataset_file = slice_dir / "corrected.json"
        if not dataset_file.exists():
            print(f"Skipping {slice_dir.name}: no enriched.json or corrected.json")
            continue

        dataset_name = f"OWI_slice_{slice_dir.name}"
        rows = load_json_file(dataset_file)
        output_path = output_path_for(args, dataset_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed, skipped = generate_outputs(rows, extractor, args.n_results, str(output_path))
        print(f"\nSuccessfully processed {processed:,} samples")
        print(f"Skipped: {skipped:,} samples")
        print(f"Output saved to: {output_path}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(f"Using extractor: {args.extractor}")
    extractor = get_extractor(args.extractor)

    if args.local or args.extractor in {"owi", "wili", "wiki"}:
        dataset_path = Path(args.dataset)
        if args.extractor == "owi" and dataset_path.is_dir():
            process_owi_directory(args, extractor, dataset_path)
            return

        dataset_name = dataset_name_for(args.extractor, args.dataset)
        print(f"Loading local dataset: {dataset_name}")
        dataset = load_local_dataset(args.dataset, args.extractor, split=args.split)
    else:
        from datasets import load_dataset

        dataset_name = dataset_name_for(args.extractor, args.dataset)
        print(f"Loading dataset: {dataset_name} (split: {args.split})")
        dataset = load_dataset(args.dataset)[args.split]

    print(f"\nGenerating outputs (n_results={args.n_results})...")
    output_path = output_path_for(args, dataset_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed, skipped = generate_outputs(dataset, extractor, args.n_results, str(output_path))

    print(f"\nSuccessfully processed {processed:,} samples")
    print(f"Skipped: {skipped:,} samples")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()

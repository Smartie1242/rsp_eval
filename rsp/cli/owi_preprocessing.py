"""CLI for enriching corrected OWI annotations with source metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..pipeline import enrich_owi_slices


def slice_names(slice_dir: Path, dataset_tag: str | None):
    """Return slice names to process from an OWI slice root."""
    if dataset_tag:
        return [dataset_tag]
    if not slice_dir.exists() or not slice_dir.is_dir():
        raise FileNotFoundError(f"OWI slice directory not found: {slice_dir}")
    return sorted(
        path.name
        for path in slice_dir.iterdir()
        if path.is_dir()
        and (
            (path / "raw.jsonl").exists()
            or (path / "cleaned.json").exists()
            or (path / "corrected.json").exists()
        )
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Enrich OWI corrected annotations with raw and cleaned text metadata."
    )
    parser.add_argument(
        "--slice-dir",
        type=Path,
        default=Path("data/OWI_slice"),
        help="Directory containing OWI slice folders.",
    )
    parser.add_argument(
        "--dataset-tag",
        default=None,
        help="Optional slice name to process, e.g. dutch, frisian, or random.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned enrichment without writing files.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    slices = slice_names(args.slice_dir, args.dataset_tag)
    written = enrich_owi_slices(args.slice_dir, slices=slices, dry_run=args.dry_run)
    if not written:
        print("No OWI slices were enriched.")


if __name__ == "__main__":
    main()

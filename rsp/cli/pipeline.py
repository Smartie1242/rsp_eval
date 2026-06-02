"""Run automated RSP pipeline stages up to publication visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..pipeline import (
    DEFAULT_OWI_SLICES,
    ensure_output_roots,
    enrich_owi_slices,
    run_analysis,
    run_balance_glc,
    run_detector_outputs,
    run_prepare_owi,
)


def parse_slices(value: str):
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run automated RSP pipeline stages up to publication visualization."
    )
    parser.add_argument("--slice-root", default="data/OWI_slice", help="OWI slice root directory.")
    parser.add_argument("--slices", default=",".join(DEFAULT_OWI_SLICES), help="Comma-separated OWI slices to process.")
    parser.add_argument("--glc", default="data/GLC", help="Raw GLC dataset directory.")
    parser.add_argument("--glc-balanced", default="data/GLC_balanced", help="Balanced GLC output/input directory.")
    parser.add_argument("--wili", default="data/WiLI_2018", help="WiLI dataset directory.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for GLC balancing.")
    parser.add_argument("--n-results", type=int, default=5, help="Number of ranked detector outputs.")
    parser.add_argument("--skip-commonlid", action="store_true", help="Skip CommonLID output generation.")
    parser.add_argument("--include-commonlid", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-prepare", action="store_true", help="Skip OWI Label Studio preparation.")
    parser.add_argument("--skip-enrich", action="store_true", help="Skip OWI corrected annotation enrichment.")
    parser.add_argument("--skip-balance-glc", action="store_true", help="Skip GLC balancing.")
    parser.add_argument("--skip-outputs", action="store_true", help="Skip detector output generation.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip score-gap/cutoff/length analysis.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without running them.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    slice_root = Path(args.slice_root)
    slices = parse_slices(args.slices)
    glc = Path(args.glc)
    glc_balanced = Path(args.glc_balanced)
    wili = Path(args.wili)

    ensure_output_roots(Path("extracted"), Path("results"), dry_run=args.dry_run)

    if not args.skip_prepare:
        print("\n=== Prepare OWI slices ===")
        run_prepare_owi(slice_root, slices=slices, dry_run=args.dry_run)

    if not args.skip_enrich:
        print("\n=== Enrich corrected OWI slices ===")
        enrich_owi_slices(slice_root, slices=slices, dry_run=args.dry_run)

    if not args.skip_balance_glc:
        print("\n=== Balance GLC ===")
        run_balance_glc(glc, glc_balanced, seed=args.seed, dry_run=args.dry_run)

    if not args.skip_outputs:
        print("\n=== Generate detector outputs ===")
        run_detector_outputs(
            slice_root,
            glc,
            glc_balanced,
            wili,
            n_results=args.n_results,
            include_commonlid=not args.skip_commonlid or args.include_commonlid,
            dry_run=args.dry_run,
        )

    if not args.skip_analysis:
        print("\n=== Run analysis outputs ===")
        run_analysis(dry_run=args.dry_run)

    print("\nPipeline complete up to visualization.")
    print("Next command: python -m rsp.cli.publication_visuals")


if __name__ == "__main__":
    main()

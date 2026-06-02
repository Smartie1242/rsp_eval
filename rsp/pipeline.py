"""High-level orchestration for the automated RSP research pipeline."""

from __future__ import annotations

from pathlib import Path

from .datasets import (
    enrich_owi_corrected_rows,
    ensure_dir,
    load_json,
    load_jsonl_to_dict,
    save_json,
)

DEFAULT_OWI_SLICES = ("frisian", "dutch", "random")


def existing_raw_inputs(slice_root: Path, slices=DEFAULT_OWI_SLICES):
    """Return existing OWI raw files in canonical slice folders."""
    return [slice_root / name / "raw.jsonl" for name in slices if (slice_root / name / "raw.jsonl").exists()]


def load_cleaned_lookup(path: Path):
    """Load cleaned OWI records keyed by doc_id."""
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return {row.get("doc_id"): row for row in rows if row.get("doc_id")}


def enrich_owi_slices(slice_root: Path, slices=DEFAULT_OWI_SLICES, dry_run=False):
    """Write enriched.json for OWI slices that have raw, cleaned, and corrected files."""
    written = []
    for name in slices:
        slice_dir = slice_root / name
        raw_file = slice_dir / "raw.jsonl"
        cleaned_file = slice_dir / "cleaned.json"
        corrected_file = slice_dir / "corrected.json"
        output_file = slice_dir / "enriched.json"

        missing = [path.name for path in (raw_file, cleaned_file, corrected_file) if not path.exists()]
        if missing:
            print(f"Skipping enrichment for {name}: missing {', '.join(missing)}")
            continue

        if dry_run:
            print(f"Would enrich {name}: {corrected_file} -> {output_file}")
            written.append(output_file)
            continue

        raw_lookup = load_jsonl_to_dict(raw_file, key_field="url")
        clean_lookup = load_cleaned_lookup(cleaned_file)
        corrected_rows = load_json(corrected_file)
        if not isinstance(corrected_rows, list):
            raise ValueError(f"Expected a JSON list in {corrected_file}")

        enriched_rows = enrich_owi_corrected_rows(corrected_rows, raw_lookup, clean_lookup)
        save_json(enriched_rows, output_file)
        print(f"Wrote enriched OWI file: {output_file}")
        written.append(output_file)

    return written


def run_prepare_owi(slice_root: Path, slices=DEFAULT_OWI_SLICES, dry_run=False):
    """Prepare existing OWI raw files for Label Studio."""
    inputs = existing_raw_inputs(slice_root, slices=slices)
    if not inputs:
        print(f"Skipping OWI preparation: no raw.jsonl files found under {slice_root}")
        return

    if dry_run:
        print("Would prepare OWI Label Studio files from:")
        for path in inputs:
            print(f"  {path}")
        return

    from .preannotation import process_inputs

    process_inputs([str(path) for path in inputs], slice_root)


def run_balance_glc(input_dir: Path, output_dir: Path, seed: int, dry_run=False):
    """Balance GLC if the input directory exists."""
    if not input_dir.exists():
        print(f"Skipping GLC balancing: missing {input_dir}")
        return

    if dry_run:
        print(f"Would balance GLC: {input_dir} -> {output_dir} (seed={seed})")
        return

    from .glc import balance_glc

    balance_glc(input_dir, output_dir, seed=seed)
    print(f"Balanced GLC written to: {output_dir}")


def run_detector_outputs(
    slice_root: Path,
    glc_dir: Path,
    glc_balanced_dir: Path,
    wili_dir: Path,
    n_results: int,
    include_commonlid: bool,
    dry_run=False,
):
    """Generate rp_outputs.csv for local datasets and optionally CommonLID."""
    from .cli import resiliparse_outputs

    jobs = []
    if slice_root.exists():
        jobs.append(["--extractor", "owi", "--dataset", str(slice_root), "--local", "--n-results", str(n_results)])

    glc_source = glc_balanced_dir if glc_balanced_dir.exists() else glc_dir
    if glc_source.exists():
        if glc_source == glc_balanced_dir:
            print(f"Using balanced GLC source: {glc_source}")
        else:
            print(f"Using unbalanced GLC source: {glc_source}")
        jobs.append(["--extractor", "wiki", "--dataset", str(glc_source), "--local", "--split", "test", "--n-results", str(n_results)])

    if wili_dir.exists():
        jobs.append(["--extractor", "wili", "--dataset", str(wili_dir), "--local", "--n-results", str(n_results)])

    if include_commonlid:
        jobs.append(["--extractor", "commonlid", "--dataset", "commoncrawl/CommonLID", "--split", "test", "--n-results", str(n_results)])

    if not jobs:
        print("Skipping detector outputs: no dataset inputs found.")
        return

    for argv in jobs:
        if dry_run:
            print("Would run: python -m rsp.cli.resiliparse_outputs " + " ".join(argv))
            continue
        resiliparse_outputs.main(argv)


def run_analysis(dry_run=False):
    """Run analysis commands that consume rp_outputs.csv files."""
    commands = [
        ("score_gap", "python -m rsp.cli.score_gap"),
        ("cutoff_sweep", "python -m rsp.cli.cutoff_sweep"),
        ("text_length_sweep", "python -m rsp.cli.text_length_sweep"),
    ]
    if dry_run:
        for _, command in commands:
            print(f"Would run: {command}")
        return

    from .cli import cutoff_sweep, score_gap, text_length_sweep

    score_gap.main([])
    cutoff_sweep.main([])
    text_length_sweep.main([])


def ensure_output_roots(*paths: Path, dry_run=False):
    """Create non-data output roots used by the orchestrator."""
    if dry_run:
        for path in paths:
            print(f"Would ensure directory: {path}")
        return
    for path in paths:
        ensure_dir(path)

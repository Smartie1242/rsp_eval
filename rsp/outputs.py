"""CSV output generation for detector results."""

from __future__ import annotations

import csv

from .detectors import run_model
from .extractors import DatasetExtractor
from .languages import normalize_to_iso3


def build_output_fieldnames(extractor: DatasetExtractor, n_results: int):
    """Build the rp_outputs.csv header while preserving the legacy schema."""
    fieldnames = ["label"]
    fieldnames.extend(extractor.metadata_headers())
    fieldnames.extend(["text_length", "runtime_rp", "runtime_ft", "runtime_url"])
    for i in range(1, n_results + 1):
        fieldnames.append(f"rank_{i}_lang_rp")
        fieldnames.append(f"rank_{i}_oop_score")
        fieldnames.append(f"rank_{i}_lang_ft")
        fieldnames.append(f"rank_{i}_probs")
    fieldnames.append("lang_url")
    fieldnames.append("url_score")
    return fieldnames


def build_output_row(row, extractor: DatasetExtractor, n_results: int):
    """Run detectors for one input row and build a CSV row."""
    text = extractor.extract_text(row)
    label = extractor.extract_label(row)
    url = extractor.extract_url(row)

    if label is None:
        return None

    results_rp, runtime_rp = run_model(text, "rp", n_results)
    results_ft, runtime_ft = run_model(text, "ft", n_results)
    (url_lang, url_score), runtime_url = run_model(url, "url", n_results)

    if not (results_rp or results_ft):
        return None

    row_data = {"label": label}
    row_data.update(extractor.extract_metadata(row))
    row_data["text_length"] = len(text)
    row_data["runtime_rp"] = runtime_rp
    row_data["runtime_ft"] = runtime_ft
    row_data["runtime_url"] = runtime_url

    for rank, (lang, oop_score) in enumerate(results_rp, 1):
        row_data[f"rank_{rank}_lang_rp"] = normalize_to_iso3(lang)
        row_data[f"rank_{rank}_oop_score"] = oop_score

    for rank, (lang, probs) in enumerate(results_ft, 1):
        row_data[f"rank_{rank}_lang_ft"] = normalize_to_iso3(lang)
        row_data[f"rank_{rank}_probs"] = probs

    row_data["lang_url"] = normalize_to_iso3(url_lang)
    row_data["url_score"] = url_score
    return row_data


def generate_outputs(
    dataset, extractor: DatasetExtractor, n_results=5, output_filename="resiliparse_outputs.csv"
):
    """Generate detector outputs and write them to CSV."""
    processed = 0
    skipped = 0

    with open(output_filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=build_output_fieldnames(extractor, n_results)
        )
        writer.writeheader()

        for row in dataset:
            row_data = build_output_row(row, extractor, n_results)
            if row_data is None:
                skipped += 1
                continue

            writer.writerow(row_data)
            processed += 1

            if processed % 10000 == 0:
                print(f"  Processed: {processed:,} samples (skipped: {skipped:,})")

    return processed, skipped


"""Helpers for balancing GLC language datasets."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from .datasets import ensure_dir


def read_split_lines(input_dir: Path, split: str):
    """Read non-empty lines for each language in one split."""
    language_lines = {}
    for language_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        split_file = language_dir / f"{split}.txt"
        if not split_file.exists():
            continue
        with open(split_file, "r", encoding="utf-8") as handle:
            language_lines[language_dir.name] = [line for line in handle if line.strip()]
    return language_lines


def balance_split(input_dir: Path, output_dir: Path, split: str, seed: int):
    """Balance one split by downsampling each language to the smallest count."""
    language_lines = read_split_lines(input_dir, split)
    if not language_lines:
        return {}, {}

    input_counts = {language: len(lines) for language, lines in language_lines.items()}
    target_count = min(input_counts.values())
    output_counts = {}

    for language, lines in language_lines.items():
        rng = random.Random(f"{seed}:{split}:{language}")
        sampled = rng.sample(lines, target_count)
        language_dir = ensure_dir(output_dir / language)
        with open(language_dir / f"{split}.txt", "w", encoding="utf-8") as handle:
            handle.writelines(sampled)
        output_counts[language] = len(sampled)

    return input_counts, output_counts


def balance_glc(input_dir, output_dir, seed=42, splits=None):
    """Balance a GLC dataset and write a manifest."""
    input_dir = Path(input_dir)
    output_dir = ensure_dir(Path(output_dir))
    splits = splits or ["train", "val", "test"]

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"GLC input directory not found: {input_dir}")

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "strategy": "downsample_each_split_to_smallest_language_count",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": {},
    }

    for split in splits:
        input_counts, output_counts = balance_split(input_dir, output_dir, split, seed)
        if input_counts:
            manifest["splits"][split] = {
                "input_counts": input_counts,
                "output_counts": output_counts,
                "target_count": min(output_counts.values()),
            }

    manifest_path = output_dir / "balance_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    return manifest


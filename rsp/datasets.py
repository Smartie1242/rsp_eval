"""Dataset and file I/O helpers used by the research commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_DATASET_PATHS = {
    "OWI_slice_dutch": "extracted/OWI_slice/dutch/rp_outputs.csv",
    "OWI_slice_frisian": "extracted/OWI_slice/frisian/rp_outputs.csv",
    "OWI_slice_random": "extracted/OWI_slice/random/rp_outputs.csv",
    "commonlid": "extracted/commonlid_CommonLID/rp_outputs.csv",
    "WiLI_2018": "extracted/WiLI_2018/rp_outputs.csv",
    "GLC": "extracted/GLC/rp_outputs.csv",
}


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv(path):
    """Load a CSV file into a pandas DataFrame."""
    import pandas as pd

    return pd.read_csv(path)


def load_jsonl(path: Path) -> List[Dict]:
    """Load a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(data: List[Dict], path: Path) -> None:
    """Save records to JSONL."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_json(path: Path):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path) -> None:
    """Save JSON data."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_jsonl_to_dict(path: Path, key_field: str = "doc_id") -> Dict[str, Dict]:
    """Load JSONL records keyed by a selected field."""
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            key = data.get(key_field)
            if key is not None:
                records[key] = data
    return records


def load_folder_jsonl_to_dict(path: Path, key_field: str = "doc_id") -> Dict[str, Dict]:
    """Load all JSONL files in a folder keyed by a selected field."""
    records = {}
    for file_path in sorted(path.iterdir()):
        if file_path.suffix.lower() == ".jsonl":
            records.update(load_jsonl_to_dict(file_path, key_field=key_field))
    return records


def load_jsonl_file(path: Path) -> List[Dict]:
    """Compatibility alias for JSONL loading."""
    return load_jsonl(path)


def load_json_file(path: Path) -> List[Dict]:
    """Load a JSON file containing one record or a list of records."""
    data = load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON content in {path}")


def load_wili_dataset(dataset_path: str) -> List[Dict]:
    """Load WiLI-2018 x_test/y_test files as records."""
    dataset_dir = Path(dataset_path)
    x_path = dataset_dir / "x_test.txt"
    y_path = dataset_dir / "y_test.txt"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"WiLI dataset not found at {dataset_path}")

    with open(x_path, "r", encoding="utf-8") as fx, open(
        y_path, "r", encoding="utf-8"
    ) as fy:
        texts = [line.rstrip("\n") for line in fx]
        labels = [line.strip() for line in fy]

    if len(texts) != len(labels):
        raise ValueError(
            f"WiLI x_test and y_test length mismatch: {len(texts)} vs {len(labels)}"
        )

    return [
        {"text": text, "label": label, "split": "test", "source_index": idx}
        for idx, (text, label) in enumerate(zip(texts, labels), start=1)
    ]


def load_wiki_dataset(dataset_path: str, split: str = "train") -> List[Dict]:
    """Load a wiki language-directory dataset."""
    dataset_dir = Path(dataset_path)
    rows = []

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Wiki dataset directory not found: {dataset_path}")

    for language_dir in sorted(dataset_dir.iterdir()):
        if not language_dir.is_dir():
            continue

        language = language_dir.name
        split_file = language_dir / f"{split}.txt"
        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, start=1):
                    text = line.strip()
                    if text:
                        rows.append(
                            {
                                "doc_id": f"{language}/{split}_{idx}",
                                "text": text,
                                "label": language,
                                "language": language,
                                "source_file": str(split_file),
                            }
                        )
            continue

        for path in sorted(language_dir.rglob("*")):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                rows.append(
                    {
                        "doc_id": str(path.relative_to(dataset_dir)),
                        "text": text,
                        "label": language,
                        "language": language,
                        "source_file": str(path),
                    }
                )

    return rows


def load_local_dataset(dataset_path: str, extractor_type: str, split: str = None):
    """Load a local dataset based on extractor type and file extension."""
    path = Path(dataset_path)

    if extractor_type == "wili":
        return load_wili_dataset(dataset_path)
    if extractor_type == "wiki":
        return load_wiki_dataset(dataset_path, split=split or "train")
    if path.is_dir():
        rows = []
        for local_path in sorted(path.iterdir()):
            if local_path.suffix.lower() == ".jsonl":
                rows.extend(load_jsonl_file(local_path))
            elif local_path.suffix.lower() == ".json":
                rows.extend(load_json_file(local_path))
        return rows

    if path.suffix.lower() == ".jsonl":
        return load_jsonl_file(path)
    if path.suffix.lower() == ".json":
        return load_json_file(path)

    raise ValueError(f"Unsupported local dataset path: {dataset_path}")


def get_dataset_tag(dataset_path: str) -> str:
    """Convert a dataset path into the tag used by result folder names."""
    base = os.path.basename(dataset_path)
    base = os.path.splitext(base)[0]
    return base.split("_")[0]


def parse_dataset_overrides(overrides: List[str] | None) -> Dict[str, str]:
    """Parse repeated NAME=PATH dataset override values."""
    parsed = {}
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Dataset override must use NAME=PATH: {override}")
        name, path = override.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Dataset override must use NAME=PATH: {override}")
        parsed[name] = path
    return parsed


def resolve_dataset_paths(overrides: List[str] | None = None) -> Dict[str, str]:
    """Return default dataset paths with optional NAME=PATH replacements/additions."""
    paths = dict(DEFAULT_DATASET_PATHS)
    paths.update(parse_dataset_overrides(overrides))
    return paths


def iter_existing_datasets(dataset_paths: Dict[str, str]):
    """Yield existing dataset CSV paths and print skips for missing paths."""
    for dataset_name, path in dataset_paths.items():
        if not os.path.exists(path):
            print(f"Skipping {dataset_name} (file not found: {path})")
            continue
        yield dataset_name, path


def enrich_owi_corrected_rows(
    rows: List[Dict], raw_lookup: Dict[str, Dict], clean_lookup: Dict[str, Dict]
) -> List[Dict]:
    """Enrich corrected OWI annotation rows with raw and cleaned text metadata."""
    enriched = []
    for row in rows:
        data = row.get("data", {})
        doc_id = data.get("doc_id")
        raw_record = raw_lookup.get(doc_id, raw_lookup.get(data.get("url", ""), {}))
        clean_record = clean_lookup.get(doc_id, {}) if doc_id else {}

        data["raw_text"] = raw_record.get(
            "main_content", raw_record.get("text", raw_record.get("raw_text", ""))
        )
        data["clean_text"] = clean_record.get("text", clean_record.get("clean_text", ""))
        data["raw_length"] = raw_record.get(
            "raw_length", raw_record.get("clean_length", len(data.get("raw_text", "")))
        )
        data["clean_length"] = clean_record.get(
            "clean_length", len(data.get("clean_text", ""))
        )

        row["data"] = data
        enriched.append(row)
    return enriched

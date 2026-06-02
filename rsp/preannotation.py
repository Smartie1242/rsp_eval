"""Pre-annotation dataset preparation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from .datasets import ensure_dir, load_jsonl, save_json


def clean_document(doc: Dict, source_name: str, idx: int) -> Dict:
    """Clean a raw OWI document into plain text metadata."""
    from resiliparse.extract.html2text import extract_plain_text

    url = doc.get("url", "")
    title = doc.get("title", "")
    raw_html = doc.get("main_content", "")

    try:
        clean_text = extract_plain_text(raw_html)
    except Exception as exc:
        print(f"Error cleaning HTML for {source_name}_{idx:07d}: {exc}")
        clean_text = ""

    return {
        "doc_id": f"{source_name}_{idx:07d}",
        "source": source_name,
        "url": url,
        "title": title,
        "text": clean_text,
        "raw_length": len(raw_html),
        "clean_length": len(clean_text),
    }


def create_labelstudio_task(doc: Dict) -> Dict:
    """Convert a cleaned document into a Label Studio task."""
    return {
        "data": {
            "doc_id": doc["doc_id"],
            "url": doc.get("url", ""),
            "title": doc.get("title", ""),
            "text": doc.get("text", ""),
            "source": doc.get("source", ""),
        }
    }


def source_name_from_path(input_path: Path) -> str:
    """Derive a slice/source name from an input filename or slice folder."""
    if input_path.name == "raw.jsonl" and input_path.parent.name:
        return input_path.parent.name
    return input_path.stem.replace("_sample", "").replace("_clean", "")


def process_dataset(input_path: Path, source_name: str, output_dir: Path) -> int:
    """Clean one raw JSONL dataset and write cleaned/Label Studio JSON files."""
    slice_dir = ensure_dir(output_dir / source_name)
    print(f"\n[{source_name}] Processing {input_path.name}...")

    raw_docs = load_jsonl(input_path)
    cleaned_docs = []
    for idx, doc in enumerate(raw_docs):
        if not doc.get("main_content"):
            continue
        cleaned_docs.append(clean_document(doc, source_name, idx))

    print(f"Cleaned {len(cleaned_docs)} documents")

    labelstudio_tasks = [create_labelstudio_task(doc) for doc in cleaned_docs]

    cleaned_output_file = slice_dir / "cleaned.json"
    save_json(cleaned_docs, cleaned_output_file)
    print(f"Saved cleaned dataset to {cleaned_output_file}")

    labelstudio_output_file = slice_dir / "labelstudio.json"
    save_json(labelstudio_tasks, labelstudio_output_file)
    print(f"Saved Label Studio tasks to {labelstudio_output_file}")

    return len(labelstudio_tasks)


def process_inputs(input_files: Iterable[str], output_dir: Path) -> int:
    """Process multiple input files and return the total task count."""
    ensure_dir(output_dir)
    total_count = 0
    for input_file in input_files:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"File not found: {input_path}")
            continue
        total_count += process_dataset(
            input_path, source_name_from_path(input_path), output_dir
        )
    return total_count


class PreAnnotationPipeline:
    """Compatibility facade for the previous class-based API."""

    clean_document = staticmethod(clean_document)
    create_labelstudio_task = staticmethod(create_labelstudio_task)
    process_dataset = staticmethod(process_dataset)

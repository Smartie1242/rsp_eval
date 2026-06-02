"""Annotation comparison helpers for Label Studio exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComparisonSummary:
    """Agreement summary for two annotation exports."""

    total: int
    agree: int
    disagreements: list[dict]

    @property
    def disagreement_count(self) -> int:
        return self.total - self.agree

    @property
    def agreement_rate(self) -> float:
        return self.agree / self.total if self.total else 0.0


def extract_label(item):
    """Extract the first selected Label Studio choice from one task."""
    try:
        return item["annotations"][0]["result"][0]["value"]["choices"][0]
    except (IndexError, KeyError):
        return None


def load_annotations(path: Path):
    """Load a Label Studio export into normalized comparison records."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    normalized = []
    for new_id, item in enumerate(data, start=1):
        normalized.append(
            {
                "id": new_id,
                "task_id": item.get("id"),
                "doc_id": item.get("data", {}).get("doc_id"),
                "label": extract_label(item),
            }
        )

    return normalized


def align_annotations(a1, a2):
    """Align annotations by stable doc_id when possible, otherwise by row order."""
    if all(item.get("doc_id") for item in a1 + a2):
        by_doc_1 = {item["doc_id"]: item for item in a1}
        by_doc_2 = {item["doc_id"]: item for item in a2}
        if set(by_doc_1) != set(by_doc_2):
            raise ValueError("Files have different doc_id sets")
        return [(by_doc_1[doc_id], by_doc_2[doc_id]) for doc_id in sorted(by_doc_1)]

    if len(a1) != len(a2):
        raise ValueError("Files have different number of samples")
    return list(zip(a1, a2))


def compare_annotations(a1, a2) -> ComparisonSummary:
    """Compare two normalized annotation lists."""
    aligned = align_annotations(a1, a2)
    agree = 0
    disagreements = []

    for x, y in aligned:
        if x["label"] == y["label"]:
            agree += 1
        else:
            disagreements.append(
                {
                    "id": x["id"],
                    "doc_id": x.get("doc_id"),
                    "annotator1_task_id": x.get("task_id"),
                    "annotator2_task_id": y.get("task_id"),
                    "annotator1": x["label"],
                    "annotator2": y["label"],
                }
            )

    return ComparisonSummary(total=len(aligned), agree=agree, disagreements=disagreements)


def compare_annotation_files(file1: Path, file2: Path) -> ComparisonSummary:
    """Load and compare two Label Studio annotation export files."""
    return compare_annotations(load_annotations(file1), load_annotations(file2))


def save_disagreements(disagreements: list[dict], output_path: Path) -> None:
    """Write disagreement records as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(disagreements, handle, indent=2, ensure_ascii=False)

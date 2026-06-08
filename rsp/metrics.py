"""Metric and feature helpers for analysis commands."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

UNEVALUABLE_LANGUAGE_VALUES = {"unknown", "unk", "none", "mixed", "", None}
EXCLUDED_EVALUATION_LABELS = {"mixed", "unknown"}
WILI_SOURCE_DATASET = "WiLI_2018"
WILI_EVAL_OVERLAP_DATASET = "WiLI_2018_eval_overlap"


def filter_evaluation_rows(df, dataset: str = "", label_column: str = "label"):
    """Drop labels that cannot be evaluated by single-label LID models."""
    if label_column not in df.columns:
        return df

    before = len(df)
    labels = df[label_column].fillna("unknown").astype(str).str.lower()
    filtered = df[~labels.isin(EXCLUDED_EVALUATION_LABELS)].copy().reset_index(drop=True)
    dropped = before - len(filtered)
    if dropped:
        prefix = f"{dataset}: " if dataset else ""
        print(f"[filter] {prefix}dropped {dropped:,} unevaluable row(s)")
    return filtered


def is_wili_dataset_name(dataset: str) -> bool:
    """Return whether a dataset name refers to the WiLI source or filtered view."""
    return str(dataset) in {WILI_SOURCE_DATASET, WILI_EVAL_OVERLAP_DATASET}


def filter_wili_eval_overlap_frames(
    frames,
    label_column: str = "label",
    rename_wili: bool = True,
):
    """Filter WiLI to the gold-label universe covered by the other evaluation datasets.

    The source WiLI output is intentionally left unchanged on disk. This helper
    creates the publication-facing comparison view used by tables and figures:
    WiLI labels are retained only when they also appear as gold labels in the
    non-WiLI evaluation datasets.
    """
    if not frames:
        return frames, set()

    label_universe = set()
    wili_names = []
    for dataset, df in frames.items():
        if label_column not in df.columns:
            continue
        if is_wili_dataset_name(dataset):
            wili_names.append(dataset)
            continue
        labels = df[label_column].fillna("unknown").astype(str).str.lower()
        label_universe.update(labels[~labels.isin(EXCLUDED_EVALUATION_LABELS)])

    if not wili_names or not label_universe:
        return frames, label_universe

    filtered_frames = {}
    for dataset, df in frames.items():
        if not is_wili_dataset_name(dataset):
            filtered_frames[dataset] = df.reset_index(drop=True)
            continue

        labels = df[label_column].fillna("unknown").astype(str).str.lower()
        retained = df[labels.isin(label_universe)].copy().reset_index(drop=True)
        retained_labels = retained[label_column].fillna("unknown").astype(str).str.lower().nunique()
        output_name = WILI_EVAL_OVERLAP_DATASET if rename_wili else dataset
        print(
            "[filter] "
            f"{dataset}: retained {len(retained):,}/{len(df):,} row(s) "
            f"across {retained_labels:,} label(s) for thesis evaluation overlap"
        )
        filtered_frames[output_name] = retained

    return filtered_frames, label_universe


def safe_f1(y_true, y_pred):
    """Closed-set macro F1 over the gold labels that returns 0.0 for empty inputs."""
    if len(y_true) == 0:
        return 0.0
    from sklearn.metrics import f1_score

    labels = sorted(set(y_true))
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def safe_accuracy(y_true, y_pred):
    """Accuracy that returns 0.0 for empty inputs."""
    if len(y_true) == 0:
        return 0.0
    from sklearn.metrics import accuracy_score

    return accuracy_score(y_true, y_pred)


def add_k_score_features(df, max_rank=5):
    """Add Resiliparse score-gap features."""
    df = df.copy()
    df["is_correct"] = df["label"] == df["rank_1_lang_rp"]
    top1_scores = df["rank_1_oop_score"]

    for rank in range(2, max_rank + 1):
        score_col = f"rank_{rank}_oop_score"
        if score_col not in df.columns:
            print(f"Skipping missing column: {score_col}")
            continue
        df[f"gap_1_{rank}"] = df[score_col] - top1_scores

    return df


def add_k_score_features_ft(df, max_rank=5):
    """Add FastText probability-gap features."""
    df = df.copy()
    df["is_correct"] = df["label"] == df["rank_1_lang_ft"]
    top1_scores = df["rank_1_probs"]

    for rank in range(2, max_rank + 1):
        score_col = f"rank_{rank}_probs"
        if score_col not in df.columns:
            print(f"Skipping missing column: {score_col}")
            continue
        df[f"gap_1_{rank}"] = top1_scores - df[score_col]

    return df


def compute_summary(df):
    """Create descriptive statistics for score gaps."""
    gap_cols = [col for col in df.columns if col.startswith("gap_1_")]
    return df.groupby("is_correct")[gap_cols].agg(["mean", "median", "std"]).round(2)


def normalize_prediction_columns(df):
    """Fill missing prediction labels with `unknown` strings."""
    df = df.copy()
    for col in ["label", "rank_1_lang_rp", "rank_1_lang_ft", "lang_url"]:
        df[col] = df[col].fillna("unknown").astype(str)
    return df


def compute_confusion_matrix(records: List[Dict]) -> Tuple[Dict, List[str]]:
    """Compute confusion matrix from prediction records."""
    from .languages import normalize_language

    matrix = defaultdict(lambda: defaultdict(int))
    labels = set()

    for record in records:
        annotated = record.get("annotated_label")
        predicted = record.get("predicted_label")
        annotated = normalize_language(annotated) if annotated else None

        if annotated is None or predicted is None:
            continue

        matrix[annotated][predicted] += 1
        labels.add(annotated)
        labels.add(predicted)

    return dict(matrix), sorted(labels)


def compute_accuracy(records: List[Dict]) -> Tuple[int, int, float]:
    """Compute record-level accuracy from prediction records."""
    from .languages import normalize_language

    correct = 0
    total = 0
    for record in records:
        annotated = record.get("annotated_label")
        predicted = record.get("predicted_label")
        annotated = normalize_language(annotated) if annotated else None
        if annotated is None or predicted is None:
            continue
        total += 1
        if annotated == predicted:
            correct += 1

    accuracy = correct / total if total else 0
    return correct, total, accuracy


def compute_metrics(matrix: Dict, labels: List[str]) -> Dict:
    """Compute per-label precision, recall, F1, and support."""
    metrics = {}
    for label in labels:
        tp = matrix.get(label, {}).get(label, 0)
        fp = sum(matrix.get(a, {}).get(label, 0) for a in labels if a != label)
        fn = sum(matrix.get(label, {}).get(p, 0) for p in labels if p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
    return metrics


def collect_errors(records: List[Dict]) -> List[Dict]:
    """Collect misclassified samples."""
    from .languages import normalize_language

    errors = []
    for record in records:
        annotated = record.get("annotated_label")
        predicted = record.get("predicted_label")
        annotated = normalize_language(annotated) if annotated else None
        if annotated != predicted and annotated is not None and predicted is not None:
            errors.append(
                {
                    "doc_id": record.get("doc_id"),
                    "annotated_label": annotated,
                    "predicted_label": predicted,
                    "prediction_score": record.get("prediction_score"),
                }
            )
    return errors


TEXT_BINS = [
    (0, 50, "0-50"),
    (50, 100, "50-100"),
    (100, 200, "100-200"),
    (200, 500, "200-500"),
    (500, float("inf"), "500+"),
]


def get_text_bin(length: int) -> str:
    """Map a text length to a coarse bin label."""
    for low, high, label in TEXT_BINS:
        if low <= length < high:
            return label
    return "unknown"

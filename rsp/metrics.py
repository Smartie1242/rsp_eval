"""Metric and feature helpers for analysis commands."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple


def safe_f1(y_true, y_pred):
    """Macro F1 that returns 0.0 for empty inputs."""
    if len(y_true) == 0:
        return 0.0
    from sklearn.metrics import f1_score

    return f1_score(y_true, y_pred, average="macro", zero_division=0)


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


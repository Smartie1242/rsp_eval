#!/usr/bin/env python3
"""Publication-ready visualizations for Resiliparse evaluation results.

This script can either:
1. derive normalized metrics and confusion tables from existing ``rp_outputs.csv``
   files, or
2. load precomputed metrics/confusion CSVs with the normalized schemas.

The plotting and table output are intentionally kept here as a single
self-contained script because the outputs are publication artifacts rather than
pipeline primitives.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from functools import lru_cache
from pathlib import Path

from .languages import OWI_LABEL_TO_ISO3, safe_tag_distance
from .metrics import (
    UNEVALUABLE_LANGUAGE_VALUES,
    WILI_EVAL_OVERLAP_DATASET,
    WILI_SOURCE_DATASET,
    filter_evaluation_rows,
    filter_wili_eval_overlap_frames,
    is_wili_dataset_name,
)
from .routing import combined_url_fasttext_decision, split_trigger_url_fasttext_decision


# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = Path("results/publication_visuals")
METRICS_FILE = OUTPUT_DIR / "normalized_metrics.csv"
CONFUSION_FILE = OUTPUT_DIR / "normalized_confusion.csv"
TABLES_DIRNAME = "tables"
NORMALIZED_DIRNAME = "normalized"
CONFUSION_DIRNAME = "confusion_matrices"
FAMILY_CONFUSION_DIRNAME = "families"
GLOBAL_CONFUSION_DIRNAME = "global_top20"
TEXT_LENGTH_DIRNAME = "text_length"
RESOURCE_LEVEL_DIRNAME = "resource_levels"
RUNTIME_DIRNAME = "runtime"
LANGUAGE_SIMILARITY_DIRNAME = "language_similarity"
SUMMARY_PLOTS_DIRNAME = "summary_plots"

RP_OUTPUT_FILES = {
    "OWI_slice_dutch": Path("extracted/OWI_slice/dutch/rp_outputs.csv"),
    "OWI_slice_frisian": Path("extracted/OWI_slice/frisian/rp_outputs.csv"),
    "OWI_slice_random": Path("extracted/OWI_slice/random/rp_outputs.csv"),
    "GLC": Path("extracted/GLC/rp_outputs.csv"),
    "WiLI_2018": Path("extracted/WiLI_2018/rp_outputs.csv"),
    "CommonLID": Path("extracted/commonlid_CommonLID/rp_outputs.csv"),
}

LENGTH_SWEEP_FILES = {
    "OWI_slice_dutch": Path("results/length_sweep/OWI_slice_dutch/length_sweep.csv"),
    "OWI_slice_frisian": Path("results/length_sweep/OWI_slice_frisian/length_sweep.csv"),
    "OWI_slice_random": Path("results/length_sweep/OWI_slice_random/length_sweep.csv"),
    "GLC": Path("results/length_sweep/GLC/length_sweep.csv"),
    "WiLI_2018_eval_overlap": Path("results/length_sweep/WiLI_2018_eval_overlap/length_sweep.csv"),
    "CommonLID": Path("results/length_sweep/commonlid/length_sweep.csv"),
}

HYBRID_CUTOFF = 1200.0
LANG_AWARE_DISTANCE_THRESHOLD = 30
HIGH_RESOURCE_ARTICLES = 1_000_000
LOW_RESOURCE_ARTICLES = 100_000
INCLUDE_UNKNOWN_RESOURCE = True

MODEL_DEFINITIONS = {
    "Resiliparse": "rank_1_lang_rp",
    "FastText": "rank_1_lang_ft",
    "RP+FastText": "hybrid",
}

EXTENDED_MODEL_ORDER = [
    "Resiliparse",
    "FastText",
    "RP+FastText",
    "RP+URL",
    "RP+FastText lang-aware",
    "RP+URL lang-aware",
    "RP+FastText+URL lang-aware",
    "RP+FastText+URL split-trigger",
]

# Resource-level grouping is intentionally limited to languages supported by the
# local detector pipeline. Counts are deliberately editable; supported languages
# without a local count are shown as "unknown" resource level.
RESILIPARSE_SUPPORTED_LANGUAGE_CODES = {
    code
    for code in OWI_LABEL_TO_ISO3.values()
    if code not in {"mixed", "unknown", "other"}
}
WIKIPEDIA_ARTICLE_COUNTS = {
    "eng": 6_900_000,
    "deu": 2_900_000,
    "nld": 2_200_000,
    "fra": 2_600_000,
    "spa": 1_900_000,
    "ita": 1_900_000,
    "por": 1_100_000,
    "rus": 2_000_000,
    "zho": 1_400_000,
    "jpn": 1_400_000,
    "pol": 1_600_000,
    "ukr": 1_300_000,
    "ara": 1_200_000,
    "fas": 1_000_000,
    "vie": 1_300_000,
    "ind": 700_000,
    "tur": 600_000,
    "kor": 690_000,
    "hin": 165_000,
    "ben": 160_000,
    "fry": 55_000,
    "mlg": 100_000,
    "swh": 80_000,
    "hau": 30_000,
    "bel": 250_000,
    "bul": 300_000,
    "mon": 25_000,
    "nds": 85_000,
    "mwl": 5_000,
    "ava": 5_000,
    "tcy": 3_000,
    "bjn": 4_000,
    "glk": 10_000,
    "lez": 12_000,
}

FASTTEXT_SUPPORTED_LANGUAGE_CODES = set(WIKIPEDIA_ARTICLE_COUNTS)
RESOURCE_SUPPORTED_LANGUAGE_CODES = RESILIPARSE_SUPPORTED_LANGUAGE_CODES | FASTTEXT_SUPPORTED_LANGUAGE_CODES

SUPPORTED_WIKIPEDIA_ARTICLE_COUNTS = {
    language: count
    for language, count in WIKIPEDIA_ARTICLE_COUNTS.items()
    if language in RESOURCE_SUPPORTED_LANGUAGE_CODES
}

LANGUAGE_FAMILIES = {
    "Germanic": {"eng", "deu", "nld", "fry", "nds", "dan", "swe", "nor", "isl"},
    "Romance": {"fra", "spa", "ita", "por", "ron", "cat", "glg", "oci", "mwl"},
    "Slavic": {"rus", "ukr", "bel", "pol", "ces", "slk", "slv", "bul", "srp", "hrv"},
    "Indo-Aryan": {"hin", "ben", "urd", "mar", "guj", "pan", "nep", "asm", "ori", "snd"},
    "Sino-Tibetan": {"zho", "mya", "bod"},
    "Bantu (Niger-Congo)": {"swh", "zul", "xho", "kin", "run", "lin"},
}

REQUIRED_METRIC_COLUMNS = {
    "dataset",
    "model",
    "language",
    "language_family",
    "resource_level",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "precision_weighted",
    "recall_weighted",
}

REQUIRED_CONFUSION_COLUMNS = {
    "dataset",
    "model",
    "true_label",
    "predicted_label",
    "count",
}

REQUIRED_LENGTH_SWEEP_COLUMNS = {
    "text_length_bin",
    "model",
    "accuracy",
    "f1_macro",
}
TEXT_LENGTH_PLOT_MODELS = {"Resiliparse", "FastText", "RP+FastText"}

REQUIRED_RUNTIME_COLUMNS = {
    "runtime_rp",
    "runtime_ft",
}

REQUIRED_LANGUAGE_SIMILARITY_COLUMNS = {
    "label",
    "rank_1_lang_rp",
    "rank_2_lang_rp",
    "rank_1_oop_score",
    "rank_2_oop_score",
}

EXCLUDED_EVALUATION_LABELS = {"mixed", "unknown"}
URL_EVALUATION_MODEL_ORDER = [
    "Always English",
    "URL",
    "Resiliparse",
    "FastText",
    "RP+URL",
    "RP+URL lang-aware",
    "RP+FastText+URL lang-aware",
    "RP+FastText+URL split-trigger",
]


# =============================================================================
# DATA NORMALIZATION
# =============================================================================


def log(stage: str, message: str) -> None:
    """Print a compact, consistently formatted status message."""
    print(f"[{stage}] {message}", flush=True)


def safe_name(value: str) -> str:
    """Convert a label to a filesystem-safe name."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def has_url_metadata(dataset: str) -> bool:
    """Return whether a dataset has meaningful URL metadata for URL models."""
    return str(dataset).startswith("OWI_slice")


def is_url_model(model: str) -> bool:
    """Return whether a model depends on URL-derived language."""
    return "URL" in str(model)


def language_family(language: str) -> str:
    for family, languages in LANGUAGE_FAMILIES.items():
        if language in languages:
            return family
    return "Other"


def resource_level(language: str) -> str:
    if language not in RESOURCE_SUPPORTED_LANGUAGE_CODES:
        return "unknown"
    count = SUPPORTED_WIKIPEDIA_ARTICLE_COUNTS.get(language)
    if count is None:
        return "unknown"
    if count >= HIGH_RESOURCE_ARTICLES:
        return "high"
    if count < LOW_RESOURCE_ARTICLES:
        return "low"
    return "mid"


def load_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path)


def require_columns(df, required: set[str], path: Path) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def dataset_display_label(dataset: str) -> str:
    """Return the thesis-facing display name for dataset identifiers."""
    if dataset == WILI_EVAL_OVERLAP_DATASET:
        return "WiLI_Filtered"
    if dataset == WILI_SOURCE_DATASET:
        return "WiLI-2018"
    return str(dataset)


def filter_normalized_wili_eval_overlap(metrics_df, confusion_df):
    """Apply the WiLI evaluation-overlap view to pre-normalized input tables."""
    import pandas as pd

    label_universe = set()
    if confusion_df is not None and not confusion_df.empty and "true_label" in confusion_df.columns:
        non_wili = ~confusion_df["dataset"].astype(str).map(is_wili_dataset_name)
        labels = confusion_df.loc[non_wili, "true_label"].fillna("unknown").astype(str).str.lower()
        label_universe.update(labels[~labels.isin(EXCLUDED_EVALUATION_LABELS)])
    if metrics_df is not None and not metrics_df.empty and "language" in metrics_df.columns:
        non_wili = ~metrics_df["dataset"].astype(str).map(is_wili_dataset_name)
        labels = metrics_df.loc[non_wili, "language"].fillna("unknown").astype(str).str.lower()
        label_universe.update(labels[~labels.isin(EXCLUDED_EVALUATION_LABELS)])

    if not label_universe:
        return metrics_df, confusion_df

    if metrics_df is not None and not metrics_df.empty and "language" in metrics_df.columns:
        wili_mask = metrics_df["dataset"].astype(str).map(is_wili_dataset_name)
        keep_mask = ~wili_mask | metrics_df["language"].fillna("unknown").astype(str).str.lower().isin(label_universe)
        before = int(wili_mask.sum())
        metrics_df = metrics_df[keep_mask].copy()
        metrics_df.loc[metrics_df["dataset"].astype(str).map(is_wili_dataset_name), "dataset"] = WILI_EVAL_OVERLAP_DATASET
        after = int((metrics_df["dataset"] == WILI_EVAL_OVERLAP_DATASET).sum())
        if before:
            log("filter", f"{WILI_SOURCE_DATASET}: retained {after:,}/{before:,} normalized metric row(s)")

    if confusion_df is not None and not confusion_df.empty and "true_label" in confusion_df.columns:
        wili_mask = confusion_df["dataset"].astype(str).map(is_wili_dataset_name)
        keep_mask = ~wili_mask | confusion_df["true_label"].fillna("unknown").astype(str).str.lower().isin(label_universe)
        before = int(wili_mask.sum())
        confusion_df = confusion_df[keep_mask].copy()
        confusion_df.loc[confusion_df["dataset"].astype(str).map(is_wili_dataset_name), "dataset"] = WILI_EVAL_OVERLAP_DATASET
        after = int((confusion_df["dataset"] == WILI_EVAL_OVERLAP_DATASET).sum())
        if before:
            log("filter", f"{WILI_SOURCE_DATASET}: retained {after:,}/{before:,} normalized confusion row(s)")

    return metrics_df, confusion_df


def prepare_wili_eval_overlap_frames(frames: dict[str, object]):
    """Filter loaded rp_outputs frames to the publication-facing WiLI view."""
    filtered_frames, label_universe = filter_wili_eval_overlap_frames(frames)
    if label_universe:
        log("filter", f"evaluation universe: {len(label_universe):,} non-WiLI label(s)")
    return filtered_frames


def filter_confusion_rows(confusion_df):
    """Drop unevaluable true-label rows from normalized confusion data."""
    if confusion_df is None or confusion_df.empty or "true_label" not in confusion_df.columns:
        return confusion_df

    before = len(confusion_df)
    labels = confusion_df["true_label"].fillna("unknown").astype(str).str.lower()
    confusion_df = confusion_df[~labels.isin(EXCLUDED_EVALUATION_LABELS)].copy()
    dropped = before - len(confusion_df)
    if dropped:
        log("filter", f"dropped {dropped:,} unevaluable confusion row(s)")
    return confusion_df


def filter_metrics_rows(metrics_df):
    """Drop unevaluable language rows from normalized metric data."""
    if metrics_df is None or metrics_df.empty or "language" not in metrics_df.columns:
        return metrics_df

    before = len(metrics_df)
    labels = metrics_df["language"].fillna("unknown").astype(str).str.lower()
    metrics_df = metrics_df[~labels.isin(EXCLUDED_EVALUATION_LABELS)].copy()
    dropped = before - len(metrics_df)
    if dropped:
        log("filter", f"dropped {dropped:,} unevaluable metric row(s)")
    return metrics_df


def model_predictions(df, model: str, cutoff: float):
    import numpy as np

    if model == "RP+FastText":
        scores = df["rank_1_oop_score"].fillna(math.inf).astype(float)
        fallback = scores >= cutoff
        return np.where(fallback, df["rank_1_lang_ft"], df["rank_1_lang_rp"])
    return df[MODEL_DEFINITIONS[model]]


def rank_distance_trigger_mask(df, lang_threshold: int = LANG_AWARE_DISTANCE_THRESHOLD):
    distances = df.apply(lambda row: safe_tag_distance(row["rank_1_lang_rp"], row["rank_2_lang_rp"]), axis=1)
    safe_distance = distances < 100
    return (~safe_distance) | (distances > lang_threshold)


def extended_model_predictions(df, model: str, cutoff: float, lang_threshold: int = LANG_AWARE_DISTANCE_THRESHOLD):
    import numpy as np
    import pandas as pd

    if model == "Resiliparse":
        return df["rank_1_lang_rp"]
    if model == "FastText":
        return df["rank_1_lang_ft"]

    scores = pd.to_numeric(df["rank_1_oop_score"], errors="coerce").fillna(math.inf)
    cutoff_fallback = scores >= cutoff
    if model == "RP+FastText":
        fallback = cutoff_fallback
        return np.where(fallback, df["rank_1_lang_ft"], df["rank_1_lang_rp"])
    if model == "RP+URL":
        fallback = cutoff_fallback
        return np.where(fallback, df["lang_url"], df["rank_1_lang_rp"])

    distance_fallback = rank_distance_trigger_mask(df, lang_threshold)
    if model == "RP+FastText lang-aware":
        fallback = cutoff_fallback | distance_fallback
        return np.where(fallback, df["rank_1_lang_ft"], df["rank_1_lang_rp"])
    if model == "RP+URL lang-aware":
        fallback = cutoff_fallback | distance_fallback
        return np.where(fallback, df["lang_url"], df["rank_1_lang_rp"])
    if model == "RP+FastText+URL lang-aware":
        return combined_url_fasttext_decision(df, cutoff, lang_threshold)["prediction"]
    if model == "RP+FastText+URL split-trigger":
        return split_trigger_url_fasttext_decision(df, cutoff, lang_threshold)["prediction"]
    raise ValueError(f"Unknown extended model: {model}")


def model_runtime_seconds(df, model: str, cutoff: float):
    import numpy as np
    import pandas as pd

    runtime_rp = pd.to_numeric(df.get("runtime_rp", 0), errors="coerce").fillna(0.0)
    runtime_ft = pd.to_numeric(df.get("runtime_ft", 0), errors="coerce").fillna(0.0)
    runtime_url = pd.to_numeric(df.get("runtime_url", 0), errors="coerce").fillna(0.0)

    if model == "Resiliparse":
        return runtime_rp
    if model == "FastText":
        return runtime_ft
    if model == "URL":
        return runtime_url

    scores = pd.to_numeric(df["rank_1_oop_score"], errors="coerce").fillna(math.inf)
    cutoff_fallback = scores >= cutoff
    if model == "RP+FastText":
        fallback = cutoff_fallback
        return runtime_rp + pd.Series(np.where(fallback, runtime_ft, 0.0), index=df.index)
    if model == "RP+URL":
        fallback = cutoff_fallback
        return runtime_rp + pd.Series(np.where(fallback, runtime_url, 0.0), index=df.index)
    if model == "RP+FastText lang-aware":
        fallback = cutoff_fallback | rank_distance_trigger_mask(df)
        return runtime_rp + pd.Series(np.where(fallback, runtime_ft, 0.0), index=df.index)
    if model == "RP+URL lang-aware":
        fallback = cutoff_fallback | rank_distance_trigger_mask(df)
        return runtime_rp + pd.Series(np.where(fallback, runtime_url, 0.0), index=df.index)
    if model == "RP+FastText+URL lang-aware":
        decision = combined_url_fasttext_decision(df, cutoff)
        return (
            runtime_rp
            + pd.Series(np.where(decision["fallback"], runtime_url, 0.0), index=df.index)
            + pd.Series(np.where(decision["use_fasttext"], runtime_ft, 0.0), index=df.index)
        )
    if model == "RP+FastText+URL split-trigger":
        decision = split_trigger_url_fasttext_decision(df, cutoff)
        return (
            runtime_rp
            + pd.Series(np.where(decision["use_url"], runtime_url, 0.0), index=df.index)
            + pd.Series(np.where(decision["use_fasttext"], runtime_ft, 0.0), index=df.index)
        )
    raise ValueError(f"Unknown runtime model: {model}")


def derive_confusion_from_rp_outputs(paths: dict[str, Path], hybrid_cutoff: float):
    import pandas as pd

    rows = []
    frames = {}
    log("derive", f"{len(paths)} configured rp_outputs file(s)")
    for dataset, path in paths.items():
        if not path.exists():
            log("skip", f"{dataset}: missing {path}")
            continue

        log("load", f"{dataset}: {path}")
        df = pd.read_csv(path)
        log("load", f"{dataset}: {len(df):,} rows")
        needed = {"label", "rank_1_lang_rp", "rank_1_lang_ft", "rank_1_oop_score"}
        require_columns(df, needed, path)
        df = df.copy()
        df["label"] = df["label"].fillna("unknown").astype(str)
        df["rank_1_lang_rp"] = df["rank_1_lang_rp"].fillna("unknown").astype(str)
        df["rank_1_lang_ft"] = df["rank_1_lang_ft"].fillna("unknown").astype(str)
        df = filter_evaluation_rows(df, dataset)
        frames[dataset] = df

    frames = prepare_wili_eval_overlap_frames(frames)
    for dataset, df in frames.items():
        for model in MODEL_DEFINITIONS:
            log("derive", f"{dataset} / {model}")
            pred = pd.Series(model_predictions(df, model, hybrid_cutoff)).fillna("unknown").astype(str)
            counts = (
                pd.DataFrame({"true_label": df["label"], "predicted_label": pred})
                .groupby(["true_label", "predicted_label"])
                .size()
                .reset_index(name="count")
            )
            counts.insert(0, "model", model)
            counts.insert(0, "dataset", dataset)
            rows.append(counts)

    if not rows:
        raise FileNotFoundError("No rp_outputs files were found; provide --metrics-file/--confusion-file or fix RP_OUTPUT_FILES.")

    confusion_df = pd.concat(rows, ignore_index=True)
    log("derive", f"confusion rows: {len(confusion_df):,}")
    return confusion_df


def derive_extended_confusion_from_rp_outputs(paths: dict[str, Path], hybrid_cutoff: float):
    import pandas as pd

    rows = []
    frames = {}
    log("derive", f"extended summary from {len(paths)} rp_outputs file(s)")
    for dataset, path in paths.items():
        if not path.exists():
            log("skip", f"{dataset}: missing extended summary source {path}")
            continue

        df = pd.read_csv(path)
        needed = {"label", "rank_1_lang_rp", "rank_1_lang_ft", "rank_1_oop_score"}
        try:
            require_columns(df, needed, path)
        except ValueError as exc:
            log("skip", f"{dataset}: incompatible extended summary schema ({exc})")
            continue

        df = df.copy()
        for column in ["label", "rank_1_lang_rp", "rank_1_lang_ft"]:
            df[column] = df[column].fillna("unknown").astype(str)
        if "lang_url" in df.columns:
            df["lang_url"] = df["lang_url"].fillna("unknown").astype(str)
        if "rank_2_lang_rp" in df.columns:
            df["rank_2_lang_rp"] = df["rank_2_lang_rp"].fillna("unknown").astype(str)
        df = filter_evaluation_rows(df, dataset)
        frames[dataset] = df

    frames = prepare_wili_eval_overlap_frames(frames)
    for dataset, df in frames.items():
        available_models = list(EXTENDED_MODEL_ORDER)
        if "lang_url" not in df.columns or not has_url_metadata(dataset):
            available_models = [model for model in available_models if "URL" not in model]
            log("skip", f"{dataset}: URL models require OWI URL metadata")
        if "rank_2_lang_rp" not in df.columns:
            available_models = [
                model
                for model in available_models
                if "lang-aware" not in model and model != "RP+FastText+URL split-trigger"
            ]
            log("skip", f"{dataset}: lang-aware models require rank_2_lang_rp")

        for model in available_models:
            log("derive", f"{dataset} / {model}")
            pred = pd.Series(extended_model_predictions(df, model, hybrid_cutoff)).fillna("unknown").astype(str)
            counts = (
                pd.DataFrame({"true_label": df["label"], "predicted_label": pred})
                .groupby(["true_label", "predicted_label"])
                .size()
                .reset_index(name="count")
            )
            counts.insert(0, "model", model)
            counts.insert(0, "dataset", dataset)
            rows.append(counts)

    if not rows:
        log("skip", "extended model summary: no compatible rp_outputs rows")
        return pd.DataFrame(columns=["dataset", "model", "true_label", "predicted_label", "count"])

    confusion_df = pd.concat(rows, ignore_index=True)
    log("derive", f"extended confusion rows: {len(confusion_df):,}")
    return confusion_df


def derive_runtime_from_rp_outputs(paths: dict[str, Path], hybrid_cutoff: float):
    import pandas as pd

    rows = []
    frames = {}
    runtime_models = [
        "Resiliparse",
        "FastText",
        "URL",
        "RP+FastText",
        "RP+URL",
        "RP+FastText lang-aware",
        "RP+URL lang-aware",
        "RP+FastText+URL lang-aware",
        "RP+FastText+URL split-trigger",
    ]
    log("runtime", f"{len(paths)} configured rp_outputs file(s)")
    for dataset, path in paths.items():
        if not path.exists():
            log("skip", f"{dataset}: missing runtime source {path}")
            continue

        df = pd.read_csv(path)
        try:
            require_columns(df, REQUIRED_RUNTIME_COLUMNS | {"rank_1_oop_score"}, path)
        except ValueError as exc:
            log("skip", f"{dataset}: incompatible runtime schema ({exc})")
            continue
        df = filter_evaluation_rows(df, dataset)
        frames[dataset] = df

    frames = prepare_wili_eval_overlap_frames(frames)
    for dataset, df in frames.items():
        available_models = list(runtime_models)
        if "runtime_url" not in df.columns:
            available_models = [
                model
                for model in available_models
                if model not in {"URL", "RP+URL", "RP+URL lang-aware", "RP+FastText+URL split-trigger"}
            ]
        if "lang_url" not in df.columns or not has_url_metadata(dataset):
            available_models = [model for model in available_models if "URL" not in model and model != "URL"]
        if "rank_2_lang_rp" not in df.columns:
            available_models = [
                model
                for model in available_models
                if "lang-aware" not in model and model != "RP+FastText+URL split-trigger"
            ]

        for model in available_models:
            runtimes = model_runtime_seconds(df, model, hybrid_cutoff)
            documents = int(runtimes.count())
            total_seconds = float(runtimes.sum())
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "documents": documents,
                    "mean_seconds": float(runtimes.mean()) if documents else 0.0,
                    "median_seconds": float(runtimes.median()) if documents else 0.0,
                    "p95_seconds": float(runtimes.quantile(0.95)) if documents else 0.0,
                    "total_seconds": total_seconds,
                    "docs_per_second": documents / total_seconds if total_seconds > 0 else 0.0,
                }
            )

    if not rows:
        log("skip", "runtime: no compatible rp_outputs timing columns")
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "documents",
                "mean_seconds",
                "median_seconds",
                "p95_seconds",
                "total_seconds",
                "docs_per_second",
            ]
        )

    runtime_df = pd.DataFrame(rows)
    log("runtime", f"rows: {len(runtime_df):,}")
    return runtime_df


@lru_cache(maxsize=512)
def language_tag_is_valid(value) -> bool:
    if value in UNEVALUABLE_LANGUAGE_VALUES:
        return False
    try:
        from langcodes import tag_distance

        tag_distance(str(value), str(value))
        return True
    except Exception:
        return False


def derive_language_similarity_from_rp_outputs(paths: dict[str, Path]):
    import pandas as pd

    rows = []
    frames = {}
    log("similarity", f"{len(paths)} configured rp_outputs file(s)")
    for dataset, path in paths.items():
        if not path.exists():
            log("skip", f"{dataset}: missing similarity source {path}")
            continue

        df = pd.read_csv(path)
        try:
            require_columns(df, REQUIRED_LANGUAGE_SIMILARITY_COLUMNS, path)
        except ValueError as exc:
            if "rank_2_lang_rp" in str(exc):
                log("skip", f"{dataset}: language similarity requires rank_2_lang_rp")
            else:
                log("skip", f"{dataset}: incompatible similarity schema ({exc})")
            continue

        log("similarity", dataset)
        df = filter_evaluation_rows(df, dataset)
        frames[dataset] = df

    frames = prepare_wili_eval_overlap_frames(frames)
    for dataset, df in frames.items():
        log("similarity", dataset)
        for record in df.itertuples(index=False):
            true_lang = str(getattr(record, "label", "unknown") or "unknown")
            rank_1_lang = str(getattr(record, "rank_1_lang_rp", "unknown") or "unknown")
            rank_2_lang = str(getattr(record, "rank_2_lang_rp", "unknown") or "unknown")
            rank_3_lang = str(getattr(record, "rank_3_lang_rp", "unknown") or "unknown")
            oop_score = getattr(record, "rank_1_oop_score", math.nan)
            rank_2_oop_score = getattr(record, "rank_2_oop_score", math.nan)
            rank_3_oop_score = getattr(record, "rank_3_oop_score", math.nan)
            oop_score_numeric = pd.to_numeric(oop_score, errors="coerce")
            rank_2_oop_score_numeric = pd.to_numeric(rank_2_oop_score, errors="coerce")
            rank_3_oop_score_numeric = pd.to_numeric(rank_3_oop_score, errors="coerce")
            oop_gap = rank_2_oop_score_numeric - oop_score_numeric
            rank_1_valid = language_tag_is_valid(rank_1_lang)
            rank_2_valid = language_tag_is_valid(rank_2_lang)
            invalid = not (rank_1_valid and rank_2_valid)
            distance = safe_tag_distance(rank_1_lang, rank_2_lang)
            true_family = language_family(true_lang)
            predicted_family = language_family(rank_1_lang)
            rank_1_family = language_family(rank_1_lang)
            rank_2_family = language_family(rank_2_lang)
            rank_3_family = language_family(rank_3_lang)
            rank_1_correct = true_lang == rank_1_lang
            rank_2_correct = true_lang == rank_2_lang
            rank_3_correct = true_lang == rank_3_lang
            if rank_1_correct:
                rank_correctness = "Rank 1 correct"
            elif rank_2_correct:
                rank_correctness = "Rank 2 correct"
            elif rank_3_correct:
                rank_correctness = "Rank 3 correct"
            else:
                rank_correctness = "Neither rank correct"
            rows.append(
                {
                    "dataset": dataset,
                    "true_lang": true_lang,
                    "rank_1_lang": rank_1_lang,
                    "rank_2_lang": rank_2_lang,
                    "rank_3_lang": rank_3_lang,
                    "predicted_lang": rank_1_lang,
                    "rank_1_family": rank_1_family,
                    "rank_2_family": rank_2_family,
                    "rank_3_family": rank_3_family,
                    "true_family": true_family,
                    "predicted_family": predicted_family,
                    "same_rank_family": rank_1_family == rank_2_family,
                    "same_family": true_family == predicted_family,
                    "is_correct": rank_1_correct,
                    "rank_1_correct": rank_1_correct,
                    "rank_2_correct": rank_2_correct,
                    "rank_3_correct": rank_3_correct,
                    "rank_correctness": rank_correctness,
                    "rank_1_rank_2_distance": distance,
                    "lang_distance": distance,
                    "rank_1_oop_score": oop_score_numeric,
                    "rank_2_oop_score": rank_2_oop_score_numeric,
                    "rank_3_oop_score": rank_3_oop_score_numeric,
                    "rank_1_rank_2_oop_gap": oop_gap,
                    "oop_gap_invalid": pd.isna(oop_gap),
                    "rank_distance_invalid": invalid,
                    "lang_distance_invalid": invalid,
                }
            )

    if not rows:
        log("skip", "similarity: no compatible rp_outputs rows")
        return pd.DataFrame(
            columns=[
                "dataset",
                "true_lang",
                "rank_1_lang",
                "rank_2_lang",
                "rank_3_lang",
                "predicted_lang",
                "rank_1_family",
                "rank_2_family",
                "rank_3_family",
                "true_family",
                "predicted_family",
                "same_rank_family",
                "same_family",
                "is_correct",
                "rank_1_correct",
                "rank_2_correct",
                "rank_3_correct",
                "rank_correctness",
                "rank_1_rank_2_distance",
                "lang_distance",
                "rank_1_oop_score",
                "rank_2_oop_score",
                "rank_3_oop_score",
                "rank_1_rank_2_oop_gap",
                "oop_gap_invalid",
                "rank_distance_invalid",
                "lang_distance_invalid",
            ]
        )

    similarity_df = pd.DataFrame(rows)
    similarity_df["rank_1_oop_score"] = pd.to_numeric(similarity_df["rank_1_oop_score"], errors="coerce")
    similarity_df["rank_2_oop_score"] = pd.to_numeric(similarity_df["rank_2_oop_score"], errors="coerce")
    if "rank_3_oop_score" in similarity_df.columns:
        similarity_df["rank_3_oop_score"] = pd.to_numeric(similarity_df["rank_3_oop_score"], errors="coerce")
    similarity_df["rank_1_rank_2_oop_gap"] = pd.to_numeric(similarity_df["rank_1_rank_2_oop_gap"], errors="coerce")
    similarity_df["oop_gap_invalid"] = similarity_df["rank_1_rank_2_oop_gap"].isna()
    log("similarity", f"rows: {len(similarity_df):,}")
    return similarity_df


def metrics_from_confusion(confusion_df):
    import pandas as pd

    rows = []
    log("metrics", "computing per-language scores")
    for (dataset, model), group in confusion_df.groupby(["dataset", "model"], sort=True):
        log("metrics", f"{dataset} / {model}")
        labels = sorted(set(group["true_label"]))
        total = int(group["count"].sum())

        for language in sorted(set(group["true_label"])):
            true_mask = group["true_label"] == language
            pred_mask = group["predicted_label"] == language
            tp = int(group.loc[true_mask & pred_mask, "count"].sum())
            fp = int(group.loc[~true_mask & pred_mask, "count"].sum())
            fn = int(group.loc[true_mask & ~pred_mask, "count"].sum())
            tn = total - tp - fp - fn
            support = tp + fn
            if support == 0:
                continue

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            accuracy = (tp + tn) / total if total else 0.0
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "language": language,
                    "language_family": language_family(language),
                    "resource_level": resource_level(language),
                    "accuracy": accuracy,
                    "precision_macro": precision,
                    "recall_macro": recall,
                    "f1_macro": f1,
                    "f1_weighted": f1,
                    "precision_weighted": precision,
                    "recall_weighted": recall,
                    "support": support,
                    "n_languages_dataset": len(labels),
                }
            )

    metrics_df = pd.DataFrame(rows)
    log("metrics", f"rows: {len(metrics_df):,}")
    return metrics_df


def model_metric_rows_from_confusion(confusion_df):
    """Compute one multiclass metric row per dataset/model from count data."""
    rows = []
    for (dataset, model), group in confusion_df.groupby(["dataset", "model"], sort=True):
        labels = sorted(set(group["true_label"]))
        total = int(group["count"].sum())
        if total == 0 or not labels:
            continue

        metric_rows = []
        correct = 0
        for language in labels:
            true_mask = group["true_label"] == language
            pred_mask = group["predicted_label"] == language
            tp = int(group.loc[true_mask & pred_mask, "count"].sum())
            fp = int(group.loc[~true_mask & pred_mask, "count"].sum())
            fn = int(group.loc[true_mask & ~pred_mask, "count"].sum())
            support = int(group.loc[true_mask, "count"].sum())
            correct += tp

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            metric_rows.append(
                {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": support,
                }
            )

        support_total = sum(row["support"] for row in metric_rows)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "accuracy": correct / total,
                "precision_macro": sum(row["precision"] for row in metric_rows) / len(metric_rows),
                "recall_macro": sum(row["recall"] for row in metric_rows) / len(metric_rows),
                "f1_macro": sum(row["f1"] for row in metric_rows) / len(metric_rows),
                "f1_weighted": (
                    sum(row["f1"] * row["support"] for row in metric_rows) / support_total
                    if support_total
                    else 0.0
                ),
                "support": support_total,
                "languages": group["true_label"].nunique(),
            }
        )

    import pandas as pd

    return pd.DataFrame(rows)


def aggregate_for_baseline(metrics_df):
    rows = []
    grouped = metrics_df.copy()
    if not INCLUDE_UNKNOWN_RESOURCE:
        grouped = grouped[grouped["resource_level"] != "unknown"]

    for (dataset, model, level), group in grouped.groupby(
        ["dataset", "model", "resource_level"], sort=True
    ):
        weights = group["support"].clip(lower=1)
        row = {
            "dataset": dataset,
            "model": model,
            "resource_level": level,
            "accuracy": (group["accuracy"] * weights).sum() / weights.sum(),
            "precision_macro": group["precision_macro"].mean(),
            "recall_macro": group["recall_macro"].mean(),
            "f1_macro": group["f1_macro"].mean(),
            "f1_weighted": (group["f1_weighted"] * weights).sum() / weights.sum(),
            "n_languages": group["language"].nunique(),
        }
        rows.append(row)

    import pandas as pd

    return pd.DataFrame(rows)


def parse_rp_output_overrides(overrides):
    """Parse repeated NAME=PATH rp_outputs overrides."""
    paths = dict(RP_OUTPUT_FILES)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"--rp-output must use NAME=PATH, got: {override}")
        name, path = override.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"--rp-output must use NAME=PATH, got: {override}")
        paths[name] = Path(path)
    return paths


def parse_length_sweep_overrides(overrides):
    """Parse repeated NAME=PATH length-sweep overrides."""
    paths = dict(LENGTH_SWEEP_FILES)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"--length-sweep must use NAME=PATH, got: {override}")
        name, path = override.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"--length-sweep must use NAME=PATH, got: {override}")
        paths[name] = Path(path)
    return paths


def load_or_derive_inputs(args):
    metrics_file = Path(args.metrics_file)
    confusion_file = Path(args.confusion_file)

    if args.generate_demo_data:
        log("input", "demo data")
        return generate_demo_data()

    if not args.normalized_only:
        log("input", "derive from rp_outputs.csv")
        paths = parse_rp_output_overrides(args.rp_output)
        confusion_df = derive_confusion_from_rp_outputs(paths, args.hybrid_cutoff)
        confusion_df = filter_confusion_rows(confusion_df)
        metrics_df = metrics_from_confusion(confusion_df)
        return metrics_df, confusion_df

    log("input", "normalized CSVs")
    log("load", f"metrics: {metrics_file}")
    metrics_df = load_csv(metrics_file)
    log("load", f"metrics rows: {len(metrics_df):,}")
    log("load", f"confusion: {confusion_file}")
    confusion_df = load_csv(confusion_file)
    log("load", f"confusion rows: {len(confusion_df):,}")
    require_columns(metrics_df, REQUIRED_METRIC_COLUMNS, metrics_file)
    require_columns(confusion_df, REQUIRED_CONFUSION_COLUMNS, confusion_file)
    if "support" not in metrics_df.columns:
        metrics_df["support"] = 1
    metrics_df = filter_metrics_rows(metrics_df)
    confusion_df = filter_confusion_rows(confusion_df)
    metrics_df, confusion_df = filter_normalized_wili_eval_overlap(metrics_df, confusion_df)
    return metrics_df, confusion_df


def write_normalized_tables(metrics_df, confusion_df, output_dir: Path):
    normalized_dir = output_dir / NORMALIZED_DIRNAME
    normalized_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = normalized_dir / "normalized_metrics.csv"
    confusion_path = normalized_dir / "normalized_confusion.csv"
    log("write", f"metrics: {metrics_path}")
    metrics_df.to_csv(metrics_path, index=False)
    log("write", f"confusion: {confusion_path}")
    confusion_df.to_csv(confusion_path, index=False)


# =============================================================================
# LATEX TABLE
# =============================================================================


def latex_escape(value) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def format_metric(value: float, best: bool) -> str:
    formatted = f"{value:.3f}"
    if best:
        return r"\textbf{" + formatted + "}"
    return formatted


def format_plain_value(value, column: str) -> str:
    """Format non-highlighted table values for compact thesis tables."""
    if value is None:
        return "n/a"
    try:
        if math.isnan(float(value)):
            return "n/a"
    except (TypeError, ValueError):
        pass
    if column in {"Accuracy", "F1 (macro)", "F1 (weighted)", "Mean runtime (ms)"}:
        return f"{float(value):.3f}"
    if column == "Docs/s":
        return "n/a" if float(value) == 0.0 else f"{float(value):.1f}"
    if column in {"Support", "Languages"}:
        return str(int(value))
    return latex_escape(value)


def write_compact_latex_table(display_df, columns, tex_path: Path, csv_path: Path | None = None) -> None:
    """Write a compact booktabs table grouped by dataset."""
    if csv_path is not None:
        display_df[columns].to_csv(csv_path, index=False)
        log("saved", f"compact table csv: {csv_path}")

    align = "ll" + ("r" * (len(columns) - 2))
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write(r"\begin{tabular}{" + align + "}" + "\n")
        handle.write(r"\toprule" + "\n")
        handle.write(" & ".join(columns) + r" \\" + "\n")
        handle.write(r"\midrule" + "\n")

        previous_dataset = None
        for record in display_df[columns].to_dict("records"):
            dataset = record["Dataset"]
            if previous_dataset is not None and previous_dataset != dataset:
                handle.write(r"\midrule" + "\n")
            dataset_label = latex_escape(dataset) if previous_dataset != dataset else ""
            previous_dataset = dataset
            row_values = [dataset_label, latex_escape(record["Model"])]
            row_values.extend(format_plain_value(record[column], column) for column in columns[2:])
            handle.write(" & ".join(row_values) + r" \\" + "\n")

        handle.write(r"\bottomrule" + "\n")
        handle.write(r"\end{tabular}" + "\n")
    log("saved", f"compact table: {tex_path}")


RANK_CORRECTNESS_ORDER = ["Rank 1 correct", "Rank 2 correct", "Rank 3 correct", "Neither rank correct"]
OOP_GAP_THRESHOLDS = [10, 25, 50, 100]
LANG_DISTANCE_BUCKETS = ["<=20", "21-80", "81-100", ">100"]


def language_distance_bucket(distance) -> str:
    value = float(distance)
    if value <= 20:
        return "<=20"
    if value <= 80:
        return "21-80"
    if value <= 100:
        return "81-100"
    return ">100"


def summarize_oop_gap_correctness(similarity_df):
    import pandas as pd

    if similarity_df is None or similarity_df.empty:
        return pd.DataFrame()

    gap_df = similarity_df[
        (~similarity_df["oop_gap_invalid"].astype(bool))
        & similarity_df["rank_1_rank_2_oop_gap"].notna()
        & similarity_df["rank_correctness"].notna()
    ].copy()
    if gap_df.empty:
        return pd.DataFrame()

    gap_df["rank_1_rank_2_oop_gap"] = pd.to_numeric(
        gap_df["rank_1_rank_2_oop_gap"],
        errors="coerce",
    )
    gap_df = gap_df.dropna(subset=["rank_1_rank_2_oop_gap"])

    rows = []
    for dataset, dataset_group in gap_df.groupby("dataset", sort=True):
        for correctness in RANK_CORRECTNESS_ORDER:
            group = dataset_group[dataset_group["rank_correctness"] == correctness]
            row = {
                "dataset": dataset,
                "rank_correctness": correctness,
                "n": int(len(group)),
                "median_gap": float(group["rank_1_rank_2_oop_gap"].median()) if not group.empty else math.nan,
            }
            for threshold in OOP_GAP_THRESHOLDS:
                row[f"pct_gap_lte_{threshold}"] = (
                    float((group["rank_1_rank_2_oop_gap"] <= threshold).mean() * 100)
                    if not group.empty
                    else math.nan
                )
            rows.append(row)

    return pd.DataFrame(rows)


def format_gap_value(value) -> str:
    if value is None or math.isnan(float(value)):
        return "--"
    return f"{float(value):.1f}"


def format_percent_value(value) -> str:
    if value is None or math.isnan(float(value)):
        return "--"
    return f"{float(value):.1f}\\%"


def write_oop_gap_correctness_table(table_df, tex_path: Path, include_dataset: bool) -> None:
    columns = [
        "N",
        "Median gap",
        r"\% gap $\leq$ 10",
        r"\% gap $\leq$ 25",
        r"\% gap $\leq$ 50",
        r"\% gap $\leq$ 100",
    ]
    with open(tex_path, "w", encoding="utf-8") as handle:
        if include_dataset:
            handle.write(r"\begin{tabular}{llrrrrrr}" + "\n")
            handle.write(r"\toprule" + "\n")
            handle.write("Dataset & Rank correctness & " + " & ".join(columns) + r" \\" + "\n")
        else:
            handle.write(r"\begin{tabular}{lrrrrrr}" + "\n")
            handle.write(r"\toprule" + "\n")
            handle.write("Rank correctness & " + " & ".join(columns) + r" \\" + "\n")
        handle.write(r"\midrule" + "\n")

        previous_dataset = None
        for record in table_df.itertuples(index=False):
            if include_dataset and previous_dataset is not None and previous_dataset != record.dataset:
                handle.write(r"\midrule" + "\n")
            values = [
                str(int(record.n)),
                format_gap_value(record.median_gap),
                format_percent_value(record.pct_gap_lte_10),
                format_percent_value(record.pct_gap_lte_25),
                format_percent_value(record.pct_gap_lte_50),
                format_percent_value(record.pct_gap_lte_100),
            ]
            if include_dataset:
                dataset_label = latex_escape(dataset_display_label(record.dataset)) if previous_dataset != record.dataset else ""
                previous_dataset = record.dataset
                handle.write(
                    f"{dataset_label} & {latex_escape(record.rank_correctness)} & "
                    + " & ".join(values)
                    + r" \\"
                    + "\n"
                )
            else:
                handle.write(
                    f"{latex_escape(record.rank_correctness)} & "
                    + " & ".join(values)
                    + r" \\"
                    + "\n"
                )

        handle.write(r"\bottomrule" + "\n")
        handle.write(r"\end{tabular}" + "\n")


def write_baseline_table(metrics_df, output_dir: Path):
    log("table", "baseline")
    tables_dir = output_dir / TABLES_DIRNAME
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_df = aggregate_for_baseline(metrics_df)
    log("table", f"rows: {len(table_df):,}")
    metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    labels = {
        "accuracy": "Accuracy",
        "precision_macro": "Precision (macro)",
        "recall_macro": "Recall (macro)",
        "f1_macro": "F1 (macro)",
        "f1_weighted": "F1 (weighted)",
    }

    best_lookup = {}
    for dataset, group in table_df.groupby("dataset"):
        for metric in metric_cols:
            best_lookup[(dataset, metric)] = group[metric].max()

    output_path = tables_dir / "baseline_table.tex"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(r"\begin{tabular}{lllrrrrrr}" + "\n")
        handle.write(r"\toprule" + "\n")
        handle.write(
            "Dataset & Model & Resource & "
            + " & ".join(labels[col] for col in metric_cols)
            + r" & Languages \\"
            + "\n"
        )
        handle.write(r"\midrule" + "\n")

        previous_dataset = None
        for record in table_df.sort_values(["dataset", "model", "resource_level"]).itertuples(index=False):
            if previous_dataset is not None and previous_dataset != record.dataset:
                handle.write(r"\midrule" + "\n")
            dataset_label = latex_escape(dataset_display_label(record.dataset)) if previous_dataset != record.dataset else ""
            previous_dataset = record.dataset

            values = []
            for metric in metric_cols:
                value = getattr(record, metric)
                best = math.isclose(value, best_lookup[(record.dataset, metric)], rel_tol=1e-12, abs_tol=1e-12)
                values.append(format_metric(value, best))

            handle.write(
                f"{dataset_label} & {latex_escape(record.model)} & {latex_escape(record.resource_level)} & "
                + " & ".join(values)
                + f" & {int(record.n_languages)} "
                + r"\\"
                + "\n"
            )

        handle.write(r"\bottomrule" + "\n")
        handle.write(r"\end{tabular}" + "\n")

    log("saved", f"baseline: {output_path}")


def write_model_dataset_summary(confusion_df, output_dir: Path, runtime_df=None):
    log("table", "model dataset summary")
    tables_dir = output_dir / TABLES_DIRNAME
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_df = model_metric_rows_from_confusion(confusion_df)
    include_runtime = runtime_df is not None and not runtime_df.empty
    if include_runtime:
        runtime_cols = ["dataset", "model", "mean_seconds", "docs_per_second"]
        runtime_table = runtime_df[runtime_cols].copy()
        runtime_table["mean_runtime_ms"] = runtime_table["mean_seconds"] * 1000
        runtime_table = runtime_table.drop(columns=["mean_seconds"])
        runtime_table = runtime_table[["dataset", "model", "mean_runtime_ms", "docs_per_second"]]
        table_df = table_df.merge(runtime_table, on=["dataset", "model"], how="left")
    log("table", f"model rows: {len(table_df):,}")

    csv_path = tables_dir / "model_dataset_summary.csv"
    table_df.to_csv(csv_path, index=False)
    log("saved", f"model summary csv: {csv_path}")

    metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    labels = {
        "accuracy": "Accuracy",
        "precision_macro": "Precision (macro)",
        "recall_macro": "Recall (macro)",
        "f1_macro": "F1 (macro)",
        "f1_weighted": "F1 (weighted)",
    }

    best_lookup = {}
    for dataset, group in table_df.groupby("dataset"):
        for metric in metric_cols:
            best_lookup[(dataset, metric)] = group[metric].max()

    output_path = tables_dir / "model_dataset_summary.tex"
    with open(output_path, "w", encoding="utf-8") as handle:
        column_spec = "llrrrrrrr" + ("rr" if include_runtime else "")
        handle.write(r"\begin{tabular}{" + column_spec + "}" + "\n")
        handle.write(r"\toprule" + "\n")
        header_cols = ["Dataset", "Model", *[labels[col] for col in metric_cols], "Support", "Languages"]
        if include_runtime:
            header_cols.extend(["Mean runtime (ms)", "Docs/s"])
        handle.write(" & ".join(header_cols) + r" \\" + "\n")
        handle.write(r"\midrule" + "\n")

        previous_dataset = None
        for record in table_df.sort_values(["dataset", "model"]).itertuples(index=False):
            if previous_dataset is not None and previous_dataset != record.dataset:
                handle.write(r"\midrule" + "\n")
            dataset_label = latex_escape(dataset_display_label(record.dataset)) if previous_dataset != record.dataset else ""
            previous_dataset = record.dataset

            values = []
            for metric in metric_cols:
                value = getattr(record, metric)
                best = math.isclose(value, best_lookup[(record.dataset, metric)], rel_tol=1e-12, abs_tol=1e-12)
                values.append(format_metric(value, best))

            handle.write(
                f"{dataset_label} & {latex_escape(record.model)} & "
                + " & ".join(values)
                + f" & {int(record.support)} & {int(record.languages)}"
                + (
                    f" & {float(record.mean_runtime_ms):.3f} & {float(record.docs_per_second):.1f}"
                    if include_runtime and not math.isnan(float(record.mean_runtime_ms))
                    else (" & n/a & n/a" if include_runtime else "")
                )
                + " "
                + r"\\"
                + "\n"
            )

        handle.write(r"\bottomrule" + "\n")
        handle.write(r"\end{tabular}" + "\n")

    log("saved", f"model summary: {output_path}")

    write_model_dataset_summary_splits(table_df, output_dir)


def write_model_dataset_summary_splits(table_df, output_dir: Path):
    """Write compact thesis-facing baseline performance/runtime tables."""
    tables_dir = output_dir / TABLES_DIRNAME
    display_df = table_df.rename(
        columns={
            "dataset": "Dataset",
            "model": "Model",
            "accuracy": "Accuracy",
            "f1_macro": "F1 (macro)",
            "f1_weighted": "F1 (weighted)",
            "support": "Support",
            "languages": "Languages",
            "mean_runtime_ms": "Mean runtime (ms)",
            "docs_per_second": "Docs/s",
        }
    ).copy()
    display_df["Dataset"] = display_df["Dataset"].map(dataset_display_label)
    display_df = display_df.sort_values(["Dataset", "Model"])

    performance_columns = [
        "Dataset",
        "Model",
        "Accuracy",
        "F1 (macro)",
        "F1 (weighted)",
        "Support",
        "Languages",
    ]
    write_compact_latex_table(
        display_df,
        performance_columns,
        tables_dir / "model_dataset_performance_summary.tex",
        tables_dir / "model_dataset_performance_summary.csv",
    )

    if {"Mean runtime (ms)", "Docs/s"}.issubset(display_df.columns):
        runtime_columns = ["Dataset", "Model", "Mean runtime (ms)", "Docs/s"]
        write_compact_latex_table(
            display_df,
            runtime_columns,
            tables_dir / "model_dataset_runtime_summary.tex",
            tables_dir / "model_dataset_runtime_summary.csv",
        )


def write_model_dataset_summary_extended(confusion_df, output_dir: Path):
    if confusion_df is None or confusion_df.empty:
        log("skip", "extended model summary: no rows")
        return

    log("table", "extended model dataset summary")
    tables_dir = output_dir / TABLES_DIRNAME
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_df = model_metric_rows_from_confusion(confusion_df)
    log("table", f"extended model rows: {len(table_df):,}")

    csv_path = tables_dir / "model_dataset_summary_extended.csv"
    table_df.to_csv(csv_path, index=False)
    log("saved", f"extended model summary csv: {csv_path}")

    metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    labels = {
        "accuracy": "Accuracy",
        "precision_macro": "Precision (macro)",
        "recall_macro": "Recall (macro)",
        "f1_macro": "F1 (macro)",
        "f1_weighted": "F1 (weighted)",
    }

    best_lookup = {}
    for dataset, group in table_df.groupby("dataset"):
        for metric in metric_cols:
            best_lookup[(dataset, metric)] = group[metric].max()

    model_order = {model: index for index, model in enumerate(EXTENDED_MODEL_ORDER)}
    table_df = table_df.assign(
        _model_order=table_df["model"].map(lambda model: model_order.get(model, len(model_order)))
    )

    output_path = tables_dir / "model_dataset_summary_extended.tex"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(r"\begin{tabular}{llrrrrrrr}" + "\n")
        handle.write(r"\toprule" + "\n")
        handle.write(
            "Dataset & Model & "
            + " & ".join(labels[col] for col in metric_cols)
            + r" & Support & Languages \\"
            + "\n"
        )
        handle.write(r"\midrule" + "\n")

        previous_dataset = None
        for record in table_df.sort_values(["dataset", "_model_order", "model"]).itertuples(index=False):
            if previous_dataset is not None and previous_dataset != record.dataset:
                handle.write(r"\midrule" + "\n")
            dataset_label = latex_escape(dataset_display_label(record.dataset)) if previous_dataset != record.dataset else ""
            previous_dataset = record.dataset

            values = []
            for metric in metric_cols:
                value = getattr(record, metric)
                best = math.isclose(value, best_lookup[(record.dataset, metric)], rel_tol=1e-12, abs_tol=1e-12)
                values.append(format_metric(value, best))

            handle.write(
                f"{dataset_label} & {latex_escape(record.model)} & "
                + " & ".join(values)
                + f" & {int(record.support)} & {int(record.languages)} "
                + r"\\"
                + "\n"
            )

        handle.write(r"\bottomrule" + "\n")
        handle.write(r"\end{tabular}" + "\n")

    log("saved", f"extended model summary: {output_path}")

    write_extended_summary_splits(table_df, output_dir)


def write_extended_summary_splits(table_df, output_dir: Path):
    """Write compact thesis-facing adapted-system summary splits."""
    tables_dir = output_dir / TABLES_DIRNAME
    display_df = table_df.rename(
        columns={
            "dataset": "Dataset",
            "model": "Model",
            "accuracy": "Accuracy",
            "f1_macro": "F1 (macro)",
            "f1_weighted": "F1 (weighted)",
            "support": "Support",
            "languages": "Languages",
        }
    ).copy()
    display_df["Dataset"] = display_df["Dataset"].map(dataset_display_label)

    model_order = {model: index for index, model in enumerate(EXTENDED_MODEL_ORDER)}
    display_df["_model_order"] = display_df["Model"].map(lambda model: model_order.get(model, len(model_order)))
    display_df = display_df.sort_values(["Dataset", "_model_order", "Model"])
    columns = [
        "Dataset",
        "Model",
        "Accuracy",
        "F1 (macro)",
        "F1 (weighted)",
        "Support",
        "Languages",
    ]

    frisian_df = display_df[display_df["Dataset"] == "OWI_slice_frisian"].drop(columns=["_model_order"])
    if not frisian_df.empty:
        write_compact_latex_table(
            frisian_df,
            columns,
            tables_dir / "adapted_owi_frisian_summary.tex",
            tables_dir / "adapted_owi_frisian_summary.csv",
        )

    control_df = display_df[display_df["Dataset"] != "OWI_slice_frisian"].drop(columns=["_model_order"])
    if not control_df.empty:
        write_compact_latex_table(
            control_df,
            columns,
            tables_dir / "adapted_cross_dataset_summary.tex",
            tables_dir / "adapted_cross_dataset_summary.csv",
        )


def _annotate_bar_values(ax, values):
    for patch, value in zip(ax.patches, values):
        if math.isnan(value):
            continue
        ax.annotate(
            f"{value:.0%}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 3),
            textcoords="offset points",
        )


def write_adaptation_ladder_plot(output_dir: Path):
    import matplotlib.pyplot as plt
    import pandas as pd
    from .plots import MODEL_COLORS

    tables_dir = output_dir / TABLES_DIRNAME
    summary_path = tables_dir / "model_dataset_summary_extended.csv"
    if not summary_path.exists():
        log("skip", f"adaptation ladder: missing {summary_path}")
        return

    df = pd.read_csv(summary_path)
    dataset = "OWI_slice_frisian"
    ladder_order = [
        "Resiliparse",
        "RP+FastText",
        "RP+URL",
        "RP+FastText lang-aware",
        "RP+URL lang-aware",
        "RP+FastText+URL split-trigger",
        "RP+FastText+URL lang-aware",
        "FastText",
    ]
    plot_df = df[(df["dataset"] == dataset) & (df["model"].isin(ladder_order))].copy()
    if plot_df.empty:
        log("skip", f"adaptation ladder: no {dataset} rows")
        return

    plot_df["model"] = pd.Categorical(plot_df["model"], categories=ladder_order, ordered=True)
    plot_df = plot_df.sort_values("model")
    models = plot_df["model"].astype(str).tolist()
    x = list(range(len(plot_df)))

    summary_dir = output_dir / SUMMARY_PLOTS_DIRNAME
    summary_dir.mkdir(parents=True, exist_ok=True)

    log("summary", "OWI Frisian adaptation ladder")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, metric, title in [
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "f1_macro", "Macro F1"),
    ]:
        values = plot_df[metric].astype(float).tolist()
        colors = [MODEL_COLORS.get(model, "#7f7f7f") for model in models]
        ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
        _annotate_bar_values(ax, values)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right")
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
        ax.set_ylabel("Score")

    fig.suptitle("OWI Frisian adapted-model performance ladder")
    fig.tight_layout()
    output_path = summary_dir / "owi_frisian_adaptation_ladder.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    log("saved", f"summary: {output_path}")


def write_clean_vs_web_gap_plot(output_dir: Path):
    import matplotlib.pyplot as plt
    import pandas as pd
    from .plots import MODEL_COLORS

    tables_dir = output_dir / TABLES_DIRNAME
    summary_path = tables_dir / "model_dataset_summary.csv"
    if not summary_path.exists():
        log("skip", f"clean/web gap: missing {summary_path}")
        return

    df = pd.read_csv(summary_path)
    dataset_order = [
        "WiLI_2018_eval_overlap",
        "GLC",
        "CommonLID",
        "OWI_slice_dutch",
        "OWI_slice_frisian",
        "OWI_slice_random",
    ]
    label_map = {
        "WiLI_2018_eval_overlap": "WiLI_Filtered",
        "GLC": "GLC\ncontrolled",
        "CommonLID": "CommonLID\nweb",
        "OWI_slice_dutch": "OWI Dutch\nweb",
        "OWI_slice_frisian": "OWI Frisian\nweb",
        "OWI_slice_random": "OWI Random\nweb",
    }
    plot_df = df[(df["model"] == "Resiliparse") & (df["dataset"].isin(dataset_order))].copy()
    if plot_df.empty:
        log("skip", "clean/web gap: no Resiliparse rows")
        return

    plot_df["dataset"] = pd.Categorical(plot_df["dataset"], categories=dataset_order, ordered=True)
    plot_df = plot_df.sort_values("dataset")
    labels = [label_map.get(dataset, dataset) for dataset in plot_df["dataset"].astype(str)]
    x = list(range(len(plot_df)))

    summary_dir = output_dir / SUMMARY_PLOTS_DIRNAME
    summary_dir.mkdir(parents=True, exist_ok=True)

    log("summary", "Resiliparse clean vs web gap")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    colors = []
    for dataset in plot_df["dataset"].astype(str):
        if dataset in {"WiLI_2018_eval_overlap", "GLC"}:
            colors.append(MODEL_COLORS["Resiliparse"])
        else:
            colors.append("#ff7f0e")

    for ax, metric, title in [
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "f1_macro", "Macro F1"),
    ]:
        values = plot_df[metric].astype(float).tolist()
        ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
        _annotate_bar_values(ax, values)
        ax.axvline(1.5, color="#555555", linestyle="--", linewidth=0.9)
        ax.text(0.2, 1.01, "clean / controlled", transform=ax.get_xaxis_transform(), fontsize=8)
        ax.text(2.15, 1.01, "web", transform=ax.get_xaxis_transform(), fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
        ax.set_ylabel("Score")

    fig.suptitle("Resiliparse performance gap between filtered clean/controlled and web datasets")
    fig.tight_layout()
    output_path = summary_dir / "resiliparse_clean_vs_web_gap.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    log("saved", f"summary: {output_path}")


def write_summary_plots(output_dir: Path):
    write_adaptation_ladder_plot(output_dir)
    write_clean_vs_web_gap_plot(output_dir)


def summarize_seconds(seconds):
    documents = int(seconds.count())
    total_seconds = float(seconds.sum()) if documents else 0.0
    mean_seconds = float(seconds.mean()) if documents else math.nan
    docs_per_second = documents / total_seconds if total_seconds > 0 else math.nan
    return mean_seconds, docs_per_second


def derive_url_model_evaluation(paths: dict[str, Path], hybrid_cutoff: float):
    import numpy as np
    import pandas as pd

    confusion_rows = []
    runtime_rows = []
    log("table", "url model evaluation")
    for dataset, path in paths.items():
        if not has_url_metadata(dataset):
            log("skip", f"{dataset}: URL/model evaluation is limited to OWI slices")
            continue
        if not path.exists():
            log("skip", f"{dataset}: missing URL evaluation source {path}")
            continue

        df = pd.read_csv(path)
        needed = {"label", "rank_1_lang_rp", "rank_1_lang_ft", "lang_url", "rank_1_oop_score"}
        try:
            require_columns(df, needed, path)
        except ValueError as exc:
            log("skip", f"{dataset}: incompatible URL evaluation schema ({exc})")
            continue

        df = df.copy()
        for column in ["label", "rank_1_lang_rp", "rank_1_lang_ft", "lang_url"]:
            df[column] = df[column].fillna("unknown").astype(str)
        if "rank_2_lang_rp" in df.columns:
            df["rank_2_lang_rp"] = df["rank_2_lang_rp"].fillna("unknown").astype(str)
        df = filter_evaluation_rows(df, dataset)
        if df.empty:
            log("skip", f"{dataset}: no evaluable URL/model rows")
            continue

        prediction_sources = {
            "Always English": pd.Series(np.repeat("eng", len(df)), index=df.index),
            "URL": df["lang_url"],
            "Resiliparse": df["rank_1_lang_rp"],
            "FastText": df["rank_1_lang_ft"],
            "RP+URL": pd.Series(
                extended_model_predictions(df, "RP+URL", hybrid_cutoff),
                index=df.index,
            ),
        }
        if "rank_2_lang_rp" in df.columns:
            prediction_sources["RP+URL lang-aware"] = pd.Series(
                extended_model_predictions(df, "RP+URL lang-aware", hybrid_cutoff),
                index=df.index,
            )
            prediction_sources["RP+FastText+URL lang-aware"] = pd.Series(
                extended_model_predictions(df, "RP+FastText+URL lang-aware", hybrid_cutoff),
                index=df.index,
            )
            prediction_sources["RP+FastText+URL split-trigger"] = pd.Series(
                extended_model_predictions(df, "RP+FastText+URL split-trigger", hybrid_cutoff),
                index=df.index,
            )
        else:
            log("skip", f"{dataset}: combined UrlExtractor/FastText composite requires rank_2_lang_rp")

        for model in URL_EVALUATION_MODEL_ORDER:
            if model not in prediction_sources:
                continue
            pred = pd.Series(prediction_sources[model], index=df.index).fillna("unknown").astype(str)
            counts = (
                pd.DataFrame({"true_label": df["label"], "predicted_label": pred})
                .groupby(["true_label", "predicted_label"])
                .size()
                .reset_index(name="count")
            )
            counts.insert(0, "model", model)
            counts.insert(0, "dataset", dataset)
            confusion_rows.append(counts)

            if model == "Always English":
                runtimes = pd.Series(np.repeat(0.0, len(df)), index=df.index)
            elif model == "URL":
                runtimes = pd.to_numeric(df.get("runtime_url", 0.0), errors="coerce").fillna(0.0)
            elif model == "Resiliparse":
                runtimes = model_runtime_seconds(df, "Resiliparse", hybrid_cutoff)
            elif model == "FastText":
                runtimes = model_runtime_seconds(df, "FastText", hybrid_cutoff)
            elif model == "RP+URL":
                runtimes = model_runtime_seconds(df, "RP+URL", hybrid_cutoff)
            elif model == "RP+URL lang-aware":
                runtimes = model_runtime_seconds(df, "RP+URL lang-aware", hybrid_cutoff)
            elif model == "RP+FastText+URL split-trigger":
                runtimes = model_runtime_seconds(df, "RP+FastText+URL split-trigger", hybrid_cutoff)
            else:
                runtimes = model_runtime_seconds(df, "RP+FastText+URL lang-aware", hybrid_cutoff)
            mean_seconds, docs_per_second = summarize_seconds(runtimes)
            runtime_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "mean_runtime_ms": mean_seconds * 1000 if not math.isnan(mean_seconds) else math.nan,
                    "docs_per_second": docs_per_second,
                }
            )

    if not confusion_rows:
        log("skip", "URL/model evaluation: no compatible rows")
        return None

    confusion_df = pd.concat(confusion_rows, ignore_index=True)
    metrics_df = model_metric_rows_from_confusion(confusion_df)
    runtime_df = pd.DataFrame(runtime_rows)
    table_df = metrics_df.merge(runtime_df, on=["dataset", "model"], how="left")
    model_order = {model: index for index, model in enumerate(URL_EVALUATION_MODEL_ORDER)}
    return table_df.assign(
        _model_order=table_df["model"].map(lambda model: model_order.get(model, len(model_order)))
    ).sort_values(["dataset", "_model_order", "model"]).drop(columns=["_model_order"])


def write_url_model_evaluation_table(paths: dict[str, Path], output_dir: Path, hybrid_cutoff: float):
    table_df = derive_url_model_evaluation(paths, hybrid_cutoff)
    if table_df is None or table_df.empty:
        return

    tables_dir = output_dir / TABLES_DIRNAME
    tables_dir.mkdir(parents=True, exist_ok=True)

    display_df = table_df.rename(
        columns={
            "dataset": "Dataset",
            "model": "Model",
            "accuracy": "Accuracy",
            "precision_macro": "Precision (macro)",
            "recall_macro": "Recall (macro)",
            "f1_macro": "F1 (macro)",
            "f1_weighted": "F1 (weighted)",
            "support": "Support",
            "languages": "Languages",
            "mean_runtime_ms": "Mean runtime (ms)",
            "docs_per_second": "Docs/s",
        }
    )
    display_df["Dataset"] = display_df["Dataset"].map(dataset_display_label)
    columns = [
        "Dataset",
        "Model",
        "Accuracy",
        "Precision (macro)",
        "Recall (macro)",
        "F1 (macro)",
        "F1 (weighted)",
        "Support",
        "Languages",
        "Mean runtime (ms)",
        "Docs/s",
    ]
    csv_path = tables_dir / "url_model_evaluation.csv"
    display_df[columns].to_csv(csv_path, index=False)
    log("saved", f"URL/model evaluation csv: {csv_path}")

    tex_path = tables_dir / "url_model_evaluation.tex"
    metric_cols = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "F1 (weighted)"]
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write(r"\begin{tabular}{llrrrrrrrrr}" + "\n")
        handle.write(r"\toprule" + "\n")
        handle.write(
            r"Dataset & Model & Accuracy & Precision (macro) & Recall (macro) & F1 (macro) & F1 (weighted) & Support & Languages & Mean runtime (ms) & Docs/s \\"
            + "\n"
        )
        handle.write(r"\midrule" + "\n")

        previous_dataset = None
        for record in display_df[columns].to_dict("records"):
            dataset = record["Dataset"]
            if previous_dataset is not None and previous_dataset != dataset:
                handle.write(r"\midrule" + "\n")
            dataset_label = latex_escape(dataset_display_label(dataset)) if previous_dataset != dataset else ""
            previous_dataset = dataset
            values = [format_metric(float(record[col]), False) for col in metric_cols]
            mean_ms = record["Mean runtime (ms)"]
            docs_s = record["Docs/s"]
            mean_ms_text = f"{float(mean_ms):.3f}" if not math.isnan(float(mean_ms)) else "n/a"
            docs_s_text = f"{float(docs_s):.1f}" if not math.isnan(float(docs_s)) else "n/a"
            handle.write(
                f"{dataset_label} & {latex_escape(record['Model'])} & "
                + " & ".join(values)
                + f" & {int(record['Support'])} & {int(record['Languages'])}"
                + f" & {mean_ms_text} & {docs_s_text} "
                + r"\\"
                + "\n"
            )

        handle.write(r"\bottomrule" + "\n")
        handle.write(r"\end{tabular}" + "\n")

    log("saved", f"URL/model evaluation: {tex_path}")

    write_url_model_evaluation_splits(display_df, output_dir)


def write_url_model_evaluation_splits(display_df, output_dir: Path):
    """Write compact URL/model evaluation tables for target and control slices."""
    tables_dir = output_dir / TABLES_DIRNAME
    columns = [
        "Dataset",
        "Model",
        "Accuracy",
        "F1 (macro)",
        "F1 (weighted)",
        "Support",
        "Mean runtime (ms)",
        "Docs/s",
    ]

    frisian_df = display_df[display_df["Dataset"] == "OWI_slice_frisian"].copy()
    if not frisian_df.empty:
        write_compact_latex_table(
            frisian_df,
            columns,
            tables_dir / "url_model_evaluation_frisian.tex",
            tables_dir / "url_model_evaluation_frisian.csv",
        )

    controls_df = display_df[display_df["Dataset"].isin(["OWI_slice_dutch", "OWI_slice_random"])].copy()
    if not controls_df.empty:
        write_compact_latex_table(
            controls_df,
            columns,
            tables_dir / "url_model_evaluation_controls.tex",
            tables_dir / "url_model_evaluation_controls.csv",
        )


def write_resource_level_summary_table(metrics_df, output_dir: Path):
    log("table", "resource level summary")
    tables_dir = output_dir / TABLES_DIRNAME
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_df = aggregate_for_baseline(metrics_df)
    log("table", f"resource rows: {len(table_df):,}")

    csv_path = tables_dir / "resource_level_summary.csv"
    table_df.to_csv(csv_path, index=False)
    log("saved", f"resource summary csv: {csv_path}")

    metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    labels = {
        "accuracy": "Accuracy",
        "precision_macro": "Precision (macro)",
        "recall_macro": "Recall (macro)",
        "f1_macro": "F1 (macro)",
        "f1_weighted": "F1 (weighted)",
    }

    best_lookup = {}
    for (dataset, level), group in table_df.groupby(["dataset", "resource_level"]):
        for metric in metric_cols:
            best_lookup[(dataset, level, metric)] = group[metric].max()

    output_path = tables_dir / "resource_level_summary.tex"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(r"\begin{tabular}{lllrrrrrr}" + "\n")
        handle.write(r"\toprule" + "\n")
        handle.write(
            "Dataset & Resource & Model & "
            + " & ".join(labels[col] for col in metric_cols)
            + r" & Languages \\"
            + "\n"
        )
        handle.write(r"\midrule" + "\n")

        previous_dataset = None
        for record in table_df.sort_values(["dataset", "resource_level", "model"]).itertuples(index=False):
            if previous_dataset is not None and previous_dataset != record.dataset:
                handle.write(r"\midrule" + "\n")
            dataset_label = latex_escape(dataset_display_label(record.dataset)) if previous_dataset != record.dataset else ""
            previous_dataset = record.dataset

            values = []
            for metric in metric_cols:
                value = getattr(record, metric)
                best = math.isclose(
                    value,
                    best_lookup[(record.dataset, record.resource_level, metric)],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                values.append(format_metric(value, best))

            handle.write(
                f"{dataset_label} & {latex_escape(record.resource_level)} & {latex_escape(record.model)} & "
                + " & ".join(values)
                + f" & {int(record.n_languages)} "
                + r"\\"
                + "\n"
            )

        handle.write(r"\bottomrule" + "\n")
        handle.write(r"\end{tabular}" + "\n")

    log("saved", f"resource summary: {output_path}")


def write_resource_level_plots(metrics_df, output_dir: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from .plots import model_palette, ordered_models, style_model_axis

    resource_dir = output_dir / RESOURCE_LEVEL_DIRNAME
    resource_dir.mkdir(parents=True, exist_ok=True)
    table_df = aggregate_for_baseline(metrics_df)
    if table_df.empty:
        log("skip", "resource levels: no non-unknown rows")
        return
    table_df = table_df.assign(dataset_label=table_df["dataset"].map(dataset_display_label))

    levels = [level for level in ["low", "mid", "high", "unknown"] if level in set(table_df["resource_level"])]
    models = ordered_models(table_df["model"].dropna().astype(str))
    palette = model_palette(models)
    for metric, filename, ylabel in [
        ("f1_macro", "resource_f1_macro.pdf", "Macro F1"),
        ("accuracy", "resource_accuracy.pdf", "Accuracy"),
    ]:
        log("resource", metric)
        grid = sns.catplot(
            data=table_df,
            x="resource_level",
            y=metric,
            hue="model",
            col="dataset_label",
            col_wrap=3,
            kind="bar",
            order=levels,
            hue_order=models,
            palette=palette,
            sharey=True,
            height=3.2,
            aspect=1.15,
        )
        grid.set_axis_labels("Resource level", ylabel)
        grid.set_titles("{col_name}")
        for ax in grid.axes.flat:
            style_model_axis(ax)
        grid.tight_layout()
        output_path = resource_dir / filename
        grid.figure.savefig(output_path, bbox_inches="tight")
        plt.close(grid.figure)
        log("saved", f"resource: {output_path}")


def format_seconds(value: float, best: bool = False) -> str:
    formatted = f"{value:.4f}"
    if best:
        return r"\textbf{" + formatted + "}"
    return formatted


def format_speedup(value: float, best: bool = False) -> str:
    if math.isnan(value) or math.isinf(value):
        return "n/a"
    formatted = f"{value:.2f}x"
    if best:
        return r"\textbf{" + formatted + "}"
    return formatted


RUNTIME_BASELINE_MODELS = ["Resiliparse", "FastText"]


def speedup_column_name(model: str) -> str:
    return f"speedup_vs_{safe_name(model).lower()}"


def build_runtime_speedup_table(runtime_df):
    import pandas as pd
    from .plots import ordered_models

    rows = []
    for dataset, group in runtime_df.groupby("dataset", sort=True):
        mean_by_model = {
            record.model: float(record.mean_seconds)
            for record in group.itertuples(index=False)
            if float(record.mean_seconds) > 0
        }
        for model in ordered_models(group["model"].dropna().astype(str)):
            model_mean = mean_by_model.get(model)
            docs_per_second = math.nan
            model_rows = group[group["model"].astype(str) == model]
            if not model_rows.empty and "docs_per_second" in model_rows.columns:
                docs_per_second = float(model_rows.iloc[0]["docs_per_second"])
            row = {
                "dataset": dataset,
                "model": model,
                "mean_seconds": model_mean if model_mean is not None else math.nan,
                "mean_ms": model_mean * 1000 if model_mean is not None else math.nan,
                "docs_per_second": docs_per_second,
            }
            for comparison_model in RUNTIME_BASELINE_MODELS:
                comparison_mean = mean_by_model.get(comparison_model)
                if model_mean is None or comparison_mean is None:
                    row[speedup_column_name(comparison_model)] = math.nan
                elif model == comparison_model:
                    row[speedup_column_name(comparison_model)] = 1.0
                else:
                    row[speedup_column_name(comparison_model)] = comparison_mean / model_mean
            rows.append(row)

    return pd.DataFrame(rows), RUNTIME_BASELINE_MODELS


def write_runtime_speedup_table(runtime_df, tables_dir: Path):
    log("table", "runtime speedup comparison")
    speedup_df, models = build_runtime_speedup_table(runtime_df)
    csv_path = tables_dir / "runtime_speedup_comparison.csv"
    speedup_df.to_csv(csv_path, index=False)
    log("saved", f"runtime speedup csv: {csv_path}")

    best_lookup = {}
    for dataset, group in speedup_df.groupby("dataset", sort=True):
        comparable = group[~group["model"].astype(str).isin({"URL"})]
        for comparison_model in models:
            column = speedup_column_name(comparison_model)
            best_lookup[(dataset, comparison_model)] = comparable[column].max(skipna=True)

    tex_path = tables_dir / "runtime_speedup_comparison.tex"
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write(r"\begin{tabular}{llrrrr}" + "\n")
        handle.write(r"\toprule" + "\n")
        handle.write(
            r"Dataset & Model & Mean (ms) & Docs/s & Speedup vs Resiliparse & Speedup vs FastText \\"
            + "\n"
        )
        handle.write(r"\midrule" + "\n")

        previous_dataset = None
        for record in speedup_df.itertuples(index=False):
            if previous_dataset is not None and previous_dataset != record.dataset:
                handle.write(r"\midrule" + "\n")
            dataset_label = latex_escape(dataset_display_label(record.dataset)) if previous_dataset != record.dataset else ""
            previous_dataset = record.dataset

            values = [
                f"{float(record.mean_ms):.3f}" if not math.isnan(float(record.mean_ms)) else "n/a",
                f"{float(record.docs_per_second):.1f}" if not math.isnan(float(record.docs_per_second)) else "n/a",
            ]
            for comparison_model in models:
                column = speedup_column_name(comparison_model)
                value = float(getattr(record, column))
                best_value = best_lookup[(record.dataset, comparison_model)]
                best = (
                    record.model not in {"URL"}
                    and not math.isnan(value)
                    and not math.isnan(float(best_value))
                    and math.isclose(value, float(best_value), rel_tol=1e-12, abs_tol=1e-12)
                )
                values.append(format_speedup(value, best))

            handle.write(
                f"{dataset_label} & {latex_escape(record.model)} & "
                + " & ".join(values)
                + r"\\"
                + "\n"
            )

        handle.write(r"\bottomrule" + "\n")
        handle.write(r"\end{tabular}" + "\n")

    log("saved", f"runtime speedup: {tex_path}")


def write_runtime_outputs(runtime_df, output_dir: Path):
    if runtime_df is None or runtime_df.empty:
        log("skip", "runtime: no runtime rows")
        return

    log("table", "runtime summary")
    tables_dir = output_dir / TABLES_DIRNAME
    tables_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = output_dir / RUNTIME_DIRNAME
    runtime_dir.mkdir(parents=True, exist_ok=True)

    table_df = runtime_df.sort_values(["dataset", "model"]).copy()
    csv_path = tables_dir / "runtime_summary.csv"
    table_df.to_csv(csv_path, index=False)
    log("saved", f"runtime csv: {csv_path}")
    write_runtime_speedup_table(table_df, tables_dir)

    best_lookup = {}
    for dataset, group in table_df.groupby("dataset"):
        best_lookup[dataset] = group["mean_seconds"].min()

    tex_path = tables_dir / "runtime_summary.tex"
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write(r"\begin{tabular}{llrrrrrr}" + "\n")
        handle.write(r"\toprule" + "\n")
        handle.write(
            r"Dataset & Model & Mean (s) & Median (s) & P95 (s) & Total (s) & Docs/s & Documents \\"
            + "\n"
        )
        handle.write(r"\midrule" + "\n")

        previous_dataset = None
        for record in table_df.itertuples(index=False):
            if previous_dataset is not None and previous_dataset != record.dataset:
                handle.write(r"\midrule" + "\n")
            dataset_label = latex_escape(dataset_display_label(record.dataset)) if previous_dataset != record.dataset else ""
            previous_dataset = record.dataset
            best = math.isclose(
                float(record.mean_seconds),
                float(best_lookup[record.dataset]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            handle.write(
                f"{dataset_label} & {latex_escape(record.model)} & "
                + format_seconds(float(record.mean_seconds), best)
                + f" & {float(record.median_seconds):.4f}"
                + f" & {float(record.p95_seconds):.4f}"
                + f" & {float(record.total_seconds):.2f}"
                + f" & {float(record.docs_per_second):.1f}"
                + f" & {int(record.documents)} "
                + r"\\"
                + "\n"
            )

        handle.write(r"\bottomrule" + "\n")
        handle.write(r"\end{tabular}" + "\n")

    log("saved", f"runtime table: {tex_path}")

    import matplotlib.pyplot as plt
    import seaborn as sns
    from .plots import model_palette, ordered_models, style_model_axis

    log("runtime", "mean_seconds")
    models = ordered_models(table_df["model"].dropna().astype(str))
    palette = model_palette(models)
    table_df = table_df.assign(dataset_label=table_df["dataset"].map(dataset_display_label))
    fig, ax = plt.subplots(figsize=(max(8.0, 1.25 * table_df["dataset_label"].nunique()), 4.8))
    sns.barplot(
        data=table_df,
        x="dataset_label",
        y="mean_seconds",
        hue="model",
        hue_order=models,
        palette=palette,
        ax=ax,
    )
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Mean runtime per document (s)")
    ax.set_title("Detector Runtime by Dataset")
    ax.tick_params(axis="x", rotation=30)
    style_model_axis(ax)
    fig.tight_layout()
    plot_path = runtime_dir / "runtime_mean_seconds.pdf"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    log("saved", f"runtime: {plot_path}")


def write_text_length_plots(length_sweep_paths: dict[str, Path], output_dir: Path):
    from .plots import plot_length_comparison

    text_length_dir = output_dir / TEXT_LENGTH_DIRNAME
    for dataset, path in length_sweep_paths.items():
        if not path.exists():
            log("skip", f"{dataset}: missing length sweep {path}")
            continue

        log("length", dataset)
        length_df = load_csv(path)
        try:
            require_columns(length_df, REQUIRED_LENGTH_SWEEP_COLUMNS, path)
        except ValueError as exc:
            log("skip", f"{dataset}: incompatible length sweep schema ({exc})")
            continue
        dataset_dir = text_length_dir / safe_name(dataset)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        if not has_url_metadata(dataset):
            length_df = length_df[~length_df["model"].map(is_url_model)].copy()
        length_df = length_df[length_df["model"].isin(TEXT_LENGTH_PLOT_MODELS)].copy()
        if length_df.empty:
            log("skip", f"{dataset}: no supported text-length plot models")
            continue
        plot_length_comparison(length_df, dataset, dataset_dir)
        log("saved", f"length plots: {dataset_dir}")


def write_language_similarity_outputs(similarity_df, output_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    if similarity_df is None or similarity_df.empty:
        log("skip", "similarity: no rows")
        return

    normalized_dir = output_dir / NORMALIZED_DIRNAME
    normalized_dir.mkdir(parents=True, exist_ok=True)
    similarity_path = normalized_dir / "language_similarity.csv"
    similarity_df.to_csv(similarity_path, index=False)
    log("saved", f"similarity: {similarity_path}")

    similarity_dir = output_dir / LANGUAGE_SIMILARITY_DIRNAME
    similarity_dir.mkdir(parents=True, exist_ok=True)
    gap_summary = summarize_oop_gap_correctness(similarity_df)
    if gap_summary.empty:
        log("skip", "similarity: no valid OOP gap rows for correctness table")
    else:
        summary_csv = similarity_dir / "oop_score_gap_correctness_summary.csv"
        summary_tex = similarity_dir / "oop_score_gap_correctness_summary.tex"
        gap_summary.to_csv(summary_csv, index=False)
        write_oop_gap_correctness_table(gap_summary, summary_tex, include_dataset=True)
        log("saved", f"similarity: {summary_csv}")
        log("saved", f"similarity: {summary_tex}")

    for dataset, group in similarity_df.groupby("dataset", sort=True):
        dataset_dir = similarity_dir / safe_name(dataset)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        if not gap_summary.empty:
            dataset_gap_summary = gap_summary[gap_summary["dataset"] == dataset].copy()
            if not dataset_gap_summary.empty:
                dataset_csv = dataset_dir / "oop_score_gap_correctness_table.csv"
                dataset_tex = dataset_dir / "oop_score_gap_correctness_table.tex"
                dataset_gap_summary.to_csv(dataset_csv, index=False)
                write_oop_gap_correctness_table(dataset_gap_summary, dataset_tex, include_dataset=False)
                log("saved", f"similarity: {dataset_csv}")
                log("saved", f"similarity: {dataset_tex}")
        valid = group[
            (~group["lang_distance_invalid"].astype(bool))
            & group["lang_distance"].notna()
            & group["rank_1_oop_score"].notna()
        ].copy()
        if valid.empty:
            log("skip", f"{dataset}: no valid language-distance rows")
            continue

        log("similarity", dataset)
        valid["Correctness"] = valid["is_correct"].map({True: "Correct", False: "Incorrect"})
        valid["Family relation"] = valid["same_rank_family"].map({True: "Same rank family", False: "Different rank family"})

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        sns.histplot(
            data=valid,
            x="lang_distance",
            hue="Correctness",
            stat="density",
            common_norm=False,
            bins=25,
            element="step",
            fill=True,
            alpha=0.28,
            ax=ax,
        )
        ax.set_xlabel("Language distance: Resiliparse rank 1 vs rank 2")
        ax.set_ylabel("Density")
        ax.set_title(f"{dataset} - Rank-1 vs rank-2 language distance by correctness")
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
        fig.tight_layout()
        output_path = dataset_dir / "distance_by_correctness.pdf"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        log("saved", f"similarity: {output_path}")

        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        sns.scatterplot(
            data=valid,
            x="lang_distance",
            y="rank_1_oop_score",
            hue="Correctness",
            style="Family relation",
            alpha=0.62,
            s=35,
            edgecolor="none",
            ax=ax,
        )
        ax.set_xlabel("Language distance: Resiliparse rank 1 vs rank 2")
        ax.set_ylabel("Resiliparse rank-1 OOP score")
        ax.set_title(f"{dataset} - OOP score vs language distance")
        ax.grid(axis="both", linestyle=":", linewidth=0.7, alpha=0.7)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_alpha(0.92)
        fig.tight_layout()
        output_path = dataset_dir / "oop_score_vs_language_distance.pdf"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        log("saved", f"similarity: {output_path}")

        gap_valid = group[
            (~group["oop_gap_invalid"].astype(bool))
            & group["rank_1_rank_2_oop_gap"].notna()
            & group["rank_correctness"].notna()
        ].copy()
        if gap_valid.empty:
            log("skip", f"{dataset}: no valid rank-1/rank-2 OOP gap rows")
        else:
            gap_valid["rank_1_rank_2_oop_gap"] = pd.to_numeric(
                gap_valid["rank_1_rank_2_oop_gap"],
                errors="coerce",
            )
            gap_valid = gap_valid.dropna(subset=["rank_1_rank_2_oop_gap"])
            distance_gap_valid = group[
                (~group["lang_distance_invalid"].astype(bool))
                & (~group["oop_gap_invalid"].astype(bool))
                & group["lang_distance"].notna()
                & group["rank_1_rank_2_oop_gap"].notna()
                & group["rank_correctness"].notna()
            ].copy()
            distance_gap_valid["rank_1_rank_2_oop_gap"] = pd.to_numeric(
                distance_gap_valid["rank_1_rank_2_oop_gap"],
                errors="coerce",
            )
            distance_gap_valid = distance_gap_valid.dropna(subset=["rank_1_rank_2_oop_gap"])
            if distance_gap_valid.empty:
                log("skip", f"{dataset}: no valid binned distance/OOP gap rows")
            else:
                distance_gap_valid["Distance bucket"] = distance_gap_valid["lang_distance"].map(language_distance_bucket)
                distance_gap_valid["Distance bucket"] = pd.Categorical(
                    distance_gap_valid["Distance bucket"],
                    categories=LANG_DISTANCE_BUCKETS,
                    ordered=True,
                )
                present_buckets = [
                    bucket
                    for bucket in LANG_DISTANCE_BUCKETS
                    if bucket in set(distance_gap_valid["Distance bucket"].dropna().astype(str))
                ]
                present_correctness = [
                    label
                    for label in RANK_CORRECTNESS_ORDER
                    if label in set(distance_gap_valid["rank_correctness"])
                ]
                fig, ax = plt.subplots(figsize=(7.2, 4.8))
                sns.pointplot(
                    data=distance_gap_valid,
                    x="Distance bucket",
                    y="rank_1_rank_2_oop_gap",
                    hue="rank_correctness",
                    order=present_buckets,
                    hue_order=present_correctness,
                    estimator=np.median,
                    errorbar=None,
                    dodge=0.35,
                    markers=["o", "s", "^", "D"][: len(present_correctness)],
                    linestyles=["-", "--", ":", "-."][: len(present_correctness)],
                    ax=ax,
                )
                ax.set_xlabel("Language distance bucket: Resiliparse rank 1 vs rank 2")
                ax.set_ylabel("Median OOP score gap: rank 2 minus rank 1")
                ax.set_title(f"{dataset} - OOP gap by discrete language distance")
                ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
                legend = ax.get_legend()
                if legend is not None:
                    legend.set_title("Correct rank")
                    legend.get_frame().set_alpha(0.92)
                fig.tight_layout()
                output_path = dataset_dir / "oop_gap_by_language_distance_bin.pdf"
                fig.savefig(output_path, bbox_inches="tight")
                plt.close(fig)
                log("saved", f"similarity: {output_path}")

            present_order = [label for label in RANK_CORRECTNESS_ORDER if label in set(gap_valid["rank_correctness"])]
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            sns.boxplot(
                data=gap_valid,
                x="rank_correctness",
                y="rank_1_rank_2_oop_gap",
                order=present_order,
                color="#d8e6f3",
                width=0.55,
                showfliers=False,
                ax=ax,
            )
            sns.stripplot(
                data=gap_valid,
                x="rank_correctness",
                y="rank_1_rank_2_oop_gap",
                order=present_order,
                color="#24476b",
                alpha=0.36,
                size=3.0,
                jitter=0.22,
                ax=ax,
            )
            ax.axhline(0, color="#333333", linewidth=0.9, linestyle=":")
            ax.set_xlabel("Correct Resiliparse rank")
            ax.set_ylabel("OOP score gap: rank 2 minus rank 1")
            ax.set_title(f"{dataset} - Rank-1/rank-2 OOP score gap by correctness")
            ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
            ax.tick_params(axis="x", rotation=12)
            fig.tight_layout()
            output_path = dataset_dir / "oop_score_gap_by_rank_correctness.pdf"
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            log("saved", f"similarity: {output_path}")

        wrong = valid[~valid["is_correct"].astype(bool)].copy()
        if wrong.empty:
            log("skip", f"{dataset}: no misclassifications for family heatmap")
            continue
        matrix = wrong.pivot_table(
            index="true_family",
            columns="predicted_family",
            values="true_lang",
            aggfunc="count",
            fill_value=0,
        )
        row_sums = matrix.sum(axis=1).replace(0, 1)
        normalized = matrix.div(row_sums, axis=0) * 100

        fig_width = max(6.0, 0.7 * len(normalized.columns) + 3.0)
        fig_height = max(4.5, 0.55 * len(normalized.index) + 2.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.heatmap(
            normalized,
            cmap="viridis",
            vmin=0,
            vmax=100,
            annot=True,
            fmt=".1f",
            linewidths=0.3,
            cbar_kws={"label": "Row-normalized incorrect predictions (%)"},
            ax=ax,
        )
        ax.set_xlabel("Predicted language family")
        ax.set_ylabel("True language family")
        ax.set_title(f"{dataset} - Family-level misclassifications")
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
        output_path = dataset_dir / "family_error_heatmap.pdf"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        log("saved", f"similarity: {output_path}")


# =============================================================================
# CONFUSION MATRICES
# =============================================================================


def row_normalized_matrix(confusion_df, labels):
    import pandas as pd

    matrix = confusion_df.pivot_table(
        index="true_label",
        columns="predicted_label",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    matrix = matrix.reindex(index=labels, columns=labels, fill_value=0)
    row_sums = matrix.sum(axis=1).replace(0, 1)
    return matrix.div(row_sums, axis=0) * 100


def plot_confusion_matrix(matrix, title: str, output_path: Path, pdf_pages=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_labels = len(matrix.index)
    size = max(5.0, min(18.0, 0.45 * n_labels + 3.0))
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        matrix,
        cmap="viridis",
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".1f",
        square=True,
        linewidths=0.3,
        cbar_kws={"label": "Row-normalized (%)"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted language")
    ax.set_ylabel("True language")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    if pdf_pages is not None:
        pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    log("saved", f"matrix: {output_path}")


def family_labels(confusion_df, family: str):
    family_langs = LANGUAGE_FAMILIES[family]
    true_langs = set(confusion_df["true_label"])
    labels = sorted(true_langs & family_langs)
    return labels if len(labels) >= 2 else []


def top_confused_matrix(confusion_df, top_n=20):
    import pandas as pd

    off_diag = confusion_df[confusion_df["true_label"] != confusion_df["predicted_label"]].copy()
    if off_diag.empty:
        return pd.DataFrame()

    off_diag = off_diag.sort_values("count", ascending=False).head(top_n)
    labels = sorted(set(off_diag["true_label"]) | set(off_diag["predicted_label"]))
    return row_normalized_matrix(off_diag, labels)


def write_confusion_matrices(confusion_df, output_dir: Path):
    from matplotlib.backends.backend_pdf import PdfPages

    confusion_dir = output_dir / CONFUSION_DIRNAME
    family_dir = confusion_dir / FAMILY_CONFUSION_DIRNAME
    global_dir = confusion_dir / GLOBAL_CONFUSION_DIRNAME
    family_dir.mkdir(parents=True, exist_ok=True)
    global_dir.mkdir(parents=True, exist_ok=True)

    combined_path = confusion_dir / "all_confusion_matrices.pdf"
    total_groups = confusion_df.groupby(["dataset", "model"]).ngroups
    log("render", f"{total_groups} dataset/model group(s)")
    with PdfPages(combined_path) as pdf_pages:
        for group_index, ((dataset, model), group) in enumerate(
            confusion_df.groupby(["dataset", "model"], sort=True),
            start=1,
        ):
            log("render", f"{group_index}/{total_groups} {dataset} / {model}")
            for family in LANGUAGE_FAMILIES:
                labels = family_labels(group, family)
                if not labels:
                    log("skip", f"{dataset} / {model} / {family}: <2 true languages")
                    continue
                log("render", f"{dataset} / {model} / {family}: {len(labels)} labels")
                filtered = group[
                    group["true_label"].isin(labels) & group["predicted_label"].isin(labels)
                ]
                matrix = row_normalized_matrix(filtered, labels)
                output_path = family_dir / f"confmat_{safe_name(dataset)}_{safe_name(model)}_{safe_name(family)}.pdf"
                plot_confusion_matrix(
                    matrix,
                    f"{dataset} - {model} - {family}",
                    output_path,
                    pdf_pages,
                )

            matrix = top_confused_matrix(group, top_n=20)
            if not matrix.empty:
                log("render", f"{dataset} / {model} / global top-20: {len(matrix.index)} labels")
                output_path = global_dir / f"confmat_{safe_name(dataset)}_{safe_name(model)}_global_top20.pdf"
                plot_confusion_matrix(
                    matrix,
                    f"{dataset} - {model} - Global top-20 confusions",
                    output_path,
                    pdf_pages,
                )
            else:
                log("skip", f"{dataset} / {model} / global top-20: no off-diagonal confusions")

    log("saved", f"combined PDF: {combined_path}")


# =============================================================================
# DEMO DATA
# =============================================================================


def generate_demo_data():
    import pandas as pd

    confusion_rows = [
        ("DemoSet", "Resiliparse", "eng", "eng", 92),
        ("DemoSet", "Resiliparse", "eng", "deu", 8),
        ("DemoSet", "Resiliparse", "deu", "deu", 88),
        ("DemoSet", "Resiliparse", "deu", "nld", 12),
        ("DemoSet", "Resiliparse", "nld", "nld", 75),
        ("DemoSet", "Resiliparse", "nld", "deu", 25),
        ("DemoSet", "FastText", "eng", "eng", 95),
        ("DemoSet", "FastText", "eng", "deu", 5),
        ("DemoSet", "FastText", "deu", "deu", 84),
        ("DemoSet", "FastText", "deu", "nld", 16),
        ("DemoSet", "FastText", "nld", "nld", 82),
        ("DemoSet", "FastText", "nld", "deu", 18),
        ("DemoSet", "RP+FastText", "eng", "eng", 96),
        ("DemoSet", "RP+FastText", "eng", "deu", 4),
        ("DemoSet", "RP+FastText", "deu", "deu", 90),
        ("DemoSet", "RP+FastText", "deu", "nld", 10),
        ("DemoSet", "RP+FastText", "nld", "nld", 85),
        ("DemoSet", "RP+FastText", "nld", "deu", 15),
    ]
    confusion_df = pd.DataFrame(
        confusion_rows,
        columns=["dataset", "model", "true_label", "predicted_label", "count"],
    )
    return metrics_from_confusion(confusion_df), confusion_df


# =============================================================================
# CLI
# =============================================================================


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create publication-ready Resiliparse evaluation visualizations."
    )
    parser.add_argument(
        "--metrics-file",
        default=str(METRICS_FILE),
        help="Normalized metrics CSV path. Used as input only with --normalized-only.",
    )
    parser.add_argument(
        "--confusion-file",
        default=str(CONFUSION_FILE),
        help="Normalized confusion CSV path. Used as input only with --normalized-only.",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory.")
    parser.add_argument(
        "--rp-output",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="Override or add an rp_outputs.csv source. Can be repeated.",
    )
    parser.add_argument(
        "--length-sweep",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="Override or add a length_sweep.csv source for publication plots. Can be repeated.",
    )
    parser.add_argument(
        "--normalized-only",
        action="store_true",
        help="Load --metrics-file and --confusion-file instead of deriving from rp_outputs.csv.",
    )
    parser.add_argument("--generate-demo-data", action="store_true", help="Generate placeholder demo data instead of loading real results.")
    parser.add_argument("--hybrid-cutoff", type=float, default=HYBRID_CUTOFF, help="OOP cutoff for RP+FastText fallback.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    log("output", str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df, confusion_df = load_or_derive_inputs(args)
    log("write", "normalized tables and artifacts")
    runtime_df = None
    rp_output_paths = None
    if not args.normalized_only and not args.generate_demo_data:
        rp_output_paths = parse_rp_output_overrides(args.rp_output)
        runtime_df = derive_runtime_from_rp_outputs(rp_output_paths, args.hybrid_cutoff)
    write_normalized_tables(metrics_df, confusion_df, output_dir)
    write_baseline_table(metrics_df, output_dir)
    write_model_dataset_summary(confusion_df, output_dir, runtime_df)
    write_resource_level_summary_table(metrics_df, output_dir)
    write_resource_level_plots(metrics_df, output_dir)
    if args.normalized_only or args.generate_demo_data:
        log("skip", "extended model summary: requires rp_outputs.csv routing columns")
        log("skip", "runtime: requires rp_outputs.csv timing columns")
        log("skip", "similarity: requires rp_outputs.csv language columns")
        confusion_for_matrices = confusion_df
    else:
        extended_confusion_df = derive_extended_confusion_from_rp_outputs(rp_output_paths, args.hybrid_cutoff)
        write_model_dataset_summary_extended(extended_confusion_df, output_dir)
        write_url_model_evaluation_table(rp_output_paths, output_dir, args.hybrid_cutoff)
        write_runtime_outputs(runtime_df, output_dir)
        similarity_df = derive_language_similarity_from_rp_outputs(rp_output_paths)
        write_language_similarity_outputs(similarity_df, output_dir)
        confusion_for_matrices = extended_confusion_df if not extended_confusion_df.empty else confusion_df
    write_summary_plots(output_dir)
    write_text_length_plots(parse_length_sweep_overrides(args.length_sweep), output_dir)
    write_confusion_matrices(confusion_for_matrices, output_dir)

    log("done", "publication visuals complete")


if __name__ == "__main__":
    main()

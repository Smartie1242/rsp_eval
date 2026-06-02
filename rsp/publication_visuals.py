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

from .languages import safe_tag_distance


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
    "WiLI_2018": Path("results/length_sweep/WiLI_2018/length_sweep.csv"),
    "CommonLID": Path("results/length_sweep/commonlid/length_sweep.csv"),
}

HYBRID_CUTOFF = 1200.0
LANG_AWARE_DISTANCE_THRESHOLD = 30
HIGH_RESOURCE_ARTICLES = 1_000_000
LOW_RESOURCE_ARTICLES = 100_000
INCLUDE_UNKNOWN_RESOURCE = False

MODEL_DEFINITIONS = {
    "Resiliparse": "rank_1_lang_rp",
    "FastText": "rank_1_lang_ft",
    "RP+FastText": "hybrid",
}

EXTENDED_MODEL_ORDER = [
    "Resiliparse",
    "FastText",
    "URL",
    "RP+FastText",
    "RP+URL",
    "RP+FastText lang-aware",
    "RP+URL lang-aware",
]

# Wikipedia article counts are deliberately editable. Values below cover common
# languages in the current repository outputs; unknown languages are categorized
# as "unknown" resource level.
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

REQUIRED_RUNTIME_COLUMNS = {
    "runtime_rp",
    "runtime_ft",
}

REQUIRED_LANGUAGE_SIMILARITY_COLUMNS = {
    "label",
    "rank_1_lang_rp",
    "rank_2_lang_rp",
    "rank_1_oop_score",
}

BAD_LANGUAGE_VALUES = {"unknown", "unk", "none", "", None}


# =============================================================================
# DATA NORMALIZATION
# =============================================================================


def log(stage: str, message: str) -> None:
    """Print a compact, consistently formatted status message."""
    print(f"[{stage}] {message}", flush=True)


def safe_name(value: str) -> str:
    """Convert a label to a filesystem-safe name."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def language_family(language: str) -> str:
    for family, languages in LANGUAGE_FAMILIES.items():
        if language in languages:
            return family
    return "Other"


def resource_level(language: str) -> str:
    count = WIKIPEDIA_ARTICLE_COUNTS.get(language)
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


def model_predictions(df, model: str, cutoff: float):
    import numpy as np

    if model == "RP+FastText":
        scores = df["rank_1_oop_score"].fillna(math.inf).astype(float)
        return np.where(scores >= cutoff, df["rank_1_lang_ft"], df["rank_1_lang_rp"])
    return df[MODEL_DEFINITIONS[model]]


def extended_model_predictions(df, model: str, cutoff: float, lang_threshold: int = LANG_AWARE_DISTANCE_THRESHOLD):
    import numpy as np
    import pandas as pd

    if model == "Resiliparse":
        return df["rank_1_lang_rp"]
    if model == "FastText":
        return df["rank_1_lang_ft"]
    if model == "URL":
        return df["lang_url"]

    scores = pd.to_numeric(df["rank_1_oop_score"], errors="coerce").fillna(math.inf)
    cutoff_fallback = scores >= cutoff
    if model == "RP+FastText":
        return np.where(cutoff_fallback, df["rank_1_lang_ft"], df["rank_1_lang_rp"])
    if model == "RP+URL":
        return np.where(cutoff_fallback, df["lang_url"], df["rank_1_lang_rp"])

    distances = df.apply(lambda row: safe_tag_distance(row["rank_1_lang_rp"], row["rank_2_lang_rp"]), axis=1)
    distance_fallback = distances > lang_threshold
    fallback = cutoff_fallback | distance_fallback
    if model == "RP+FastText lang-aware":
        return np.where(fallback, df["rank_1_lang_ft"], df["rank_1_lang_rp"])
    if model == "RP+URL lang-aware":
        return np.where(fallback, df["lang_url"], df["rank_1_lang_rp"])
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
    fallback = scores >= cutoff
    if model == "RP+FastText":
        return runtime_rp + pd.Series(np.where(fallback, runtime_ft, 0.0), index=df.index)
    if model == "RP+URL":
        return runtime_rp + pd.Series(np.where(fallback, runtime_url, 0.0), index=df.index)
    raise ValueError(f"Unknown runtime model: {model}")


def derive_confusion_from_rp_outputs(paths: dict[str, Path], hybrid_cutoff: float):
    import pandas as pd

    rows = []
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

        available_models = list(EXTENDED_MODEL_ORDER)
        if "lang_url" not in df.columns:
            available_models = [
                model for model in available_models if model not in {"URL", "RP+URL", "RP+URL lang-aware"}
            ]
            log("skip", f"{dataset}: URL models require lang_url")
        if "rank_2_lang_rp" not in df.columns:
            available_models = [
                model for model in available_models if not model.endswith("lang-aware")
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
    runtime_models = ["Resiliparse", "FastText", "URL", "RP+FastText", "RP+URL"]
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

        available_models = list(runtime_models)
        if "runtime_url" not in df.columns:
            available_models = [model for model in available_models if model not in {"URL", "RP+URL"}]

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
    if value in BAD_LANGUAGE_VALUES:
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
        for record in df.itertuples(index=False):
            true_lang = str(getattr(record, "label", "unknown") or "unknown")
            rank_1_lang = str(getattr(record, "rank_1_lang_rp", "unknown") or "unknown")
            rank_2_lang = str(getattr(record, "rank_2_lang_rp", "unknown") or "unknown")
            oop_score = getattr(record, "rank_1_oop_score", math.nan)
            rank_1_valid = language_tag_is_valid(rank_1_lang)
            rank_2_valid = language_tag_is_valid(rank_2_lang)
            invalid = not (rank_1_valid and rank_2_valid)
            distance = safe_tag_distance(rank_1_lang, rank_2_lang)
            true_family = language_family(true_lang)
            predicted_family = language_family(rank_1_lang)
            rank_1_family = language_family(rank_1_lang)
            rank_2_family = language_family(rank_2_lang)
            rows.append(
                {
                    "dataset": dataset,
                    "true_lang": true_lang,
                    "rank_1_lang": rank_1_lang,
                    "rank_2_lang": rank_2_lang,
                    "predicted_lang": rank_1_lang,
                    "rank_1_family": rank_1_family,
                    "rank_2_family": rank_2_family,
                    "true_family": true_family,
                    "predicted_family": predicted_family,
                    "same_rank_family": rank_1_family == rank_2_family,
                    "same_family": true_family == predicted_family,
                    "is_correct": true_lang == rank_1_lang,
                    "rank_1_rank_2_distance": distance,
                    "lang_distance": distance,
                    "rank_1_oop_score": oop_score,
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
                "predicted_lang",
                "rank_1_family",
                "rank_2_family",
                "true_family",
                "predicted_family",
                "same_rank_family",
                "same_family",
                "is_correct",
                "rank_1_rank_2_distance",
                "lang_distance",
                "rank_1_oop_score",
                "rank_distance_invalid",
                "lang_distance_invalid",
            ]
        )

    similarity_df = pd.DataFrame(rows)
    similarity_df["rank_1_oop_score"] = pd.to_numeric(similarity_df["rank_1_oop_score"], errors="coerce")
    log("similarity", f"rows: {len(similarity_df):,}")
    return similarity_df


def metrics_from_confusion(confusion_df):
    import pandas as pd

    rows = []
    log("metrics", "computing per-language scores")
    for (dataset, model), group in confusion_df.groupby(["dataset", "model"], sort=True):
        log("metrics", f"{dataset} / {model}")
        labels = sorted(set(group["true_label"]) | set(group["predicted_label"]))
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
        labels = sorted(set(group["true_label"]) | set(group["predicted_label"]))
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
            dataset_label = latex_escape(record.dataset) if previous_dataset != record.dataset else ""
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


def write_model_dataset_summary(confusion_df, output_dir: Path):
    log("table", "model dataset summary")
    tables_dir = output_dir / TABLES_DIRNAME
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_df = model_metric_rows_from_confusion(confusion_df)
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
        for record in table_df.sort_values(["dataset", "model"]).itertuples(index=False):
            if previous_dataset is not None and previous_dataset != record.dataset:
                handle.write(r"\midrule" + "\n")
            dataset_label = latex_escape(record.dataset) if previous_dataset != record.dataset else ""
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

    log("saved", f"model summary: {output_path}")


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
            dataset_label = latex_escape(record.dataset) if previous_dataset != record.dataset else ""
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
            dataset_label = latex_escape(record.dataset) if previous_dataset != record.dataset else ""
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

    levels = [level for level in ["low", "mid", "high"] if level in set(table_df["resource_level"])]
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
            col="dataset",
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


def speedup_column_name(model: str) -> str:
    return f"speedup_vs_{safe_name(model).lower()}"


def build_runtime_speedup_table(runtime_df):
    import pandas as pd
    from .plots import ordered_models

    rows = []
    models = ordered_models(runtime_df["model"].dropna().astype(str))
    for dataset, group in runtime_df.groupby("dataset", sort=True):
        mean_by_model = {
            record.model: float(record.mean_seconds)
            for record in group.itertuples(index=False)
            if float(record.mean_seconds) > 0
        }
        for model in ordered_models(group["model"].dropna().astype(str)):
            model_mean = mean_by_model.get(model)
            row = {
                "dataset": dataset,
                "model": model,
                "mean_seconds": model_mean if model_mean is not None else math.nan,
            }
            for comparison_model in models:
                comparison_mean = mean_by_model.get(comparison_model)
                if model_mean is None or comparison_mean is None:
                    row[speedup_column_name(comparison_model)] = math.nan
                else:
                    row[speedup_column_name(comparison_model)] = comparison_mean / model_mean
            rows.append(row)

    return pd.DataFrame(rows), models


def write_runtime_speedup_table(runtime_df, tables_dir: Path):
    log("table", "runtime speedup comparison")
    speedup_df, models = build_runtime_speedup_table(runtime_df)
    csv_path = tables_dir / "runtime_speedup_comparison.csv"
    speedup_df.to_csv(csv_path, index=False)
    log("saved", f"runtime speedup csv: {csv_path}")

    best_lookup = {}
    for (dataset, comparison_model), group in [
        ((dataset, comparison_model), group)
        for dataset, dataset_group in speedup_df.groupby("dataset", sort=True)
        for comparison_model in models
        for group in [dataset_group]
    ]:
        column = speedup_column_name(comparison_model)
        best_lookup[(dataset, comparison_model)] = group[column].max(skipna=True)

    tex_path = tables_dir / "runtime_speedup_comparison.tex"
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write(r"\begin{tabular}{ll" + "r" * len(models) + "}" + "\n")
        handle.write(r"\toprule" + "\n")
        handle.write(
            "Dataset & Model & "
            + " & ".join(f"vs {latex_escape(model)}" for model in models)
            + r" \\"
            + "\n"
        )
        handle.write(r"\midrule" + "\n")

        previous_dataset = None
        for record in speedup_df.itertuples(index=False):
            if previous_dataset is not None and previous_dataset != record.dataset:
                handle.write(r"\midrule" + "\n")
            dataset_label = latex_escape(record.dataset) if previous_dataset != record.dataset else ""
            previous_dataset = record.dataset

            values = []
            for comparison_model in models:
                column = speedup_column_name(comparison_model)
                value = float(getattr(record, column))
                best = math.isclose(
                    value,
                    float(best_lookup[(record.dataset, comparison_model)]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
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
            dataset_label = latex_escape(record.dataset) if previous_dataset != record.dataset else ""
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
    fig, ax = plt.subplots(figsize=(max(8.0, 1.25 * table_df["dataset"].nunique()), 4.8))
    sns.barplot(
        data=table_df,
        x="dataset",
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
        plot_length_comparison(length_df, dataset, dataset_dir)
        log("saved", f"length plots: {dataset_dir}")


def write_language_similarity_outputs(similarity_df, output_dir: Path):
    import matplotlib.pyplot as plt
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

    for dataset, group in similarity_df.groupby("dataset", sort=True):
        dataset_dir = similarity_dir / safe_name(dataset)
        dataset_dir.mkdir(parents=True, exist_ok=True)
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
    write_normalized_tables(metrics_df, confusion_df, output_dir)
    write_baseline_table(metrics_df, output_dir)
    write_model_dataset_summary(confusion_df, output_dir)
    write_resource_level_summary_table(metrics_df, output_dir)
    write_resource_level_plots(metrics_df, output_dir)
    if args.normalized_only or args.generate_demo_data:
        log("skip", "extended model summary: requires rp_outputs.csv routing columns")
        log("skip", "runtime: requires rp_outputs.csv timing columns")
        log("skip", "similarity: requires rp_outputs.csv language columns")
    else:
        rp_output_paths = parse_rp_output_overrides(args.rp_output)
        extended_confusion_df = derive_extended_confusion_from_rp_outputs(rp_output_paths, args.hybrid_cutoff)
        write_model_dataset_summary_extended(extended_confusion_df, output_dir)
        runtime_df = derive_runtime_from_rp_outputs(rp_output_paths, args.hybrid_cutoff)
        write_runtime_outputs(runtime_df, output_dir)
        similarity_df = derive_language_similarity_from_rp_outputs(rp_output_paths)
        write_language_similarity_outputs(similarity_df, output_dir)
    write_text_length_plots(parse_length_sweep_overrides(args.length_sweep), output_dir)
    write_confusion_matrices(confusion_df, output_dir)

    log("done", "publication visuals complete")


if __name__ == "__main__":
    main()

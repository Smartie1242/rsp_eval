"""Routing and cutoff sweep computations."""

from __future__ import annotations

from .languages import safe_tag_distance
from .metrics import normalize_prediction_columns, safe_accuracy, safe_f1

LENGTH_CUTOFFS = [300, 600, 1200, 2400, 4800]
SCORE_CUTOFFS = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 2000]
HYBRID_CUTOFF = 1200.0
LENGTH_BINS = [0, 300, 600, 1200, 2400, 4800, 100000]
LENGTH_BIN_LABELS = [
    "0-300",
    "300-600",
    "600-1200",
    "1200-2400",
    "2400-4800",
    "4800+",
]


def add_length_features(df):
    """Add character-length bin labels."""
    import pandas as pd

    df = df.copy()
    df["text_length"] = df["text_length"].fillna(0).astype(int)
    df["text_length_bin"] = pd.cut(
        df["text_length"], bins=LENGTH_BINS, labels=LENGTH_BIN_LABELS, include_lowest=True
    )
    return df


def length_model_predictions(df, hybrid_cutoff=HYBRID_CUTOFF):
    """Return standalone and composite model predictions for length analysis."""
    import numpy as np

    df = normalize_prediction_columns(df)
    oop_score = df["rank_1_oop_score"].fillna(float("inf")).astype(float)
    fallback_mask = oop_score >= hybrid_cutoff

    return {
        "Resiliparse": df["rank_1_lang_rp"],
        "FastText": df["rank_1_lang_ft"],
        "URL": df["lang_url"],
        "RP+FastText": np.where(fallback_mask, df["rank_1_lang_ft"], df["rank_1_lang_rp"]),
        "RP+URL": np.where(fallback_mask, df["lang_url"], df["rank_1_lang_rp"]),
    }


def evaluate_length_bin(df, length_bin, hybrid_cutoff=HYBRID_CUTOFF):
    """Evaluate detector performance for one text-length bin."""
    df = normalize_prediction_columns(df)
    df["text_length"] = df["text_length"].fillna(0).astype(int)
    if "text_length_bin" not in df.columns:
        df = add_length_features(df)
    model_predictions = length_model_predictions(df, hybrid_cutoff=hybrid_cutoff)
    mask = df["text_length_bin"].astype(str) == str(length_bin)
    support = int(mask.sum())
    total = len(df)

    rows = []
    for model, predictions in model_predictions.items():
        rows.append(
            {
                "text_length_bin": str(length_bin),
                "model": model,
                "support": support,
                "coverage": support / total if total else 0.0,
                "accuracy": safe_accuracy(df.loc[mask, "label"], predictions[mask]),
                "f1_macro": safe_f1(df.loc[mask, "label"], predictions[mask]),
                "hybrid_cutoff": hybrid_cutoff,
            }
        )
    return rows


def evaluate_length_cutoff(df, cutoff, hybrid_cutoff=HYBRID_CUTOFF):
    """Compatibility wrapper for old callers; evaluates the bin ending at cutoff."""
    for label in LENGTH_BIN_LABELS:
        if label.endswith(f"-{cutoff}") or label == f"{cutoff}+":
            return evaluate_length_bin(df, label, hybrid_cutoff=hybrid_cutoff)
    return evaluate_length_bin(df, str(cutoff), hybrid_cutoff=hybrid_cutoff)


def run_length_sweep(df, cutoffs=None, hybrid_cutoff=HYBRID_CUTOFF):
    """Evaluate detector performance across text-length bins."""
    import pandas as pd

    df = add_length_features(df)
    rows = []
    for length_bin in LENGTH_BIN_LABELS:
        if (df["text_length_bin"].astype(str) == length_bin).any():
            rows.extend(
                evaluate_length_bin(df, length_bin, hybrid_cutoff=hybrid_cutoff)
            )
    return pd.DataFrame(rows)


def evaluate_score_cutoff(df, cutoff, lang_thresh=30, score_col="rank_1_oop_score"):
    """Evaluate routing by Resiliparse score cutoff, with lang-aware variant."""
    import numpy as np

    df = normalize_prediction_columns(df)
    df["lang_dist"] = df.apply(
        lambda r: safe_tag_distance(r["rank_1_lang_rp"], r["rank_2_lang_rp"]), axis=1
    )

    rp_mask_base = df[score_col] >= cutoff
    fb_mask_base = ~rp_mask_base

    base_rp_f1 = safe_f1(
        df.loc[rp_mask_base, "label"], df.loc[rp_mask_base, "rank_1_lang_rp"]
    )
    base_ft_f1 = safe_f1(
        df.loc[fb_mask_base, "label"], df.loc[fb_mask_base, "rank_1_lang_ft"]
    )
    base_url_f1 = safe_f1(
        df.loc[fb_mask_base, "label"], df.loc[fb_mask_base, "lang_url"]
    )

    system_ft_base = np.where(rp_mask_base, df["rank_1_lang_rp"], df["rank_1_lang_ft"])
    system_url_base = np.where(rp_mask_base, df["rank_1_lang_rp"], df["lang_url"])

    rp_mask_lang = (df[score_col] >= cutoff) & (df["lang_dist"] <= lang_thresh)
    fb_mask_lang = ~rp_mask_lang

    lang_rp_f1 = safe_f1(
        df.loc[rp_mask_lang, "label"], df.loc[rp_mask_lang, "rank_1_lang_rp"]
    )
    lang_ft_f1 = safe_f1(
        df.loc[fb_mask_lang, "label"], df.loc[fb_mask_lang, "rank_1_lang_ft"]
    )
    lang_url_f1 = safe_f1(
        df.loc[fb_mask_lang, "label"], df.loc[fb_mask_lang, "lang_url"]
    )

    system_ft_lang = np.where(rp_mask_lang, df["rank_1_lang_rp"], df["rank_1_lang_ft"])
    system_url_lang = np.where(rp_mask_lang, df["rank_1_lang_rp"], df["lang_url"])

    return {
        "cutoff": cutoff,
        "base_rp_f1": base_rp_f1,
        "base_ft_f1": base_ft_f1,
        "base_url_f1": base_url_f1,
        "system_ft_f1": safe_f1(df["label"], system_ft_base),
        "system_url_f1": safe_f1(df["label"], system_url_base),
        "lang_rp_f1": lang_rp_f1,
        "lang_ft_f1": lang_ft_f1,
        "lang_url_f1": lang_url_f1,
        "system_ft_lang_f1": safe_f1(df["label"], system_ft_lang),
        "system_url_lang_f1": safe_f1(df["label"], system_url_lang),
        "rp_coverage": rp_mask_base.mean(),
        "lang_rp_coverage": rp_mask_lang.mean(),
        "rp_acc": safe_accuracy(
            df.loc[rp_mask_base, "label"], df.loc[rp_mask_base, "rank_1_lang_rp"]
        ),
        "system_ft_acc": safe_accuracy(df["label"], system_ft_base),
        "system_url_acc": safe_accuracy(df["label"], system_url_base),
        "system_ft_lang_acc": safe_accuracy(df["label"], system_ft_lang),
        "system_url_lang_acc": safe_accuracy(df["label"], system_url_lang),
    }


def run_score_cutoff_sweep(df, cutoffs=None):
    """Evaluate all score cutoffs and return a results DataFrame."""
    import pandas as pd

    cutoffs = cutoffs or SCORE_CUTOFFS
    return pd.DataFrame([evaluate_score_cutoff(df, cutoff) for cutoff in cutoffs])

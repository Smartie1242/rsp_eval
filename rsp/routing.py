"""Routing and cutoff sweep computations."""

from __future__ import annotations

from .languages import safe_tag_distance
from .metrics import normalize_prediction_columns, safe_accuracy, safe_f1

LENGTH_CUTOFFS = [300, 600, 1200, 2400, 4800]
SCORE_CUTOFFS = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 2000]
HYBRID_CUTOFF = 1200.0
LANG_AWARE_DISTANCE_THRESHOLD = 30
URL_ENGLISH_FALLBACK_SCORE = 0.05
LENGTH_BINS = [0, 300, 600, 1200, 2400, 4800, 100000]
LENGTH_BIN_LABELS = [
    "0-300",
    "300-600",
    "600-1200",
    "1200-2400",
    "2400-4800",
    "4800+",
]


def valid_language_mask(series):
    """Return a boolean mask for language tags that can be compared safely."""
    return series.map(lambda value: safe_tag_distance(value, value) < 100)


def rank_distance_trigger_mask(df, lang_thresh=LANG_AWARE_DISTANCE_THRESHOLD):
    """Trigger fallback when Resiliparse rank 1 and rank 2 are linguistically distant."""
    distances = df.apply(
        lambda row: safe_tag_distance(row["rank_1_lang_rp"], row["rank_2_lang_rp"]),
        axis=1,
    )
    valid_distance = distances < 100
    return (~valid_distance) | (distances > lang_thresh)


def combined_url_fasttext_decision(
    df,
    cutoff=HYBRID_CUTOFF,
    lang_thresh=LANG_AWARE_DISTANCE_THRESHOLD,
    score_col="rank_1_oop_score",
):
    """Route fragile Resiliparse cases to UrlExtractor when it supports rank 1/2, else FastText."""
    import pandas as pd

    df = normalize_prediction_columns(df)
    rank_2 = df["rank_2_lang_rp"].fillna("unknown").astype(str)
    scores = pd.to_numeric(df[score_col], errors="coerce").fillna(float("inf"))
    url_scores = pd.to_numeric(df.get("url_score", 0.0), errors="coerce").fillna(0.0)

    fallback = (scores >= cutoff) | rank_distance_trigger_mask(df, lang_thresh=lang_thresh)
    valid_url = valid_language_mask(df["lang_url"])
    explicit_url = url_scores > URL_ENGLISH_FALLBACK_SCORE
    url_supports_rank_1 = valid_url & explicit_url & (df["lang_url"] == df["rank_1_lang_rp"])
    url_supports_rank_2 = valid_url & (df["lang_url"] == rank_2)

    use_url = fallback & (url_supports_rank_1 | url_supports_rank_2)
    use_fasttext = fallback & ~use_url

    prediction = df["rank_1_lang_rp"].copy()
    prediction.loc[use_url] = df.loc[use_url, "lang_url"]
    prediction.loc[use_fasttext] = df.loc[use_fasttext, "rank_1_lang_ft"]

    return pd.DataFrame(
        {
            "prediction": prediction,
            "fallback": fallback,
            "use_url": use_url,
            "use_fasttext": use_fasttext,
        },
        index=df.index,
    )


def split_trigger_url_fasttext_decision(
    df,
    cutoff=HYBRID_CUTOFF,
    lang_thresh=LANG_AWARE_DISTANCE_THRESHOLD,
    score_col="rank_1_oop_score",
):
    """Use FastText for OOP fallback; use explicit UrlExtractor evidence for distance-only fallback."""
    import pandas as pd

    df = normalize_prediction_columns(df)
    scores = pd.to_numeric(df[score_col], errors="coerce").fillna(float("inf"))
    url_scores = pd.to_numeric(df.get("url_score", 0.0), errors="coerce").fillna(0.0)

    cutoff_fallback = scores >= cutoff
    distance_fallback = rank_distance_trigger_mask(df, lang_thresh=lang_thresh)
    distance_only = distance_fallback & ~cutoff_fallback
    explicit_url = valid_language_mask(df["lang_url"]) & (url_scores > URL_ENGLISH_FALLBACK_SCORE)

    use_fasttext = cutoff_fallback
    use_url = distance_only & explicit_url

    prediction = df["rank_1_lang_rp"].copy()
    prediction.loc[use_fasttext] = df.loc[use_fasttext, "rank_1_lang_ft"]
    prediction.loc[use_url] = df.loc[use_url, "lang_url"]

    return pd.DataFrame(
        {
            "prediction": prediction,
            "fallback": cutoff_fallback | distance_fallback,
            "use_url": use_url,
            "use_fasttext": use_fasttext,
            "cutoff_fallback": cutoff_fallback,
            "distance_fallback": distance_fallback,
        },
        index=df.index,
    )


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
    cutoff_fallback = oop_score >= hybrid_cutoff
    rank_distance_fallback = rank_distance_trigger_mask(df)
    combined = combined_url_fasttext_decision(df, hybrid_cutoff)
    split = split_trigger_url_fasttext_decision(df, hybrid_cutoff)

    return {
        "Resiliparse": df["rank_1_lang_rp"],
        "FastText": df["rank_1_lang_ft"],
        "RP+FastText": np.where(cutoff_fallback, df["rank_1_lang_ft"], df["rank_1_lang_rp"]),
        "RP+URL": np.where(cutoff_fallback, df["lang_url"], df["rank_1_lang_rp"]),
        "RP+FastText lang-aware": np.where(
            cutoff_fallback | rank_distance_fallback, df["rank_1_lang_ft"], df["rank_1_lang_rp"]
        ),
        "RP+URL lang-aware": np.where(
            cutoff_fallback | rank_distance_fallback, df["lang_url"], df["rank_1_lang_rp"]
        ),
        "RP+FastText+URL lang-aware": combined["prediction"],
        "RP+FastText+URL split-trigger": split["prediction"],
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
    """Evaluate cutoff, lang-aware, URL fallback, and combined routing variants."""
    import numpy as np

    df = normalize_prediction_columns(df)
    df["lang_dist"] = df.apply(
        lambda r: safe_tag_distance(r["rank_1_lang_rp"], r["rank_2_lang_rp"]), axis=1
    )

    cutoff_fallback = df[score_col] >= cutoff
    rank_distance_fallback = rank_distance_trigger_mask(df, lang_thresh=lang_thresh)
    combined = combined_url_fasttext_decision(df, cutoff, lang_thresh=lang_thresh, score_col=score_col)
    split = split_trigger_url_fasttext_decision(df, cutoff, lang_thresh=lang_thresh, score_col=score_col)

    fb_mask_base = cutoff_fallback
    rp_mask_base = ~fb_mask_base

    base_rp_f1 = safe_f1(
        df.loc[rp_mask_base, "label"], df.loc[rp_mask_base, "rank_1_lang_rp"]
    )
    base_ft_f1 = safe_f1(
        df.loc[fb_mask_base, "label"], df.loc[fb_mask_base, "rank_1_lang_ft"]
    )

    system_ft_base = np.where(fb_mask_base, df["rank_1_lang_ft"], df["rank_1_lang_rp"])
    system_url_base = np.where(fb_mask_base, df["lang_url"], df["rank_1_lang_rp"])

    fb_mask_lang = cutoff_fallback | rank_distance_fallback
    rp_mask_lang = ~fb_mask_lang

    lang_rp_f1 = safe_f1(
        df.loc[rp_mask_lang, "label"], df.loc[rp_mask_lang, "rank_1_lang_rp"]
    )
    lang_ft_f1 = safe_f1(
        df.loc[fb_mask_lang, "label"], df.loc[fb_mask_lang, "rank_1_lang_ft"]
    )

    system_ft_lang = np.where(fb_mask_lang, df["rank_1_lang_ft"], df["rank_1_lang_rp"])
    system_url_lang = np.where(fb_mask_lang, df["lang_url"], df["rank_1_lang_rp"])
    system_ft_url_lang = combined["prediction"]
    system_ft_url_split = split["prediction"]
    resiliparse_prediction = df["rank_1_lang_rp"]
    fasttext_prediction = df["rank_1_lang_ft"]

    return {
        "cutoff": cutoff,
        "resiliparse_f1": safe_f1(df["label"], resiliparse_prediction),
        "fasttext_f1": safe_f1(df["label"], fasttext_prediction),
        "base_rp_f1": base_rp_f1,
        "base_ft_f1": base_ft_f1,
        "system_ft_f1": safe_f1(df["label"], system_ft_base),
        "system_url_f1": safe_f1(df["label"], system_url_base),
        "lang_rp_f1": lang_rp_f1,
        "lang_ft_f1": lang_ft_f1,
        "system_ft_lang_f1": safe_f1(df["label"], system_ft_lang),
        "system_url_lang_f1": safe_f1(df["label"], system_url_lang),
        "system_ft_url_lang_f1": safe_f1(df["label"], system_ft_url_lang),
        "system_ft_url_split_f1": safe_f1(df["label"], system_ft_url_split),
        "rp_coverage": rp_mask_base.mean(),
        "lang_rp_coverage": rp_mask_lang.mean(),
        "combined_url_coverage": combined["use_url"].mean(),
        "combined_fasttext_coverage": combined["use_fasttext"].mean(),
        "split_url_coverage": split["use_url"].mean(),
        "split_fasttext_coverage": split["use_fasttext"].mean(),
        "resiliparse_acc": safe_accuracy(df["label"], resiliparse_prediction),
        "fasttext_acc": safe_accuracy(df["label"], fasttext_prediction),
        "rp_acc": safe_accuracy(
            df.loc[rp_mask_base, "label"], df.loc[rp_mask_base, "rank_1_lang_rp"]
        ),
        "system_ft_acc": safe_accuracy(df["label"], system_ft_base),
        "system_url_acc": safe_accuracy(df["label"], system_url_base),
        "system_ft_lang_acc": safe_accuracy(df["label"], system_ft_lang),
        "system_url_lang_acc": safe_accuracy(df["label"], system_url_lang),
        "system_ft_url_lang_acc": safe_accuracy(df["label"], system_ft_url_lang),
        "system_ft_url_split_acc": safe_accuracy(df["label"], system_ft_url_split),
    }


def run_score_cutoff_sweep(df, cutoffs=None):
    """Evaluate all score cutoffs and return a results DataFrame."""
    import pandas as pd

    cutoffs = cutoffs or SCORE_CUTOFFS
    return pd.DataFrame([evaluate_score_cutoff(df, cutoff) for cutoff in cutoffs])

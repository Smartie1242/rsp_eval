"""Plot helpers for analysis and visualisation commands."""

from __future__ import annotations

import itertools
import os

from .languages import safe_tag_distance


MODEL_ORDER = [
    "Resiliparse",
    "FastText",
    "URL",
    "RP+FastText",
    "RP+URL",
    "RP+FastText lang-aware",
    "RP+URL lang-aware",
    "RP+FastText+URL split-trigger",
    "RP+FastText+URL lang-aware",
]
MODEL_COLORS = {
    "Resiliparse": "#1f77b4",
    "FastText": "#d62728",
    "URL": "#2ca02c",
    "RP+FastText": "#9467bd",
    "RP+URL": "#bcbd22",
    "RP+FastText lang-aware": "#ff7f0e",
    "RP+URL lang-aware": "#e377c2",
    "RP+FastText+URL split-trigger": "#8c564b",
    "RP+FastText+URL lang-aware": "#17becf",
}
MODEL_LINESTYLES = {
    "Resiliparse": "-",
    "FastText": "--",
    "URL": ":",
    "RP+FastText": "-.",
    "RP+URL": (0, (4, 2)),
    "RP+FastText lang-aware": (0, (5, 2)),
    "RP+URL lang-aware": (0, (2, 1)),
    "RP+FastText+URL split-trigger": (0, (3, 1, 1, 1)),
    "RP+FastText+URL lang-aware": (0, (1, 1)),
}
MODEL_MARKERS = {
    "Resiliparse": "o",
    "FastText": "s",
    "URL": "^",
    "RP+FastText": "D",
    "RP+URL": "h",
    "RP+FastText lang-aware": "X",
    "RP+URL lang-aware": "<",
    "RP+FastText+URL split-trigger": ">",
    "RP+FastText+URL lang-aware": "v",
}


def ordered_models(models):
    """Return present model names in stable publication order."""
    present = [model for model in MODEL_ORDER if model in set(models)]
    extras = sorted(set(models) - set(MODEL_ORDER))
    return present + extras


def model_palette(models):
    """Return a color mapping for present model names."""
    fallback_colors = ["#17becf", "#8c564b", "#e377c2", "#7f7f7f"]
    palette = {}
    for index, model in enumerate(ordered_models(models)):
        palette[model] = MODEL_COLORS.get(model, fallback_colors[index % len(fallback_colors)])
    return palette


def model_marker_offsets(models, width=0.18):
    """Return small categorical x-axis offsets to reveal overlapping markers."""
    model_names = ordered_models(models)
    if len(model_names) <= 1:
        return {model_names[0]: 0.0} if model_names else {}
    step = width / (len(model_names) - 1)
    start = -width / 2
    return {model: start + index * step for index, model in enumerate(model_names)}


def style_model_axis(ax):
    """Apply shared grid and legend styling for model comparison plots."""
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Model")
        legend.get_frame().set_alpha(0.92)


def plot_histogram(df, dataset_name, output_dir, gap="gap_1_2"):
    """Histogram comparing score-gap distributions."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    gap_label_map = {
        "gap_1_2": "Score difference (Rank 1 vs Rank 2)",
        "gap_1_3": "Score difference (Rank 1 vs Rank 3)",
        "gap_1_4": "Score difference (Rank 1 vs Rank 4)",
        "gap_1_5": "Score difference (Rank 1 vs Rank 5)",
    }

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x=gap,
        hue="is_correct",
        bins=40,
        kde=True,
        stat="density",
        common_norm=False,
    )
    plt.xlabel(gap_label_map.get(gap, gap.replace("_", " ")))
    plt.ylabel("Density")
    plt.title(f"{dataset_name} - {gap} Histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_{gap}_histogram.png"), dpi=300)
    plt.close()


def plot_simple_gap_summary(df, dataset_name, output_dir, score_type="oop"):
    """Plot mean score gap by rank and correctness."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    rows = []
    for rank in range(2, 6):
        col = f"rank_{rank}_{score_type}_score"
        if col not in df.columns:
            continue
        for label in [True, False]:
            subset = df[df["is_correct"] == label]
            gap = subset[col] - subset[f"rank_1_{score_type}_score"]
            rows.append(
                {
                    "rank": f"Rank {rank}",
                    "is_correct": "Correct" if label else "Wrong",
                    "mean_gap": gap.mean(),
                }
            )

    plot_df = pd.DataFrame(rows)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="rank", y="mean_gap", hue="is_correct")
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Mean Score Gap (Rank k - Rank 1)")
    plt.xlabel("Rank")
    plt.title(f"{dataset_name} - Score Gap vs Rank 1")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_simple_gap_summary.png"), dpi=300)
    plt.close()


def plot_rank_correctness_share(df, dataset_name, output_dir):
    """Plot share of correct vs wrong predictions per rank position."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    rows = []
    for rank in range(1, 6):
        col = f"rank_{rank}_lang_rp"
        if col not in df.columns:
            continue
        correct = (df["label"] == df[col]).mean()
        rows.append({"rank": f"Rank {rank}", "Correct": correct, "Wrong": 1 - correct})

    plot_df = pd.DataFrame(rows).melt(
        id_vars="rank", var_name="Outcome", value_name="Share"
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="rank", y="Share", hue="Outcome")
    plt.ylabel("Share")
    plt.title(f"{dataset_name} - Rank correctness distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_rank_correctness_share.png"), dpi=300)
    plt.close()


def plot_gap_vs_lang_distance(df, dataset_name, output_dir, score_type="oop"):
    """Scatter plot plus binned CSV summary for language distance vs score gap."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = df.copy()
    lang1_col = "rank_1_lang_rp"
    lang2_col = "rank_2_lang_rp"
    if lang1_col not in df.columns:
        raise ValueError(f"Missing column: {lang1_col}")
    if lang2_col not in df.columns:
        raise ValueError(f"Missing column: {lang2_col}")

    df["score_gap"] = df[f"rank_2_{score_type}_score"] - df[f"rank_1_{score_type}_score"]
    df["lang_dist"] = df.apply(
        lambda r: safe_tag_distance(r[lang1_col], r[lang2_col]), axis=1
    )
    df["is_correct_int"] = df["is_correct"].astype(int)

    bins = [0, 10, 30, 60, 100, 135]
    labels = [
        "0-10 (near identical)",
        "10-30 (close)",
        "30-60 (moderate)",
        "60-100 (distant)",
        "100-135 (very distant)",
    ]
    df["dist_bin"] = pd.cut(df["lang_dist"], bins=bins, labels=labels, include_lowest=True)

    summary = (
        df.groupby("dist_bin")
        .agg(
            mean_score_gap=("score_gap", "mean"),
            median_score_gap=("score_gap", "median"),
            correctness_rate=("is_correct_int", "mean"),
            count=("score_gap", "size"),
        )
        .reset_index()
    )
    summary["dist_bin"] = summary["dist_bin"].astype(str)
    summary.to_csv(
        os.path.join(output_dir, f"{dataset_name}_gap_vs_lang_distance_binned.csv"),
        index=False,
    )

    df["outcome"] = df["is_correct"].map({True: "Correct", False: "Wrong"})
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="lang_dist", y="score_gap", hue="outcome", alpha=0.4)
    plt.xlabel("Language distance")
    plt.ylabel("Score gap (Rank 1 - Rank 2)")
    plt.title(f"{dataset_name} - Score gap vs language distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_gap_vs_lang_distance.png"), dpi=300)
    plt.close()


def plot_length_comparison(results_df, name, out_dir):
    """Plot detector performance across text-length bins."""
    import matplotlib.pyplot as plt

    def draw_metric(metric, ylabel, title, filename):
        plot_df = results_df.copy()
        bins = list(dict.fromkeys(plot_df["text_length_bin"].astype(str)))
        x_positions = {label: index for index, label in enumerate(bins)}
        support_by_bin = (
            plot_df.assign(text_length_bin=plot_df["text_length_bin"].astype(str))
            .groupby("text_length_bin")["support"]
            .max()
            .to_dict()
            if "support" in plot_df.columns
            else {}
        )
        x_labels = [
            f"{label}\nn={int(support_by_bin[label]):,}"
            if label in support_by_bin and support_by_bin[label] == support_by_bin[label]
            else label
            for label in bins
        ]
        models = ordered_models(plot_df["model"].dropna().astype(str))
        offsets = model_marker_offsets(models)
        palette = model_palette(models)

        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for zorder, model in enumerate(models, start=3):
            subset = plot_df[plot_df["model"] == model].copy()
            if subset.empty:
                continue
            subset["text_length_bin"] = subset["text_length_bin"].astype(str)
            subset["_x"] = subset["text_length_bin"].map(x_positions) + offsets.get(model, 0.0)
            subset = subset.sort_values("_x")
            ax.plot(
                subset["_x"],
                subset[metric],
                label=model,
                color=palette[model],
                linestyle=MODEL_LINESTYLES.get(model, "-"),
                linewidth=2.0,
                marker=MODEL_MARKERS.get(model, "o"),
                markersize=7,
                markerfacecolor="white",
                markeredgecolor=palette[model],
                markeredgewidth=1.6,
                zorder=zorder,
            )

        ax.set_xlabel("Text length bin (characters)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(x_labels, rotation=0, ha="center")
        style_model_axis(ax)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, filename), dpi=300)
        plt.close(fig)

    draw_metric(
        "f1_macro",
        "Macro F1",
        f"{name} - Performance by text length",
        "length_comparison.png",
    )
    draw_metric(
        "accuracy",
        "Accuracy",
        f"{name} - Accuracy by text length",
        "length_comparison_accuracy.png",
    )


def plot_cutoff_comparison(results_df, dataset_name, dataset_output):
    """Plot score-cutoff routing comparisons for F1 and accuracy."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(7, 5))
    sns.lineplot(data=results_df, x="cutoff", y="resiliparse_f1", label="Resiliparse", color="tab:blue", linestyle="-")
    sns.lineplot(data=results_df, x="cutoff", y="fasttext_f1", label="FastText", color="tab:red", linestyle="--")
    sns.lineplot(data=results_df, x="cutoff", y="system_ft_f1", label="RP+FastText", color="tab:purple", linestyle="-.")
    sns.lineplot(data=results_df, x="cutoff", y="system_ft_lang_f1", label="RP+FastText lang-aware", color="tab:orange", linestyle=":")
    plt.xlabel("Resiliparse cutoff")
    plt.ylabel("Macro F1")
    plt.title(f"{dataset_name} - Routing comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_output, "cutoff_comparison.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.lineplot(data=results_df, x="cutoff", y="resiliparse_acc", label="Resiliparse", color="tab:blue", linestyle="-")
    sns.lineplot(data=results_df, x="cutoff", y="fasttext_acc", label="FastText", color="tab:red", linestyle="--")
    sns.lineplot(data=results_df, x="cutoff", y="system_ft_acc", label="RP+FastText", color="tab:purple", linestyle="-.")
    sns.lineplot(data=results_df, x="cutoff", y="system_ft_lang_acc", label="RP+FastText lang-aware", color="tab:orange", linestyle=":")
    plt.xlabel("Resiliparse cutoff")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset_name} - Routing comparison (Accuracy)")
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_output, "cutoff_comparison_accuracy.png"), dpi=300)
    plt.close()


def compute_pca_coordinates(df, score_cols):
    """Fit two-dimensional PCA and return a DataFrame with pc1/pc2 columns."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    coords = pca.fit_transform(df[score_cols])
    plot_df = df.copy()
    plot_df["pc1"] = coords[:, 0]
    plot_df["pc2"] = coords[:, 1]
    return plot_df


def plot_pairwise_pca(plot_df, label_col, output_dir):
    """Write pairwise PCA scatter plots."""
    import matplotlib.pyplot as plt

    languages = sorted(plot_df[label_col].unique())
    cmap = plt.get_cmap("tab10")
    color_map = {lang: cmap(i) for i, lang in enumerate(languages)}

    for lang1, lang2 in itertools.combinations(languages, 2):
        plt.figure(figsize=(6, 5))
        subset1 = plot_df[plot_df[label_col] == lang1]
        subset2 = plot_df[plot_df[label_col] == lang2]
        plt.scatter(subset1["pc1"], subset1["pc2"], label=lang1, color=color_map[lang1], alpha=0.6, s=40)
        plt.scatter(subset2["pc1"], subset2["pc2"], label=lang2, color=color_map[lang2], alpha=0.6, s=40)
        plt.title(f"PCA: {lang1} vs {lang2}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pca_{lang1}_vs_{lang2}.png"), dpi=300)
        plt.close()


def plot_full_pca(plot_df, label_col, output_dir):
    """Write the full combined PCA scatter plot."""
    import matplotlib.pyplot as plt

    languages = sorted(plot_df[label_col].unique())
    cmap = plt.get_cmap("tab10")
    color_map = {lang: cmap(i) for i, lang in enumerate(languages)}

    plt.figure(figsize=(8, 6))
    for lang in languages:
        subset = plot_df[plot_df[label_col] == lang]
        plt.scatter(
            subset["pc1"],
            subset["pc2"],
            label=lang,
            color=color_map[lang],
            alpha=0.6,
            s=40,
            edgecolors="black",
            linewidths=0.4,
        )
    plt.title("PCA: Full Language Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_full_combined.png"), dpi=300)
    plt.close()

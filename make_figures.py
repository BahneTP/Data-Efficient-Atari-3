#!/usr/bin/env python3
"""Generate analysis figures for the Atari-3 selection.

The script uses the already generated files in ``analysis/``:

- Atari100k-Normalized.csv
- Atari3-candidates.csv
- Atari2-Validation-candidates.csv
- selection.json

It writes publication-style PNG and PDF figures to ``figures/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis"
OUTPUT_DIR = ROOT / "figures"
CATEGORIES_PATH = ROOT / "data" / "Atari100k-Game-Categories.csv"
TOP_N = 10
CATEGORY_ORDER = ["Combat", "Maze", "Sports", "Other", "Action"]

# Hard-code the methods you want to label in the scatter plots here.
# Use the method names exactly as they appear in the data tables.
#
# Examples:
# LABEL_METHODS = ["DER", "SWM (4 frames)", "OTRainbow (100k)", "EfficientZero-V2"]
# LABEL_METHODS = []
#
# Selected LABEL_METHODS are shown as colored points and annotated. If
# LABEL_METHODS is empty, the script labels the AUTO_LABEL_COUNT largest errors
# automatically. Set AUTO_LABEL_COUNT = 0 for no scatter labels.
LABEL_METHODS: list[str] = [
    "DER",
    "SimPLe",
    "BBF (RR8)",
    "SAC-BBF (RR2)",
    "DreamerV3",
    "EfficientZero",
    "EfficientZero-V2",
    "SPR",
]
AUTO_LABEL_COUNT = 5

# Set this to True once if you want to print all available method labels.
PRINT_AVAILABLE_METHODS = False


def inverse_transform(values: np.ndarray | pd.Series) -> np.ndarray:
    """Inverse of log10(1 + max(HNS, 0))."""

    return (10**values) - 1


def clean_label(value: str) -> str:
    """Make game-combination labels compact for figures."""

    return value.replace(", ", " + ")


def resolve_label_methods(
    requested_methods: list[str],
    available_methods: pd.Index,
) -> list[str]:
    """Resolve requested method labels against available method names."""

    available_set = set(available_methods)

    resolved: list[str] = []
    missing: list[str] = []
    for requested in requested_methods:
        if requested in available_set:
            method = requested
        else:
            method = None

        if method is None:
            missing.append(requested)
            continue
        if method not in resolved:
            resolved.append(method)

    if missing:
        print("Warning: requested labels not found:", ", ".join(missing))
    return resolved


def label_color_map(label_methods: list[str] | None) -> dict[str, tuple]:
    """Assign stable highlight colors to selected labelled methods."""

    if not label_methods:
        return {}

    color_cycle = plt.get_cmap("tab10").colors
    return {
        method: color_cycle[index % len(color_cycle)]
        for index, method in enumerate(label_methods)
    }


def highlight_methods(
    axis: plt.Axes,
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    label_methods: list[str] | None,
    colors: dict[str, tuple],
) -> None:
    """Draw selected methods as colored points on top of the base scatter."""

    if not label_methods:
        return

    for method in label_methods:
        if method not in data.index:
            continue
        row = data.loc[method]
        axis.scatter(
            row[x_column],
            row[y_column],
            s=80,
            color=colors[method],
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )


def annotate_methods(
    axis: plt.Axes,
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    ranking_column: str,
    label_methods: list[str] | None,
    auto_label_count: int,
    colors: dict[str, tuple] | None = None,
) -> None:
    """Annotate selected methods, or auto-label the largest errors."""

    if label_methods is None:
        rows = data.nlargest(auto_label_count, ranking_column)
    else:
        rows = data.loc[[method for method in label_methods if method in data.index]]

    for method, row in rows.iterrows():
        color = "black"
        if colors is not None and method in colors:
            color = colors[method]
        axis.annotate(
            method,
            (row[x_column], row[y_column]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color=color,
            zorder=5,
        )


def load_selection(analysis_dir: Path) -> dict:
    with (analysis_dir / "selection.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def load_normalized_data(analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_data = pd.read_csv(analysis_dir / "Atari100k-Normalized.csv")
    method_data = (
        long_data.drop_duplicates("Method")
        .set_index("Method")[["MedianHNS26", "LogMedianHNS26"]]
        .sort_index()
    )
    log_hns = long_data.pivot(
        index="Method",
        columns="Game",
        values="LogHNS",
    ).loc[method_data.index]
    return method_data, log_hns


def load_game_categories(path: Path) -> pd.Series:
    categories = pd.read_csv(path).set_index("Game")["Category"]
    return categories


def sort_games_by_category(games: pd.Index, categories: pd.Series) -> list[str]:
    category_rank = {
        category: rank
        for rank, category in enumerate(CATEGORY_ORDER)
    }

    missing_games = sorted(set(games) - set(categories.index))
    if missing_games:
        raise ValueError(f"Missing game categories: {missing_games}")

    unknown_categories = sorted(set(categories.loc[list(games)]) - set(CATEGORY_ORDER))
    if unknown_categories:
        raise ValueError(f"Unknown game categories: {unknown_categories}")

    return sorted(
        games,
        key=lambda game: (category_rank[categories.loc[game]], game),
    )


def predict_subset(
    selection: dict,
    subset_key: str,
    method_data: pd.DataFrame,
    log_hns: pd.DataFrame,
) -> pd.DataFrame:
    subset = selection[subset_key]
    games = subset["games"]
    coefficients = np.asarray(subset["coefficients"], dtype=float)

    predicted_log = np.asarray(log_hns.loc[:, games]) @ coefficients
    result = method_data.copy()
    result["PredictedLogMedianHNS26"] = predicted_log
    result["PredictedMedianHNS26"] = inverse_transform(predicted_log)
    result["AbsLogError"] = (
        result["PredictedLogMedianHNS26"] - result["LogMedianHNS26"]
    ).abs()
    return result


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_predicted_vs_true(
    selection: dict,
    method_data: pd.DataFrame,
    log_hns: pd.DataFrame,
    output_dir: Path,
    label_methods: list[str] | None,
    auto_label_count: int,
) -> None:
    subsets = [
        ("atari3_test", "Atari-3 test"),
        ("atari2_validation", "Atari-2 validation"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)

    all_predictions = [
        predict_subset(selection, key, method_data, log_hns)
        for key, _ in subsets
    ]
    colors = label_color_map(label_methods)
    min_value = min(
        min(prediction["MedianHNS26"].min(), prediction["PredictedMedianHNS26"].min())
        for prediction in all_predictions
    )
    max_value = max(
        max(prediction["MedianHNS26"].max(), prediction["PredictedMedianHNS26"].max())
        for prediction in all_predictions
    )
    lower = max(1e-2, min_value * 0.75)
    upper = max_value * 1.25

    for axis, (key, title), prediction in zip(axes, subsets, all_predictions):
        subset = selection[key]
        axis.scatter(
            prediction["MedianHNS26"],
            prediction["PredictedMedianHNS26"],
            s=34,
            color="#B8B8B8" if label_methods else "#4C78A8",
            alpha=0.55 if label_methods else 0.82,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        highlight_methods(
            axis=axis,
            data=prediction,
            x_column="MedianHNS26",
            y_column="PredictedMedianHNS26",
            label_methods=label_methods,
            colors=colors,
        )
        axis.plot([lower, upper], [lower, upper], color="black", linewidth=1)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.grid(True, which="both", alpha=0.22)
        axis.set_title(
            f"{title}\n"
            f"CV RMSE={subset['cv_rmse']:.3f}, "
            f"CV R²={subset['cv_r2']:.3f}"
        )
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.yaxis.set_major_formatter(ScalarFormatter())

        annotate_methods(
            axis=axis,
            data=prediction,
            x_column="MedianHNS26",
            y_column="PredictedMedianHNS26",
            ranking_column="AbsLogError",
            label_methods=label_methods,
            auto_label_count=auto_label_count,
            colors=colors,
        )

    axes[0].set_ylabel("Predicted 26-game median HNS")
    for axis in axes:
        axis.set_xlabel("True 26-game median HNS")

    fig.suptitle("Predicted vs. true Atari-100k performance", y=1.03)
    save_figure(fig, output_dir, "predicted_vs_true")


def plot_top_candidates(analysis_dir: Path, output_dir: Path, top_n: int) -> None:
    candidate_files = [
        ("Atari3-candidates.csv", "Atari-3 candidates"),
        ("Atari2-Validation-candidates.csv", "Atari-2 validation candidates"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for axis, (filename, title) in zip(axes, candidate_files):
        candidates = pd.read_csv(analysis_dir / filename).head(top_n)
        labels = [clean_label(value) for value in candidates["Games"]]
        y_positions = np.arange(len(candidates))

        axis.barh(y_positions, candidates["CV RMSE"], color="#4C78A8")
        axis.set_yticks(y_positions)
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        axis.set_xlabel("CV RMSE")
        axis.set_title(title)
        axis.grid(True, axis="x", alpha=0.25)

        best = float(candidates["CV RMSE"].iloc[0])
        axis.axvline(best, color="black", linewidth=1, alpha=0.75)

    fig.suptitle(f"Top-{top_n} candidate subsets by cross-validated RMSE", y=1.03)
    fig.tight_layout()
    save_figure(fig, output_dir, "top_candidates")


def plot_rank_and_correlation(
    selection: dict,
    method_data: pd.DataFrame,
    log_hns: pd.DataFrame,
    output_dir: Path,
    label_methods: list[str] | None,
    auto_label_count: int,
) -> None:
    prediction = predict_subset(selection, "atari3_test", method_data, log_hns)
    prediction["TrueRank"] = prediction["MedianHNS26"].rank(
        ascending=False,
        method="average",
    )
    prediction["PredictedRank"] = prediction["PredictedMedianHNS26"].rank(
        ascending=False,
        method="average",
    )
    spearman = prediction[["TrueRank", "PredictedRank"]].corr(
        method="spearman",
    ).iloc[0, 1]
    colors = label_color_map(label_methods)

    categories = load_game_categories(CATEGORIES_PATH)
    sorted_games = sort_games_by_category(log_hns.columns, categories)
    corr = log_hns.loc[:, sorted_games].corr(method="pearson")

    fig = plt.figure(figsize=(14, 6))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35])
    rank_axis = fig.add_subplot(grid[0, 0])
    heatmap_axis = fig.add_subplot(grid[0, 1])

    rank_axis.scatter(
        prediction["TrueRank"],
        prediction["PredictedRank"],
        s=34,
        color="#B8B8B8" if label_methods else "#4C78A8",
        alpha=0.55 if label_methods else 0.82,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
    )
    highlight_methods(
        axis=rank_axis,
        data=prediction,
        x_column="TrueRank",
        y_column="PredictedRank",
        label_methods=label_methods,
        colors=colors,
    )
    rank_axis.plot([1, len(prediction)], [1, len(prediction)], color="black", linewidth=1)
    rank_axis.set_xlim(0.5, len(prediction) + 0.5)
    rank_axis.set_ylim(len(prediction) + 0.5, 0.5)
    rank_axis.set_xlabel("Rank by full 26-game median HNS")
    rank_axis.set_ylabel("Rank predicted by Atari-3")
    rank_axis.set_title(f"Method ranking agreement\nSpearman ρ={spearman:.3f}")
    rank_axis.grid(True, alpha=0.25)

    prediction = prediction.assign(
        RankError=(prediction["PredictedRank"] - prediction["TrueRank"]).abs()
    )
    annotate_methods(
        axis=rank_axis,
        data=prediction,
        x_column="TrueRank",
        y_column="PredictedRank",
        ranking_column="RankError",
        label_methods=label_methods,
        auto_label_count=auto_label_count,
        colors=colors,
    )

    image = heatmap_axis.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    heatmap_axis.set_title("Game correlation across methods\nPearson r of log-HNS")
    heatmap_axis.set_xticks(np.arange(len(corr.columns)))
    heatmap_axis.set_xticklabels(corr.columns, rotation=90)
    heatmap_axis.set_yticks(np.arange(len(corr.index)))
    heatmap_axis.set_yticklabels(corr.index)
    heatmap_axis.tick_params(axis="both", labelsize=7)

    category_boundaries: list[int] = []
    previous_category = categories.loc[corr.columns[0]]
    for index, game in enumerate(corr.columns[1:], start=1):
        category = categories.loc[game]
        if category != previous_category:
            category_boundaries.append(index)
            previous_category = category

    for boundary in category_boundaries:
        heatmap_axis.axhline(boundary - 0.5, color="black", linewidth=0.8)
        heatmap_axis.axvline(boundary - 0.5, color="black", linewidth=0.8)

    category_spans: list[tuple[str, int, int]] = []
    start = 0
    previous_category = categories.loc[corr.columns[0]]
    for index, game in enumerate(corr.columns[1:], start=1):
        category = categories.loc[game]
        if category != previous_category:
            category_spans.append((previous_category, start, index - 1))
            start = index
            previous_category = category
    category_spans.append((previous_category, start, len(corr.columns) - 1))

    for category, start, end in category_spans:
        center = (start + end) / 2
        heatmap_axis.text(
            center,
            -1.8,
            category,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            clip_on=False,
        )

    colorbar = fig.colorbar(image, ax=heatmap_axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Correlation")

    fig.tight_layout()
    save_figure(fig, output_dir, "rank_and_correlation")


def main() -> None:
    if AUTO_LABEL_COUNT < 0:
        raise ValueError("AUTO_LABEL_COUNT must be non-negative")

    selection = load_selection(ANALYSIS_DIR)
    method_data, log_hns = load_normalized_data(ANALYSIS_DIR)

    if PRINT_AVAILABLE_METHODS:
        for method in method_data.index:
            print(method)
        return

    label_methods = None
    if LABEL_METHODS:
        label_methods = resolve_label_methods(LABEL_METHODS, method_data.index)

    plot_predicted_vs_true(
        selection,
        method_data,
        log_hns,
        OUTPUT_DIR,
        label_methods,
        AUTO_LABEL_COUNT,
    )
    plot_top_candidates(ANALYSIS_DIR, OUTPUT_DIR, TOP_N)
    plot_rank_and_correlation(
        selection,
        method_data,
        log_hns,
        OUTPUT_DIR,
        label_methods,
        AUTO_LABEL_COUNT,
    )

    print(f"Wrote figures to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate analysis figures for the Atari-3 selection.

The script uses the already generated files in ``analysis/``:

- Atari100k-Normalized.csv
- Atari3-candidates.csv
- Atari1-Validation-candidates.csv
- selection.json

It writes publication-style PNG and PDF figures to ``figures/``.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LogLocator, MultipleLocator, NullLocator, ScalarFormatter


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis"
OUTPUT_DIR = ROOT / "figures"
FIGURE_FONT = Path(
    "/home/bthiehl/torchrl-hydra-template/plots/fonts/NewCM08-Regular.otf"
)
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
    "DreamerV3",
    "SPR",
    "SR-SPR (RR16)",
]
AUTO_LABEL_COUNT = 5

# Display text override for annotated points, keyed by the method name as it
# appears in LABEL_METHODS / the data. Methods not listed here are annotated
# with their raw method name.
LABEL_DISPLAY_NAMES: dict[str, str] = {
    "SR-SPR (RR16)": "SR-SPR",
    "BBF (RR8)": "BBF",
}

# Set this to True once if you want to print all available method labels.
PRINT_AVAILABLE_METHODS = False


def inverse_transform(values: np.ndarray | pd.Series) -> np.ndarray:
    """Inverse of log10(1 + max(HNS, 0))."""

    return (10**values) - 1


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


METHOD_COLORS: dict[str, str] = {
    "DER": "#FFCCCC",
    "SPR": "#FFCC99",
    "SR-SPR (RR16)": "#CCE5FF",
    "BBF (RR8)": "black",
    "SimPLe": "#FFFF88",
    "DreamerV3": "#CDEB8B",
}


def label_color_map(label_methods: list[str] | None) -> dict[str, str]:
    """Assign stable highlight colors to selected labelled methods."""

    if not label_methods:
        return {}

    return {method: METHOD_COLORS[method] for method in label_methods}


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
        if method == "SR-SPR (RR16)":
            xytext, ha = (4, 4), "left"
        else:
            xytext, ha = (-4, 4), "right"
        axis.annotate(
            LABEL_DISPLAY_NAMES.get(method, method),
            (row[x_column], row[y_column]),
            xytext=xytext,
            textcoords="offset points",
            fontsize=12,
            color="black",
            ha=ha,
            va="bottom",
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
    for text in fig.findobj(match=matplotlib.text.Text):
        size = text.get_fontsize()
        font = FontProperties(fname=FIGURE_FONT, size=size)
        text.set_fontproperties(font)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_predicted_vs_true(
    axis: plt.Axes,
    prediction: pd.DataFrame,
    lower: float,
    upper: float,
    label_methods: list[str] | None,
    colors: dict[str, str],
    auto_label_count: int,
    title: str | None = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_y_ticklabels: bool = True,
) -> None:
    axis.scatter(
        prediction["MedianHNS26"],
        prediction["PredictedMedianHNS26"],
        s=34,
        color="#D6D6D6" if label_methods else "#4C78A8",
        alpha=0.45 if label_methods else 0.82,
        edgecolor="black",
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
    axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_minor_locator(NullLocator())
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.yaxis.set_major_formatter(ScalarFormatter())
    axis.tick_params(axis="both", which="major", labelsize=12)
    axis.tick_params(axis="both", which="minor", labelsize=12)

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

    if show_ylabel:
        axis.set_ylabel("Predicted median HNS", fontsize=16)
    if not show_y_ticklabels:
        axis.tick_params(axis="y", labelleft=False)
    if show_xlabel:
        axis.set_xlabel("True median HNS", fontsize=16)
    if title is not None:
        axis.set_title(title, fontsize=16)


def plot_predicted_vs_true(
    selection: dict,
    method_data: pd.DataFrame,
    log_hns: pd.DataFrame,
    output_dir: Path,
    label_methods: list[str] | None,
    auto_label_count: int,
) -> None:
    subsets = [
        ("atari3_test", "predicted_vs_true_atari3", "Test Environments"),
        ("atari1_validation", "predicted_vs_true_atari1", "Validation Environment"),
    ]

    all_predictions = [
        predict_subset(selection, key, method_data, log_hns)
        for key, _, _ in subsets
    ]
    for prediction in all_predictions:
        prediction["MedianHNS26"] /= 100
        prediction["PredictedMedianHNS26"] /= 100
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

    for (key, stem, title), prediction in zip(subsets, all_predictions):
        fig, axis = plt.subplots(figsize=(5.5, 4.3))
        draw_predicted_vs_true(
            axis, prediction, lower, upper, label_methods, colors, auto_label_count
        )
        fig.tight_layout()
        save_figure(fig, output_dir, stem)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for index, (axis, (key, stem, title), prediction) in enumerate(
        zip(axes, subsets, all_predictions)
    ):
        draw_predicted_vs_true(
            axis,
            prediction,
            lower,
            upper,
            label_methods,
            colors,
            auto_label_count,
            title,
            show_xlabel=False,
            show_ylabel=index == 0,
            show_y_ticklabels=index == 0,
        )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    fig.text(0.5, 0.02, "True median HNS", ha="center", fontsize=16)
    save_figure(fig, output_dir, "predicted_vs_true_combined")


def plot_correlation(
    log_hns: pd.DataFrame,
    output_dir: Path,
) -> None:
    corr = log_hns.corr(method="pearson")

    fig, heatmap_axis = plt.subplots(figsize=(9, 6))

    image = heatmap_axis.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    heatmap_axis.set_xticks(np.arange(len(corr.columns)))
    heatmap_axis.set_xticklabels(corr.columns, rotation=90)
    heatmap_axis.set_yticks(np.arange(len(corr.index)))
    heatmap_axis.set_yticklabels(corr.index)
    heatmap_axis.tick_params(axis="both", labelsize=7)

    colorbar = fig.colorbar(image, ax=heatmap_axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Correlation", fontsize=16)

    fig.tight_layout()
    save_figure(fig, output_dir, "correlation")


def plot_hns_over_time(analysis_dir: Path, output_dir: Path) -> None:
    """Scatter of release date vs. mean HNS for a fixed set of methods.

    Rainbow is not part of the fitted Atari100k data (it predates the
    100k-step benchmark), so its point uses the mean HNS=0.222 reported in
    the SimPLe paper, rescaled to the same 0-100+ percentage scale as the
    HNS values for the other methods.
    """

    long_data = pd.read_csv(analysis_dir / "Atari100k-Normalized.csv")
    mean_hns = long_data.groupby("Method")["HNS"].mean()

    records = [
        ("Rainbow", "2017-10", 0.222 * 100 / 100, "white"),
        ("DER", "2019-06", float(mean_hns.loc["DER"]) / 100, "#FFCCCC"),
        ("SPR", "2021-05", float(mean_hns.loc["SPR"]) / 100, "#FFCC99"),
        ("SR-SPR", "2023-02", float(mean_hns.loc["SR-SPR (RR16)"]) / 100, "#CCE5FF"),
        ("BBF", "2023-11", float(mean_hns.loc["BBF (RR8)"]) / 100, "black"),
    ]

    dates = [datetime.datetime.strptime(date, "%Y-%m") for _, date, _, _ in records]
    values = [value for _, _, value, _ in records]
    point_colors = [color for _, _, _, color in records]

    fig, axis = plt.subplots(figsize=(7, 3.5))
    axis.scatter(
        dates,
        values,
        s=60,
        color=point_colors,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )

    for (label, _, _, _), date, value in zip(records, dates, values):
        if label == "BBF":
            xytext, ha, va = (0, 10), "center", "bottom"
        else:
            xytext, ha, va = (4, 4), "left", "bottom"
        axis.annotate(
            label,
            (date, value),
            xytext=xytext,
            textcoords="offset points",
            fontsize=16,
            ha=ha,
            va=va,
            zorder=5,
        )

    axis.grid(True, alpha=0.22)
    axis.yaxis.set_major_locator(MultipleLocator(0.2))
    axis.tick_params(axis="both", which="major", labelsize=12)
    axis.set_xlabel("Release date", fontsize=16)
    axis.set_ylabel("Mean HNS", fontsize=16)

    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, output_dir, "hns_over_time")


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
    plot_correlation(log_hns, OUTPUT_DIR)
    plot_hns_over_time(ANALYSIS_DIR, OUTPUT_DIR)

    print(f"Wrote figures to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

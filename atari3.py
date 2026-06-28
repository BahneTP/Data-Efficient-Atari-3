#!/usr/bin/env python3
"""Select representative Atari-100k test and validation games.

This is a small, self-contained adaptation of the subset-search code from
Atari-5 (https://github.com/maitchison/Atari-5).

The unchanged core methodology is:

1. Human-normalize every raw game score.
2. Apply log10(1 + max(HNS, 0)).
3. Use linear regression without an intercept to predict each method's
   full-benchmark median HNS.
4. Rank candidate subsets by 10-fold cross-validated mean-squared error.

The necessary Atari-100k adaptations are:

- use the 26 Atari-100k games and the validated Atari100k-Results.csv;
- search directly over all three-game subsets because Atari-3 is the primary
  test set;
- after selecting Atari-3, search for a disjoint two-game validation set.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "data" / "Atari100k-Results.csv"
DEFAULT_HUMAN = ROOT / "data" / "Atari-Human.csv"
DEFAULT_OUTPUT = ROOT / "analysis"

N_SPLITS = 10
RANDOM_STATE = 1982
USE_INTERCEPT = False

# Atari-5 first keeps the best 57 candidates by in-sample MSE and then uses
# cross-validation to order those candidates. We retain that behavior.
TOP_K = 57


def clean_game_name(name: str) -> str:
    """Normalize a game name to lowercase ASCII letters without spaces."""

    return "".join(character for character in str(name).lower() if character.isalpha())


def transform(values: np.ndarray | pd.DataFrame | pd.Series) -> np.ndarray:
    """The biased log transform used by Atari-5."""

    return np.log10(1 + np.clip(values, 0, float("inf")))


def inverse_transform(values: np.ndarray | float) -> np.ndarray:
    return (10**values) - 1


@dataclass
class SubsetResult:
    games: tuple[str, ...]
    n_methods: int
    train_rmse: float
    cv_rmse: float | None
    cv_mae: float | None
    cv_r2: float | None
    approximate_relative_error_pct: float | None
    coefficients: list[float]


def load_data(
    results_path: Path,
    human_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load raw scores and return HNS, log-HNS, target, and long-form data."""

    raw = pd.read_csv(results_path)
    if raw.columns[0] != "Game":
        raise ValueError(f"{results_path}: first column must be 'Game'")
    if raw["Game"].duplicated().any():
        duplicates = raw.loc[raw["Game"].duplicated(), "Game"].tolist()
        raise ValueError(f"{results_path}: duplicate games: {duplicates}")

    raw["Game"] = raw["Game"].map(clean_game_name)
    raw = raw.set_index("Game")
    raw = raw.apply(pd.to_numeric, errors="raise")
    if raw.shape[0] != 26:
        raise ValueError(f"{results_path}: expected 26 games, found {raw.shape[0]}")
    if raw.isna().any().any():
        raise ValueError(f"{results_path}: missing score values")

    reference = pd.read_csv(human_path)
    reference["Game"] = reference["Game"].map(clean_game_name)
    reference = reference.set_index("Game")[["Random", "Human"]]
    reference = reference.apply(pd.to_numeric, errors="raise")

    missing_reference = sorted(set(raw.index) - set(reference.index))
    if missing_reference:
        raise ValueError(f"Missing Random/Human reference scores: {missing_reference}")
    reference = reference.loc[raw.index]
    if (reference["Human"] == reference["Random"]).any():
        games = reference.index[reference["Human"] == reference["Random"]].tolist()
        raise ValueError(f"Human and Random scores are equal for: {games}")

    # Methods are rows and games are columns from this point onward.
    scores = raw.T
    hns = 100 * (scores - reference["Random"]) / (
        reference["Human"] - reference["Random"]
    )
    log_hns = pd.DataFrame(
        transform(hns),
        index=hns.index,
        columns=hns.columns,
    )

    median_hns = hns.median(axis=1)
    log_target = pd.Series(
        transform(median_hns),
        index=median_hns.index,
        name="LogMedianHNS26",
    )

    long_rows: list[dict[str, object]] = []
    for method in scores.index:
        for game in scores.columns:
            long_rows.append(
                {
                    "Method": method,
                    "Game": game,
                    "Score": scores.loc[method, game],
                    "Random": reference.loc[game, "Random"],
                    "Human": reference.loc[game, "Human"],
                    "HNS": hns.loc[method, game],
                    "LogHNS": log_hns.loc[method, game],
                    "MedianHNS26": median_hns.loc[method],
                    "LogMedianHNS26": log_target.loc[method],
                }
            )
    long_data = pd.DataFrame(long_rows)

    return hns, log_hns, log_target, long_data


def fit_subset(
    games: tuple[str, ...],
    log_hns: pd.DataFrame,
    log_target: pd.Series,
) -> tuple[LinearRegression, np.ndarray, np.ndarray]:
    """Fit the same linear model used by Atari-5."""

    x = np.asarray(log_hns.loc[:, list(games)])
    y = np.asarray(log_target)
    model = LinearRegression(fit_intercept=USE_INTERCEPT)
    model.fit(x, y)
    return model, x, y


def evaluate_training_error(
    games: tuple[str, ...],
    log_hns: pd.DataFrame,
    log_target: pd.Series,
) -> SubsetResult:
    model, x, y = fit_subset(games, log_hns, log_target)
    errors = y - model.predict(x)
    return SubsetResult(
        games=games,
        n_methods=len(y),
        train_rmse=float(np.sqrt(np.mean(errors**2))),
        cv_rmse=None,
        cv_mae=None,
        cv_r2=None,
        approximate_relative_error_pct=None,
        coefficients=[float(value) for value in model.coef_],
    )


def add_cross_validation(
    result: SubsetResult,
    log_hns: pd.DataFrame,
    log_target: pd.Series,
) -> SubsetResult:
    model, x, y = fit_subset(result.games, log_hns, log_target)
    folds = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scores = cross_validate(
        model,
        x,
        y,
        cv=folds,
        scoring={
            "mse": "neg_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        },
    )
    cv_mse = float(-np.mean(scores["test_mse"]))
    cv_mae = float(-np.mean(scores["test_mae"]))
    target_variance = float(np.var(y, ddof=0))

    result.cv_rmse = float(np.sqrt(cv_mse))
    result.cv_mae = cv_mae
    result.cv_r2 = float(1 - cv_mse / target_variance)
    # This is the same approximation reported by the Atari-5 code.
    result.approximate_relative_error_pct = float(cv_mae * np.log(10) * 100)
    return result


def search_subsets(
    games: list[str],
    subset_size: int,
    log_hns: pd.DataFrame,
    log_target: pd.Series,
    top_k: int,
) -> tuple[SubsetResult, list[SubsetResult]]:
    """Search all subsets, then cross-validate the best training candidates."""

    combinations = list(itertools.combinations(games, subset_size))
    candidates = [
        evaluate_training_error(combination, log_hns, log_target)
        for combination in combinations
    ]
    candidates.sort(key=lambda result: result.train_rmse)

    finalists = candidates[: min(top_k, len(candidates))]
    finalists = [
        add_cross_validation(result, log_hns, log_target)
        for result in finalists
    ]
    finalists.sort(key=lambda result: result.cv_rmse)
    return finalists[0], finalists


def write_ranked_results(path: Path, results: list[SubsetResult]) -> None:
    fields = [
        "Rank",
        "Games",
        "Methods",
        "Train RMSE",
        "CV RMSE",
        "CV MAE",
        "CV R2",
        "Approx Relative Error (%)",
        "Coefficients",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            writer.writerow(
                {
                    "Rank": rank,
                    "Games": ", ".join(result.games),
                    "Methods": result.n_methods,
                    "Train RMSE": result.train_rmse,
                    "CV RMSE": result.cv_rmse,
                    "CV MAE": result.cv_mae,
                    "CV R2": result.cv_r2,
                    "Approx Relative Error (%)": result.approximate_relative_error_pct,
                    "Coefficients": ", ".join(
                        f"{coefficient:.10g}"
                        for coefficient in result.coefficients
                    ),
                }
            )


def result_for_json(result: SubsetResult) -> dict[str, object]:
    data = asdict(result)
    data["games"] = list(result.games)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Number of lowest-training-error candidates to cross-validate.",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hns, log_hns, log_target, long_data = load_data(
        args.results.resolve(),
        args.human.resolve(),
    )

    games = list(log_hns.columns)
    atari3, atari3_finalists = search_subsets(
        games,
        subset_size=3,
        log_hns=log_hns,
        log_target=log_target,
        top_k=args.top_k,
    )

    validation_candidates = [game for game in games if game not in atari3.games]
    atari2_val, validation_finalists = search_subsets(
        validation_candidates,
        subset_size=2,
        log_hns=log_hns,
        log_target=log_target,
        top_k=args.top_k,
    )

    normalized_path = args.output_dir / "Atari100k-Normalized.csv"
    atari3_path = args.output_dir / "Atari3-candidates.csv"
    validation_path = args.output_dir / "Atari2-Validation-candidates.csv"
    summary_path = args.output_dir / "selection.json"

    long_data.to_csv(normalized_path, index=False)
    write_ranked_results(atari3_path, atari3_finalists)
    write_ranked_results(validation_path, validation_finalists)

    summary = {
        "methodology": {
            "hns": "100 * (score - random) / (human - random)",
            "transform": "log10(1 + max(HNS, 0))",
            "target": "log10(1 + max(median HNS across all 26 games, 0))",
            "regression": "LinearRegression(fit_intercept=False)",
            "cross_validation": (
                f"{N_SPLITS}-fold KFold(shuffle=True, random_state={RANDOM_STATE})"
            ),
            "candidate_prefilter": (
                f"lowest {args.top_k} in-sample RMSE candidates, as in Atari-5"
            ),
        },
        "input": {
            "results": str(args.results.resolve()),
            "human_reference": str(args.human.resolve()),
            "methods": int(hns.shape[0]),
            "games": int(hns.shape[1]),
        },
        "atari3_test": result_for_json(atari3),
        "atari2_validation": result_for_json(atari2_val),
        "overlap": sorted(set(atari3.games) & set(atari2_val.games)),
        "outputs": {
            "normalized_data": str(normalized_path.resolve()),
            "atari3_candidates": str(atari3_path.resolve()),
            "validation_candidates": str(validation_path.resolve()),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Loaded {hns.shape[0]} methods across {hns.shape[1]} games.")
    print("Atari-3 test:", ", ".join(atari3.games))
    print(
        f"  CV RMSE={atari3.cv_rmse:.6f}, "
        f"approx. relative error={atari3.approximate_relative_error_pct:.2f}%"
    )
    print("Atari-2 validation:", ", ".join(atari2_val.games))
    print(
        f"  CV RMSE={atari2_val.cv_rmse:.6f}, "
        f"approx. relative error={atari2_val.approximate_relative_error_pct:.2f}%"
    )
    print("Overlap:", sorted(set(atari3.games) & set(atari2_val.games)))
    print(f"Wrote analysis files to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

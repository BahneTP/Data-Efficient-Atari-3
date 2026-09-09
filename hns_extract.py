#!/usr/bin/env python3
"""Extract MedianHNS26 summaries and a wide HNS26 table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_NORMALIZED = ROOT / "analysis" / "Atari100k-Normalized.csv"
DEFAULT_METADATA = ROOT / "data" / "Atari100k-Metadata.csv"
DEFAULT_CITATIONS = ROOT / "data" / "Atari100k-Citations.csv"
DEFAULT_OUTPUT = ROOT / "analysis" / "Atari100k-MedianHNS26.csv"
DEFAULT_TABLE_OUTPUT = ROOT / "analysis" / "Atari100k-HNS26-Table.csv"
DEFAULT_TABLE_PART1_OUTPUT = ROOT / "analysis" / "Atari100k-HNS26-Table-Part1.csv"
DEFAULT_TABLE_PART2_OUTPUT = ROOT / "analysis" / "Atari100k-HNS26-Table-Part2.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table-output", type=Path, default=DEFAULT_TABLE_OUTPUT)
    parser.add_argument("--table-part1-output", type=Path, default=DEFAULT_TABLE_PART1_OUTPUT)
    parser.add_argument("--table-part2-output", type=Path, default=DEFAULT_TABLE_PART2_OUTPUT)
    args = parser.parse_args()

    normalized = pd.read_csv(args.normalized)
    required_normalized_columns = {"Method", "Game", "HNS", "MedianHNS26"}
    missing_normalized_columns = required_normalized_columns - set(normalized.columns)
    if missing_normalized_columns:
        raise ValueError(
            f"{args.normalized}: missing columns {sorted(missing_normalized_columns)}"
        )
    if normalized[["Method", "Game"]].duplicated().any():
        duplicates = normalized.loc[
            normalized[["Method", "Game"]].duplicated(),
            ["Method", "Game"],
        ].to_dict("records")
        raise ValueError(f"{args.normalized}: duplicate method/game rows {duplicates}")

    # MedianHNS26 is repeated once per game in the long-form normalized file.
    # Verify that the repeated value is constant for each method before taking it.
    consistency = normalized.groupby("Method")["MedianHNS26"].nunique(dropna=False)
    inconsistent_methods = consistency[consistency != 1].index.tolist()
    if inconsistent_methods:
        raise ValueError(
            "MedianHNS26 is not constant for methods: "
            + ", ".join(inconsistent_methods)
        )

    method_summary = (
        normalized[["Method", "MedianHNS26"]]
        .drop_duplicates("Method", keep="first")
        .reset_index(drop=True)
    )

    metadata = pd.read_csv(args.metadata)
    required_metadata_columns = {"Abbreviation", "Paper Title", "Year"}
    missing_metadata_columns = required_metadata_columns - set(metadata.columns)
    if missing_metadata_columns:
        raise ValueError(
            f"{args.metadata}: missing columns {sorted(missing_metadata_columns)}"
        )
    if metadata["Abbreviation"].duplicated().any():
        duplicates = metadata.loc[
            metadata["Abbreviation"].duplicated(),
            "Abbreviation",
        ].tolist()
        raise ValueError(f"{args.metadata}: duplicate abbreviations {duplicates}")

    merged = method_summary.merge(
        metadata[["Abbreviation", "Paper Title", "Year"]],
        left_on="Method",
        right_on="Abbreviation",
        how="left",
        validate="one_to_one",
    )
    missing_metadata = merged.loc[merged["Paper Title"].isna(), "Method"].tolist()
    if missing_metadata:
        raise ValueError(
            "No metadata match for methods: " + ", ".join(missing_metadata)
        )

    citations = pd.read_csv(args.citations)
    required_citation_columns = {"Abbreviation", "Citation"}
    missing_citation_columns = required_citation_columns - set(citations.columns)
    if missing_citation_columns:
        raise ValueError(
            f"{args.citations}: missing columns {sorted(missing_citation_columns)}"
        )
    if citations["Abbreviation"].duplicated().any():
        duplicates = citations.loc[
            citations["Abbreviation"].duplicated(),
            "Abbreviation",
        ].tolist()
        raise ValueError(f"{args.citations}: duplicate abbreviations {duplicates}")

    merged = merged.merge(
        citations[["Abbreviation", "Citation"]],
        on="Abbreviation",
        how="left",
        validate="one_to_one",
    )
    missing_citations = merged.loc[merged["Citation"].isna(), "Method"].tolist()
    if missing_citations:
        raise ValueError(
            "No citation match for methods: " + ", ".join(missing_citations)
        )

    hns_table = normalized.pivot(index="Method", columns="Game", values="HNS")
    games = normalized["Game"].drop_duplicates().tolist()
    missing_games = hns_table.loc[merged["Method"], games].isna()
    if missing_games.any().any():
        missing = [
            f"{method}/{game}"
            for method, row in missing_games.iterrows()
            for game, is_missing in row.items()
            if is_missing
        ]
        raise ValueError("Missing HNS values for: " + ", ".join(missing))

    hns_table = hns_table.loc[merged["Method"], games].reset_index(drop=True)
    hns_table.insert(0, "Method", merged["Method"] + " " + merged["Citation"])
    hns_table["MedianHNS26"] = merged["MedianHNS26"].to_numpy()
    split_at = len(games) // 2
    hns_table_part1 = hns_table[["Method", *games[:split_at], "MedianHNS26"]]
    hns_table_part2 = hns_table[["Method", *games[split_at:], "MedianHNS26"]]

    merged = merged[["Method", "MedianHNS26", "Paper Title", "Year", "Citation"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    hns_table.to_csv(args.table_output, index=False)
    args.table_part1_output.parent.mkdir(parents=True, exist_ok=True)
    hns_table_part1.to_csv(args.table_part1_output, index=False)
    args.table_part2_output.parent.mkdir(parents=True, exist_ok=True)
    hns_table_part2.to_csv(args.table_part2_output, index=False)

    print(f"Wrote {len(merged)} rows to {args.output}")
    print(f"Wrote {len(hns_table)} rows to {args.table_output}")
    print(f"Wrote {len(hns_table_part1)} rows to {args.table_part1_output}")
    print(f"Wrote {len(hns_table_part2)} rows to {args.table_part2_output}")


if __name__ == "__main__":
    main()

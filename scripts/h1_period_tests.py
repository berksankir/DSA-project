"""
H1: Pre-war vs War period difference tests

Dataset columns (as in your panel):
- year, week, dyad
- military_next_week, protest_next_week (counts)

This script derives binary outcomes:
- has_military_next_week = 1[military_next_week > 0]
- has_protest_next_week  = 1[protest_next_week > 0]

Tests:
- Chi-square test of independence (2x2): period x binary outcome
Optional:
- Mann–Whitney U for count outcomes (kept because it's often useful to report)

Outputs:
- results/h1_tests_summary.csv
- results/h1_contingency_tables.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


DEFAULT_PANEL_PATH = Path("data_processed/panel_diplomacy_gdelt_week_2019_2024.csv")


def make_period_from_year(df: pd.DataFrame) -> pd.Series:
    if "year" not in df.columns:
        raise KeyError("Missing column: year")
    year = pd.to_numeric(df["year"], errors="coerce")
    period = np.where(year >= 2022, "war_2022_2024", "prewar_2019_2021")
    return pd.Series(period, index=df.index)


def ensure_binary_from_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "military_next_week" not in df.columns:
        raise KeyError("Missing column: military_next_week")
    if "protest_next_week" not in df.columns:
        raise KeyError("Missing column: protest_next_week")

    m = pd.to_numeric(df["military_next_week"], errors="coerce").fillna(0)
    p = pd.to_numeric(df["protest_next_week"], errors="coerce").fillna(0)

    df["has_military_next_week"] = (m > 0).astype(int)
    df["has_protest_next_week"] = (p > 0).astype(int)
    return df


def chi_square_binary(df: pd.DataFrame, period_col: str, outcome_col: str) -> Tuple[Dict, pd.DataFrame]:
    tmp = df[[period_col, outcome_col]].dropna()
    tmp[outcome_col] = pd.to_numeric(tmp[outcome_col], errors="coerce").astype("Int64")
    tmp = tmp.dropna()

    periods = ["prewar_2019_2021", "war_2022_2024"]
    table = (
        pd.crosstab(tmp[period_col], tmp[outcome_col])
        .reindex(index=periods, columns=[0, 1], fill_value=0)
    )

    chi2, p, dof, expected = chi2_contingency(table.values, correction=False)

    # event rates + effect sizes
    pre_total = int(table.loc["prewar_2019_2021"].sum())
    war_total = int(table.loc["war_2022_2024"].sum())
    pre_rate = (table.loc["prewar_2019_2021", 1] / pre_total) if pre_total else np.nan
    war_rate = (table.loc["war_2022_2024", 1] / war_total) if war_total else np.nan
    risk_diff = war_rate - pre_rate

    # odds ratio (Haldane-Anscombe correction if any cell zero)
    a = float(table.loc["war_2022_2024", 1])
    b = float(table.loc["war_2022_2024", 0])
    c = float(table.loc["prewar_2019_2021", 1])
    d = float(table.loc["prewar_2019_2021", 0])
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    odds_ratio = (a / b) / (c / d)

    res = {
        "test": "chi_square_independence",
        "outcome": outcome_col,
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "prewar_rate": float(pre_rate) if pre_rate == pre_rate else None,
        "war_rate": float(war_rate) if war_rate == war_rate else None,
        "risk_diff_war_minus_pre": float(risk_diff) if risk_diff == risk_diff else None,
        "odds_ratio_war_vs_pre": float(odds_ratio) if odds_ratio == odds_ratio else None,
        "n_prewar": pre_total,
        "n_war": war_total,
    }

    expected_df = pd.DataFrame(expected, index=table.index, columns=table.columns)
    out_table = table.copy()
    out_table.columns = [f"obs_{c}" for c in out_table.columns]
    for ccol in expected_df.columns:
        out_table[f"exp_{ccol}"] = expected_df[ccol].round(3)

    return res, out_table


def mann_whitney_count(df: pd.DataFrame, period_col: str, outcome_col: str) -> Dict:
    tmp = df[[period_col, outcome_col]].dropna()
    periods = ["prewar_2019_2021", "war_2022_2024"]
    g0 = pd.to_numeric(tmp.loc[tmp[period_col] == periods[0], outcome_col], errors="coerce").dropna()
    g1 = pd.to_numeric(tmp.loc[tmp[period_col] == periods[1], outcome_col], errors="coerce").dropna()

    u_stat, p = mannwhitneyu(g0, g1, alternative="two-sided")

    return {
        "test": "mann_whitney_u",
        "outcome": outcome_col,
        "u_stat": float(u_stat),
        "p_value": float(p),
        "prewar_n": int(len(g0)),
        "war_n": int(len(g1)),
        "prewar_mean": float(g0.mean()),
        "war_mean": float(g1.mean()),
        "prewar_median": float(g0.median()),
        "war_median": float(g1.median()),
        "mean_diff_war_minus_pre": float(g1.mean() - g0.mean()),
        "median_diff_war_minus_pre": float(g1.median() - g0.median()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_path", type=str, default=str(DEFAULT_PANEL_PATH))
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--skip_count_tests", action="store_true", help="only run chi-square on binary outcomes")
    args = ap.parse_args()

    panel_path = Path(args.panel_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(panel_path)
    df["period"] = make_period_from_year(df)
    df = ensure_binary_from_counts(df)

    bin_outcomes = ["has_military_next_week", "has_protest_next_week"]
    count_outcomes = ["military_next_week", "protest_next_week"]

    all_results = []
    contingency_tables = {}

    for col in bin_outcomes:
        res, table = chi_square_binary(df, "period", col)
        all_results.append(res)
        contingency_tables[col] = table.to_dict(orient="index")

    if not args.skip_count_tests:
        for col in count_outcomes:
            all_results.append(mann_whitney_count(df, "period", col))

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(out_dir / "h1_tests_summary.csv", index=False)

    with open(out_dir / "h1_contingency_tables.json", "w", encoding="utf-8") as f:
        json.dump(contingency_tables, f, ensure_ascii=False, indent=2)

    print("Saved:", out_dir / "h1_tests_summary.csv")
    print("Saved:", out_dir / "h1_contingency_tables.json")
    print("\n=== H1 summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

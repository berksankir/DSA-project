"""
H1 visualizations (Pre-war vs War)

Uses your panel columns:
- year
- military_next_week, protest_next_week

Derives:
- has_military_next_week = 1[military_next_week > 0]
- has_protest_next_week  = 1[protest_next_week > 0]

Outputs (figures/ by default):
- h1_military_prewar_vs_war_box.png
- h1_protest_prewar_vs_war_box.png
- h1_military_share_ge1_bar.png
- h1_protest_share_ge1_bar.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    df["has_military_next_week"] = (pd.to_numeric(df["military_next_week"], errors="coerce").fillna(0) > 0).astype(int)
    df["has_protest_next_week"] = (pd.to_numeric(df["protest_next_week"], errors="coerce").fillna(0) > 0).astype(int)
    return df


def plot_box_count(df: pd.DataFrame, period_col: str, y_col: str, out_path: Path, title: str):
    tmp = df[[period_col, y_col]].dropna()
    groups = ["prewar_2019_2021", "war_2022_2024"]
    data = [pd.to_numeric(tmp.loc[tmp[period_col] == g, y_col], errors="coerce").dropna().values for g in groups]

    plt.figure(figsize=(7, 4))
    plt.boxplot(data, labels=groups, showfliers=True)
    plt.title(title)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bar_rate(df: pd.DataFrame, period_col: str, y_bin: str, out_path: Path, title: str):
    tmp = df[[period_col, y_bin]].dropna()
    rates = tmp.groupby(period_col)[y_bin].mean().reindex(["prewar_2019_2021", "war_2022_2024"])
    ns = tmp.groupby(period_col)[y_bin].size().reindex(["prewar_2019_2021", "war_2022_2024"])

    plt.figure(figsize=(6, 4))
    plt.bar(rates.index.tolist(), rates.values)
    plt.title(title)
    plt.ylabel("event rate (P[y=1])")
    for i, (idx, val) in enumerate(rates.items()):
        n = int(ns.loc[idx]) if idx in ns.index and not pd.isna(ns.loc[idx]) else 0
        plt.text(i, float(val), f"N={n}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_path", type=str, default=str(DEFAULT_PANEL_PATH))
    ap.add_argument("--fig_dir", type=str, default="figures")
    args = ap.parse_args()

    panel_path = Path(args.panel_path)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(panel_path)
    df["period"] = make_period_from_year(df)
    df = ensure_binary_from_counts(df)

    plot_box_count(df, "period", "military_next_week", fig_dir / "h1_military_prewar_vs_war_box.png",
                   "H1: military_next_week (pre-war vs war)")
    plot_box_count(df, "period", "protest_next_week", fig_dir / "h1_protest_prewar_vs_war_box.png",
                   "H1: protest_next_week (pre-war vs war)")

    plot_bar_rate(df, "period", "has_military_next_week", fig_dir / "h1_military_share_ge1_bar.png",
                  "H1: P(military_next_week > 0) by period")
    plot_bar_rate(df, "period", "has_protest_next_week", fig_dir / "h1_protest_share_ge1_bar.png",
                  "H1: P(protest_next_week > 0) by period")

    print("Saved figures to:", fig_dir.resolve())


if __name__ == "__main__":
    main()

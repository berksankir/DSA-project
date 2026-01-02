
"""
H2 visualizations: coefficient (IRR) plot for military feature tests.

Reads:
- results/h2_military_feature_tests.csv

Outputs (default figures/):
- figures/h2_military_irr_plot.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", type=str, default="results/h2_military_feature_tests.csv")
    ap.add_argument("--fig_dir", type=str, default="figures")
    args = ap.parse_args()

    results_csv = Path(args.results_csv)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)

    # Plot IRR with 95% CI
    df = df.sort_values("p_value", ascending=True).reset_index(drop=True)

    y = np.arange(len(df))
    irr = df["IRR"].values
    lo = df["IRR_CI_low"].values
    hi = df["IRR_CI_high"].values
    labels = df["feature"].tolist()

    plt.figure(figsize=(8, max(4, 0.45 * len(df))))
    plt.hlines(y, lo, hi)
    plt.plot(irr, y, "o")
    plt.axvline(1.0)
    plt.yticks(y, labels)
    plt.xlabel("Incidence Rate Ratio (IRR) per +1 unit in scaled feature")
    plt.title("H2: Military next-week events — IRR (one-feature NB GLM)")
    plt.tight_layout()

    out_path = fig_dir / "h2_military_irr_plot.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved:", out_path.resolve())


if __name__ == "__main__":
    main()

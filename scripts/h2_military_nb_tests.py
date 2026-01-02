
"""
H2: Discourse -> next-week MILITARY events (count)

We run one-feature-at-a-time Negative Binomial GLM models to avoid multicollinearity
among discourse variables.

Outcome:
- military_next_week  (count, next-week)

Predictors (tested one at a time):
- war_ratio, peace_ratio, security_frame_ratio, economy_frame_ratio,
  humanrights_frame_ratio, support_ratio, condemn_ratio, tone_support_score

Controls:
- num_docs
- dyad fixed effects: C(dyad)
- year fixed effects: C(year)
- week control: either numeric 'week' (default) or week fixed effects C(week) via --week_fe

NB dispersion (alpha):
- estimated by method-of-moments from outcome mean/variance unless overridden by --alpha

Outputs:
- results/h2_military_feature_tests.csv
- results/h2_military_meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


DEFAULT_PANEL_PATH = Path("data_processed/panel_diplomacy_gdelt_week_2019_2024.csv")

DISCOURSE_FEATURES: List[str] = [
    "war_ratio",
    "peace_ratio",
    "security_frame_ratio",
    "economy_frame_ratio",
    "humanrights_frame_ratio",
    "support_ratio",
    "condemn_ratio",
    "tone_support_score",
]


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR adjustment."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)
    q = p * m / ranks
    q_sorted = q[order]
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_adj = np.empty(m, dtype=float)
    q_adj[order] = np.minimum(q_sorted, 1.0)
    return q_adj


def estimate_alpha_nb(y: pd.Series) -> float:
    """Method-of-moments alpha for NB2: Var(y) = mu + alpha * mu^2."""
    y = pd.to_numeric(y, errors="coerce").dropna()
    mu = float(y.mean())
    var = float(y.var(ddof=1))
    if mu <= 0:
        return 1.0
    alpha = (var - mu) / (mu ** 2)
    return float(max(alpha, 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_path", type=str, default=str(DEFAULT_PANEL_PATH))
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--week_fe", action="store_true", help="Use week fixed effects C(week) instead of numeric week.")
    ap.add_argument("--alpha", type=float, default=None, help="Override NB alpha (dispersion).")
    ap.add_argument("--scale_factor", type=float, default=10000.0, help="Scale discourse features by this factor.")
    args = ap.parse_args()

    panel_path = Path(args.panel_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(panel_path)

    required = {"dyad", "year", "week", "num_docs", "military_next_week"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    for f in DISCOURSE_FEATURES:
        if f not in df.columns:
            raise KeyError(f"Missing discourse feature column: {f}")

    # Scale discourse features to make coefficients interpretable and numerically stable.
    # With scale_factor=10000, coefficients correspond roughly to a +1e-4 change in the original ratio.
    scaled_map: Dict[str, str] = {}
    for f in DISCOURSE_FEATURES:
        new_name = f"{f}_scaled"
        df[new_name] = pd.to_numeric(df[f], errors="coerce") * float(args.scale_factor)
        scaled_map[f] = new_name

    alpha = float(args.alpha) if args.alpha is not None else estimate_alpha_nb(df["military_next_week"])

    week_term = "C(week)" if args.week_fe else "week"

    rows = []
    for f in DISCOURSE_FEATURES:
        x = scaled_map[f]
        formula = f"military_next_week ~ {x} + num_docs + C(dyad) + C(year) + {week_term}"

        model = smf.glm(
            formula,
            data=df,
            family=sm.families.NegativeBinomial(alpha=alpha),
        )

        res = model.fit(cov_type="HC1", maxiter=200)

        b = float(res.params[x])
        se = float(res.bse[x])
        p = float(res.pvalues[x])

        # IRR for +1 unit in scaled feature (i.e., +1/scale_factor in original feature)
        irr = float(np.exp(b))
        ci_low = float(np.exp(b - 1.96 * se))
        ci_high = float(np.exp(b + 1.96 * se))

        rows.append(
            {
                "feature": f,
                "feature_scaled": x,
                "scale_factor": float(args.scale_factor),
                "coef": b,
                "se": se,
                "p_value": p,
                "IRR": irr,
                "IRR_CI_low": ci_low,
                "IRR_CI_high": ci_high,
                "n": int(res.nobs),
            }
        )

    out = pd.DataFrame(rows)
    out["p_fdr"] = bh_fdr(out["p_value"].values)
    out = out.sort_values("p_value").reset_index(drop=True)

    out_csv = out_dir / "h2_military_feature_tests.csv"
    out.to_csv(out_csv, index=False)

    meta = {
        "panel_path": str(panel_path),
        "alpha_used": alpha,
        "week_term": week_term,
        "scale_factor": float(args.scale_factor),
        "models": "one-feature-at-a-time NB GLM with HC1 robust SE; controls: num_docs + dyad FE + year FE + week control",
        "features_tested": DISCOURSE_FEATURES,
    }
    meta_json = out_dir / "h2_military_meta.json"
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:", out_csv)
    print("Saved:", meta_json)
    print("\n=== H2 military (feature tests) ===")
    print(out[["feature", "coef", "IRR", "p_value", "p_fdr"]])


if __name__ == "__main__":
    main()

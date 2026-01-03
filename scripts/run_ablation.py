import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

PANEL_PATH = "data_processed/panel_diplomacy_gdelt_week_2019_2024.csv"

BASE_FEATURES = [
    "war_ratio","peace_ratio","security_frame_ratio","economy_frame_ratio",
    "humanrights_frame_ratio","support_ratio","condemn_ratio","tone_support_score",
]
DOC_FEATURE = ["num_docs"]

CORE = ["dyad","year","week"]
TARGETS = ["protest_next_week","military_next_week"]

df = pd.read_csv(PANEL_PATH, usecols=CORE + BASE_FEATURES + DOC_FEATURE + TARGETS).sort_values(["dyad","year","week"])
df["has_protest_next_week"]  = (df["protest_next_week"]  > 0).astype(int)
df["has_military_next_week"] = (df["military_next_week"] > 0).astype(int)

train = df[df["year"] <= 2022].copy()
val   = df[df["year"] == 2023].copy()
test  = df[df["year"] == 2024].copy()

def add_lags(d, feat_cols):
    d = d.sort_values(["dyad","year","week"]).copy()
    for c in feat_cols:
        d[f"{c}_lag1"] = d.groupby("dyad")[c].shift(1)
    return d

def best_thr_by_f1(y, p):
    grid = np.linspace(0.05, 0.95, 19)
    best_t, best_f = 0.5, -1
    for t in grid:
        f = f1_score(y, (p>=t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t), float(best_f)

def eval_bin(y, p, thr):
    pred = (p>=thr).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "f1": f1_score(y, pred, zero_division=0),
    }

def run_config(use_num_docs: bool, use_lag: bool):
    feats = BASE_FEATURES + (DOC_FEATURE if use_num_docs else [])
    tr = train.copy(); v = val.copy(); te = test.copy()

    if use_lag:
        tr = add_lags(tr, feats)
        v  = add_lags(v, feats)
        te = add_lags(te, feats)
        feats = feats + [f"{c}_lag1" for c in feats]
        tr = tr.dropna(subset=feats); v = v.dropna(subset=feats); te = te.dropna(subset=feats)

    Xtr, Xv, Xte = tr[feats], v[feats], te[feats]

    out_rows = []

    # Protest: HGB-Cls
    ytr, yv, yte = tr["has_protest_next_week"], v["has_protest_next_week"], te["has_protest_next_week"]
    m_p = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400)
    m_p.fit(Xtr, ytr)
    pv = m_p.predict_proba(Xv)[:,1]
    thr, _ = best_thr_by_f1(yv, pv)
    pte = m_p.predict_proba(Xte)[:,1]
    r_val  = eval_bin(yv, pv, thr)
    r_test = eval_bin(yte, pte, thr)
    out_rows += [
        {"task":"PROTEST_BINARY","model":"HGB-Cls","split":"VAL","thr":thr, **r_val},
        {"task":"PROTEST_BINARY","model":"HGB-Cls","split":"TEST","thr":thr, **r_test},
    ]

    # Military: LogReg
    ytr, yv, yte = tr["has_military_next_week"], v["has_military_next_week"], te["has_military_next_week"]
    m_m = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    m_m.fit(Xtr, ytr)
    pv = m_m.predict_proba(Xv)[:,1]
    thr, _ = best_thr_by_f1(yv, pv)
    pte = m_m.predict_proba(Xte)[:,1]
    r_val  = eval_bin(yv, pv, thr)
    r_test = eval_bin(yte, pte, thr)
    out_rows += [
        {"task":"MILITARY_BINARY","model":"LogReg","split":"VAL","thr":thr, **r_val},
        {"task":"MILITARY_BINARY","model":"LogReg","split":"TEST","thr":thr, **r_test},
    ]

    tag = f"num_docs={use_num_docs}|lag1={use_lag}"
    for r in out_rows:
        r["config"] = tag
    return out_rows

rows = []
for use_num_docs in [True, False]:
    for use_lag in [True, False]:
        rows += run_config(use_num_docs, use_lag)

res = pd.DataFrame(rows).sort_values(["task","config","split"])
res.to_csv("ablation_binary_results.csv", index=False)

print("Wrote: ablation_binary_results.csv")
print(res.to_string(index=False))

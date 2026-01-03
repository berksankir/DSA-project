import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)

PANEL_PATH = "data_processed/panel_diplomacy_gdelt_week_2019_2024.csv"

BASE_FEATURES = [
    "war_ratio","peace_ratio","security_frame_ratio","economy_frame_ratio",
    "humanrights_frame_ratio","support_ratio","condemn_ratio","tone_support_score",
]
DOC = ["num_docs"]
CORE = ["dyad","year","week"]
TARGETS = ["protest_next_week","military_next_week"]

# -----------------------
# Load
# -----------------------
df = pd.read_csv(PANEL_PATH, usecols=CORE + BASE_FEATURES + DOC + TARGETS).sort_values(["dyad","year","week"])
df["has_protest_next_week"]  = (df["protest_next_week"]  > 0).astype(int)
df["has_military_next_week"] = (df["military_next_week"] > 0).astype(int)

train = df[df["year"] <= 2022].copy()
val   = df[df["year"] == 2023].copy()
test  = df[df["year"] == 2024].copy()

def add_lag1(d, feats):
    d = d.sort_values(["dyad","year","week"]).copy()
    for c in feats:
        d[f"{c}_lag1"] = d.groupby("dyad")[c].shift(1)
    return d

def make_splits(use_num_docs: bool, use_lag: bool):
    feats = BASE_FEATURES + (DOC if use_num_docs else [])
    tr, v, te = train.copy(), val.copy(), test.copy()

    if use_lag:
        tr = add_lag1(tr, feats)
        v  = add_lag1(v, feats)
        te = add_lag1(te, feats)
        feats = feats + [f"{c}_lag1" for c in feats]
        tr = tr.dropna(subset=feats)
        v  = v.dropna(subset=feats)
        te = te.dropna(subset=feats)

    return feats, tr, v, te

def thr_by_f1(y, p):
    grid = np.linspace(0.05, 0.95, 19)
    best_t, best_f = 0.5, -1
    for t in grid:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t), float(best_f)

def thr_by_precision(y, p, target_precision=0.70):
    # choose highest recall among thresholds reaching target precision
    grid = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91), p]))  # dense + score-based
    best = None
    for t in grid:
        pred = (p >= t).astype(int)
        prec = precision_score(y, pred, zero_division=0)
        if prec + 1e-12 < target_precision:
            continue
        rec = recall_score(y, pred, zero_division=0)
        f1  = f1_score(y, pred, zero_division=0)
        row = (float(t), float(prec), float(rec), float(f1))
        if best is None or rec > best[2] or (rec == best[2] and f1 > best[3]):
            best = row
    # fallback: if target not reachable, return F1-opt
    if best is None:
        t, _ = thr_by_f1(y, p)
        pred = (p >= t).astype(int)
        return float(t), float(precision_score(y, pred, zero_division=0)), float(recall_score(y, pred, zero_division=0)), float(f1_score(y, pred, zero_division=0))
    return best  # t, prec, rec, f1

def eval_at_threshold(task, model_name, split, y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "task": task,
        "model": model_name,
        "split": split,
        "thr": float(thr),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

rows = []

# =========================================================
# FINAL CONFIGS from ablation:
#   PROTEST:  num_docs=True,  lag1=True,  model=HGB-Cls
#   MILITARY: num_docs=False, lag1=True,  model=LogReg
# =========================================================

# -----------------------
# PROTEST
# -----------------------
feats_p, tr_p, v_p, te_p = make_splits(use_num_docs=True, use_lag=True)
Xtr, Xv, Xte = tr_p[feats_p], v_p[feats_p], te_p[feats_p]
ytr, yv, yte = tr_p["has_protest_next_week"], v_p["has_protest_next_week"], te_p["has_protest_next_week"]

m_p = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400)
m_p.fit(Xtr, ytr)

pv  = m_p.predict_proba(Xv)[:, 1]
pte = m_p.predict_proba(Xte)[:, 1]

# threshold A: F1-opt on VAL
thr_f1, _ = thr_by_f1(yv, pv)

# threshold B: precision-constrained on VAL (default 0.70)
thr_prec, val_prec, val_rec, val_f1 = thr_by_precision(yv, pv, target_precision=0.70)

# report both thresholds on TEST (and also VAL rows for record)
rows += [
    {"threshold_rule":"VAL_F1_OPT", "config":"num_docs=True|lag1=True", **eval_at_threshold("PROTEST_BINARY","HGB-Cls","VAL", yv, pv,  thr_f1)},
    {"threshold_rule":"VAL_F1_OPT", "config":"num_docs=True|lag1=True", **eval_at_threshold("PROTEST_BINARY","HGB-Cls","TEST",yte,pte, thr_f1)},
    {"threshold_rule":"VAL_PREC>=0.70_MAXREC", "config":"num_docs=True|lag1=True", **eval_at_threshold("PROTEST_BINARY","HGB-Cls","VAL", yv, pv,  thr_prec)},
    {"threshold_rule":"VAL_PREC>=0.70_MAXREC", "config":"num_docs=True|lag1=True", **eval_at_threshold("PROTEST_BINARY","HGB-Cls","TEST",yte,pte, thr_prec)},
]

# -----------------------
# MILITARY
# -----------------------
feats_m, tr_m, v_m, te_m = make_splits(use_num_docs=False, use_lag=True)
Xtr, Xv, Xte = tr_m[feats_m], v_m[feats_m], te_m[feats_m]
ytr, yv, yte = tr_m["has_military_next_week"], v_m["has_military_next_week"], te_m["has_military_next_week"]

m_m = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
m_m.fit(Xtr, ytr)

pv  = m_m.predict_proba(Xv)[:, 1]
pte = m_m.predict_proba(Xte)[:, 1]

thr_f1, _ = thr_by_f1(yv, pv)
thr_prec, val_prec, val_rec, val_f1 = thr_by_precision(yv, pv, target_precision=0.70)

rows += [
    {"threshold_rule":"VAL_F1_OPT", "config":"num_docs=False|lag1=True", **eval_at_threshold("MILITARY_BINARY","LogReg","VAL", yv, pv,  thr_f1)},
    {"threshold_rule":"VAL_F1_OPT", "config":"num_docs=False|lag1=True", **eval_at_threshold("MILITARY_BINARY","LogReg","TEST",yte,pte, thr_f1)},
    {"threshold_rule":"VAL_PREC>=0.70_MAXREC", "config":"num_docs=False|lag1=True", **eval_at_threshold("MILITARY_BINARY","LogReg","VAL", yv, pv,  thr_prec)},
    {"threshold_rule":"VAL_PREC>=0.70_MAXREC", "config":"num_docs=False|lag1=True", **eval_at_threshold("MILITARY_BINARY","LogReg","TEST",yte,pte, thr_prec)},
]

report = pd.DataFrame(rows).sort_values(["task","threshold_rule","split"])
report.to_csv("final_binary_report_two_thresholds.csv", index=False)

print("Wrote: final_binary_report_two_thresholds.csv")
print(report.to_string(index=False))
    
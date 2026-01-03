import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score

PANEL_PATH = "data_processed/panel_diplomacy_gdelt_week_2019_2024.csv"

BASE_FEATURES = [
    "war_ratio","peace_ratio","security_frame_ratio","economy_frame_ratio",
    "humanrights_frame_ratio","support_ratio","condemn_ratio","tone_support_score",
]
DOC = ["num_docs"]
CORE = ["dyad","year","week"]
TARGETS = ["protest_next_week","military_next_week"]

def add_lag1(d, feats):
    d = d.sort_values(["dyad","year","week"]).copy()
    for c in feats:
        d[f"{c}_lag1"] = d.groupby("dyad")[c].shift(1)
    return d

# -----------------------
# Load + labels
# -----------------------
df = pd.read_csv(PANEL_PATH, usecols=CORE + BASE_FEATURES + DOC + TARGETS).sort_values(["dyad","year","week"])
df["has_protest_next_week"]  = (df["protest_next_week"]  > 0).astype(int)
df["has_military_next_week"] = (df["military_next_week"] > 0).astype(int)

train = df[df["year"] <= 2022].copy()
test  = df[df["year"] == 2024].copy()

# =========================================================
# A) PROTEST PR curve (HGB-Cls, num_docs=True, lag1=True)
# =========================================================
feats_p = BASE_FEATURES + DOC
train_p = add_lag1(train, feats_p).dropna()
test_p  = add_lag1(test, feats_p).dropna()
feats_p = feats_p + [f"{c}_lag1" for c in feats_p]

Xtr = train_p[feats_p]
ytr = train_p["has_protest_next_week"].values
Xte = test_p[feats_p]
yte = test_p["has_protest_next_week"].values

m_p = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400)
m_p.fit(Xtr, ytr)
pte = m_p.predict_proba(Xte)[:, 1]

prec, rec, _ = precision_recall_curve(yte, pte)
ap = average_precision_score(yte, pte)

plt.figure()
plt.plot(rec, prec, label=f"PR curve (AP={ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PROTEST_BINARY — Precision-Recall (TEST 2024)")
plt.legend(loc="best")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("fig_pr_protest_test.png", dpi=200)
plt.close()

# =========================================================
# B) MILITARY PR curve (LogReg, num_docs=False, lag1=True)
# =========================================================
feats_m = BASE_FEATURES
train_m = add_lag1(train, feats_m).dropna()
test_m  = add_lag1(test, feats_m).dropna()
feats_m = feats_m + [f"{c}_lag1" for c in feats_m]

Xtr = train_m[feats_m]
ytr = train_m["has_military_next_week"].values
Xte = test_m[feats_m]
yte = test_m["has_military_next_week"].values

m_m = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
m_m.fit(Xtr, ytr)
pte = m_m.predict_proba(Xte)[:, 1]

prec, rec, _ = precision_recall_curve(yte, pte)
ap = average_precision_score(yte, pte)

plt.figure()
plt.plot(rec, prec, label=f"PR curve (AP={ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("MILITARY_BINARY — Precision-Recall (TEST 2024)")
plt.legend(loc="best")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("fig_pr_military_test.png", dpi=200)
plt.close()

print("Wrote: fig_pr_protest_test.png, fig_pr_military_test.png")

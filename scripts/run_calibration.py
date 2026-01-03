import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

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

def reliability_table(y, p, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins, right=True)
    rows = []
    for b in range(1, n_bins + 1):
        mask = idx == b
        if mask.sum() == 0:
            continue
        yb = y[mask]
        pb = p[mask]
        rows.append({
            "bin": b,
            "n": int(mask.sum()),
            "p_mean": float(np.mean(pb)),
            "y_rate": float(np.mean(yb)),
            "abs_gap": float(abs(np.mean(pb) - np.mean(yb))),
            "p_min": float(bins[b-1]),
            "p_max": float(bins[b]),
        })
    return pd.DataFrame(rows)

# -----------------------
# Load + labels
# -----------------------
df = pd.read_csv(PANEL_PATH, usecols=CORE + BASE_FEATURES + DOC + TARGETS).sort_values(["dyad","year","week"])
df["has_protest_next_week"]  = (df["protest_next_week"]  > 0).astype(int)
df["has_military_next_week"] = (df["military_next_week"] > 0).astype(int)

train = df[df["year"] <= 2022].copy()
val   = df[df["year"] == 2023].copy()
test  = df[df["year"] == 2024].copy()

# Calibration will be fit on TRAIN+VAL only (no TEST leakage)
trainval = pd.concat([train, val], ignore_index=True)

results = []

# =========================================================
# A) PROTEST: HGB-Cls, num_docs=True, lag1=True
# =========================================================
feats_p = BASE_FEATURES + DOC

trv_p  = add_lag1(trainval, feats_p).dropna()
test_p = add_lag1(test, feats_p).dropna()

feats_p = feats_p + [f"{c}_lag1" for c in feats_p]

X_trv = trv_p[feats_p].values
y_trv = trv_p["has_protest_next_week"].values
X_te  = test_p[feats_p].values
y_te  = test_p["has_protest_next_week"].values

base_p = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400)

# Uncalibrated: fit base on train+val
base_p.fit(X_trv, y_trv)
p_te_uncal = base_p.predict_proba(X_te)[:, 1]

# Calibrated: cross-validated calibration on train+val
cal_p = CalibratedClassifierCV(estimator=HistGradientBoostingClassifier(
    max_depth=3, learning_rate=0.05, max_iter=400
), method="sigmoid", cv=5)
cal_p.fit(X_trv, y_trv)
p_te_cal = cal_p.predict_proba(X_te)[:, 1]

results += [
    {"task":"PROTEST_BINARY","model":"HGB-Cls","calibration":"none(train+val fit)",
     "roc_auc": roc_auc_score(y_te, p_te_uncal),
     "pr_auc": average_precision_score(y_te, p_te_uncal),
     "brier": brier_score_loss(y_te, p_te_uncal)},
    {"task":"PROTEST_BINARY","model":"HGB-Cls","calibration":"sigmoid(cv=5 on train+val)",
     "roc_auc": roc_auc_score(y_te, p_te_cal),
     "pr_auc": average_precision_score(y_te, p_te_cal),
     "brier": brier_score_loss(y_te, p_te_cal)},
]

reliability_table(y_te, p_te_uncal, 10).to_csv("reliability_bins_protest_uncal.csv", index=False)
reliability_table(y_te, p_te_cal,   10).to_csv("reliability_bins_protest_cal_sigmoid.csv", index=False)

# =========================================================
# B) MILITARY: LogReg, num_docs=False, lag1=True
# =========================================================
feats_m = BASE_FEATURES  # no num_docs

trv_m  = add_lag1(trainval, feats_m).dropna()
test_m = add_lag1(test, feats_m).dropna()

feats_m = feats_m + [f"{c}_lag1" for c in feats_m]

X_trv = trv_m[feats_m].values
y_trv = trv_m["has_military_next_week"].values
X_te  = test_m[feats_m].values
y_te  = test_m["has_military_next_week"].values

base_m = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
base_m.fit(X_trv, y_trv)
p_te_uncal = base_m.predict_proba(X_te)[:, 1]

cal_m = CalibratedClassifierCV(
    estimator=Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]),
    method="sigmoid",
    cv=5
)
cal_m.fit(X_trv, y_trv)
p_te_cal = cal_m.predict_proba(X_te)[:, 1]

results += [
    {"task":"MILITARY_BINARY","model":"LogReg","calibration":"none(train+val fit)",
     "roc_auc": roc_auc_score(y_te, p_te_uncal),
     "pr_auc": average_precision_score(y_te, p_te_uncal),
     "brier": brier_score_loss(y_te, p_te_uncal)},
    {"task":"MILITARY_BINARY","model":"LogReg","calibration":"sigmoid(cv=5 on train+val)",
     "roc_auc": roc_auc_score(y_te, p_te_cal),
     "pr_auc": average_precision_score(y_te, p_te_cal),
     "brier": brier_score_loss(y_te, p_te_cal)},
]

reliability_table(y_te, p_te_uncal, 10).to_csv("reliability_bins_military_uncal.csv", index=False)
reliability_table(y_te, p_te_cal,   10).to_csv("reliability_bins_military_cal_sigmoid.csv", index=False)

# -----------------------
# Save summary
# -----------------------
report = pd.DataFrame(results)
report.to_csv("calibration_report.csv", index=False)

print("Wrote: calibration_report.csv")
print("Wrote reliability bins CSVs (protest/military, uncalibrated/calibrated).")
print("\n=== Calibration summary (TEST) ===")
print(report.to_string(index=False))

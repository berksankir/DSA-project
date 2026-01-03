import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# Count baseline (NegBin) için
import statsmodels.api as sm

# -------------------------
# 0) Dosya yolu
# -------------------------
PANEL_PATH = "data_processed/panel_diplomacy_gdelt_week_2019_2024.csv"  # kendi klasörüne göre düzelt

# -------------------------
# 1) Beklenen kolon adları (otomatik yakalamaya çalışacağız)
# -------------------------
CORE_CANDIDATES = {
    "dyad": ["dyad", "dyad_id", "pair", "dyad_name"],
    "year": ["year"],
    "week": ["week", "week_id", "weekofyear"],
    "num_docs": ["num_docs", "doc_count", "n_docs", "num_statements"],
}

FEATURE_CANDIDATES = [
    "war_ratio", "peace_ratio", "security_frame_ratio", "economy_frame_ratio",
    "humanrights_frame_ratio", "support_ratio", "condemn_ratio", "tone_support_score",
    "num_docs"
]

TARGET_CANDIDATES = {
    "protest_count": ["protest_next_week", "protest_events_next_week", "protest_cnt_next_week"],
    "military_count": ["military_next_week", "military_events_next_week", "military_cnt_next_week"],
    "protest_bin": ["has_protest_next_week", "protest_has_next_week"],
    "military_bin": ["has_military_next_week", "military_has_next_week"],
}

def pick_col(df_cols, candidates):
    s = set(df_cols)
    for c in candidates:
        if c in s:
            return c
    return None

# -------------------------
# 2) Önce sadece header oku → kolonları bul
# -------------------------
header = pd.read_csv(PANEL_PATH, nrows=0)
cols = header.columns.tolist()

dyad_col = pick_col(cols, CORE_CANDIDATES["dyad"])
year_col = pick_col(cols, CORE_CANDIDATES["year"])
week_col = pick_col(cols, CORE_CANDIDATES["week"])
num_docs_col = pick_col(cols, CORE_CANDIDATES["num_docs"])

protest_count_col = pick_col(cols, TARGET_CANDIDATES["protest_count"])
military_count_col = pick_col(cols, TARGET_CANDIDATES["military_count"])
protest_bin_col = pick_col(cols, TARGET_CANDIDATES["protest_bin"])
military_bin_col = pick_col(cols, TARGET_CANDIDATES["military_bin"])

if not (dyad_col and year_col and week_col):
    raise ValueError(f"Gerekli core kolonları bulamadım. Bulunanlar: dyad={dyad_col}, year={year_col}, week={week_col}")

# Feature kolonlarını mevcutlardan seç
feature_cols = [c for c in FEATURE_CANDIDATES if c in cols]
if num_docs_col and num_docs_col not in feature_cols:
    feature_cols.append(num_docs_col)

# Minimal okuma kolonları
usecols = list(dict.fromkeys(
    [dyad_col, year_col, week_col] + feature_cols +
    [c for c in [protest_count_col, military_count_col, protest_bin_col, military_bin_col] if c]
))

print("Seçilen kolonlar:", usecols)

# -------------------------
# 3) Büyük dosya için sadece usecols oku
# -------------------------
df = pd.read_csv(PANEL_PATH, usecols=usecols)

# -------------------------
# 4) Binary label yoksa count'tan üret
# -------------------------
df = df.sort_values([dyad_col, year_col, week_col])

if protest_bin_col is None:
    if protest_count_col is None:
        raise ValueError("Protest için ne binary ne count label kolonu bulundu.")
    df["has_protest_next_week"] = (df[protest_count_col] > 0).astype(int)
    protest_bin_col = "has_protest_next_week"

if military_bin_col is None:
    if military_count_col is None:
        raise ValueError("Military için ne binary ne count label kolonu bulundu.")
    df["has_military_next_week"] = (df[military_count_col] > 0).astype(int)
    military_bin_col = "has_military_next_week"

# Count hedefleri yoksa skip edeceğiz
has_protest_count = protest_count_col is not None
has_military_count = military_count_col is not None

# -------------------------
# 5) Lag1 feature ekle
# -------------------------
for c in feature_cols:
    df[f"{c}_lag1"] = df.groupby(dyad_col)[c].shift(1)

X_cols = feature_cols + [f"{c}_lag1" for c in feature_cols]
df_model = df.dropna(subset=X_cols + [protest_bin_col, military_bin_col, year_col]).copy()

# -------------------------
# 6) Time split
# -------------------------
train = df_model[df_model[year_col] <= 2022]
val   = df_model[df_model[year_col] == 2023]
test  = df_model[df_model[year_col] == 2024]

def eval_binary(name, y_true, p):
    pred = (p >= 0.5).astype(int)
    return {
        "split": name,
        "roc_auc": roc_auc_score(y_true, p),
        "pr_auc": average_precision_score(y_true, p),
        "f1@0.5": f1_score(y_true, pred)
    }

def eval_count(name, y_true, y_hat):
    y_hat = np.clip(y_hat, 0, None)
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    return {
        "split": name,
        "mae": mean_absolute_error(y_true, y_hat),
        "rmse": rmse
    }


# =========================
# A) BINARY MODELS
# =========================
def run_binary(task_name, ycol):
    X_train, y_train = train[X_cols], train[ycol]
    X_val, y_val     = val[X_cols],   val[ycol]
    X_test, y_test   = test[X_cols],  test[ycol]

    # Baseline: Logistic Regression
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    lr.fit(X_train, y_train)
    res_lr = [
        {"model": "LogReg", **eval_binary("VAL", y_val, lr.predict_proba(X_val)[:,1])},
        {"model": "LogReg", **eval_binary("TEST", y_test, lr.predict_proba(X_test)[:,1])},
    ]

    # Non-linear: HGB Classifier
    hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400)
    hgb.fit(X_train, y_train)
    res_hgb = [
        {"model": "HGB-Cls", **eval_binary("VAL", y_val, hgb.predict_proba(X_val)[:,1])},
        {"model": "HGB-Cls", **eval_binary("TEST", y_test, hgb.predict_proba(X_test)[:,1])},
    ]

    out = pd.DataFrame(res_lr + res_hgb)
    out.insert(0, "task", task_name)
    return out

bin_results = pd.concat([
    run_binary("PROTEST_BINARY", protest_bin_col),
    run_binary("MILITARY_BINARY", military_bin_col)
], ignore_index=True)

print("\n=== BINARY RESULTS ===")
print(bin_results.to_string(index=False))

# =========================
# B) COUNT MODELS
# =========================
def run_count(task_name, ycount_col):
    X_train, y_train = train[X_cols], train[ycount_col]
    X_val, y_val     = val[X_cols],   val[ycount_col]
    X_test, y_test   = test[X_cols],  test[ycount_col]

    # Baseline 1: Negative Binomial (GLM)
    X_train_sm = sm.add_constant(X_train, has_constant="add")
    X_val_sm   = sm.add_constant(X_val, has_constant="add")
    X_test_sm  = sm.add_constant(X_test, has_constant="add")

    nb = sm.GLM(y_train, X_train_sm, family=sm.families.NegativeBinomial())
    nb_fit = nb.fit()

    nb_val  = nb_fit.predict(X_val_sm)
    nb_test = nb_fit.predict(X_test_sm)

    res_nb = [
        {"model":"NegBin", **eval_count("VAL", y_val, nb_val)},
        {"model":"NegBin", **eval_count("TEST", y_test, nb_test)},
    ]

    # Baseline 2: log1p + HGB Regressor
    y_train_log = np.log1p(y_train)

    hgb_r = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=600)
    hgb_r.fit(X_train, y_train_log)

    pred_val  = np.expm1(hgb_r.predict(X_val))
    pred_test = np.expm1(hgb_r.predict(X_test))

    res_hgb = [
        {"model":"HGB-Reg-log1p", **eval_count("VAL", y_val, pred_val)},
        {"model":"HGB-Reg-log1p", **eval_count("TEST", y_test, pred_test)},
    ]

    out = pd.DataFrame(res_nb + res_hgb)
    out.insert(0, "task", task_name)
    return out

count_frames = []
if has_protest_count:
    count_frames.append(run_count("PROTEST_COUNT", protest_count_col))
if has_military_count:
    count_frames.append(run_count("MILITARY_COUNT", military_count_col))

if count_frames:
    count_results = pd.concat(count_frames, ignore_index=True)
    print("\n=== COUNT RESULTS ===")
    print(count_results.to_string(index=False))
else:
    print("\nCount kolonları bulunamadı; count modelleri çalıştırılmadı.")

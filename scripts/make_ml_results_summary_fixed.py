import pandas as pd

COLS = [
    "section","task","model","features","note",
    "threshold_rule","thr","split",
    "roc_auc","pr_auc","precision","recall","f1","tp","fp","tn","fn",
    "calibration","brier","decision"
]

def row(**kwargs):
    r = {c: "" for c in COLS}
    for k,v in kwargs.items():
        r[k] = v
    return r

rows = []

# ---------------------------
# Final choices
# ---------------------------
rows += [
    row(section="final_choice", task="PROTEST_BINARY", model="HGB-Cls",
        features="num_docs=True, lag1=True",
        note="Selected as best protest binary model (from test PR-AUC/F1)"),

    row(section="final_choice", task="MILITARY_BINARY", model="LogReg",
        features="num_docs=False, lag1=True",
        note="Selected as best military binary model (from test F1)"),

    row(section="final_choice", task="PROTEST_COUNT", model="NegBin(GLM)",
        features="same base features",
        note="Selected as best count baseline (lowest MAE/RMSE vs log1p boosting)"),

    row(section="final_choice", task="MILITARY_COUNT", model="NegBin(GLM)",
        features="same base features",
        note="Selected as best count baseline (lowest MAE/RMSE vs log1p boosting)"),
]

# ---------------------------
# Two-threshold TEST report (final_binary_report_two_thresholds.csv)
# ---------------------------
rows += [
    row(section="threshold_report", task="MILITARY_BINARY", model="LogReg",
        features="num_docs=False, lag1=True",
        threshold_rule="VAL_F1_OPT", thr=0.55, split="TEST",
        roc_auc=0.893887, pr_auc=0.772959, precision=0.597561, recall=0.960784, f1=0.736842,
        tp=49, fp=33, tn=69, fn=2),

    row(section="threshold_report", task="MILITARY_BINARY", model="LogReg",
        features="num_docs=False, lag1=True",
        threshold_rule="VAL_PREC>=0.70_MAXREC", thr=0.41, split="TEST",
        roc_auc=0.893887, pr_auc=0.772959, precision=0.520408, recall=1.000000, f1=0.684536,
        tp=51, fp=47, tn=55, fn=0),

    row(section="threshold_report", task="PROTEST_BINARY", model="HGB-Cls",
        features="num_docs=True, lag1=True",
        threshold_rule="VAL_F1_OPT", thr=0.60, split="TEST",
        roc_auc=0.875286, pr_auc=0.697757, precision=0.538462, recall=0.736842, f1=0.622222,
        tp=28, fp=24, tn=91, fn=10),

    row(section="threshold_report", task="PROTEST_BINARY", model="HGB-Cls",
        features="num_docs=True, lag1=True",
        threshold_rule="VAL_PREC>=0.70_MAXREC", thr=0.22, split="TEST",
        roc_auc=0.875286, pr_auc=0.697757, precision=0.443038, recall=0.921053, f1=0.598291,
        tp=35, fp=44, tn=71, fn=3),
]

# ---------------------------
# Calibration summary (TEST) - sigmoid run
# ---------------------------
rows += [
    row(section="calibration", task="PROTEST_BINARY", model="HGB-Cls",
        features="num_docs=True, lag1=True",
        calibration="none(train+val fit)", split="TEST",
        roc_auc=0.908238, pr_auc=0.703526, brier=0.125235,
        decision="keep_uncalibrated",
        note="Sigmoid worsened PR-AUC and Brier; keep uncalibrated probabilities."),

    row(section="calibration", task="PROTEST_BINARY", model="HGB-Cls",
        features="num_docs=True, lag1=True",
        calibration="sigmoid(cv=5 on train+val)", split="TEST",
        roc_auc=0.881922, pr_auc=0.685044, brier=0.144708,
        decision="reject"),

    row(section="calibration", task="MILITARY_BINARY", model="LogReg",
        features="num_docs=False, lag1=True",
        calibration="none(train+val fit)", split="TEST",
        roc_auc=0.899846, pr_auc=0.781996, brier=0.173492,
        decision="use_for_PR_ranking",
        note="Best PR-AUC (ranking/detection)."),

    row(section="calibration", task="MILITARY_BINARY", model="LogReg",
        features="num_docs=False, lag1=True",
        calibration="sigmoid(cv=5 on train+val)", split="TEST",
        roc_auc=0.897155, pr_auc=0.780837, brier=0.139207,
        decision="use_if_probability_quality_needed",
        note="Much better Brier with minimal PR-AUC drop."),
]

df = pd.DataFrame(rows, columns=COLS)
df.to_csv("ml_results_summary_fixed_v2.csv", index=False, encoding="utf-8", lineterminator="\n")
print("Wrote: ml_results_summary_fixed.csv")
print(df.to_string(index=False))

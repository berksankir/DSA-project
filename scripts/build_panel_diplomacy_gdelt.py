import pandas as pd
from pathlib import Path

# Girdi dosyaları
DIP_PATH = Path("data_processed/diplomacy_dyad_week_features_2019_2024.csv")
GDELT_PATH = Path("data_processed/gdelt_dyad_week_2019_2024.csv")

OUT_PATH = Path("data_processed/panel_diplomacy_gdelt_week_2019_2024.csv")

print("Reading diplomacy features from", DIP_PATH)
dip = pd.read_csv(DIP_PATH)

print("Reading GDELT dyad-week from", GDELT_PATH)
gdl = pd.read_csv(GDELT_PATH)

# Diplomasi tarafında mutlaka olması gereken kolonlar
needed_dip = {"dyad", "year", "week",
              "war_ratio", "peace_ratio",
              "security_frame_ratio", "economy_frame_ratio",
              "humanrights_frame_ratio", "support_ratio",
              "condemn_ratio", "tone_support_score"}

# GDELT tarafında beklenen kolonlar
needed_gdl = {"dyad", "year", "week",
              "protest_count", "military_count",
              "total_events", "goldstein_sum",
              "goldstein_mean", "avg_tone_mean"}

missing_dip = needed_dip - set(dip.columns)
missing_gdl = needed_gdl - set(gdl.columns)

if missing_dip:
    raise ValueError(f"Missing columns in diplomacy file: {missing_dip}")
if missing_gdl:
    raise ValueError(f"Missing columns in GDELT file: {missing_gdl}")

# 1) Aynı hafta merge
panel = dip.merge(
    gdl,
    on=["dyad", "year", "week"],
    how="left",
    suffixes=("", "_g")
)

# 2) GDELT sayısal kolonlar NaN ise 0 ile doldur
for col in ["protest_count", "military_count",
            "total_events", "goldstein_sum",
            "goldstein_mean", "avg_tone_mean"]:
    if col in panel.columns:
        # int kolonları int, float kolonları float olarak doldur
        if panel[col].dtype.kind in "iu":  # int/unsigned
            panel[col] = panel[col].fillna(0).astype(int)
        else:
            panel[col] = panel[col].fillna(0.0)

# 3) Zaman sırası
panel = panel.sort_values(["dyad", "year", "week"]).reset_index(drop=True)

# 4) Bir sonraki hafta hedefleri (lead)
panel["protest_next_week"] = (
    panel.groupby("dyad")["protest_count"].shift(-1)
)
panel["military_next_week"] = (
    panel.groupby("dyad")["military_count"].shift(-1)
)

# Son haftalar NaN → 0
panel["protest_next_week"] = panel["protest_next_week"].fillna(0).astype(int)
panel["military_next_week"] = panel["military_next_week"].fillna(0).astype(int)

# 5) Kaydet
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
panel.to_csv(OUT_PATH, index=False)

print("Saved panel with", len(panel), "rows to", OUT_PATH)
print("Dyad counts in panel:")
print(panel["dyad"].value_counts())

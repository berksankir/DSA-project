import pandas as pd
from pathlib import Path

# Girdi / çıktı yolları
IN_PATH = Path("data_raw/gdelt/gdelt_events_us_rus_ukr_2019_2024.csv")
OUT_PATH = Path("data_processed/gdelt_dyad_week_2019_2024.csv")

print("Reading", IN_PATH)
df = pd.read_csv(IN_PATH)

# 1) Gerekli kolonlar
needed_cols = [
    "SQLDATE",
    "Actor1CountryCode",
    "Actor2CountryCode",
    "EventCode",
    "EventRootCode",
    "ActionGeo_CountryCode",
    "QuadClass",
    "GoldsteinScale",
    "AvgTone",
]
missing = [c for c in needed_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in input CSV: {missing}")

# 2) SQLDATE -> datetime
df["SQLDATE"] = pd.to_datetime(
    df["SQLDATE"].astype(str), format="%Y%m%d", errors="coerce"
)
df = df.dropna(subset=["SQLDATE"])

# 3) 2019–2024 filtresi (zaten öyle çekmiş olsan da dursun)
mask = (df["SQLDATE"] >= "2019-01-01") & (df["SQLDATE"] <= "2024-12-31")
df = df[mask].copy()
print("After date filter:", len(df), "rows")

# 4) Dyad atama: USA_RUS / USA_UKR / RUS_UKR
FOCUS = {"USA", "RUS", "UKR"}

def assign_dyad(a1, a2):
    if pd.isna(a1) or pd.isna(a2):
        return None
    a1 = str(a1).upper()
    a2 = str(a2).upper()
    # İki aktör de odak ülkelerden olmalı
    if a1 not in FOCUS or a2 not in FOCUS:
        return None
    if a1 == a2:
        return None
    # Alfabetik sıralı dyad ismi: RUS_USA, RUS_UKR, UKR_USA
    return "_".join(sorted([a1, a2]))


df["dyad"] = df.apply(
    lambda row: assign_dyad(row["Actor1CountryCode"], row["Actor2CountryCode"]),
    axis=1,
)
df = df[df["dyad"].notna()].copy()
print("After dyad filter:", len(df), "rows")

# 5) ISO year–week
iso = df["SQLDATE"].dt.isocalendar()
df["year"] = iso.year
df["week"] = iso.week

# 6) Protest / military bayrakları
df["EventRootCode"] = df["EventRootCode"].astype(str)
df["is_protest"] = (df["EventRootCode"] == "14").astype(int)

df["QuadClass"] = pd.to_numeric(df["QuadClass"], errors="coerce")
df["is_military"] = (df["QuadClass"] == 4).astype(int)

# 7) Goldstein & AvgTone numerik
df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce")
df["AvgTone"] = pd.to_numeric(df["AvgTone"], errors="coerce")

# 8) Dyad–week seviyesinde aggregate
grouped = (
    df.groupby(["dyad", "year", "week"])
      .agg(
          protest_count=("is_protest", "sum"),
          military_count=("is_military", "sum"),
          total_events=("dyad", "size"),
          goldstein_sum=("GoldsteinScale", "sum"),
          goldstein_mean=("GoldsteinScale", "mean"),
          avg_tone_mean=("AvgTone", "mean"),
      )
      .reset_index()
)

# 9) Kaydet
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
grouped.to_csv(OUT_PATH, index=False)

print("Saved", len(grouped), "rows to", OUT_PATH)

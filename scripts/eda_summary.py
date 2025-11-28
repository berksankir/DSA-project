import pandas as pd
from pathlib import Path

PANEL_PATH = Path("data_processed/panel_diplomacy_gdelt_week_2019_2024.csv")

print("Reading panel from:", PANEL_PATH)
df = pd.read_csv(PANEL_PATH)

# -------------------------
# 1) Temel kontroller
# -------------------------
print("\n=== Columns ===")
print(df.columns.tolist())

print("\n=== Dyad list ===")
print(df["dyad"].value_counts())

print("\n=== Year range ===")
print(df["year"].min(), "→", df["year"].max())

# -------------------------
# 2) Dyad bazında olay özetleri
# -------------------------

print("\n=== Dyad-level TOTAL protest & military counts (current week) ===")
print(
    df.groupby("dyad")[["protest_count", "military_count"]].sum()
)

print("\n=== Dyad-level TOTAL protest & military counts (next week targets) ===")
print(
    df.groupby("dyad")[["protest_next_week", "military_next_week"]].sum()
)

print("\n=== Dyad-level MEAN of next-week targets (probability-ish) ===")
print(
    df.groupby("dyad")[["protest_next_week", "military_next_week"]].mean()
)

# Kaç hafta en az 1 event olmuş?
def event_rate(col):
    return (df[col] > 0).groupby(df["dyad"]).mean()

print("\n=== Share of weeks with >=1 protest next week (by dyad) ===")
print(event_rate("protest_next_week"))

print("\n=== Share of weeks with >=1 military event next week (by dyad) ===")
print(event_rate("military_next_week"))

# -------------------------
# 3) Pre-war (2019–2021) vs war (2022–2024) karşılaştırma
# -------------------------
pre = df[df["year"] < 2022].copy()
war = df[df["year"] >= 2022].copy()

print("\n=== Pre-war vs War period: number of weeks ===")
print("Pre-war weeks:", len(pre))
print("War weeks   :", len(war))

print("\n=== Pre-war vs War: mean next-week military & protest counts ===")
print("Pre-war military_next_week mean:", pre["military_next_week"].mean())
print("War     military_next_week mean:", war["military_next_week"].mean())
print("Pre-war protest_next_week mean :", pre["protest_next_week"].mean())
print("War     protest_next_week mean :", war["protest_next_week"].mean())

print("\n=== Pre-war vs War: share of weeks with >=1 military next week ===")
print("Pre-war:", (pre["military_next_week"] > 0).mean())
print("War    :", (war["military_next_week"] > 0).mean())

print("\n=== Pre-war vs War: share of weeks with >=1 protest next week ===")
print("Pre-war:", (pre["protest_next_week"] > 0).mean())
print("War    :", (war["protest_next_week"] > 0).mean())

# -------------------------
# 4) Binary hedefler (isteğe bağlı ama korelasyon için faydalı)
# -------------------------
df["has_protest_next_week"] = (df["protest_next_week"] > 0).astype(int)
df["has_military_next_week"] = (df["military_next_week"] > 0).astype(int)

# -------------------------
# 5) Söylem feature'ları vs outcome korelasyonları
# -------------------------

lex_cols = [
    "war_ratio",
    "peace_ratio",
    "security_frame_ratio",
    "economy_frame_ratio",
    "humanrights_frame_ratio",
    "support_ratio",
    "condemn_ratio",
    "tone_support_score",
]

# 5.1. Sürekli hedef (count) ile korelasyon
corr_cols_mil = lex_cols + ["military_next_week"]
corr_cols_pro = lex_cols + ["protest_next_week"]

print("\n=== Correlation: lexicon features vs military_next_week ===")
print(df[corr_cols_mil].corr()["military_next_week"])

print("\n=== Correlation: lexicon features vs protest_next_week ===")
print(df[corr_cols_pro].corr()["protest_next_week"])

# 5.2. Binary hedef ile korelasyon (sadece fikir vermek için)
corr_cols_mil_bin = lex_cols + ["has_military_next_week"]
corr_cols_pro_bin = lex_cols + ["has_protest_next_week"]

print("\n=== Correlation: lexicon features vs has_military_next_week (binary) ===")
print(df[corr_cols_mil_bin].corr()["has_military_next_week"])

print("\n=== Correlation: lexicon features vs has_protest_next_week (binary) ===")
print(df[corr_cols_pro_bin].corr()["has_protest_next_week"])

print("\n=== EDA summary script finished ===")

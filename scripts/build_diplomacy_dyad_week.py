import ast
from pathlib import Path

import pandas as pd

# diplomacy_flat dosyasının yolu
FLAT_PATH = Path("data_processed/diplomacy_flat_2019_2024.csv")

OUT_PATH = Path("data_processed/diplomacy_dyad_week_2019_2024.csv")

FOCUS = {"USA", "RUS", "UKR"}

# Country name -> ISO3 map
NAME_TO_ISO = {
    "Russia": "RUS",
    "Russian Federation": "RUS",
    "Ukraine": "UKR",
    "United States": "USA",
    "the United States": "USA",
    "United States of America": "USA",
    "U.S.": "USA",
    "U.S": "USA",
    "US": "USA",
    "U.S.A.": "USA",
    "UNITED STATES": "USA",
}


def parse_mentioned(x):
    """
    diplomacy_flat'teki mentioned_countries kolonunu listeye çevir.
    Örn:
      ["United States", "Russia"]
    veya
      '["United States", "Russia"]'
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        try:
            val = ast.literal_eval(x)
            if isinstance(val, list):
                return val
            return []
        except Exception:
            return []
    return []


def make_dyad(a, b):
    """İki ülkeyi alfabetik sıralayıp dyad ismi yap: RUS_UKR, RUS_USA, UKR_USA."""
    s = sorted([a, b])
    return f"{s[0]}_{s[1]}"


def main():
    print("Reading flat diplomacy file:", FLAT_PATH)
    df = pd.read_csv(FLAT_PATH)

    print("Columns:", df.columns.tolist())

    needed = {"origin_country", "date", "content", "mentioned_countries"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in diplomacy_flat: {missing}")

    # Origin ülke
    df["origin_country"] = df["origin_country"].astype(str).str.upper()
    df = df[df["origin_country"].isin(FOCUS)].copy()
    print("After origin filter:", len(df), "rows")

    # Tarih filtresi
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    mask = (df["date"] >= "2019-01-01") & (df["date"] <= "2024-12-31")
    df = df[mask].copy()
    print("After date filter:", len(df), "rows")

    # mentioned_countries -> liste
    df["mentioned_list"] = df["mentioned_countries"].apply(parse_mentioned)

    # isimleri ISO3'e çevir
    def to_iso_list(names):
        out = set()
        for name in names:
            iso = NAME_TO_ISO.get(str(name))
            if iso is not None:
                out.add(iso)
        return list(out)

    df["mentioned_iso"] = df["mentioned_list"].apply(to_iso_list)

    # explode
    df = df.explode("mentioned_iso")
    df = df.dropna(subset=["mentioned_iso"])
    df["mentioned_iso"] = df["mentioned_iso"].astype(str).str.upper()

    # mentioned da FOCUS içinde olsun
    df = df[df["mentioned_iso"].isin(FOCUS)].copy()

    # self-dyad at
    df = df[df["origin_country"] != df["mentioned_iso"]].copy()
    print("After mentioned filter:", len(df), "rows")

    # Dyad
    df["dyad"] = df.apply(
        lambda r: make_dyad(r["origin_country"], r["mentioned_iso"]),
        axis=1,
    )

    print("\nDyad value_counts:")
    print(df["dyad"].value_counts())

    # ISO year–week
    iso = df["date"].dt.isocalendar()
    df["year"] = iso.year
    df["week"] = iso.week

    # Dyad-week bazında content + belge sayısı
    grouped = (
        df.groupby(["dyad", "year", "week"])
          .agg(
              content_concat=("content", lambda xs: " ".join(str(x) for x in xs)),
              num_docs=("content", "count"),
          )
          .reset_index()
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print("\nSaved", len(grouped), "rows to", OUT_PATH)


if __name__ == "__main__":
    main()

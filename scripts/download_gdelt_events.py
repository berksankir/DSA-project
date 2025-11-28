import gdelt
import pandas as pd
from pathlib import Path


def filter_chunk(df: pd.DataFrame, focus) -> pd.DataFrame:
    """GDELT chunk'ından sadece işimize yarayan kolonları ve ülkeleri bırak."""
    needed_cols = [
        "SQLDATE",
        "Actor1CountryCode",
        "Actor2CountryCode",
        "EventCode",
        "EventRootCode",
        "ActionGeo_CountryCode",
    ]

    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in GDELT chunk: {missing}")

    df = df[needed_cols].copy()

    mask = df["Actor1CountryCode"].isin(focus) | df["Actor2CountryCode"].isin(focus)
    df = df[mask].copy()
    return df


def main():
    print(">>> GDELT download script started")

    # Çıktı dosyası
    out_dir = Path("data_raw/gdelt")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "events_us_rus_ukr_2019_2024.csv"
    print("Output will be written to:", out_path)

    # GDELT 2.0 client
    gd = gdelt.gdelt(version=2)

    # İlgilendiğimiz ülkeler
    FOCUS = {"USA", "RUS", "UKR"}

    # Çekmek istediğimiz dönemler
    periods = [
        ("2019 01 01", "2019 06 30"),
        ("2019 07 01", "2019 12 31"),
        ("2020 01 01", "2020 06 30"),
        ("2020 07 01", "2020 12 31"),
        ("2021 01 01", "2021 06 30"),
        ("2021 07 01", "2021 12 31"),
        ("2022 01 01", "2022 06 30"),
        ("2022 07 01", "2022 12 31"),
        ("2023 01 01", "2023 06 30"),
        ("2023 07 01", "2023 12 31"),
        ("2024 01 01", "2024 06 30"),
        ("2024 07 01", "2024 12 31"),
    ]

    first_chunk = True

    for start, end in periods:
        try:
            print(f"\n>>> Pulling {start} → {end} ...")

            # table='events' : Events 2.0
            chunk = gd.Search([start, end], table="events", coverage=True)

            print("    Raw rows:", len(chunk))

            chunk_filt = filter_chunk(chunk, FOCUS)
            print("    After US/RUS/UKR filter:", len(chunk_filt))

            if len(chunk_filt) == 0:
                print("    Nothing to write for this period.")
                continue

            chunk_filt.to_csv(
                out_path,
                mode="a",
                index=False,
                header=first_chunk,
            )
            first_chunk = False
            print("    Written to CSV.")
        except Exception as e:
            print(f"!!! ERROR for period {start} → {end}: {repr(e)}")

    print("\n>>> Done. If no errors above, output should be at:", out_path)


if __name__ == "__main__":
    main()

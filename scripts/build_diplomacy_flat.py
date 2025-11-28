import json
import pandas as pd
from pathlib import Path

base = Path("data_raw/globaldiplomacy")   # gerekirse yolu değiştir
base.mkdir(parents=True, exist_ok=True)

countries = ["USA", "RUS", "UKR"]
records = []

for origin in countries:
    for source_type in ["exec", "mofa"]:
        # USA_exec, USA_exec_1, USA_exec_2, ...
        pattern = f"{origin}_{source_type}*"
        for folder in base.glob(pattern):
            news_path = folder / "news.jsonl"
            if not news_path.exists():
                continue

            print(f"Reading {news_path}")
            with news_path.open("r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)

                    # Tarih ve içerik
                    date_str = obj.get("date")
                    content = obj.get("content", "")

                    # Metinde geçen ülkeler
                    entities = obj.get("entities", {})
                    mentioned_countries = entities.get("countries", [])

                    records.append({
                        "origin_country": origin,
                        "source_type": source_type,
                        "date": date_str,
                        "content": content,
                        "mentioned_countries": mentioned_countries,
                    })

# DataFrame'e çevir
df = pd.DataFrame(records)

# Tarihi datetime'a çevir
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 2019–2024 filtresi
mask = (df["date"] >= "2019-01-01") & (df["date"] <= "2024-12-31")
df = df[mask].dropna(subset=["date"])

# İsteğe bağlı: sadece İngilizce
# df = df[df["lang"] == "en"]  # 'lang' kolonunu eklemek istersen yukarıdan çekmen gerekir

# Kaydet
out_path = Path("data_processed")
out_path.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path / "diplomacy_flat_2019_2024.csv", index=False)

print("Saved", len(df), "rows to data_processed/diplomacy_flat_2019_2024.csv")

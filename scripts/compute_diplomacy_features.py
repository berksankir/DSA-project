import json
import re
import pandas as pd
from pathlib import Path

# -------------------------
# DOSYALAR
# -------------------------

IN_PATH  = Path("data_processed/diplomacy_dyad_week_2019_2024.csv")
OUT_PATH = Path("data_processed/diplomacy_dyad_week_features_2019_2024.csv")
LEX_PATH = Path("lexicons/diplomacy_tone_lexicon.json")   
# -------------------------
# LEXICON'U YÜKLE
# -------------------------

with LEX_PATH.open("r", encoding="utf-8") as f:
    raw_lex = json.load(f)

# Beklenen key'ler:
# war_escalation, peace_deescalation, security_frame,
# economy_sanctions_frame, human_rights_frame,
# support_stance, condemnation_stance

categories = [
    "war_escalation",
    "peace_deescalation",
    "security_frame",
    "economy_sanctions_frame",
    "human_rights_frame",
    "support_stance",
    "condemnation_stance",
]

# Tek kelimeli terimler ile çok kelimeli ifadeleri ayıralım
lex_tokens = {}   # {category: set(single-word-terms)}
lex_phrases = {}  # {category: list(multi-word-phrases)}

for cat in categories:
    items = raw_lex.get(cat, [])
    token_set = set()
    phrase_list = []
    for it in items:
        if not isinstance(it, str):
            continue
        it = it.strip().lower()
        if not it:
            continue
        if " " in it:
            phrase_list.append(it)    # multi-word ifade
        else:
            token_set.add(it)         # tek kelime
    lex_tokens[cat] = token_set
    lex_phrases[cat] = phrase_list

# -------------------------
# METİN ÖN İŞLEME
# -------------------------

# Basit tokenizer: harf dışı her şeyi boşluk yap
token_pattern = re.compile(r"[^a-z]+")

def text_to_tokens(text: str):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = token_pattern.sub(" ", text)
    tokens = text.split()
    return tokens

# Çok kelimeli ifadeleri aramak için clean edilmiş küçük harfli string tutacağız.
# Phrase'ler için kaba bir yaklaşım: her phrase'i içeriyorsa 1 say (kaç defa geçtiğine bakmadan).
def count_phrases(text_lower: str, phrases):
    count = 0
    for ph in phrases:
        if ph in text_lower:
            count += 1
    return count

# -------------------------
# FEATURE HESAPLAMA
# -------------------------

def compute_features(text: str):
    tokens = text_to_tokens(text)
    if not tokens:
        return {
            "war_ratio": 0.0,
            "peace_ratio": 0.0,
            "security_frame_ratio": 0.0,
            "economy_frame_ratio": 0.0,
            "humanrights_frame_ratio": 0.0,
            "support_ratio": 0.0,
            "condemn_ratio": 0.0,
            "tone_support_score": 0.0,   # support - condemn normalizasyonu
        }

    total = len(tokens)
    text_lower = " ".join(tokens)  # phrase aramak için

    # Her kategori için token ve phrase sayıları
    counts = {}

    for cat in categories:
        token_set = lex_tokens[cat]
        phrase_list = lex_phrases[cat]

        # tek kelimeli terimler
        c_tokens = sum(1 for t in tokens if t in token_set)

        # çok kelimeli ifadeler (basit: varsa 1 sayıyoruz)
        c_phrases = count_phrases(text_lower, phrase_list)

        counts[cat] = c_tokens + c_phrases

    # Oranlar
    war_ratio   = counts["war_escalation"] / total
    peace_ratio = counts["peace_deescalation"] / total
    sec_ratio   = counts["security_frame"] / total
    eco_ratio   = counts["economy_sanctions_frame"] / total
    hr_ratio    = counts["human_rights_frame"] / total
    supp_ratio  = counts["support_stance"] / total
    cond_ratio  = counts["condemnation_stance"] / total

    # "Tone" benzeri bir destek skoru: support - condemn
    tone_support = (counts["support_stance"] - counts["condemnation_stance"]) / total

    return {
        "war_ratio": war_ratio,
        "peace_ratio": peace_ratio,
        "security_frame_ratio": sec_ratio,
        "economy_frame_ratio": eco_ratio,
        "humanrights_frame_ratio": hr_ratio,
        "support_ratio": supp_ratio,
        "condemn_ratio": cond_ratio,
        "tone_support_score": tone_support,
    }

# -------------------------
# DATAFRAME'E UYGULA
# -------------------------

df = pd.read_csv(IN_PATH)

features = df["content_concat"].apply(compute_features).apply(pd.Series)
df_feat = pd.concat([df, features], axis=1)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_feat.to_csv(OUT_PATH, index=False)

print("Saved", len(df_feat), "rows to", OUT_PATH)

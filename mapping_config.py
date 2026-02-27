"""Configuration for audio-text mapping."""

# Base URLs for the kol-yid platform
BASE_URLS = [
    "https://kol-yid.on-forge.com",
    "https://kolyid.dynamiq.dev",
]

# Eval set -- these audio-transcript pairs are EXCLUDED (reserved for training data)
EVAL_SET = [
    {"audio_id": 28, "transcript_id": 21, "title": "5711-tamuz-12-sicha"},
    {"audio_id": 30, "transcript_id": 23, "title": "5715-Tamuz 13d Sicha"},
    {"audio_id": 80, "transcript_id": None, "title": "5741-Nissan 11 Mamar"},
    {"audio_id": 1,  "transcript_id": None, "title": "5711 10 Shevat Maamar"},
    {"audio_id": 10, "transcript_id": None, "title": "5742 19 Kislev Sicha"},
]

EVAL_AUDIO_IDS = {item["audio_id"] for item in EVAL_SET}

# Target: 50 hours of audio distributed across all available years and months
TARGET_HOURS = 50

# Hebrew month order for sorting and distribution
HEBREW_MONTHS = [
    "Tishrei", "Cheshvan", "Kislev", "Teves", "Shevat", "Adar",
    "Adar I", "Adar II", "Nissan", "Iyar", "Sivan",
    "Tamuz", "Av", "Elul",
]

# Title parsing patterns
MONTH_ALIASES = {
    "tishrei": "Tishrei", "tishri": "Tishrei",
    "cheshvan": "Cheshvan", "marcheshvan": "Cheshvan",
    "kislev": "Kislev",
    "teves": "Teves", "tevet": "Teves",
    "shevat": "Shevat", "shvat": "Shevat",
    "adar": "Adar", "adar i": "Adar I", "adar ii": "Adar II",
    "adar1": "Adar I", "adar2": "Adar II",
    "nissan": "Nissan", "nisan": "Nissan",
    "iyar": "Iyar",
    "sivan": "Sivan",
    "tamuz": "Tamuz", "tammuz": "Tamuz",
    "av": "Av", "menachem av": "Av",
    "elul": "Elul",
}

TYPE_KEYWORDS = {
    "sicha": "sicha",
    "sichos": "sicha",
    "maamar": "maamar",
    "mamar": "maamar",
    "maamorim": "maamar",
    "farbrengen": "farbrengen",
}

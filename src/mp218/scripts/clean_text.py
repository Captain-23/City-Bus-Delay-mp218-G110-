import pandas as pd

# every raw spelling we accept, mapped to the real level
WEATHER_MAP = {"fog": "fog", "rain": "rain", "rai": "rain",
               "clear": "clear", "cle": "clear"}

def normalise_weather(s: pd.Series) -> pd.Series:
    # strip spaces, lowercase, remove periods, then map to the real level
    return s.str.strip().str.lower().str.replace(".", "", regex=False).map(WEATHER_MAP)

def spelling_variants(train: pd.DataFrame):
    raw_levels = train["weather"].nunique()
    clean = normalise_weather(train["weather"])
    unmapped = int(clean.isna().sum())   # must be 0, or a variant is missing from the map
    return raw_levels, clean.nunique(), unmapped
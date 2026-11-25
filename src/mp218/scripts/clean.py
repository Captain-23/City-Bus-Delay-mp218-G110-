import numpy as np
import pandas as pd

from mp218.scripts.clean_text import normalise_weather

TARGET = "delay_min"
NUMERIC = ["route_km", "stops", "passengers", "hour", "temp_c", "driver_exp_yr"]
CATEGORICAL = ["weather", "depot_zone"]
FEATURES = NUMERIC + CATEGORICAL
TEMP_SENTINEL = 999  # Stage 1 finding: 999 typed instead of a blank


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()          # 51 extra copies
    df.loc[df["temp_c"] == TEMP_SENTINEL, "temp_c"] = np.nan
    df["weather"] = normalise_weather(df["weather"])
    return df
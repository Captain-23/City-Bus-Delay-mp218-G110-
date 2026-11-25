import numpy as np
import pandas as pd

from mp218.scripts.clean import clean


def make_df(rows):
    base = {"record_id": "R1", "date": "2026-01-01", "route_km": 10.0, "stops": 5,
            "passengers": 20, "hour": 8, "temp_c": 25.0, "driver_exp_yr": 3,
            "weather": "fog", "depot_zone": "North", "delay_min": 12.0}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_duplicates_removed():
    df = make_df([{}, {}])
    assert len(clean(df)) == 1


def test_temp_sentinel_becomes_nan():
    df = make_df([{"temp_c": 999.0}])
    assert np.isnan(clean(df)["temp_c"].iloc[0])


def test_weather_variants_mapped():
    df = make_df([
        {"record_id": "R1", "weather": " FOG."},
        {"record_id": "R2", "weather": "rai."},
        {"record_id": "R3", "weather": "cle."},
    ])
    assert sorted(clean(df)["weather"].tolist()) == ["clear", "fog", "rain"]
import pandas as pd

TARGET = "delay_min"

def leaking_column(train: pd.DataFrame, holdout: pd.DataFrame):
    # columns in train that are missing from the holdout file (ignoring the target)
    missing = sorted(set(train.columns) - set(holdout.columns) - {TARGET})
    # the numeric column most strongly correlated with the target
    corr = train.select_dtypes("number").corr()[TARGET].drop(TARGET).abs()
    return missing, corr.idxmax(), round(corr.max(), 3)

def unit_mixup(train: pd.DataFrame):
    # medians per depot; a zone off by 10x to 1000x stands out
    return train.groupby("depot_zone")[["route_km", "driver_exp_yr"]].median()

def sentinel_values(train: pd.DataFrame):
    num = train.select_dtypes("number").drop(columns=[TARGET])
    # the three most common values in each numeric column
    return {c: num[c].value_counts().head(3) for c in num.columns}
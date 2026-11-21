import pandas as pd

NUMERIC = ["route_km", "stops", "passengers", "hour", "temp_c", "driver_exp_yr"]

def duplicate_rows(train: pd.DataFrame) -> int:
    # number of EXTRA copies, i.e. rows you would delete
    return int(train.duplicated().sum())

def stuck_sensor(train: pd.DataFrame):
    # remove duplicates, sort by date, then find the longest run of one value per depot
    d = train.drop_duplicates().sort_values(["date", "record_id"])
    results = []
    for zone, g in d.groupby("depot_zone"):
        for c in NUMERIC:
            run = (g[c] != g[c].shift()).cumsum()
            r = g.groupby(run).agg(value=(c, "first"), n=(c, "size"),
                                   start=("date", "first"), end=("date", "last"))
            top = r.loc[r["n"].idxmax()]
            results.append((zone, c, top["value"], int(top["n"]), top["start"], top["end"]))
    return sorted(results, key=lambda x: -x[3])[:5]

def target_typos(train: pd.DataFrame):
    # the 20 largest targets; look for a small cluster far above the rest
    return train["delay_min"].sort_values(ascending=False).head(20).tolist()
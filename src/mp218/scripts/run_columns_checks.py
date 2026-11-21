import sys
sys.path.insert(0, "src")

import pandas as pd
from mp218.scripts.checks_columns import leaking_column, unit_mixup, sentinel_values

train = pd.read_csv("data/raw/train.csv")
hold = pd.read_csv("data/raw/holdout_features.csv")

print(leaking_column(train, hold))
print(unit_mixup(train))

for c, vc in sentinel_values(train).items():
    print(c, vc.to_dict())
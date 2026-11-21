import pandas as pd
from mp218.scripts.checks_row import duplicate_rows, stuck_sensor, target_typos

train = pd.read_csv("data/raw/train.csv")
print("duplicates:", duplicate_rows(train))
for row in stuck_sensor(train):
    print(row)
print(target_typos(train))  
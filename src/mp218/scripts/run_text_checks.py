import pandas as pd
from mp218.scripts.clean_text import spelling_variants

train = pd.read_csv("data/raw/train.csv")
print(spelling_variants(train))

later = pd.read_csv("data/raw/later.csv")
print(later["weather"].unique())
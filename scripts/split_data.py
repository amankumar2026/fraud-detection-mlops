"""Time-based train/val/test split -- NOT a random shuffle split. Fraud
detection is a forecasting problem in practice (train on the past, score
transactions that happen next), so splitting by Time mimics real deployment
and avoids leakage a random split wouldn't necessarily catch.
"""

from pathlib import Path

import pandas as pd

df = pd.read_csv("data/processed/creditcard.csv")
df = df.sort_values("Time").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
    path = Path(f"data/processed/{name}.csv")
    split.to_csv(path, index=False)
    n_fraud = split["Class"].sum()
    print(f"{name}: {len(split):,} rows, Time range [{split['Time'].min():.0f}, "
          f"{split['Time'].max():.0f}], {n_fraud} frauds ({n_fraud/len(split):.4%})")

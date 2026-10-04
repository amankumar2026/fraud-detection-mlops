"""Parses the real ULB Credit Card Fraud dataset (ARFF format, fetched via
OpenML's API, dataset id 1597) into a clean CSV, and reports real stats.
"""

from pathlib import Path

import pandas as pd
from scipy.io import arff

RAW_PATH = Path("data/raw/creditcard.arff")
OUT_PATH = Path("data/processed/creditcard.csv")


def main():
    print(f"Loading {RAW_PATH} ...")
    data, meta = arff.loadarff(RAW_PATH)
    df = pd.DataFrame(data)
    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")

    # ARFF nominal/string columns load as bytes in scipy; Class is the target
    df["Class"] = df["Class"].apply(lambda x: int(x.decode() if isinstance(x, bytes) else x))

    print(f"\nClass distribution:\n{df['Class'].value_counts()}")
    fraud_rate = df["Class"].mean()
    print(f"Fraud rate: {fraud_rate:.4%}")

    print(f"\nMissing values: {df.isnull().sum().sum()}")
    n_dupes = df.duplicated().sum()
    print(f"Duplicate rows: {n_dupes}")
    if n_dupes > 0:
        # A known characteristic of this exact dataset. Dropped because exact
        # duplicates across all 31 continuous features are statistically
        # implausible to occur independently -- almost certainly double-logged
        # transactions, not genuinely distinct events -- and leaving them in
        # risks a duplicate pair straddling the train/test time-split boundary,
        # which would leak a test-set label into training.
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  dropped {before - len(df)} duplicate rows -> {len(df):,} remaining")

    print(f"\nAmount stats:\n{df['Amount'].describe()}")
    print(f"\nTime stats (seconds from first transaction):\n{df['Time'].describe()}")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {df.shape} to {OUT_PATH}")


if __name__ == "__main__":
    main()

"""Feature engineering for the fraud model.

V1-V28 are already PCA components (roughly standardized by construction), so
they're used as-is. HourOfDay is derived from Time -- fraud/legitimate
transactions have different time-of-day patterns in this well-studied
dataset, a real signal, not a speculative addition. Amount scaling
(RobustScaler, since fraud amounts include real outliers a mean/std scaler
would be distorted by) is handled inside the sklearn Pipeline built in
scripts/train.py, not here -- bundling the scaler INTO the logged model
artifact means serving only has to load one object and call .predict_proba
on a dataframe with the right raw columns, instead of separately managing a
scaler file that could silently drift out of sync with the model.
"""

import pandas as pd

PCA_COLS = [f"V{i}" for i in range(1, 29)]
MODEL_INPUT_COLS = PCA_COLS + ["HourOfDay", "Amount"]


def add_hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HourOfDay"] = (df["Time"] % 86400) // 3600
    return df


def prepare_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature prep shared by training and serving -- no fitted
    state here (that's inside the Pipeline), so there's no scaler-leakage
    risk from calling this on a single incoming prediction request."""
    df = add_hour_of_day(df)
    return df[MODEL_INPUT_COLS]

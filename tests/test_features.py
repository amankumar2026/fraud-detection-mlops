import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import MODEL_INPUT_COLS, add_hour_of_day, prepare_model_input


def _sample_row(**overrides):
    row = {"Time": 3661.0, "Amount": 50.0}
    for i in range(1, 29):
        row[f"V{i}"] = 0.0
    row.update(overrides)
    return pd.DataFrame([row])


def test_add_hour_of_day_wraps_correctly():
    df = _sample_row(Time=3661.0)  # 1 hour, 1 min, 1 sec -> hour 1
    out = add_hour_of_day(df)
    assert out["HourOfDay"].iloc[0] == 1


def test_add_hour_of_day_wraps_past_one_day():
    df = _sample_row(Time=86400.0 + 7200.0)  # 1 day + 2 hours -> hour 2
    out = add_hour_of_day(df)
    assert out["HourOfDay"].iloc[0] == 2


def test_prepare_model_input_has_expected_columns():
    df = _sample_row()
    X = prepare_model_input(df)
    assert list(X.columns) == MODEL_INPUT_COLS
    assert len(X) == 1


def test_prepare_model_input_does_not_mutate_original():
    df = _sample_row()
    original_cols = list(df.columns)
    prepare_model_input(df)
    assert list(df.columns) == original_cols  # HourOfDay not added to caller's df

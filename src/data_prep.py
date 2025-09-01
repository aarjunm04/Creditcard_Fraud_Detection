# src/data_prep.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic sanity checks
    assert "Class" in df.columns, "Target column 'Class' not found."
    return df


def build_preprocessor() -> ColumnTransformer:
    # Scale only Time & Amount; leave PCA components as-is
    time_amount = ["Time", "Amount"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale_num", StandardScaler(), time_amount),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    return preprocessor


def get_feature_target(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    return X, y

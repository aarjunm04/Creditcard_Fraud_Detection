# src/data_prep.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src import logger


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the raw credit card fraud dataset from CSV.

    Args:
        csv_path (str | Path): Path to raw CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe with a required 'Class' column.

    Raises:
        AssertionError: If 'Class' column is missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found at {csv_path}.")
    df = pd.read_csv(csv_path)
    assert "Class" in df.columns, "Target column 'Class' not found."
    logger.info(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing transformer:
    - Scales only 'Time' and 'Amount' columns
    - Leaves PCA features as-is (already standardized in dataset)

    Returns:
        ColumnTransformer: Configured preprocessor.
    """
    time_amount = ["Time", "Amount"]
    preprocessor = ColumnTransformer(
        transformers=[("scale_num", StandardScaler(), time_amount)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    logger.info("🔧 Preprocessor built: scaling ['Time', 'Amount']")
    return preprocessor


def get_feature_target(df: pd.DataFrame):
    """
    Split features (X) and target (y).

    Args:
        df (pd.DataFrame): Input dataframe with 'Class' column.

    Returns:
        X (pd.DataFrame): Features without target.
        y (pd.Series): Target labels (0 = non-fraud, 1 = fraud).
    """
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    logger.info(f"📊 Split into X={X.shape}, y={y.shape}")
    return X, y

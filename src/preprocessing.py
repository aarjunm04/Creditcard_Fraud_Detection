# src/preprocessing.py
"""
Reusable preprocessing utilities for Credit Card Fraud project.
Keeps logic separate from data_prep (raw load) and train_model (pipeline).
"""

import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def build_preprocessor() -> ColumnTransformer:
    """
    Preprocess data:
    - Scale 'Time' and 'Amount'
    - Leave PCA features unchanged
    """
    time_amount = ["Time", "Amount"]
    preprocessor = ColumnTransformer(
        transformers=[("scale_num", StandardScaler(), time_amount)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return preprocessor


def save_preprocessor(preprocessor, path: str | Path):
    """Persist preprocessor to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: str | Path):
    """Reload preprocessor from disk."""
    return joblib.load(path)
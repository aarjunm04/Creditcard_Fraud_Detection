# gan/ctgan_wrapper.py
"""
CTGAN wrapper for synthetic fraud data generation.

Provides:
- train_ctgan: fit CTGAN on real data
- sample_synthetic: generate synthetic transactions
- save_model / load_model: persist CTGAN state
"""

from pathlib import Path
import joblib
import pandas as pd
from ctgan import CTGAN

try:
    from src import logger
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ctgan_wrapper")


def train_ctgan(df: pd.DataFrame, discrete_columns: list[str], epochs: int = 10) -> CTGAN:
    """
    Train CTGAN on given dataframe.

    Args:
        df (pd.DataFrame): input real dataset
        discrete_columns (list[str]): categorical/discrete column names
        epochs (int): number of training epochs

    Returns:
        trained CTGAN instance
    """
    model = CTGAN(epochs=epochs, batch_size=500, verbose=True)
    logger.info("🚀 Training CTGAN with epochs=%s, discrete=%s", epochs, discrete_columns)
    model.fit(df, discrete_columns)
    logger.info("✅ CTGAN training complete")
    return model


def sample_synthetic(model: CTGAN, n: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic samples using trained CTGAN.
    """
    logger.info("🎲 Sampling %s synthetic rows...", n)
    return model.sample(n)


def save_model(model: CTGAN, path: str | Path):
    """Persist CTGAN using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("💾 CTGAN model saved to %s", path)


def load_model(path: str | Path) -> CTGAN:
    """Reload CTGAN from joblib file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CTGAN model not found: {path}")
    model = joblib.load(path)
    logger.info("📂 CTGAN model loaded from %s", path)
    return model
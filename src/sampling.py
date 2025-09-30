# src/sampling.py
"""
Utility for creating lightweight sample datasets for demos (e.g., GitHub CI, quick tests).
"""

from pathlib import Path

import pandas as pd

from src import logger

# Paths
RAW_DATA_PATH = Path("data/raw/creditcard.csv")
SAMPLE_RAW_PATH = Path("data/raw/sample_raw.csv")
SAMPLE_PROCESSED_PATH = Path("data/processed/sample_processed.csv")


def create_samples(n: int = 500, stratify: bool = True):
    """
    Create lightweight sample datasets for GitHub demo & CI.

    Args:
        n (int): Number of rows to sample.
        stratify (bool): Whether to preserve fraud/non-fraud ratio.
    """
    if not RAW_DATA_PATH.exists():
        logger.error(f"❌ Raw dataset not found at {RAW_DATA_PATH}")
        return

    logger.info(f"📥 Loading dataset from {RAW_DATA_PATH} ...")
    df = pd.read_csv(RAW_DATA_PATH)

    if stratify and "Class" in df.columns:
        # stratified sample (preserves fraud ratio)
        sample_df = (
            df.groupby("Class", group_keys=False)
            .apply(
                lambda x: x.sample(int(len(x) / len(df) * n), random_state=42),
                include_groups=False,
            )
            .reset_index(drop=True)
        )
        logger.info("✂️ Stratified sampling applied.")
    else:
        sample_df = df.sample(n=n, random_state=42)
        logger.info("✂️ Random sampling applied.")

    SAMPLE_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"💾 Saving sample_raw → {SAMPLE_RAW_PATH}")
    sample_df.to_csv(SAMPLE_RAW_PATH, index=False)

    logger.info(f"💾 Saving sample_processed → {SAMPLE_PROCESSED_PATH}")
    sample_df.to_csv(SAMPLE_PROCESSED_PATH, index=False)

    logger.info("✅ Samples created successfully!")


if __name__ == "__main__":
    create_samples()

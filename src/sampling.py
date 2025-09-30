# src/sampling.py
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional

# Default paths (tests may override these at runtime)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH: Optional[Path] = PROJECT_ROOT / "data" / "raw" / "sample_raw.csv"
SAMPLE_RAW_PATH: Path = PROJECT_ROOT / "data" / "raw" / "sample_raw.csv"
SAMPLE_PROCESSED_PATH: Path = PROJECT_ROOT / "data" / "processed" / "sample_processed.csv"


def _find_source_path() -> Optional[Path]:
    """
    Determine the source/raw CSV path to use.
    Priority:
    1) If the module-level RAW_DATA_PATH is set and exists -> use it.
    2) SAMPLE_RAW_PATH if exists.
    3) creditcard.csv in data/raw/
    4) None if nothing found.
    """
    # Respect any runtime override to RAW_DATA_PATH
    global RAW_DATA_PATH, SAMPLE_RAW_PATH
    if RAW_DATA_PATH and Path(RAW_DATA_PATH).exists():
        return Path(RAW_DATA_PATH)

    if SAMPLE_RAW_PATH and Path(SAMPLE_RAW_PATH).exists():
        return Path(SAMPLE_RAW_PATH)

    fallback = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
    if fallback.exists():
        return fallback

    return None


def create_samples(n: int = 100, random_state: int = 42) -> None:
    """
    Create small sample CSVs used by tests/CI. Keeps the 'Class' label column.
    If no raw dataset is found, bootstrap a small synthetic one (safe for CI).
    This function respects runtime overrides of RAW_DATA_PATH, SAMPLE_RAW_PATH,
    and SAMPLE_PROCESSED_PATH (tests set these directly).
    """
    rng = np.random.default_rng(random_state)

    # Resolve final paths (tests may have reassigned these module globals)
    src_path = _find_source_path()

    if src_path is not None:
        df = pd.read_csv(src_path)
    else:
        # Bootstrap tiny synthetic data for CI environments
        cols = ["Time"] + [f"V{i}" for i in range(1, 29 if False else 4)] + ["Amount", "Class"]
        # Here we default to a small set of V1..V3 for speed in CI
        df = pd.DataFrame({
            "Time": rng.integers(0, 1000, size=n),
            "V1": rng.normal(0, 1, size=n),
            "V2": rng.normal(0, 1, size=n),
            "V3": rng.normal(0, 1, size=n),
            "Amount": rng.uniform(1, 500, size=n),
            "Class": rng.integers(0, 2, size=n),
        })

    # Ensure 'Class' column exists
    if "Class" not in df.columns:
        df["Class"] = 0

    # Create a stratified-ish sample preserving class ratios when possible
    if "Class" in df.columns:
        try:
            df_sample = (
                df.groupby("Class", group_keys=False)
                .apply(lambda g: g.sample(max(1, int(len(g) / len(df) * n)), random_state=random_state))
                .reset_index(drop=True)
            )
        except Exception:
            df_sample = df.sample(n=min(n, len(df)), random_state=random_state).reset_index(drop=True)
    else:
        df_sample = df.sample(n=min(n, len(df)), random_state=random_state).reset_index(drop=True)

    # Reorder columns: put 'Class' at the end for consistency with tests
    if "Class" in df_sample.columns:
        cols = [c for c in df_sample.columns if c != "Class"] + ["Class"]
        df_sample = df_sample[cols]

    # Ensure directories exist and write with index=False
    Path(SAMPLE_RAW_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(SAMPLE_PROCESSED_PATH).parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(SAMPLE_RAW_PATH, index=False)
    df_sample.to_csv(SAMPLE_PROCESSED_PATH, index=False)


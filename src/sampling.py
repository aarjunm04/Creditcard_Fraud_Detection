from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "sample_raw.csv"
SAMPLE_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "sample_processed.csv"

def create_samples(n: int = 100, random_state: int = 42) -> None:
    """
    Create small sample CSVs used by tests/CI. Keeps the 'Class' label column
    (if present) and writes with index=False so columns remain exact.
    """
    # read raw (fall back to creditcard.csv if sample not present)
    raw_candidates = [
        PROJECT_ROOT / "data" / "raw" / "sample_raw.csv",
        PROJECT_ROOT / "data" / "raw" / "creditcard.csv",
    ]
    src_path = next((p for p in raw_candidates if p.exists()), None)
    if src_path is None:
        raise FileNotFoundError("No raw dataset found for sampling (data/raw/*.csv).")

    df = pd.read_csv(src_path)

    # If dataset missing 'Class', try to infer or add placeholder (tests expect Class)
    if "Class" not in df.columns:
        # If there's a label column under other names, try common alternatives (none found -> set zeros)
        df["Class"] = 0

    # Stratified-ish sampling: preserve class ratio if possible
    try:
        df_sample = (
            df.groupby("Class", group_keys=False)
            .apply(lambda g: g.sample(max(1, int(len(g) / len(df) * n)), random_state=random_state))
            .reset_index(drop=True)
        )
    except Exception:
        # fallback to simple sample
        df_sample = df.sample(n=min(n, len(df)), random_state=random_state).reset_index(drop=True)

    # Ensure Class column exists and order columns consistently
    cols = list(df_sample.columns)
    if "Class" in cols:
        cols = [c for c in cols if c != "Class"] + ["Class"]  # put Class last (or adjust as you prefer)
        df_sample = df_sample[cols]

    # Create parent dirs if needed and write without index so tests can assert columns list
    SAMPLE_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(SAMPLE_RAW_PATH, index=False)
    df_sample.to_csv(SAMPLE_PROCESSED_PATH, index=False)

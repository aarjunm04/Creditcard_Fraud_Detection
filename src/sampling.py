import pandas as pd
from pathlib import Path

# Paths
RAW_DATA_PATH = Path("data/raw/creditcard.csv")
SAMPLE_RAW_PATH = Path("data/raw/sample_raw.csv")
SAMPLE_PROCESSED_PATH = Path("data/processed/sample_processed.csv")

def create_samples(n=500):
    """Create lightweight sample datasets for GitHub demo."""
    if not RAW_DATA_PATH.exists():
        print(f"❌ Raw dataset not found at {RAW_DATA_PATH}")
        return
    
    print(f"📥 Loading dataset from {RAW_DATA_PATH} ...")
    df = pd.read_csv(RAW_DATA_PATH)

    print(f"✂️ Creating sample of {n} rows ...")
    sample_df = df.sample(n=n, random_state=42)

    print(f"💾 Saving sample_raw -> {SAMPLE_RAW_PATH}")
    sample_df.to_csv(SAMPLE_RAW_PATH, index=False)

    print(f"💾 Saving sample_processed -> {SAMPLE_PROCESSED_PATH}")
    sample_df.to_csv(SAMPLE_PROCESSED_PATH, index=False)

    print("✅ Samples created successfully!")

if __name__ == "__main__":
    create_samples()

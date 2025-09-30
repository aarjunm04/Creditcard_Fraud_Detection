# gan/hf_push.py
"""
Push synthetic datasets or trained GANs to HuggingFace Hub.
"""

from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi, HfFolder, Repository

try:
    from src import logger
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("hf_push")


def push_dataset_to_hub(df: pd.DataFrame, repo_id: str, filename: str = "synthetic.csv", private: bool = True):
    """
    Push synthetic dataset to HuggingFace Hub as CSV.

    Args:
        df: synthetic dataframe
        repo_id: e.g. "username/creditcard-synth"
        filename: name inside repo
        private: whether repo should be private
    """
    api = HfApi()
    token = HfFolder.get_token()
    if not token:
        raise RuntimeError("No HuggingFace token found. Run `huggingface-cli login` first.")

    repo_url = api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    repo = Repository(local_dir=Path("hf_tmp"), clone_from=repo_url, use_auth_token=token)

    df.to_csv(repo.local_dir / filename, index=False)
    repo.push_to_hub(commit_message="Add synthetic dataset")
    logger.info("✅ Synthetic dataset pushed to HF Hub: %s/%s", repo_id, filename)


def push_model_to_hub(model_path: str | Path, repo_id: str, filename: str = "ctgan_model.joblib", private: bool = True):
    """
    Push trained GAN model file to HuggingFace Hub.
    """
    api = HfApi()
    token = HfFolder.get_token()
    if not token:
        raise RuntimeError("No HuggingFace token found. Run `huggingface-cli login` first.")

    repo_url = api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    repo = Repository(local_dir=Path("hf_tmp"), clone_from=repo_url, use_auth_token=token)

    import shutil
    shutil.copy(model_path, repo.local_dir / filename)
    repo.push_to_hub(commit_message="Upload trained CTGAN model")
    logger.info("✅ CTGAN model pushed to HF Hub: %s/%s", repo_id, filename)
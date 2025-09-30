# models/trainer.py
"""
Trainer utilities that orchestrate training for XGBoost and Torch models.

This module connects the model wrappers (models/xgb.py and models/torch_nn.py)
to the project's data pipeline. It provides:
- high-level functions to train XGBoost and Torch models from numpy arrays
- evaluation, saving, and basic reporting
- a small CLI for quick smoke training (not for large-scale runs)

Intended usage (examples):
    from models.trainer import Trainer
    tr = Trainer(artifacts_dir="artifacts")
    tr.train_xgb(X_train, y_train, X_val, y_val, name="xgb_v1")
    tr.train_torch(X_train_np, y_train_np, X_val_np, y_val_np, name="torch_v1")
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Dict, Optional, Tuple, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from models.xgb import build_xgb, train_xgb, evaluate_model_predictions, save_model as save_xgb
from models.torch_nn import build_model, train_nn, save_model as save_torch, evaluate_model

try:
    from src import logger
except Exception:
    import logging

    logger = logging.getLogger("trainer")
    logging.basicConfig(level=logging.INFO)


@dataclass
class TrainerConfig:
    artifacts_dir: Path = Path("artifacts")
    xgb_early_stopping_rounds: Optional[int] = 50
    xgb_eval_kwargs: Dict = None
    torch_epochs: int = 30
    torch_batch_size: int = 512
    torch_lr: float = 1e-3
    torch_weight_decay: float = 1e-5
    device: Optional[str] = None  # "cpu" | "cuda" | None (auto)


class Trainer:
    def __init__(self, cfg: Optional[TrainerConfig] = None):
        self.cfg = cfg or TrainerConfig()
        self.artifacts_dir = Path(self.cfg.artifacts_dir)
        self.models_dir = self.artifacts_dir / "models"
        self.reports_dir = self.artifacts_dir / "reports"
        for d in (self.artifacts_dir, self.models_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # XGBoost training pipeline
    # -------------------------
    def train_xgb(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        name: str = "xgb",
        params: Optional[Dict] = None,
    ) -> Dict:
        """
        Train an XGBoost model using the models.xgb wrapper and save artifacts.

        Args:
            X_train, y_train: training arrays (numpy)
            X_val, y_val: optional validation arrays for early stopping / monitoring
            name: base name used for saving
            params: optional override params for XGB builder

        Returns:
            metrics dict containing evaluation on validation (if provided) or train set.
        """
        t0 = time.time()
        logger.info("Starting XGB training (%s)", name)
        model = build_xgb(params)
        model = train_xgb(
            model,
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            early_stopping_rounds=self.cfg.xgb_early_stopping_rounds,
        )

        # If validation provided evaluate on it, otherwise evaluate on train (not ideal)
        eval_X = X_val if X_val is not None else X_train
        eval_y = y_val if y_val is not None else y_train

        proba = model.predict_proba(eval_X)[:, 1]
        metrics = evaluate_model_predictions(eval_y, proba, threshold=0.5)

        # persist model
        model_path = self.models_dir / f"{name}.joblib"
        save_xgb(model, model_path)

        # save metrics
        metrics_path = self.reports_dir / f"{name}_metrics.json"
        metrics_payload = {"model": name, "framework": "xgboost", "metrics": metrics}
        metrics_payload["train_time_seconds"] = time.time() - t0
        metrics_path.write_text(json.dumps(metrics_payload, indent=2))

        logger.info("XGB training complete (%s). F1=%.4f roc_auc=%.4f", name, metrics["f1"], metrics["roc_auc"])
        return metrics_payload

    # -------------------------
    # Torch training pipeline
    # -------------------------
    def train_torch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        name: str = "torch",
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> Dict:
        """
        Train a torch MLP model and save artifacts and metrics.

        Returns:
            dict with simple metrics evaluated on validation (or train if val not provided).
        """
        t0 = time.time()
        epochs = epochs or self.cfg.torch_epochs
        batch_size = batch_size or self.cfg.torch_batch_size
        lr = lr or self.cfg.torch_lr

        logger.info("Starting Torch training (%s) epochs=%s batch_size=%s lr=%s", name, epochs, batch_size, lr)
        input_dim = int(X_train.shape[1])
        model = build_model(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)

        train_res = train_nn(
            model,
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=self.cfg.torch_weight_decay,
            device=self.cfg.device,
        )

        # Evaluate
        eval_X = X_val if X_val is not None else X_train
        eval_y = y_val if y_val is not None else y_train
        proba, preds = evaluate_model(train_res.model, eval_X, device=self.cfg.device)
        # proba is (n,) numpy
        metrics = {
            "precision": float(np.nan),
            "recall": float(np.nan),
            "f1": float(np.nan),
            "roc_auc": float(np.nan),
        }
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

            metrics["precision"] = float(precision_score(eval_y, preds, zero_division=0))
            metrics["recall"] = float(recall_score(eval_y, preds, zero_division=0))
            metrics["f1"] = float(f1_score(eval_y, preds, zero_division=0))
            metrics["roc_auc"] = float(roc_auc_score(eval_y, proba))
        except Exception as e:
            logger.warning("Could not compute full metrics for Torch: %s", e)

        # Save model (state_dict + metadata)
        model_path = self.models_dir / f"{name}.pt"
        metadata = {"input_dim": input_dim, "hidden_dims": tuple(hidden_dims), "dropout": float(dropout)}
        save_torch(train_res.model, model_path, metadata=metadata)

        # Save metrics & history
        metrics_payload = {"model": name, "framework": "torch", "metrics": metrics}
        metrics_payload["train_time_seconds"] = time.time() - t0
        metrics_payload["history"] = train_res.history
        (self.reports_dir / f"{name}_metrics.json").write_text(json.dumps(metrics_payload, indent=2))

        logger.info("Torch training complete (%s). F1=%.4f roc_auc=%.4f", name, metrics["f1"], metrics["roc_auc"])
        return metrics_payload

    # -------------------------
    # Convenience: train both with a simple split
    # -------------------------
    def train_both_with_split(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
        """
        Convenience method: split dataset and run both XGB and Torch training with same split.
        Saves artifacts under artifacts/models and artifacts/reports.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

        # Train XGB with validation
        xgb_metrics = self.train_xgb(X_train, y_train, X_val=X_test, y_val=y_test, name="xgb_auto")

        # Train Torch with validation
        torch_metrics = self.train_torch(X_train, y_train, X_val=X_test, y_val=y_test, name="torch_auto")

        return {"xgb": xgb_metrics, "torch": torch_metrics}


# -------------------------
# Simple CLI for quick smoke runs
# -------------------------
def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="models.trainer - smoke train CLI")
    p.add_argument("--mode", choices=["xgb", "torch", "both"], default="both")
    p.add_argument("--raw_csv", type=str, default="data/raw/creditcard.csv")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _cli_main():
    args = _parse_args()
    raw = Path(args.raw_csv)
    if not raw.exists():
        logger.error("Raw CSV not found: %s", raw)
        return

    import pandas as pd

    df = pd.read_csv(raw)
    if "Class" not in df.columns:
        logger.error("No 'Class' column found in dataset.")
        return

    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    trainer = Trainer()
    if args.mode == "xgb":
        trainer.train_xgb(*train_test_split(X, y, test_size=args.test_size, random_state=args.seed)[:2], name="xgb_cli")
    elif args.mode == "torch":
        trainer.train_torch(*train_test_split(X, y, test_size=args.test_size, random_state=args.seed)[:2], name="torch_cli")
    else:
        trainer.train_both_with_split(X, y, test_size=args.test_size, seed=args.seed)


if __name__ == "__main__":
    _cli_main()
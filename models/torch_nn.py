# models/torch_nn.py
"""
Torch-based neural network utilities for Credit Card Fraud Detection.

Provides:
- FraudNet: simple fully-connected MLP for tabular binary classification
- build_model: construct model from config
- train_nn: training loop with validation, optional early stopping & scheduler
- evaluate_model: compute probs and basic metrics
- save_model / load_model: persist model state + metadata for inference
- predict_proba / predict: helpers for inference

Designed to plug into the repo's pipeline (expects numpy arrays for X/y).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from src import logger
except Exception:
    import logging

    logger = logging.getLogger("torch_nn")
    logging.basicConfig(level=logging.INFO)


@dataclass
class TrainResult:
    model: nn.Module
    history: Dict[str, list]
    best_epoch: int
    best_val_loss: float
    path: Optional[Path] = None


class FraudNet(nn.Module):
    """A simple feed-forward network for tabular fraud detection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))  # output logits
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # returns logits


def build_model(
    input_dim: int, hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.2
) -> FraudNet:
    """Factory for FraudNet."""
    model = FraudNet(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    logger.info(
        "Built FraudNet: input_dim=%s hidden=%s dropout=%.3f",
        input_dim,
        hidden_dims,
        dropout,
    )
    return model


def _to_tensor(x: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32))


def train_nn(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: Optional[str] = None,
    early_stopping_patience: Optional[int] = 5,
    scheduler_step: Optional[int] = None,
) -> TrainResult:
    """
    Train the neural network.

    Args:
        model: nn.Module (uninitialized weights OK)
        X_train, y_train: numpy arrays
        X_val, y_val: optional validation arrays (if provided, used for early stopping)
        epochs: max epochs
        batch_size: batch size
        lr: learning rate
        weight_decay: L2 regularization
        device: "cuda" | "cpu" | None (auto-detect)
        early_stopping_patience: stop if no val loss improvement for N epochs; None disables
        scheduler_step: if set, use StepLR with this step size (gamma=0.5)

    Returns:
        TrainResult with history and best model (state in memory)
    """
    # device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info("Training on device: %s", device)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if scheduler_step:
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=scheduler_step, gamma=0.5
        )

    loss_fn = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(_to_tensor(X_train), _to_tensor(y_train))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )

    val_loader = None
    if X_val is not None and y_val is not None:
        val_ds = TensorDataset(_to_tensor(X_val), _to_tensor(y_val))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        val_loss = None
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    total_val_loss += loss.item() * xb.size(0)
            val_loss = total_val_loss / len(val_loader.dataset)
            history["val_loss"].append(val_loss)

            logger.info(
                "Epoch %d/%d — train_loss=%.6f val_loss=%.6f",
                epoch,
                epochs,
                train_loss,
                val_loss,
            )

            # early stopping
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if (
                    early_stopping_patience
                    and epochs_no_improve >= early_stopping_patience
                ):
                    logger.info(
                        "Early stopping triggered (patience=%d)",
                        early_stopping_patience,
                    )
                    break
        else:
            logger.info("Epoch %d/%d — train_loss=%.6f", epoch, epochs, train_loss)
            # Save best by train loss if no val provided
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_epoch = epoch
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if scheduler:
            scheduler.step()

    # load best state back
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    logger.info(
        "Training complete. Best epoch=%d best_loss=%.6f", best_epoch, best_val_loss
    )

    return TrainResult(
        model=model, history=history, best_epoch=best_epoch, best_val_loss=best_val_loss
    )


def evaluate_model(
    model: nn.Module, X: np.ndarray, device: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute predicted probabilities and binary predictions for given data.

    Returns:
        (probs, preds) where probs is shape (n,) with values in [0,1], preds is binary 0/1 using 0.5 threshold.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    X_t = _to_tensor(X).to(device)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    return probs, preds


def save_model(model: nn.Module, path: Path, metadata: Optional[Dict] = None):
    """
    Save model state_dict and metadata to disk (torch.save).

    Args:
        model: trained nn.Module
        path: destination file path (.pt recommended)
        metadata: any auxiliary info (input_dim, hidden_dims, scaler path etc.)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    logger.info("Saved Torch model to %s", path)


def load_model(path: Path, device: Optional[str] = None) -> Tuple[nn.Module, Dict]:
    """
    Load model payload saved by save_model.
    Returns (model, metadata). Caller must reconstruct model architecture if needed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    payload = torch.load(path, map_location="cpu")
    metadata = payload.get("metadata", {})
    state_dict = payload["state_dict"]
    # Basic: if metadata contains architecture info, rebuild model
    if "input_dim" in metadata and "hidden_dims" in metadata:
        model = build_model(
            int(metadata["input_dim"]),
            tuple(metadata["hidden_dims"]),
            float(metadata.get("dropout", 0.2)),
        )
        model.load_state_dict(state_dict)
        logger.info("Loaded model and rebuilt architecture from metadata.")
        return model, metadata
    else:
        # return raw module with state dict loaded into a generic FraudNet if possible
        model = FraudNet(1)  # placeholder; user should rebuild
        try:
            model.load_state_dict(state_dict)
            logger.warning(
                "Loaded state_dict into default FraudNet(1). Consider providing metadata to rebuild architecture."
            )
        except Exception:
            logger.warning(
                "Loaded state_dict but failed to map to default architecture; returning raw payload."
            )
        return model, metadata


def predict_proba(
    model: nn.Module, X: np.ndarray, device: Optional[str] = None
) -> np.ndarray:
    """Convenience wrapper returning positive-class probabilities."""
    probs, _ = evaluate_model(model, X, device=device)
    return probs


def predict(
    model: nn.Module,
    X: np.ndarray,
    threshold: float = 0.5,
    device: Optional[str] = None,
) -> np.ndarray:
    """Convenience wrapper returning binary predictions using a threshold on predicted probabilities."""
    probs = predict_proba(model, X, device=device)
    return (probs >= threshold).astype(int)

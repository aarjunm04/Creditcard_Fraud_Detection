# models/xgb.py
"""
XGBoost wrapper utilities for the Credit Card Fraud Detection project.

Provides:
- builder/getter for XGBClassifier with sensible defaults
- training wrapper (with optional early stopping)
- optional Optuna tuning helper (if optuna available in env)
- save / load / predict helpers

This file is intentionally self-contained and returns sklearn-compatible objects
(XGBClassifier) so they plug into pipelines (imblearn / sklearn).
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report

try:
    from src import logger
except Exception:
    # fallback if imported standalone
    import logging
    logger = logging.getLogger("xgb_wrapper")
    logging.basicConfig(level=logging.INFO)


def get_xgb_default_params() -> Dict[str, Any]:
    """
    Return a set of sensible default params for XGBClassifier used in this project.
    """
    return {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }


def build_xgb(params: Optional[Dict[str, Any]] = None) -> XGBClassifier:
    """
    Build an XGBClassifier with given params merged on top of defaults.

    Args:
        params: optional parameter overrides

    Returns:
        XGBClassifier instance (unfitted)
    """
    cfg = get_xgb_default_params()
    if params:
        cfg.update(params)
    model = XGBClassifier(**cfg)
    logger.info("Built XGBClassifier with params: %s", {k: cfg[k] for k in ("n_estimators", "max_depth", "learning_rate") if k in cfg})
    return model


def train_xgb(
    model: XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    early_stopping_rounds: Optional[int] = 50,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> XGBClassifier:
    """
    Train XGBoost model with optional early stopping using a validation set.

    Args:
        model: XGBClassifier instance (unfitted)
        X_train, y_train: training data (numpy arrays)
        X_val, y_val: optional validation data for early stopping
        early_stopping_rounds: number of rounds for early stopping (None to disable)
        fit_kwargs: extra kwargs forwarded to model.fit()

    Returns:
        Trained XGBClassifier
    """
    fit_kwargs = fit_kwargs or {}
    if X_val is not None and y_val is not None and early_stopping_rounds:
        logger.info("Training XGB with early stopping (rounds=%s)", early_stopping_rounds)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
            **fit_kwargs,
        )
    else:
        logger.info("Training XGB without early stopping")
        model.fit(X_train, y_train, **fit_kwargs)
    return model


def evaluate_model_predictions(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute basic evaluation metrics for binary classification (fraud is positive class).

    Args:
        y_true: ground truth labels
        y_proba: predicted probabilities for positive class
        threshold: decision threshold to convert probs -> labels

    Returns:
        dict with precision, recall, f1, roc_auc and classification_report (str)
    """
    y_pred = (y_proba >= threshold).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_proba)
    except Exception:
        roc = float("nan")
    rpt = classification_report(y_true, y_pred, digits=4)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "roc_auc": float(roc), "report": rpt}


def save_model(model: XGBClassifier, path: str | Path):
    """
    Persist model to disk using joblib (sklearn wrapper). Creates parent dirs if needed.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    logger.info("Saved XGB model to %s", p)


def load_model(path: str | Path) -> XGBClassifier:
    """
    Load a model previously saved with save_model.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    model = joblib.load(p)
    logger.info("Loaded XGB model from %s", p)
    return model


def predict_proba(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    """
    Return predicted probability for positive class.
    """
    # sklearn wrapper: predict_proba returns [n_samples, 2]
    proba = model.predict_proba(X)[:, 1]
    return proba


def predict(model: XGBClassifier, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Return class predictions using a probability threshold.
    """
    proba = predict_proba(model, X)
    return (proba >= threshold).astype(int)


# ----- Optional: Optuna tuning helper (only used if optuna installed) -----
def tune_xgb_optuna(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    cv_splits: int = 3,
    direction: str = "maximize",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Simple Optuna tuning for XGBoost hyperparameters (F1 oriented).
    Returns best params dict. Requires optuna in the environment.

    Note: This function is optional — the repo includes RandomizedSearchCV elsewhere.
    """
    try:
        import optuna
        from sklearn.model_selection import StratifiedKFold
        from sklearn.model_selection import cross_val_score
    except Exception as e:
        raise RuntimeError("optuna (or sklearn) is required for tune_xgb_optuna") from e

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": random_state,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
        clf = XGBClassifier(**params)
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        # use cross_val_score on roc_auc as a proxy; for F1 you'd need custom scorer with probabilities/thresholding
        scores = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    logger.info("Optuna tuning completed. Best value: %s", study.best_value)
    return study.best_params


# If module executed directly demonstrate a tiny smoke-run (do not run heavy ops)
if __name__ == "__main__":
    logger.info("xgb.py module loaded as script - no direct demo to run here.")
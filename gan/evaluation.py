# gan/evaluation.py
"""
Evaluation utilities for synthetic vs real fraud data.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

try:
    from src import logger
except Exception:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("gan_eval")


def compare_distributions(
    real: pd.DataFrame, synth: pd.DataFrame, cols: list[str]
) -> pd.DataFrame:
    """
    Compare distributions of numeric columns between real and synthetic data.
    Returns a dataframe of means and stds.
    """
    metrics = []
    for c in cols:
        r_mean, r_std = real[c].mean(), real[c].std()
        s_mean, s_std = synth[c].mean(), synth[c].std()
        metrics.append(
            {
                "column": c,
                "real_mean": r_mean,
                "synth_mean": s_mean,
                "real_std": r_std,
                "synth_std": s_std,
            }
        )
    return pd.DataFrame(metrics)


def train_classifier_test(
    real: pd.DataFrame, synth: pd.DataFrame, label_col: str = "Class"
) -> float:
    """
    Train a classifier on synthetic data, test on real data (TSTR metric).
    Returns F1 score on fraud class.
    """
    X_s, y_s = synth.drop(columns=[label_col]), synth[label_col]
    X_r, y_r = real.drop(columns=[label_col]), real[label_col]

    X_train, _, y_train, _ = train_test_split(
        X_s, y_s, test_size=0.2, stratify=y_s, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_r)

    f1 = f1_score(y_r, y_pred, zero_division=0)
    logger.info("📊 TSTR F1 on fraud: %.4f", f1)
    return f1

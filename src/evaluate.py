from __future__ import annotations
import json
from pathlib import Path
from typing import Dict
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve


def evaluate_classification(y_true, y_proba, threshold: float = 0.5) -> Dict:
    y_pred = (y_proba >= threshold).astype(int)
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    roc_auc = metrics.roc_auc_score(y_true, y_proba)
    cm = metrics.confusion_matrix(y_true, y_pred).tolist()
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }


def plot_confusion_matrix_save(y_true, y_pred, out_path: Path, normalize: bool = False):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, normalize="true" if normalize else None
    )
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()


def plot_roc_save(y_true, y_proba, out_path: Path):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_proba)
    auc = metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()


def plot_pr_curve_save(y_true, y_proba, out_path: Path):
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.figure()
    plt.plot(recalls, precisions, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()


def plot_calibration_curve_save(y_true, y_proba, out_path: Path, n_bins: int = 10):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="Calibration curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()


def find_best_threshold(y_true, y_proba, metric: str = "f1") -> dict:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = f1_scores.argmax()
    return {
        "best_threshold": float(thresholds[best_idx]),
        "best_f1": float(f1_scores[best_idx]),
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx])
    }


def save_json(obj: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

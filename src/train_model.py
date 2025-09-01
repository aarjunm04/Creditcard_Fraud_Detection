# src/train_model.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.data_prep import load_raw, build_preprocessor, get_feature_target
from src.evaluate import evaluate_classification, plot_confusion_matrix_save, plot_roc_save, save_json

from sklearn.calibration import CalibratedClassifierCV
from src.evaluate import find_best_threshold, plot_pr_curve_save, plot_calibration_curve_save


SEED = 42
N_JOBS = -1
CV_SPLITS = 5

MODELS = {
    "logreg": LogisticRegression(max_iter=2000, n_jobs=None, solver="liblinear"),
    "rf": RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS),
    "gbdt": GradientBoostingClassifier(random_state=SEED),
    "mlp": MLPClassifier(random_state=SEED, early_stopping=True, max_iter=300)
}

PARAMS = {
    "logreg": {
        "smote__sampling_strategy": [0.2, 0.3, 0.4, 0.5],
        "clf__C": np.logspace(-2, 2, 20),
        "clf__penalty": ["l1", "l2"]
    },
    "rf": {
        "smote__sampling_strategy": [0.2, 0.3, 0.4, 0.5],
        "clf__n_estimators": [200, 400, 600, 800, 1000],
        "clf__max_depth": [None, 6, 10, 14, 18, 24],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2"]
    },
    "gbdt": {
        "smote__sampling_strategy": [0.2, 0.3, 0.4, 0.5],
        "clf__n_estimators": [100, 200, 400],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__max_depth": [2, 3, 4, 5],
        "clf__subsample": [0.6, 0.8, 1.0]
    },
    "mlp": {
        "smote__sampling_strategy": [0.2, 0.3, 0.4, 0.5],
        "clf__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
        "clf__alpha": np.logspace(-5, -1, 9),
        "clf__learning_rate_init": [1e-4, 3e-4, 1e-3],
    }
}


def build_pipeline(model_key: str):
    pre = build_preprocessor()
    clf = MODELS[model_key]
    pipe = ImbPipeline(steps=[
        ("pre", pre),
        ("smote", SMOTE(random_state=SEED)),
        ("clf", clf),
    ])
    return pipe


def train_one(model_key: str, df: pd.DataFrame, artifacts_dir: Path):
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    best_thr = find_best_threshold(y_test, y_proba, metric="f1")
    metrics_dict.update({"tuned_threshold": best_thr})

    # --- Calibration (fit post-hoc isotonic & sigmoid) ---
    cal_models = {
        "isotonic": CalibratedClassifierCV(best_model, cv="prefit", method="isotonic"),
        "sigmoid": CalibratedClassifierCV(best_model, cv="prefit", method="sigmoid"),
    }

    cal_metrics = {}
    for name, cal in cal_models.items():
        cal.fit(X_train, y_train)
        cal_proba = cal.predict_proba(X_test)[:, 1]
        cal_metrics[name] = evaluate_classification(y_test, cal_proba, threshold=0.5)

        # save calibration curve plots
        plot_calibration_curve_save(
            y_test, cal_proba,
            plots_dir / f"{model_key}_calibration_{name}.png"
        )

    metrics_dict["calibration"] = cal_metrics

    # Save updated metrics
    save_json({**metrics_dict, "best_params": search.best_params_}, metrics_path)

    # --- New plots ---
    plot_pr_curve_save(y_test, y_proba, plots_dir / f"{model_key}_pr_curve.png")

    pipe = build_pipeline(model_key)
    param_dist = PARAMS[model_key]

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=30,
        scoring="f1",
        n_jobs=N_JOBS,
        cv=cv,
        refit=True,
        random_state=SEED,
        verbose=1,
        return_train_score=False
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    # Evaluate on held-out test
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics_dict = evaluate_classification(y_test, y_proba, threshold=0.5)
    report_text = classification_report(y_test, y_pred, digits=4)

    # Save artifacts
    models_dir = artifacts_dir / "models"
    reports_dir = artifacts_dir / "reports"
    plots_dir = artifacts_dir / "reports" / "plots"
    cv_dir = artifacts_dir / "reports" / "cv"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    cv_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, models_dir / f"{model_key}_pipeline.joblib")

    metrics_path = reports_dir / f"{model_key}_metrics.json"
    save_json({**metrics_dict, "best_params": search.best_params_}, metrics_path)

    with open(reports_dir / f"{model_key}_classification_report.txt", "w") as f:
        f.write(report_text)

    # Plots
    plot_confusion_matrix_save(y_test, y_pred, plots_dir / f"{model_key}_cm.png", normalize=False)
    plot_confusion_matrix_save(y_test, y_pred, plots_dir / f"{model_key}_cm_norm.png", normalize=True)
    plot_roc_save(y_test, y_proba, plots_dir / f"{model_key}_roc.png")

    # CV results
    cv_df = pd.DataFrame(search.cv_results_)
    cv_df.to_csv(cv_dir / f"{model_key}_cv_results.csv", index=False)

    print(f"[{model_key}] F1: {metrics_dict['f1']:.4f} | Recall: {metrics_dict['recall']:.4f} | "
          f"Precision: {metrics_dict['precision']:.4f} | ROC-AUC: {metrics_dict['roc_auc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train and benchmark fraud models.")
    parser.add_argument("--raw_csv", type=str, default="data/raw/creditcard.csv")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "logreg", "rf", "gbdt", "mlp"])
    parser.add_argument("--artifacts_dir", type=str, default=".")
    args = parser.parse_args()

    df = load_raw(args.raw_csv)
    artifacts_dir = Path(args.artifacts_dir)

    keys = ["logreg", "rf", "gbdt", "mlp"] if args.model == "all" else [args.model]
    for k in keys:
        train_one(k, df, artifacts_dir=artifacts_dir)


if __name__ == "__main__":
    main()

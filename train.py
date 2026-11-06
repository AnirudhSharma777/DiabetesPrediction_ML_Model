"""Train, compare and save the diabetes prediction model.

Usage:  python train.py
Outputs: model/diabetes_model.joblib and model/metrics.json
"""
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from diabetes_model import (
    FEATURES,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    build_pipeline,
    candidate_models,
    load_data,
)

CV_METRICS = ["roc_auc", "accuracy", "precision", "recall", "f1"]


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 1. Compare every candidate with 5-fold cross-validation on the TRAINING set only.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison = []
    for name, estimator, scale in candidate_models():
        scores = cross_validate(build_pipeline(estimator, scale), X_train, y_train, cv=cv, scoring=CV_METRICS)
        row = {"model": name, "scaled": scale}
        for m in CV_METRICS:
            row[m] = float(np.mean(scores[f"test_{m}"]))
            row[f"{m}_std"] = float(np.std(scores[f"test_{m}"]))
        comparison.append(row)
        print(f"{name:<24} ROC-AUC {row['roc_auc']:.3f} ± {row['roc_auc_std']:.3f}   accuracy {row['accuracy']:.3f}")

    comparison.sort(key=lambda r: r["roc_auc"], reverse=True)
    best = comparison[0]["model"]
    _, best_estimator, best_scale = next(c for c in candidate_models() if c[0] == best)
    print(f"\nBest model by cross-validated ROC-AUC: {best}")

    # 2. Fit the chosen pipeline on the full training set and evaluate ONCE on the held-out test set.
    pipeline = build_pipeline(best_estimator, best_scale).fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, proba)
    test_metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    print("Held-out test set:", {k: round(v, 3) for k, v in test_metrics.items()})

    importance = permutation_importance(
        pipeline, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=RANDOM_STATE
    )

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    metrics = {
        "best_model": best,
        "threshold": 0.5,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_balance": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "cv_comparison": comparison,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "roc_curve": {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()},
        "test_predictions": {"proba": proba.tolist(), "label": y_test.astype(int).tolist()},
        "feature_importance": dict(
            sorted(
                zip(FEATURES, importance.importances_mean.round(4).tolist()),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ),
        "training_medians": {k: float(v) for k, v in X_train.replace(0, np.nan).median().items()},
        "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved {MODEL_PATH.name} and {METRICS_PATH.name} to {MODEL_DIR}/")


if __name__ == "__main__":
    main()

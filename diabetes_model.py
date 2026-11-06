"""Shared feature definitions, preprocessing and model helpers."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "diabetes.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "diabetes_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET = "Outcome"
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
# In this dataset a 0 in these columns means "not measured", not a real value.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
RANDOM_STATE = 42


def zeros_to_nan(X):
    X = np.asarray(X, dtype=float).copy()
    X[X == 0] = np.nan
    return X


def load_data(path=DATA_PATH):
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    return df[FEATURES], df[TARGET]


def build_preprocessor(scale):
    zero_cols = [
        ("to_nan", FunctionTransformer(zeros_to_nan, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median")),
    ]
    other_cols = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        zero_cols.append(("scale", StandardScaler()))
        other_cols.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("zero_as_missing", Pipeline(zero_cols), ZERO_AS_MISSING),
            ("other", Pipeline(other_cols), [f for f in FEATURES if f not in ZERO_AS_MISSING]),
        ]
    )


def candidate_models():
    """(name, estimator, needs_scaling) for every model we compare."""
    return [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True),
        ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE), False),
        ("Random Forest", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE), False),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=RANDOM_STATE), False),
        ("Support Vector Machine", SVC(probability=True, random_state=RANDOM_STATE), True),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=15), True),
        (
            "XGBoost",
            XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=RANDOM_STATE,
            ),
            False,
        ),
    ]


def build_pipeline(estimator, scale):
    return Pipeline([("preprocess", build_preprocessor(scale)), ("model", estimator)])


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict_proba(model, record):
    """Probability of diabetes for a single record given as a dict of feature values."""
    row = pd.DataFrame([{f: float(record[f]) for f in FEATURES}])
    return float(model.predict_proba(row)[0, 1])

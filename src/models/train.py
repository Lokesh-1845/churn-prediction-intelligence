from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from xgboost import XGBClassifier

from src.features.build_features import (
    build_preprocessing_pipeline
)


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "cell2celltrain.csv"
)

MODEL_DIR = (
    ROOT
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "churn_pipeline.joblib"
)

PREPROCESSOR_PATH = (
    ROOT
    / "src"
    / "models"
    / "preprocessing_pipeline.pkl"
)


TARGET = "churn"
ID_COL = "customerid"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# TARGET
# ============================================================

if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


def convert_target(value):

    text = str(value).strip().lower()

    if text in {
        "yes",
        "y",
        "1",
        "true",
        "churn",
        "churned"
    }:

        return 1

    if text in {
        "no",
        "n",
        "0",
        "false",
        "no churn",
        "not churn"
    }:

        return 0

    return np.nan


y = (
    df[TARGET]
    .apply(convert_target)
)

valid = y.notna()

df = df.loc[valid].copy()
y = y.loc[valid].astype(int)


# ============================================================
# FEATURES
# ============================================================

drop_columns = [
    TARGET,
    ID_COL
]

X = df.drop(
    columns=drop_columns,
    errors="ignore"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


# ============================================================
# DETERMINE COLUMN TYPES
# ============================================================

num_cols = (
    X_train
    .select_dtypes(
        include=np.number
    )
    .columns
    .tolist()
)

cat_cols = (
    X_train
    .select_dtypes(
        include=[
            "object",
            "category"
        ]
    )
    .columns
    .tolist()
)


# ============================================================
# PREPROCESS
# ============================================================

preprocessor = build_preprocessing_pipeline(
    num_cols,
    cat_cols
)


X_train_transformed = (
    preprocessor.fit_transform(
        X_train
    )
)

X_test_transformed = (
    preprocessor.transform(
        X_test
    )
)


# ============================================================
# MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1
)


# ============================================================
# TRAIN
# ============================================================

model.fit(
    X_train_transformed,
    y_train
)


# ============================================================
# EVALUATION
# ============================================================

y_pred = model.predict(
    X_test_transformed
)

y_prob = model.predict_proba(
    X_test_transformed
)[:, 1]


metrics = {
    "accuracy": accuracy_score(
        y_test,
        y_pred
    ),
    "precision": precision_score(
        y_test,
        y_pred,
        zero_division=0
    ),
    "recall": recall_score(
        y_test,
        y_pred,
        zero_division=0
    ),
    "f1_score": f1_score(
        y_test,
        y_pred,
        zero_division=0
    ),
    "roc_auc": roc_auc_score(
        y_test,
        y_prob
    )
}


print("\nMODEL RESULTS")
print("-" * 50)

for name, value in metrics.items():

    print(
        f"{name:<12}: {value:.4f}"
    )


# ============================================================
# SAVE ARTIFACTS
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_DIR / "churn_model.joblib"
)

joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH
)


# ------------------------------------------------------------
# Save combined pipeline
# ------------------------------------------------------------

artifact = {
    "model": model,
    "preprocessor": preprocessor,
    "model_input_columns": list(
        X_train.columns
    ),
    "numeric_columns": num_cols,
    "categorical_columns": cat_cols,
    "metrics": metrics,
}

joblib.dump(
    artifact,
    MODEL_PATH
)


print("\nTraining completed.")
print(
    f"Saved combined model:\n{MODEL_PATH}"
)
print(
    f"Saved preprocessor:\n{PREPROCESSOR_PATH}"
)
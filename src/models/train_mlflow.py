
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import joblib
import numpy as np

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import mlflow
import mlflow.sklearn


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

X_PATH = ROOT / "src" / "models" / "transformed_features.npy"
Y_PATH = ROOT / "src" / "models" / "target.npy"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"


# ============================================================
# MLflow CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5050"
EXPERIMENT_NAME = "Customer Churn Prediction"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("CUSTOMER CHURN MODEL TRAINING")
print("=" * 70)

if not X_PATH.exists():
    raise FileNotFoundError(
        f"Transformed features file not found:\n{X_PATH}"
    )

if not Y_PATH.exists():
    raise FileNotFoundError(
        f"Target file not found:\n{Y_PATH}"
    )

print(f"\n✓ X file found : {X_PATH}")
print(f"✓ Y file found : {Y_PATH}")


# ============================================================
# LOAD X DATA
# ============================================================

print("\nLoading transformed features...")

X = np.load(X_PATH, allow_pickle=True)

print(f"X shape       : {X.shape}")
print(f"X dtype       : {X.dtype}")


# ============================================================
# LOAD TARGET DATA
# ============================================================

print("\nLoading target data...")

# allow_pickle=True is required because target.npy
# may contain strings/object dtype such as Yes/No.
y = np.load(Y_PATH, allow_pickle=True)

print(f"Y shape       : {y.shape}")
print(f"Y dtype       : {y.dtype}")


# ============================================================
# CONVERT TARGET INTO NUMERIC BINARY VALUES
# ============================================================

print("\nProcessing target variable...")

# Convert possible object/bytes/string values into strings
y = np.asarray(y).reshape(-1)

if y.dtype.kind in {"U", "S", "O"}:

    y_text = np.array([
        str(value).strip().lower()
        for value in y
    ])

    unique_values = np.unique(y_text)

    print(f"Original target values: {unique_values}")

    # Common churn representations
    positive_values = {
        "yes",
        "y",
        "true",
        "1",
        "churn",
        "churned"
    }

    negative_values = {
        "no",
        "n",
        "false",
        "0",
        "no churn",
        "not churned"
    }

    converted_y = []

    for value in y_text:

        if value in positive_values:
            converted_y.append(1)

        elif value in negative_values:
            converted_y.append(0)

        else:
            raise ValueError(
                f"Unknown target value found: '{value}'\n"
                f"Expected values similar to Yes/No or 1/0."
            )

    y = np.asarray(converted_y, dtype=np.int64)

else:

    # Numeric target
    y = np.asarray(y, dtype=np.int64)

    unique_values = np.unique(y)

    print(f"Original numeric target values: {unique_values}")


# ============================================================
# VALIDATE TARGET
# ============================================================

if len(X) != len(y):
    raise ValueError(
        f"Feature/target length mismatch!\n"
        f"X samples: {len(X)}\n"
        f"Y samples: {len(y)}"
    )

unique_y = np.unique(y)

if not np.all(np.isin(unique_y, [0, 1])):
    raise ValueError(
        f"Target must contain only 0 and 1.\n"
        f"Found: {unique_y}"
    )

print(f"Final target dtype : {y.dtype}")
print(f"Final target values: {np.unique(y)}")

print("\nTarget distribution:")
print(f"Class 0: {np.sum(y == 0)}")
print(f"Class 1: {np.sum(y == 1)}")


# ============================================================
# CONVERT FEATURE DATA
# ============================================================

print("\nChecking feature datatype...")

X = np.asarray(X)

# If features are object dtype, attempt numeric conversion.
if X.dtype.kind == "O":

    try:
        X = X.astype(np.float32)
        print("✓ Object features converted to float32")

    except (ValueError, TypeError) as error:

        raise ValueError(
            "X contains non-numeric object values and cannot "
            "be converted to float32."
        ) from error

elif X.dtype.kind in {"U", "S"}:

    try:
        X = X.astype(np.float32)
        print("✓ String features converted to float32")

    except (ValueError, TypeError) as error:

        raise ValueError(
            "X contains non-numeric string values."
        ) from error

else:

    X = X.astype(np.float32)
    print("✓ Features converted to float32")


# ============================================================
# CHECK FOR INVALID VALUES
# ============================================================

if not np.isfinite(X).all():

    print("\n⚠ Invalid values detected in X.")

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    print("✓ NaN and infinite values replaced.")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nCreating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")
print(f"Number features  : {X_train.shape[1]}")


# ============================================================
# XGBOOST MODEL
# ============================================================

print("\nCreating XGBoost classifier...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# MLflow TRAINING RUN
# ============================================================

with mlflow.start_run():

    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Log parameters
    # --------------------------------------------------------

    mlflow.log_param("model", "XGBClassifier")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("colsample_bytree", 0.8)
    mlflow.log_param("test_size", 0.20)
    mlflow.log_param("random_state", 42)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    print("\n✓ Model training completed.")


    # ========================================================
    # PREDICTIONS
    # ========================================================

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Churn", "Churn"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


    # ========================================================
    # LOG METRICS TO MLFLOW
    # ========================================================

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)


    # ========================================================
    # SAVE LOCAL MODEL
    # ========================================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(f"\n✓ Model saved:")
    print(MODEL_PATH)


    # ========================================================
    # LOG MODEL TO MLFLOW
    # ========================================================
    mlflow.sklearn.log_model(
        model,
        name="churn_model",
        skops_trusted_types=[
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier"
        ]
    )

    print("✓ Model logged to MLflow.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"\nModel file:")
print(MODEL_PATH)

print("\nDataset:")
print(f"X shape = {X.shape}")
print(f"Y shape = {y.shape}")

print("\nMetrics:")
print(f"Accuracy  = {accuracy:.4f}")
print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1 Score  = {f1:.4f}")
print(f"ROC-AUC   = {roc_auc:.4f}")

print("\n✓ Ready for prediction.")
print("=" * 70)
'''

This version specifically handles your `.npy` files:

| File                       | Handling                                           |
| -------------------------- | -------------------------------------------------- |
| `transformed_features.npy` | `np.load(..., allow_pickle=True)`                  |
| `target.npy`               | `np.load(..., allow_pickle=True)`                  |
| Object/string target       | Converts `Yes/No`, `Churn/No Churn`, `1/0` → `1/0` |
| Object feature array       | Converts to `float32`                              |
| NaN/Inf                    | Replaces invalid numeric values                    |
| Feature/target mismatch    | Explicitly detected                                |
| Model output               | `models/churn_model.joblib`                        |
| MLflow                     | Logs parameters, metrics and model                 |

### Your original project flow stays intact

```text
E:\PROJECTS_FILE\CCP
│
├── src
│   └── models
│       ├── transformed_features.npy   ← X
│       ├── target.npy                 ← y
│       └── train.py
│
└── models
    └── churn_model.joblib              ← output
```

The key change that fixes your previous error is:

```python
y = np.load(Y_PATH, allow_pickle=True)
```

**Do not use only**:

```python
y = np.load(Y_PATH).ravel().astype(int)
```

because your `target.npy` is an object/string array, and even after loading it, values such as `"Yes"` and `"No"` cannot directly be converted with `.astype(int)`.

One important point: this `.npy` approach trains directly on **already-transformed features**, so the resulting `churn_model.joblib` is an XGBoost model, not your earlier complete preprocessing + XGBoost pipeline. If your existing FastAPI/SHAP code expects `churn_pipeline.joblib`, that compatibility needs to be handled separately.
'''
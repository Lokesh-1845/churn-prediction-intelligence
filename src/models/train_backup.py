import joblib
import numpy as np

from pathlib import Path

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# PATHS
# ============================================================

X_PATH = (
    ROOT
    / "src"
    / "models"
    / "transformed_features.npy"
)

Y_PATH = (
    ROOT
    / "src"
    / "models"
    / "target.npy"
)

MODEL_PATH = (
    ROOT
    /"src"
    / "models"
    / "churn_model.joblib"
)


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("              XGBOOST CHURN MODEL TRAINING")
print("=" * 70)

print("\nChecking input files...")

print(f"X file: {X_PATH}")
print(f"Y file: {Y_PATH}")

if not X_PATH.exists():
    raise FileNotFoundError(
        f"\nTransformed feature file not found:\n{X_PATH}"
    )

if not Y_PATH.exists():
    raise FileNotFoundError(
        f"\nTarget file not found:\n{Y_PATH}"
    )


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading transformed features...")

X = np.load(X_PATH)

print(f"X shape   : {X.shape}")
print(f"X dtype   : {X.dtype}")


# ============================================================
# LOAD TARGET
# ============================================================

print("\nLoading target...")

# target.npy is an object array.
# allow_pickle=True is required to read the existing file.
y_raw = np.load(
    Y_PATH,
    allow_pickle=True
)

print(f"Raw y shape : {y_raw.shape}")
print(f"Raw y dtype : {y_raw.dtype}")


# ============================================================
# CONVERT TARGET TO BINARY
# ============================================================

y_raw = y_raw.ravel()

print(
    f"\nUnique target values before conversion:"
)
print(
    np.unique(y_raw)
)


# ------------------------------------------------------------
# Handle Yes / No target
# ------------------------------------------------------------

if y_raw.dtype == object:

    y = np.array([
        {
            "Yes": 1,
            "yes": 1,
            "YES": 1,
            "No": 0,
            "no": 0,
            "NO": 0,
            1: 1,
            0: 0
        }.get(value, value)
        for value in y_raw
    ])

else:

    y = y_raw


# ============================================================
# CONVERT TO INTEGER
# ============================================================

try:

    y = y.astype(int)

except (ValueError, TypeError) as e:

    print("\nERROR: Could not convert target to integers.")

    print(
        "\nTarget values found:"
    )

    print(
        np.unique(y_raw)
    )

    raise ValueError(
        "Target contains values that are not "
        "compatible with binary classification."
    ) from e


# ============================================================
# VALIDATE TARGET
# ============================================================

unique_targets = np.unique(y)

print(
    "\nTarget values after conversion:"
)

print(
    unique_targets
)

if not np.all(
    np.isin(unique_targets, [0, 1])
):

    raise ValueError(
        f"Target must contain only 0 and 1. "
        f"Found: {unique_targets}"
    )


print(
    f"Target shape : {y.shape}"
)

print(
    f"Target dtype : {y.dtype}"
)


# ============================================================
# VALIDATE X AND Y LENGTH
# ============================================================

if len(X) != len(y):

    raise ValueError(
        "\nFeature and target row counts do not match.\n"
        f"X rows: {len(X)}\n"
        f"y rows: {len(y)}"
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nCreating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print(
    f"Training samples : {len(X_train)}"
)

print(
    f"Testing samples  : {len(X_test)}"
)

print(
    f"Training churn rate: {y_train.mean():.4f}"
)

print(
    f"Testing churn rate : {y_test.mean():.4f}"
)


# ============================================================
# CREATE XGBOOST MODEL
# ============================================================

print("\nCreating XGBoost model...")

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

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train
)

print(
    "Training completed successfully."
)


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

pred = model.predict(
    X_test
)

prob = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    pred
)

precision = precision_score(
    y_test,
    pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    prob
)


# ============================================================
# MODEL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("                    MODEL RESULTS")
print("=" * 70)

print(
    f"\nAccuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("                 TRAINING COMPLETED")
print("=" * 70)

print(
    f"\nModel saved:"
    f"\n{MODEL_PATH}"
)

print("\n" + "=" * 70)
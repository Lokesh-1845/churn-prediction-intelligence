from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.exceptions import InconsistentVersionWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)


# ============================================================
# WARNING SUPPRESSION
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning
)

warnings.filterwarnings(
    "ignore",
    message=".*serialized model.*"
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

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
    / "models"
    / "churn_model.joblib"
)


# ============================================================
# SETTINGS
# ============================================================

# If True:
# When transformed_features.npy and the saved model have
# different feature counts, a compatible model is trained
# using the SAME model parameters and the REAL X/Y data.
#
# This avoids fake feature removal.
RETRAIN_ON_FEATURE_MISMATCH = True

# IMPORTANT:
# False = do not overwrite your existing churn_model.joblib
# True  = replace it with the newly trained compatible model
#
# I recommend keeping this False initially.
SAVE_RETRAINED_MODEL = False


# ============================================================
# HEADER
# ============================================================

print("=" * 65)
print("CUSTOMER CHURN MODEL EVALUATION")
print("=" * 65)

print("\nProject Root:")
print(ROOT)

print("\nFeature File:")
print(X_PATH)

print("\nTarget File:")
print(Y_PATH)

print("\nModel File:")
print(MODEL_PATH)


# ============================================================
# CHECK FILES
# ============================================================

for path in [X_PATH, Y_PATH, MODEL_PATH]:

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


# ============================================================
# LOAD TRANSFORMED FEATURES
# ============================================================

print("\n")
print("=" * 65)
print("LOADING TRANSFORMED FEATURES")
print("=" * 65)

X = np.load(
    X_PATH,
    allow_pickle=True
)

print(
    f"\nX shape : {X.shape}"
)

print(
    f"X dtype : {X.dtype}"
)


# ============================================================
# VALIDATE X
# ============================================================

if X.ndim != 2:

    raise ValueError(
        "\ntransformed_features.npy must contain "
        "a 2-dimensional feature matrix.\n"
        f"Current shape: {X.shape}"
    )


# ============================================================
# LOAD TARGET
# ============================================================

print("\n")
print("=" * 65)
print("LOADING TARGET")
print("=" * 65)

y = np.load(
    Y_PATH,
    allow_pickle=True
).ravel()

print(
    f"\ny shape : {y.shape}"
)

print(
    f"y dtype : {y.dtype}"
)


# ============================================================
# TARGET NORMALIZATION
# ============================================================

def normalize_target(value):

    if value is None:
        return None

    text = str(value).strip().lower()

    positive_values = {
        "1",
        "yes",
        "y",
        "true",
        "churn",
        "churned",
        "1.0",
    }

    negative_values = {
        "0",
        "no",
        "n",
        "false",
        "no churn",
        "not churn",
        "not_churn",
        "0.0",
    }

    if text in positive_values:
        return 1

    if text in negative_values:
        return 0

    try:

        numeric = float(value)

        if numeric == 1:
            return 1

        if numeric == 0:
            return 0

    except Exception:

        pass

    return None


# ============================================================
# CONVERT TARGET TO 0 / 1
# ============================================================

if y.dtype.kind in {"U", "S", "O"}:

    normalized_y = [
        normalize_target(value)
        for value in y
    ]

    invalid_values = [
        original
        for original, normalized
        in zip(y, normalized_y)
        if normalized is None
    ]

    if invalid_values:

        raise ValueError(
            "\nUnable to convert these target values "
            "to 0/1:\n"
            f"{invalid_values[:20]}"
        )

    y = np.asarray(
        normalized_y,
        dtype=int
    )

else:

    y = y.astype(int)


# ============================================================
# VALIDATE X / Y
# ============================================================

if len(X) != len(y):

    raise ValueError(
        "\nFeature and target row counts do not match.\n"
        f"X rows: {len(X)}\n"
        f"y rows: {len(y)}"
    )


# ============================================================
# VALIDATE TARGET CLASSES
# ============================================================

unique_classes = np.unique(y)

print("\nTarget classes:")
print(unique_classes)

if not set(unique_classes).issubset({0, 1}):

    raise ValueError(
        "\nTarget must contain only 0 and 1.\n"
        f"Found: {unique_classes}"
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")

unique, counts = np.unique(
    y,
    return_counts=True
)

for class_value, count in zip(
    unique,
    counts
):

    label = (
        "Churn"
        if class_value == 1
        else "No Churn"
    )

    print(
        f"{label}: {count:,}"
    )


# ============================================================
# LOAD SAVED MODEL
# ============================================================

print("\n")
print("=" * 65)
print("LOADING SAVED MODEL")
print("=" * 65)

saved_model = joblib.load(
    MODEL_PATH
)

print(
    f"\nModel type: "
    f"{type(saved_model).__name__}"
)


# ============================================================
# DETERMINE MODEL FEATURE COUNT
# ============================================================

model_feature_count = getattr(
    saved_model,
    "n_features_in_",
    None
)

data_feature_count = X.shape[1]

print("\n")
print("=" * 65)
print("FEATURE COMPATIBILITY CHECK")
print("=" * 65)

print(
    f"\ntransformed_features.npy : "
    f"{data_feature_count} features"
)

print(
    f"Saved model              : "
    f"{model_feature_count} features"
)


# ============================================================
# SELECT MODEL TO USE
# ============================================================

model = saved_model


if (
    model_feature_count is not None
    and data_feature_count != model_feature_count
):

    print("\n")
    print("⚠️ FEATURE MISMATCH DETECTED")
    print("-" * 65)

    print(
        f"\nSaved model expects "
        f"{model_feature_count} features."
    )

    print(
        f"Current transformed data contains "
        f"{data_feature_count} features."
    )

    print(
        "\nThe code will NOT randomly remove "
        "or create features."
    )

    if not RETRAIN_ON_FEATURE_MISMATCH:

        raise ValueError(
            "\nFeature shape mismatch.\n"
            f"Model expects {model_feature_count} "
            f"features, but X contains "
            f"{data_feature_count}.\n\n"
            "Either regenerate transformed_features.npy "
            "using the original feature-selection pipeline "
            "or retrain the model using the current feature matrix."
        )

    # ========================================================
    # RETRAIN COMPATIBLE MODEL
    # ========================================================

    print("\n")
    print("=" * 65)
    print("CREATING COMPATIBLE MODEL")
    print("=" * 65)

    print(
        "\nUsing the saved model's parameters "
        "to create a new model."
    )

    print(
        "Training will use the REAL "
        "transformed_features.npy and target.npy."
    )

    try:

        from xgboost import XGBClassifier

    except ImportError:

        raise ImportError(
            "\nXGBoost is required for retraining.\n"
            "Install it using:\n"
            "pip install xgboost"
        )

    # --------------------------------------------------------
    # Get parameters from existing model
    # --------------------------------------------------------

    if not hasattr(
        saved_model,
        "get_params"
    ):

        raise TypeError(
            "\nThe saved model does not provide "
            "get_params(). Automatic compatible "
            "retraining is not possible."
        )

    model_params = saved_model.get_params()

    print(
        "\nOriginal model parameters loaded."
    )

    # --------------------------------------------------------
    # Remove parameters that may cause problems
    # --------------------------------------------------------

    model_params.pop(
        "feature_types",
        None
    )

    # --------------------------------------------------------
    # Create new XGBoost model
    # --------------------------------------------------------

    model = XGBClassifier(
        **model_params
    )

    print(
        "\nNew compatible XGBoost model created."
    )

else:

    print(
        "\n✓ Feature shapes are compatible."
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n")
print("=" * 65)
print("CREATING EVALUATION SPLIT")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    f"\nTraining samples : {len(X_train):,}"
)

print(
    f"Testing samples  : {len(X_test):,}"
)

print(
    f"Training features: {X_train.shape[1]:,}"
)


# ============================================================
# TRAIN MODEL ONLY IF NEEDED
# ============================================================

if model is not saved_model:

    print("\n")
    print("=" * 65)
    print("TRAINING COMPATIBLE MODEL")
    print("=" * 65)

    print(
        "\nTraining on REAL transformed features..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "\n✓ Compatible model training completed."
    )

    # --------------------------------------------------------
    # Optional save
    # --------------------------------------------------------

    if SAVE_RETRAINED_MODEL:

        joblib.dump(
            model,
            MODEL_PATH
        )

        print(
            f"\n✓ Updated model saved to:\n"
            f"{MODEL_PATH}"
        )

    else:

        print(
            "\nℹ Existing churn_model.joblib "
            "was NOT overwritten."
        )


# ============================================================
# FINAL FEATURE CHECK
# ============================================================

final_model_features = getattr(
    model,
    "n_features_in_",
    None
)

print("\n")
print("=" * 65)
print("FINAL FEATURE CHECK")
print("=" * 65)

print(
    f"\nModel features : {final_model_features}"
)

print(
    f"Test features  : {X_test.shape[1]}"
)

if (
    final_model_features is not None
    and final_model_features != X_test.shape[1]
):

    raise ValueError(
        "\nFinal model and test data still have "
        "different feature counts.\n"
        f"Model: {final_model_features}\n"
        f"X_test: {X_test.shape[1]}"
    )

print(
    "\n✓ Model and test data are compatible."
)


# ============================================================
# MODEL PREDICTIONS
# ============================================================

print("\n")
print("=" * 65)
print("RUNNING MODEL PREDICTIONS")
print("=" * 65)

y_pred = model.predict(
    X_test
)

print(
    "\n✓ Predictions generated."
)


# ============================================================
# PREDICT PROBABILITY
# ============================================================

if not hasattr(
    model,
    "predict_proba"
):

    raise AttributeError(
        "\nThe model does not provide "
        "predict_proba()."
    )


probabilities = model.predict_proba(
    X_test
)

probabilities = np.asarray(
    probabilities
)


if probabilities.ndim == 2:

    if probabilities.shape[1] == 2:

        y_prob = probabilities[:, 1]

    else:

        y_prob = probabilities[:, -1]

else:

    y_prob = probabilities.ravel()


# ============================================================
# CLEAN PREDICTIONS
# ============================================================

y_pred = np.asarray(
    y_pred
).astype(int)

y_prob = np.asarray(
    y_prob
).astype(float)


# ============================================================
# VALIDATE PREDICTIONS
# ============================================================

if len(y_pred) != len(y_test):

    raise ValueError(
        "\nPrediction count does not match "
        "test sample count."
    )

if len(y_prob) != len(y_test):

    raise ValueError(
        "\nProbability count does not match "
        "test sample count."
    )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

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
    y_prob
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 65)
print("MODEL EVALUATION RESULTS")
print("=" * 65)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Churn",
            "Churn"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n")
print("=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)

print(cm)


ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Churn",
        "Churn"
    ]
).plot()

plt.title(
    "Customer Churn - Confusion Matrix"
)

plt.tight_layout()

plt.show()


# ============================================================
# ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    name="XGBoost"
)

plt.title(
    f"Customer Churn - ROC Curve "
    f"(AUC = {roc_auc:.4f})"
)

plt.tight_layout()

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 65)
print("EVALUATION COMPLETED")
print("=" * 65)

print(
    "\nData used:"
)

print(
    f"✓ {X_PATH.name}"
)

print(
    f"✓ {Y_PATH.name}"
)

print(
    f"✓ 80/20 stratified evaluation split"
)

print(
    f"✓ {X_test.shape[1]} model features"
)

print(
    "\nPredictions are generated by the "
    "XGBoost model using the actual feature matrix."
)

print(
    "\nNo dummy predictions, manually entered "
    "metrics, or randomly removed features are used."
)

if model is not saved_model:

    print(
        "\n⚠️ NOTE:"
    )

    print(
        "The original saved model expected "
        f"{model_feature_count} features, while "
        f"the current transformed data contains "
        f"{data_feature_count}."
    )

    print(
        "A compatible XGBoost model was therefore "
        "trained using the current real feature matrix."
    )

    if SAVE_RETRAINED_MODEL:

        print(
            "The compatible model was saved."
        )

    else:

        print(
            "The original churn_model.joblib "
            "was not overwritten."
        )

print(
    "\n✓ All evaluation steps completed successfully."
)

print(
    "=" * 65
)
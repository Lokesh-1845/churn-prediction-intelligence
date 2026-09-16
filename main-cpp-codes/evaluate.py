import joblib
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# Paths
ROOT = Path(__file__).resolve().parents[2]

X = np.load(
    ROOT / "src" / "models" / "transformed_features.npy"
)

y = np.load(
    ROOT / "src" / "models" / "target.npy",
    allow_pickle=True
).ravel()

# Convert target to 0/1
if y.dtype.kind in {"U", "S", "O"}:
    y = np.array([
        1 if str(v).lower() in ["1", "yes", "churn", "true"] else 0
        for v in y
    ])

y = y.astype(int)

# Test split
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load model
model = joblib.load(
    ROOT / "models" / "churn_model.joblib"
)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
print("\nCustomer Churn Model Results")
print("-" * 35)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No Churn", "Churn"]
)

plt.title("Confusion Matrix")
plt.show()

# ROC Curve
RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    name="XGBoost"
)

plt.title("ROC Curve")
plt.show()
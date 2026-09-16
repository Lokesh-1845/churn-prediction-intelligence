import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Paths
ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "churn_model.joblib"
X_PATH = ROOT / "src" / "models" / "transformed_features.npy"
Y_PATH = ROOT / "src" / "models" / "target.npy"


class ChurnPredictor:

    def __init__(self, model_path=MODEL_PATH):
        self.model = joblib.load(model_path)

    def predict(self, X=None):
        if X is None:
            X = np.load(X_PATH)

        X = np.asarray(X, dtype=np.float32)

        probability = self.model.predict_proba(X)[:, 1]
        prediction = (probability >= 0.5).astype(int)

        return pd.DataFrame({
            "churn_probability": probability,
            "churn_prediction": prediction
        })

    def evaluate(self):
        X = np.load(X_PATH)
        y = np.load(Y_PATH, allow_pickle=True).ravel()

        # Convert Yes/No if required
        if y.dtype.kind in {"O", "U", "S"}:
            y = np.array([
                1 if str(v).lower() in
                ["1", "yes", "churn", "true"]
                else 0
                for v in y
            ])

        y = y.astype(int)

        result = self.predict(X)

        pred = result["churn_prediction"]
        prob = result["churn_probability"]

        metrics = {
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1_score": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, prob)
        }

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        return metrics


if __name__ == "__main__":
    predictor = ChurnPredictor()

    print("\nCustomer Churn Prediction")
    print("-" * 30)

    results = predictor.evaluate()
import time
import joblib
import mlflow
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
X_PATH = ROOT / "src" / "models" / "transformed_features.npy"
Y_PATH = ROOT / "src" / "models" / "target.npy"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"

# MLflow
MLFLOW_URI = "http://127.0.0.1:5050"
EXPERIMENT = "Customer Churn Prediction"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)


class ChurnPredictor:

    def __init__(self):
        print("\nLoading model...")
        self.model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")

    def load_data(self):
        X = np.load(X_PATH)
        y = np.load(Y_PATH, allow_pickle=True).ravel()

        if y.dtype.kind in {"O", "U", "S"}:
            y = np.array([
                1 if str(v).lower() in
                ["1", "yes", "churn", "true"]
                else 0
                for v in y
            ])

        return np.asarray(X, dtype=np.float32), y.astype(int)

    def predict(self, X):
        start = time.perf_counter()

        probability = self.model.predict_proba(X)[:, 1]
        prediction = (probability >= 0.5).astype(int)

        elapsed = time.perf_counter() - start

        return prediction, probability, elapsed

    def evaluate(self):

        print("\n" + "=" * 50)
        print("       CUSTOMER CHURN - MLFLOW PREDICTION")
        print("=" * 50)

        X, y = self.load_data()

        if len(X) != len(y):
            raise ValueError("X and y have different row counts.")

        print(f"\nFeatures     : {X.shape}")
        print(f"Target rows  : {len(y)}")

        try:
            mlflow.search_experiments()
        except Exception:
            print("\nMLflow server is not running.")
            print("Start it with:")
            print("mlflow server --host 127.0.0.1 --port 5050")
            return

        print("\nGenerating predictions...")

        prediction, probability, elapsed = self.predict(X)

        accuracy = accuracy_score(y, prediction)
        precision = precision_score(
            y, prediction, zero_division=0
        )
        recall = recall_score(
            y, prediction, zero_division=0
        )
        f1 = f1_score(
            y, prediction, zero_division=0
        )
        roc_auc = roc_auc_score(
            y, probability
        )

        total = len(prediction)
        churn = int(prediction.sum())
        no_churn = total - churn

        avg_probability = float(probability.mean())
        max_probability = float(probability.max())
        min_probability = float(probability.min())

        total_time_ms = elapsed * 1000
        avg_time_ms = total_time_ms / total

        print("\n" + "-" * 50)
        print("MODEL RESULTS")
        print("-" * 50)
        print(f"Accuracy          : {accuracy:.4f}")
        print(f"Precision         : {precision:.4f}")
        print(f"Recall            : {recall:.4f}")
        print(f"F1 Score          : {f1:.4f}")
        print(f"ROC-AUC           : {roc_auc:.4f}")
        print(f"Total Predictions : {total}")
        print(f"Predicted Churn   : {churn}")
        print(f"Predicted No Churn: {no_churn}")
        print(f"Avg Probability   : {avg_probability:.4f}")
        print(f"Prediction Time   : {total_time_ms:.2f} ms")
        print(f"Avg/Customer      : {avg_time_ms:.4f} ms")

        with mlflow.start_run(
            run_name="Churn_Prediction_Performance"
        ) as run:

            mlflow.log_params({
                "model": "XGBoost",
                "input_rows": len(X),
                "input_features": X.shape[1],
                "threshold": 0.5
            })

            mlflow.log_metrics({
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc),
                "total_predictions": total,
                "predicted_churn": churn,
                "predicted_no_churn": no_churn,
                "average_churn_probability": avg_probability,
                "maximum_churn_probability": max_probability,
                "minimum_churn_probability": min_probability,
                "prediction_time_ms": total_time_ms,
                "average_prediction_time_ms": avg_time_ms
            })

            mlflow.log_dict({
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc),
                "total_predictions": total,
                "predicted_churn": churn,
                "predicted_no_churn": no_churn
            }, "prediction_summary.json")

            mlflow.set_tags({
                "run_type": "prediction",
                "model_type": "XGBClassifier",
                "status": "completed",
                "model_logging": "disabled"
            })

            print("\n" + "=" * 50)
            print("MLFLOW PREDICTION COMPLETED")
            print("=" * 50)
            print(f"Run ID : {run.info.run_id}")
            print(f"MLflow : {MLFLOW_URI}")
            print("=" * 50)


if __name__ == "__main__":
    ChurnPredictor().evaluate()
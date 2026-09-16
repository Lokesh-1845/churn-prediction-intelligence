import time
import joblib
import mlflow
import shap
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay
)

# Paths
ROOT = Path(__file__).resolve().parents[2]
X_PATH = ROOT / "src" / "models" / "transformed_features.npy"
Y_PATH = ROOT / "src" / "models" / "target.npy"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"

ARTIFACTS = ROOT / "mlflow_artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5050")
mlflow.set_experiment("Customer Churn Prediction")


def load_data():
    X = np.load(X_PATH)
    y = np.load(Y_PATH, allow_pickle=True).ravel()

    if y.dtype.kind in {"O", "U", "S"}:
        y = np.array([
            1 if str(v).lower() in
            ["1", "yes", "churn", "true"]
            else 0
            for v in y
        ])

    return X, y.astype(int)


def save_plots(y_test, pred, prob):

    cm_path = ARTIFACTS / "confusion_matrix.png"

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        pred,
        display_labels=["No Churn", "Churn"]
    )
    plt.title("Customer Churn - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    roc_path = ARTIFACTS / "roc_curve.png"

    RocCurveDisplay.from_predictions(
        y_test,
        prob
    )
    plt.title("Customer Churn - ROC Curve")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    return cm_path, roc_path


def save_shap(model, X_test):

    try:
        sample = X_test[:min(100, len(X_test))]

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample)

        if isinstance(values, list):
            values = values[-1]

        if hasattr(values, "values"):
            values = values.values

        if values.ndim == 3:
            values = values[:, :, -1]

        shap_path = ARTIFACTS / "shap_summary.png"

        shap.summary_plot(
            values,
            sample,
            max_display=20,
            show=False
        )

        plt.tight_layout()
        plt.savefig(
            shap_path,
            dpi=150,
            bbox_inches="tight"
        )
        plt.close()

        return shap_path, "TreeExplainer"

    except Exception as e:

        print("SHAP skipped:", e)
        return None, "Failed"


def main():

    print("\nCustomer Churn - MLflow Evaluation")
    print("-" * 45)

    start = time.perf_counter()

    X, y = load_data()

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Testing samples: {len(X_test)}")

    model = joblib.load(MODEL_PATH)

    print("\nGenerating predictions...")

    prediction_start = time.perf_counter()

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    prediction_time = (
        time.perf_counter() - prediction_start
    )

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(
        y_test, pred, zero_division=0
    )
    recall = recall_score(
        y_test, pred, zero_division=0
    )
    f1 = f1_score(
        y_test, pred, zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test, prob
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        pred,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else 0
    )

    report = classification_report(
        y_test,
        pred,
        target_names=["No Churn", "Churn"],
        zero_division=0
    )

    print("\nResults")
    print("-" * 45)
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1 Score    : {f1:.4f}")
    print(f"ROC-AUC     : {roc_auc:.4f}")
    print(f"Specificity : {specificity:.4f}")

    print("\nConfusion Matrix")
    print(f"TN: {tn}  FP: {fp}")
    print(f"FN: {fn}  TP: {tp}")

    print("\nClassification Report")
    print(report)

    cm_path, roc_path = save_plots(
        y_test,
        pred,
        prob
    )

    print("\nGenerating SHAP...")

    shap_path, shap_method = save_shap(
        model,
        X_test
    )

    total_time = (
        time.perf_counter() - start
    )

    with mlflow.start_run(
        run_name="XGBoost_Churn_Evaluation"
    ):

        mlflow.log_params({
            "model": "XGBoost",
            "test_size": 0.20,
            "random_state": 42,
            "test_samples": len(X_test),
            "features": X.shape[1]
        })

        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "specificity": specificity,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "prediction_time_ms":
                prediction_time * 1000,
            "avg_prediction_time_ms":
                prediction_time * 1000 / len(X_test),
            "total_evaluation_time_seconds":
                total_time
        })

        mlflow.log_text(
            report,
            "evaluation/classification_report.txt"
        )

        mlflow.log_artifact(
            str(cm_path),
            "evaluation"
        )

        mlflow.log_artifact(
            str(roc_path),
            "evaluation"
        )

        if shap_path:
            mlflow.log_artifact(
                str(shap_path),
                "explainability"
            )

        mlflow.set_tags({
            "evaluation_status": "completed",
            "shap_status": shap_method
        })

        run_id = mlflow.active_run().info.run_id

    print("\n" + "=" * 45)
    print("MLFLOW EVALUATION COMPLETED")
    print("=" * 45)
    print(f"Run ID : {run_id}")
    print(f"Time   : {total_time:.2f} seconds")
    print("MLflow : http://127.0.0.1:5050")


if __name__ == "__main__":
    main()
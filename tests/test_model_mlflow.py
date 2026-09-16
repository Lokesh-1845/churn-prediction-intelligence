import warnings
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore",category=InconsistentVersionWarning)
warnings.filterwarnings("ignore",message=".*WARNING:.*If you are loading a serialized model.*")
# ============================================================
# PATHS
# ============================================================
#mlflow server --host 127.0.0.1 --port 5050
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "src" / "models" / "churn_pipeline.joblib"
FEATURES_PATH = ROOT / "src" / "models" / "transformed_features.npy"
TARGET_PATH = ROOT / "src" / "models" / "target.npy"
# ============================================================
# MLFLOW
# ============================================================
MLFLOW_URI = "http://127.0.0.1:5050"
EXPERIMENT = "Customer Churn Prediction"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)
# ============================================================
# TEST
# ============================================================
def test_model_prediction():
    print("\n" + "=" * 50)
    print("        MODEL PREDICTION TEST RUNNER")
    print("=" * 50)
    print(f"Model    : {MODEL_PATH}")
    print(f"Features : {FEATURES_PATH}")
    print(f"Target   : {TARGET_PATH}")
    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------
    if not MODEL_PATH.exists():
        pytest.fail(f"Model not found: {MODEL_PATH}")
    if not FEATURES_PATH.exists():
        pytest.fail(f"Features not found: {FEATURES_PATH}")
    if not TARGET_PATH.exists():
        pytest.fail(f"Target not found: {TARGET_PATH}")
    print("\n✓ All required files found.")
    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    print("\nLoading model...")
    model = joblib.load(MODEL_PATH)
    print("✓ Model loaded successfully.")
    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    X = np.load(FEATURES_PATH,allow_pickle=False)
    y = np.load(TARGET_PATH,allow_pickle=True).ravel()
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape  : {y.shape}")
    # --------------------------------------------------------
    # Sample input
    # --------------------------------------------------------
    real_input = X[:5]
    print("\n--- Input Sample ---")
    print(real_input)
    print(f"Input shape: {real_input.shape}")
    # --------------------------------------------------------
    # Get classifier
    # --------------------------------------------------------
    if hasattr(model, "steps"):
        classifier = model.steps[-1][1]
    else:
        classifier = model
    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------
    print("\nRunning prediction...")
    probabilities = classifier.predict_proba(real_input)[:, 1]
    predictions = classifier.predict(real_input)
    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    output = pd.DataFrame({
        "churn_probability": probabilities,
        "churn_prediction": predictions
    })
    print("\n--- Prediction Output ---")
    print(output)
    # --------------------------------------------------------
    # Test validations
    # --------------------------------------------------------
    assert len(output) == 5
    assert "churn_probability" in output.columns
    assert "churn_prediction" in output.columns
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert len(predictions) == 5
    # --------------------------------------------------------
    # Test statistics
    # --------------------------------------------------------
    total_predictions = len(predictions)
    churn_count = int(np.sum(predictions))
    no_churn_count = (total_predictions - churn_count)
    average_probability = float(np.mean(probabilities))
    maximum_probability = float(np.max(probabilities))
    minimum_probability = float(np.min(probabilities))
    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------
    print("\n--- Test Results ---")
    print(f"Total predictions : {total_predictions}")
    print(f"Predicted churn   : {churn_count}")
    print(f"Predicted no churn: {no_churn_count}")
    print(f"Average probability: {average_probability:.4f}")
    print("\nFirst Customer")
    print(f"Probability: {probabilities[0]:.4f}")
    print(f"Prediction : {'CHURN' if predictions[0] == 1 else 'NO CHURN'}")
    # ========================================================
    # MLFLOW TRACKING
    # ========================================================
    print("\nLogging test results to MLflow...")
    with mlflow.start_run(run_name="Model_Prediction_Test") as run:
        # Parameters
        mlflow.log_params({
            "test_type": "prediction_test",
            "model": "churn_pipeline",
            "sample_size": 5,
            "feature_count": X.shape[1],
            "prediction_threshold": 0.5
        })
        # Metrics
        mlflow.log_metrics({
            "total_predictions": total_predictions,
            "predicted_churn": churn_count,
            "predicted_no_churn": no_churn_count,
            "average_churn_probability": average_probability,
            "maximum_churn_probability": maximum_probability,
            "minimum_churn_probability": minimum_probability
        })
        # Tags
        mlflow.set_tags({
            "test": "model_prediction",
            "model_type": "XGBoost",
            "status": "passed"
        })
        # Save prediction output
        output_path = ROOT / "prediction_test_results.csv"
        output.to_csv(output_path,index=False)
        mlflow.log_artifact(str(output_path),artifact_path="prediction_test")
        print("\n" + "=" * 50)
        print("       MLFLOW TEST COMPLETED")
        print("=" * 50)
        print(f"Run ID : {run.info.run_id}")
        print(f"MLflow : {MLFLOW_URI}")
        print("=" * 50)
    print("\n✓ MODEL PREDICTION TEST PASSED")
if __name__ == "__main__":
    test_model_prediction()

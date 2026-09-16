import sys
from pathlib import Path

# Add project root (CCP directory) to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Existing imports follow
import os
import pytest
import pandas as pd
import numpy as np
from src.models.predict import ChurnPredictor


def test_model_prediction():
    # 1. Resolve absolute paths relative to project root (CCP folder)
    model_path = PROJECT_ROOT / "src" / "models" / "churn_model.joblib"
    features_path = PROJECT_ROOT / "src" / "models" / "transformed_features.npy"
    target_path = PROJECT_ROOT / "src" / "models" / "target.npy"

    print("\n==========================================")
    print("        MODEL PREDICTION TEST RUNNER       ")
    print("==========================================")
    print(f"Checking model binary at: {model_path}")
    print(f"Checking transformed features at: {features_path}")
    print(f"Checking target data at: {target_path}")

    # 2. Check if the trained model exists
    if not model_path.exists():
        pytest.fail(
            f"❌ Trained model binary not found at '{model_path}'. "
            "Please run 'python -m src.models.train' first."
        )

    # 3. Check if the transformed features and target data exist
    if not features_path.exists():
        pytest.fail(f"❌ Transformed features not found at '{features_path}'. Please check file path.")
    if not target_path.exists():
        pytest.fail(f"❌ Target data not found at '{target_path}'. Please check file path.")

    # 4. Load trained model
    print("✓ Model file found. Instantiating ChurnPredictor...")
    predictor = ChurnPredictor(str(model_path))

    # 5. Load transformed features and target data
    print("✓ Loading transformed features and target data...")
    X = np.load(features_path, allow_pickle=False)
    y = np.load(target_path, allow_pickle=True)

    print(f"Transformed Features Shape: {X.shape}")
    print(f"Target Data Shape: {y.shape}")

    # 6. Select a sample of input data for prediction
    print("✓ Selecting a sample of input data for prediction...")
    real_input = pd.DataFrame(X[:5])  # Take the first 5 rows as input
    print("\n--- Real Input DataFrame Sample ---")
    print(real_input)
    print(f"Input Shape: {real_input.shape}")
    print(f"Input Columns: {list(real_input.columns)}")

    # 7. Run prediction pipeline
    print("\n--- Running Prediction Pipeline ---")
    out = predictor.predict(real_input)

    print("\n--- Model Output DataFrame ---")
    print(out)
    print(f"Output Columns: {list(out.columns)}")
    print(f"Output Shape: {out.shape}")

    # 8. Print prediction results summary for the first row
    prob = out["churn_probability"].iloc[0]
    pred = out["churn_prediction"].iloc[0]
    print("\n--- Prediction Results Summary (Row 0) ---")
    print(f"Calculated Churn Probability: {prob:.4f}")
    print(f"Final Churn Class Prediction: {pred}")
    print("==========================================\n")


if __name__ == "__main__":
    test_model_prediction()
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.predict import ChurnPredictor
import warnings
# Suppress warnings before importing other modules
from sklearn.exceptions import InconsistentVersionWarning

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Suppress XGBoost serialization warnings
warnings.filterwarnings("ignore", message=".*WARNING:.*If you are loading a serialized model.*")

def test_model_prediction():
    # 1. Define paths to model and transformed data
    model_path = Path("E:/PROJECTS_FILE/CCP/src/models/churn_pipeline.joblib")
    features_path = Path("E:/PROJECTS_FILE/CCP/src/models/transformed_features.npy")
    target_path = Path("E:/PROJECTS_FILE/CCP/src/models/target.npy")

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
    real_input = X[:5]  # Take the first 5 rows as input
    print("\n--- Real Input DataFrame Sample ---")
    print(real_input)
    print(f"Input Shape: {real_input.shape}")

    # 7. Run prediction pipeline (skip preprocessing)
    print("\n--- Running Prediction Pipeline ---")
    # Directly use the model for prediction since the data is already transformed
    probabilities = predictor.model.steps[-1][1].predict_proba(real_input)[:, 1]  # Access the classifier directly
    predictions = predictor.model.steps[-1][1].predict(real_input)

    # 8. Create output DataFrame
    out = pd.DataFrame({
        "churn_probability": probabilities,
        "churn_prediction": predictions
    })

    print("\n--- Model Output DataFrame ---")
    print(out)
    print(f"Output Columns: {list(out.columns)}")
    print(f"Output Shape: {out.shape}")

    # 9. Print prediction results summary for the first row
    prob = out["churn_probability"].iloc[0]
    pred = out["churn_prediction"].iloc[0]
    print("\n--- Prediction Results Summary (Row 0) ---")
    print(f"Calculated Churn Probability: {prob:.4f}")
    print(f"Final Churn Class Prediction: {pred}")
    print("==========================================\n")


if __name__ == "__main__":
    test_model_prediction()
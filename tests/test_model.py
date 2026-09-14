import os
import pytest
import pandas as pd
from pathlib import Path
from src.models.predict import ChurnPredictor

def test_model_prediction():
    # 1. Define absolute paths directly
    model_path = Path("E:/PROJECTS_FILE/CCP/models/models/churn_pipeline.joblib")
    data_path = Path("E:/PROJECTS_FILE/CCP/data/raw/cell2celltrain.csv")
    
    print("\n==========================================")
    print("        MODEL PREDICTION TEST RUNNER       ")
    print("==========================================")
    print(f"Checking model binary at: {model_path}")
    print(f"Checking dataset at:      {data_path}")
    
    # 2. Check if the trained model exists
    if not model_path.exists():
        pytest.fail(
            f"❌ Trained model binary not found at '{model_path}'. "
            "Please run 'python -m src.models.train' first."
        )
        
    # 3. Check if the real dataset exists
    if not data_path.exists():
        pytest.fail(f"❌ Original dataset not found at '{data_path}'. Please check file path.")

    # 4. Load trained model
    print("✓ Model file found. Instantiating ChurnPredictor...")
    predictor = ChurnPredictor(str(model_path))
    
    # 5. Read real dataset sample
    print("✓ Reading sample from original dataset...")
    full_df = pd.read_csv(data_path)
    
    # Exclude the target column if it exists in raw data
    target_column = "churn"  # Replace with your actual target column name if present
    if target_column in full_df.columns:
        real_input = full_df.drop(columns=[target_column]).head(5)
    else:
        real_input = full_df.head(5)
        
    print("\n--- Real Input DataFrame Sample ---")
    print(real_input)
    print(f"Input Shape: {real_input.shape}")
    print(f"Input Columns: {list(real_input.columns)}")
    
    # 6. Run prediction pipeline
    print("\n--- Running Prediction Pipeline ---")
    out = predictor.predict(real_input)
    
    print("\n--- Model Output DataFrame ---")
    print(out)
    print(f"Output Columns: {list(out.columns)}")
    print(f"Output Shape: {out.shape}")
    
    # 7. Print prediction results summary for the first row
    prob = out["churn_probability"].iloc[0]
    pred = out["churn_prediction"].iloc[0]
    print("\n--- Prediction Results Summary (Row 0) ---")
    print(f"Calculated Churn Probability: {prob:.4f}")
    print(f"Final Churn Class Prediction: {pred}")
    print("==========================================\n")


if __name__ == "__main__":
    test_model_prediction()

import joblib
import pandas as pd
import yaml
import os
from pathlib import Path
class ChurnPredictor:
    def __init__(self, model_path: str = "models/churn_pipeline.joblib"):
        # Load the trained pipeline (preprocessing + model)
        self.pipeline = joblib.load(model_path)
        print(f"Model loaded successfully from: {model_path}")

    def predict(self, input_df: pd.DataFrame):
        # Ensure column names are lowercase to match the training pipeline
        input_df.columns = input_df.columns.str.lower()
        print("Input DataFrame columns converted to lowercase.")

        # Get the list of required columns from the pipeline
        required_columns = self.pipeline.named_steps['preprocessor'].feature_names_in_

        # Add missing columns with default values
        for col in required_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Ensure the input DataFrame has the exact column order required
        input_df = input_df[required_columns]

        # Calculate probabilities and predictions
        probabilities = self.pipeline.predict_proba(input_df)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        # Return a DataFrame with predictions and probabilities
        return pd.DataFrame({
            'churn_probability': probabilities,
            'churn_prediction': predictions
        })

if __name__ == "__main__":
    # Hardcoded paths for model and dataset
    model_path = "models/models/models\churn_pipeline.joblib"
    raw_data_path = Path("E:/PROJECTS_FILE/CCP/data/raw/cell2celltrain.csv")

    print(f"\n--- Loading Model from '{model_path}' ---")
    predictor = ChurnPredictor(model_path=model_path)

    print(f"\n--- Loading Real Dataset from '{raw_data_path}' ---")
    real_data = pd.read_csv(raw_data_path)
    print(f"Loaded Raw Dataset Shape: {real_data.shape}")

    # Remove target and ID columns if present before running inference
    target_col = "churn".lower()
    id_col = "customerid".lower()
    
    inference_data = real_data.drop(columns=[target_col, id_col], errors='ignore')

    # Execute predictions on the real dataset
    print("\n--- Running Predictions on Real Dataset ---")
    output = predictor.predict(inference_data)

    # Attach customer ID back to predictions for identification
    if id_col in real_data.columns:
        output.insert(0, "customerid", real_data["customerid"])

    print("\n--- Real Dataset Predictions Head (Top 10 Results) ---")
    print(output.head(10))

    print("\n--- Prediction Summary ---")
    print(output['churn_prediction'].value_counts())

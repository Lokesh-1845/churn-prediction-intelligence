import pandas as pd
from typing import List, Tuple
from src.data.ingestion import DataIngestion  # Import the DataIngestion class
import os

class DataValidation:
    def __init__(self, required_columns: List[str]):
        # Initialize with required columns and convert them to lowercase
        self.required_columns = [col.lower() for col in required_columns]
        print(f"Required columns for validation: {self.required_columns}")

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        # Convert DataFrame columns to lowercase
        df_cols = [col.lower() for col in df.columns]
        print(f"Columns in the provided DataFrame: {df_cols}")

        # Identify missing columns
        missing_cols = [col for col in self.required_columns if col not in df_cols]
        is_valid = len(missing_cols) == 0

        # Print validation results
        if is_valid:
            print("Validation successful: All required columns are present.")
        else:
            print(f"Validation failed: Missing columns: {missing_cols}")

        return is_valid, missing_cols

if __name__ == "__main__":
    # Initialize DataIngestion
    ingestion = DataIngestion()

    # Load raw data using DataIngestion
    print("Loading raw data using DataIngestion...")
    df = ingestion.load_raw_data()
    print(f"Data loaded successfully. Shape: {df.shape}")

    # Define required columns
    required_columns = ["customerid", "churn", "monthlyrevenue", "monthlyminutes"]
    validator = DataValidation(required_columns)

    # Validate schema
    is_valid, missing_cols = validator.validate_schema(df)
    print(f"Is the DataFrame valid? {is_valid}")
    if not is_valid:
        print(f"Missing columns: {missing_cols}")
    else:
        # Save the validated DataFrame to the specified folder
        output_folder = "E:/PROJECTS_FILE/CCP/src/features/validated_data"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "validated_data.csv")
        df.to_csv(output_path, index=False)
        print(f"Validated data saved to: {output_path}")

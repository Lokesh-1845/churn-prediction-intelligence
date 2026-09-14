import pandas as pd
from typing import List, Tuple

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
    # Load configuration
    '''    config_path = "configs/config.yaml"
    def load_config(config_path: str) -> dict:
        import yaml
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    config = load_config(config_path)
    print(f"Configuration loaded from: {config_path}")'''

    # Load the real project data
    raw_data_path = "E:/PROJECTS_FILE/CCP/data/raw/cell2celltrain.csv"
    print(f"Loading raw data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    print(f"Raw data loaded successfully. Shape: {df.shape}")

    # Define required columns
    required_columns = ["customerid", "churn", "monthlyrevenue", "monthlyminutes"]
    validator = DataValidation(required_columns)

    # Validate schema
    is_valid, missing_cols = validator.validate_schema(df)
    print(f"Is the DataFrame valid? {is_valid}")
    if not is_valid:
        print(f"Missing columns: {missing_cols}")

'''
if __name__ == "__main__":
    # Example usage
    required_columns = ["customerid", "churn", "monthlyrevenue", "monthlyminutes"]
    validator = DataValidation(required_columns)

    # Example DataFrame
    data = {
        "CustomerID": [1, 2, 3],
        "Churn": ["Yes", "No", "Yes"],
        "MonthlyRevenue": [50.0, 70.0, 60.0]
    }
    df = pd.DataFrame(data)

    # Validate schema
    is_valid, missing_cols = validator.validate_schema(df)
    print(f"Is the DataFrame valid? {is_valid}")
    if not is_valid:
        print(f"Missing columns: {missing_cols}")
'''
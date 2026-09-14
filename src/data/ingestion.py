import pandas as pd

class DataIngestion:
    def __init__(self):
        # Directly define the configuration dictionary
        self.config = {
            "raw_data_path": "E:/PROJECTS_FILE/CCP/data/raw/cell2celltrain.csv",
            "processed_data_path": "data/processed/cleaned_churn_data.csv"            }
        print("Configuration loaded directly from the script.")
        print(f"Raw data path: {self.config['raw_data_path']}")
        print(f"Processed data path: {self.config['processed_data_path']}")

    def load_raw_data(self) -> pd.DataFrame:
        raw_path = self.config["raw_data_path"]
        print(f"Loading raw data from: {raw_path}")
        df = pd.read_csv(raw_path)
        print(f"Raw data loaded successfully. Shape: {df.shape}")
        df.columns = df.columns.str.lower()
        print("Column names converted to lowercase.")
        return df

    def save_processed_data(self, df: pd.DataFrame):
        processed_path = self.config["processed_data_path"]
        print(f"Saving processed data to: {processed_path}")
        df.to_csv(processed_path, index=False)
        print(f"Processed data saved successfully. Shape: {df.shape}")

if __name__ == "__main__":
    # Initialize DataIngestion
    ingestion = DataIngestion()

    # Load raw data
    raw_data = ingestion.load_raw_data()

    # Save processed data
    ingestion.save_processed_data(raw_data)
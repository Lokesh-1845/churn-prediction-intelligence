import pandas as pd


class DataIngestion:
    """
    Handles all customer dataset ingestion.

    Important:
    - Original customer IDs are preserved.
    - Customer names are never generated.
    - Uploaded CSV files are loaded through this class.
    """

    def __init__(self):
        self.config = {
            "raw_data_path": (
                "E:/PROJECTS_FILE/CCP/data/raw/"
                "cell2celltrain.csv"
            ),
            "processed_data_path": (
                "data/processed/"
                "cleaned_churn_data.csv"
            ),
        }

        print(
            "Configuration loaded directly from the script."
        )

        print(
            f"Raw data path: "
            f"{self.config['raw_data_path']}"
        )

        print(
            f"Processed data path: "
            f"{self.config['processed_data_path']}"
        )

    # ========================================================
    # ORIGINAL PROJECT DATASET
    # ========================================================

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load the original project dataset.
        """

        raw_path = self.config["raw_data_path"]

        print(
            f"Loading raw data from: {raw_path}"
        )

        df = pd.read_csv(raw_path)

        print(
            "Raw data loaded successfully. "
            f"Shape: {df.shape}"
        )

        # Normalize column names
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.lower()
        )

        print(
            "Column names normalized to lowercase."
        )

        return df

    # ========================================================
    # UPLOADED DATASET
    # ========================================================

    def load_uploaded_data(
        self,
        uploaded_file
    ) -> pd.DataFrame:
        """
        Load a CSV file uploaded through Streamlit.

        The uploaded file is read directly here so that
        dashboard.py does not handle CSV reading itself.
        """

        if uploaded_file is None:
            raise ValueError(
                "No file was uploaded."
            )

        file_name = getattr(
            uploaded_file,
            "name",
            ""
        )

        if not str(file_name).lower().endswith(".csv"):
            raise ValueError(
                "Only CSV files are supported."
            )

        print(
            f"Loading uploaded dataset: {file_name}"
        )

        df = pd.read_csv(uploaded_file)

        print(
            "Uploaded dataset loaded successfully. "
            f"Shape: {df.shape}"
        )

        # Normalize column names
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.lower()
        )

        print(
            "Uploaded column names normalized "
            "to lowercase."
        )

        return df

    # ========================================================
    # SAVE PROCESSED DATA
    # ========================================================

    def save_processed_data(
        self,
        df: pd.DataFrame
    ):
        """
        Save a DataFrame to the configured processed-data path.
        """

        processed_path = self.config[
            "processed_data_path"
        ]

        print(
            f"Saving processed data: "
            f"{processed_path}"
        )

        df.to_csv(
            processed_path,
            index=False
        )

        print(
            "Processed data saved successfully. "
            f"Shape: {df.shape}"
        )


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    ingestion = DataIngestion()

    raw_data = (
        ingestion.load_raw_data()
    )

    ingestion.save_processed_data(
        raw_data
    )
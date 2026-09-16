import pandas as pd


class DataIngestion:

    def __init__(self):

        self.config = {
            "raw_data_path":
                "E:/PROJECTS_FILE/CCP/data/raw/cell2celltrain.csv",

            "processed_data_path":
                "data/processed/cleaned_churn_data.csv"
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
    # LOAD ORIGINAL DATASET
    # ========================================================

    def load_raw_data(self) -> pd.DataFrame:

        raw_path = self.config[
            "raw_data_path"
        ]

        print(
            f"Loading raw data from: {raw_path}"
        )

        df = pd.read_csv(
            raw_path
        )

        print(
            f"Raw data loaded successfully. "
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
            "Column names converted to lowercase."
        )

        return df


    # ========================================================
    # LOAD UPLOADED CSV
    # ========================================================

    def load_uploaded_data(
        self,
        uploaded_file
    ) -> pd.DataFrame:

        """
        Load a CSV uploaded from Streamlit.

        IMPORTANT:
        The dashboard does NOT directly read the uploaded CSV.
        The uploaded file comes here first.
        """

        if uploaded_file is None:

            raise ValueError(
                "No uploaded file was provided."
            )


        file_name = getattr(
            uploaded_file,
            "name",
            "uploaded_file.csv"
        )


        # ----------------------------------------------------
        # Validate file type
        # ----------------------------------------------------

        if not file_name.lower().endswith(
            ".csv"
        ):

            raise ValueError(
                "Only CSV files are supported."
            )


        print(
            f"Loading uploaded file: {file_name}"
        )


        # ----------------------------------------------------
        # Read CSV
        # ----------------------------------------------------

        df = pd.read_csv(
            uploaded_file
        )


        # ----------------------------------------------------
        # Normalize columns
        # ----------------------------------------------------

        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.lower()
        )


        print(
            "Uploaded CSV loaded successfully."
        )

        print(
            f"Uploaded dataset shape: {df.shape}"
        )

        print(
            "Uploaded column names converted "
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

        processed_path = self.config[
            "processed_data_path"
        ]

        print(
            f"Saving processed data to: "
            f"{processed_path}"
        )

        df.to_csv(
            processed_path,
            index=False
        )

        print(
            f"Processed data saved successfully. "
            f"Shape: {df.shape}"
        )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    ingestion = DataIngestion()

    raw_data = (
        ingestion.load_raw_data()
    )

    ingestion.save_processed_data(
        raw_data
    )
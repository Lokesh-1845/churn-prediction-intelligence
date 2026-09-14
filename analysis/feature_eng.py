import pandas as pd
import numpy as np
from pathlib import Path
def clean_raw_df(df, config):
    df = df.copy()
    df.columns = df.columns.str.lower()
    print("Columns converted to lowercase.")
    
    df = df.drop_duplicates()
    print(f"Duplicate rows removed. Remaining rows: {df.shape[0]}")

    # Standardize unknown markers to NaN
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace(config['unknown_markers'], np.nan)
    print("Unknown markers standardized to NaN.")

    # Binary target encoding
    if config['target_column'] in df.columns:
        df[config['target_column']] = df[config['target_column']].map({'Yes': 1, 'No': 0})
        print(f"Target column '{config['target_column']}' encoded as binary.")

    print("Final DataFrame after cleaning:")
    print(df.head())
    print(f"Shape of cleaned DataFrame: {df.shape}")
    print(f"Columns in cleaned DataFrame: {df.columns.tolist()}")
    return df


if __name__ == "__main__":
    # Load configuration
    config_path =   Path("E:/PROJECTS_FILE/CCP/data/raw/cell2celltrain.csv")
    config = {
        'unknown_markers': ['Unknown', 'NA', 'None', 'null', '?'],
        'target_column': 'churn',
        'raw_data_path': 'data/raw/cell2celltrain.csv',
        'processed_data_path': 'data/processed/cleaned_churn_data.csv'
    }
    print(f"Configuration loaded from: {config_path}")

    # Load raw data
    raw_data_path = config["raw_data_path"]
    print(f"Loading raw data from: {raw_data_path}")
    raw_df = pd.read_csv(raw_data_path)
    print(f"Raw data loaded successfully. Shape: {raw_df.shape}")

    # Clean the raw DataFrame
    cleaned_df = clean_raw_df(raw_df, config)

    # Save the cleaned data
    processed_data_path = config["processed_data_path"]
    print(f"Saving cleaned data to: {processed_data_path}")
    cleaned_df.to_csv(processed_data_path, index=False)
    print(f"Cleaned data saved successfully. Shape: {cleaned_df.shape}")





'''''
    if __name__ == "__main__":
    # Example configuration
    config = {
        'unknown_markers': ['Unknown', 'NA', 'None', 'null', '?'],
        'target_column': 'churn'
    }

    # Example DataFrame
    data = {
        'CustomerID': [1, 2, 3, 4, 5],
        'Churn': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'MonthlyRevenue': [50.0, 70.0, 60.0, 80.0, 90.0],
        'UnknownColumn': ['Unknown', 'NA', 'None', 'null', '?']
    }
    df = pd.DataFrame(data)

    print("Original DataFrame:")
    print(df)

    # Clean the raw DataFrame
    cleaned_df = clean_raw_df(df, config)

    print("\nCleaned DataFrame:")
    print(cleaned_df)'''
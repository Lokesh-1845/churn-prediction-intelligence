import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class UnknownMarkerCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, markers=None):
        self.markers = markers or ["Unknown", "?", "NA", "null", "none"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = pd.DataFrame(X).copy()
        print("\n--- [UnknownMarkerCleaner] Starting Cleaning ---")
        print(f"Targeting unknown markers: {self.markers}")
        
        replaced_count = 0
        for col in X_out.columns:
            if X_out[col].dtype == 'object':
                matches = X_out[col].isin(self.markers).sum()
                replaced_count += matches
                X_out[col] = X_out[col].replace(self.markers, np.nan)
                
        print(f"Cleaned {replaced_count} unknown values -> replaced with NaN.")
        return X_out


def build_preprocessing_pipeline(num_cols: list, cat_cols: list) -> ColumnTransformer:
    print("\n==========================================")
    print("      BUILDING PREPROCESSING PIPELINE     ")
    print("==========================================")
    print(f"Numerical Columns ({len(num_cols)}): {len(num_cols)}")
    print(f"Categorical Columns ({len(cat_cols)}): {len(cat_cols)}")

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('cleaner', UnknownMarkerCleaner()),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )
    print("Pipeline construct successfully built.")
    return preprocessor


# =========================================================
# Real Dataset Execution Block
# =========================================================
if __name__ == "__main__":
    raw_data_path = "data/raw/cell2celltrain.csv"

    if os.path.exists(raw_data_path):
        print(f"\nLoading real dataset from: {raw_data_path}")
        df_real = pd.read_csv(raw_data_path)
        df_real.columns = df_real.columns.str.lower()

        # Define target and ID columns directly
        target_col = "churn"
        id_col = "customerid"

        X_real = df_real.drop(columns=[target_col, id_col], errors='ignore')

        # Automatically partition columns present in the dataset
        num_cols = X_real.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X_real.select_dtypes(include=['object', 'category']).columns.tolist()

        # 1. Build pipeline
        pipeline = build_preprocessing_pipeline(num_cols, cat_cols)

        # 2. Process production data
        print("\nExecuting fit_transform() on real dataset...")
        transformed_matrix = pipeline.fit_transform(X_real)

        # 3. Print production outputs
        print("\n--- Real Dataset Transformation Complete ---")
        print(f"Original Raw Data Shape: {df_real.shape}")
        print(f"Transformed Feature Matrix Shape: {transformed_matrix.shape}")
        print(f"Total Features Generated: {transformed_matrix.shape[1]}")
    else:
        print(f"\n❌ Dataset file not found at '{raw_data_path}'. Ensure 'cell2celltrain.csv' is saved under data/raw/")
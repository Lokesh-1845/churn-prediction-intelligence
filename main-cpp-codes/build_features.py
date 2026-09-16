import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib  # For saving the pipeline

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
    print(f"Numerical Columns ({len(num_cols)}): {num_cols}")
    print(f"Categorical Columns ({len(cat_cols)}): {cat_cols}")

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
    # Dynamically construct the path to the validated data
    base_dir = "E:/PROJECTS_FILE/CCP/src/features/validated_data"
    validated_data_filename = "validated_data.csv"
    validated_data_path = os.path.join(base_dir, validated_data_filename)

    if os.path.exists(validated_data_path):
        print(f"\nLoading validated dataset from: {validated_data_path}")
        df_validated = pd.read_csv(validated_data_path)
        df_validated.columns = df_validated.columns.str.lower()

        # Define target and ID columns directly
        target_col = "churn"
        id_col = "customerid"

        X_validated = df_validated.drop(columns=[target_col, id_col], errors='ignore')
        y_validated = df_validated[target_col]

        # Automatically partition columns present in the dataset
        num_cols = X_validated.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X_validated.select_dtypes(include=['object', 'category']).columns.tolist()

        # 1. Build pipeline
        pipeline = build_preprocessing_pipeline(num_cols, cat_cols)

        # 2. Process validated data
        print("\nExecuting fit_transform() on validated dataset...")
        transformed_matrix = pipeline.fit_transform(X_validated)

        # 3. Save the transformed data and pipeline
        output_dir = "E:/PROJECTS_FILE/CCP/src/models"
        os.makedirs(output_dir, exist_ok=True)

        # Save the transformed feature matrix
        transformed_data_path = os.path.join(output_dir, "transformed_features.npy")
        np.save(transformed_data_path, transformed_matrix)
        print(f"Transformed feature matrix saved to: {transformed_data_path}")

        # Save the target column
        target_data_path = os.path.join(output_dir, "target.npy")
        np.save(target_data_path, y_validated.values)
        print(f"Target data saved to: {target_data_path}")

        # Save the preprocessing pipeline
        pipeline_path = os.path.join(output_dir, "preprocessing_pipeline.pkl")
        joblib.dump(pipeline, pipeline_path)
        print(f"Preprocessing pipeline saved to: {pipeline_path}")

        # 4. Print transformation outputs
        print("\n--- Validated Dataset Transformation Complete ---")
        print(f"Original Validated Data Shape: {df_validated.shape}")
        print(f"Transformed Feature Matrix Shape: {transformed_matrix.shape}")
        print(f"Total Features Generated: {transformed_matrix.shape[1]}")
    else:
        print(f"\n❌ Validated dataset file not found at '{validated_data_path}'. Ensure 'validated_data.csv' is saved under the correct directory.")
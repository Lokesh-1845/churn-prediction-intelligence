'''import os
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from src.utils.helpers import load_config
from src.data.ingestion import DataIngestion
from src.features.build_features import build_preprocessing_pipeline

def train_model(config_path="configs/config.yaml"):
    config = load_config(config_path)
    ingestion = DataIngestion(config_path)
    df = ingestion.load_raw_data()

    target = config["target_column"]
    df = df.dropna(subset=[target])
    
    y = df[target].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
    X = df.drop(columns=[target, config["id_column"]], errors='ignore')

    num_cols = [c for c in config["numerical_columns"] if c in X.columns]
    cat_cols = [c for c in config["categorical_columns"] if c in X.columns]

    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    model = LGBMClassifier(
        random_state=config["random_state"],
        n_estimators=100,
        learning_rate=0.05
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(config["model_path"]), exist_ok=True)
    joblib.dump(pipeline, config["model_path"])
    print(f"Model successfully saved to {config['model_path']}")

if __name__ == "__main__":
    train_model()'''




import sys
import joblib
import pandas as pd

from pathlib import Path
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from src.features.build_features import build_preprocessing_pipeline


# ============================================================
# PROJECT ROOT
# ============================================================

# train.py:
# CCP/
#   src/
#     models/
#       train.py
#
# parent        -> models
# parent.parent -> src
# parent.parent.parent -> CCP

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    data_path = PROJECT_ROOT / "data" / "raw" / "cell2celltrain.csv"

    print("=" * 60)
    print("       CUSTOMER CHURN MODEL TRAINING")
    print("=" * 60)

    print(f"\nDataset path:")
    print(data_path)

    if not data_path.exists():

        raise FileNotFoundError(
            f"Dataset file not found:\n{data_path}"
        )

    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Standardize column names
    # --------------------------------------------------------

    df.columns = df.columns.str.lower()

    # --------------------------------------------------------
    # Target + ID
    # --------------------------------------------------------

    target = "churn"
    id_col = "customerid"

    if target not in df.columns:

        raise ValueError(
            f"Target column '{target}' not found."
        )

    # Remove rows with missing target

    df = df.dropna(subset=[target])

    # --------------------------------------------------------
    # Convert target to binary
    # --------------------------------------------------------

    y = df[target].map({
        "Yes": 1,
        "No": 0,
        "yes": 1,
        "no": 0,
        1: 1,
        0: 0
    })

    # Check for unmapped values

    if y.isna().any():

        unknown_values = df.loc[
            y.isna(),
            target
        ].unique()

        raise ValueError(
            f"Unknown target values found: {unknown_values}"
        )

    y = y.astype(int)

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = df.drop(
        columns=[target, id_col],
        errors="ignore"
    )

    print(f"Feature matrix shape: {X.shape}")

    # --------------------------------------------------------
    # Detect feature types
    # --------------------------------------------------------

    num_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    cat_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print(f"\nNumerical features : {len(num_cols)}")
    print(f"Categorical features: {len(cat_cols)}")

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    preprocessor = build_preprocessing_pipeline(
        num_cols,
        cat_cols
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    print("\nInitializing XGBoost...")

    model = XGBClassifier(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        eval_metric="logloss"
    )

    # --------------------------------------------------------
    # Complete Pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n--- Training Model Pipeline ---")

    pipeline.fit(X, y)

    print("✓ Model training completed successfully.")

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_dir = PROJECT_ROOT / "models"

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = model_dir / "churn_pipeline.joblib"

    joblib.dump(
        pipeline,
        model_path
    )

    print(
        f"\n✓ Trained model pipeline saved at:"
        f"\n  {model_path}"
    )

    print("\nXGBoost version used for training:")

    import xgboost

    print(f"  {xgboost.__version__}")

    print("\n" + "=" * 60)
    print("       TRAINING COMPLETED")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_model()



    
'''
import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.utils.helpers import load_config  # Assuming this loads the config file
from src.data.ingestion import DataIngestion  # Assuming this handles data ingestion
from src.features.build_features import build_preprocessing_pipeline  # Assuming this builds preprocessing pipeline

def train_model(config_path="configs/config.yaml"):
    # Load configuration
    config = load_config(config_path)
    ingestion = DataIngestion(config_path)
    df = ingestion.load_raw_data()

    # Prepare target and features
    target = config["target_column"]
    df = df.dropna(subset=[target])
    
    y = df[target].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
    X = df.drop(columns=[target, config["id_column"]], errors='ignore')

    # Identify numerical and categorical columns
    num_cols = [c for c in config["numerical_columns"] if c in X.columns]
    cat_cols = [c for c in config["categorical_columns"] if c in X.columns]

    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)

    # Define the Gradient Boosting model
    model = GradientBoostingClassifier(
        random_state=config["random_state"],
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6
    )

    # Create a pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Train the model
    pipeline.fit(X, y)
    print("Model training completed.")
    # Save the trained pipeline
    model_path = os.path.join(config["model_dir"], config["model_path"])
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure directory exists
    joblib.dump(pipeline, model_path)
    print(f"Trained model saved at: {model_path}")

if __name__ == "__main__":
    train_model(config_path="configs/config.yaml")
    '''
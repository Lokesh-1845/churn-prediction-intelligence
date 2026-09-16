'''# ============================================================
# CUSTOMER CHURN INTELLIGENCE DASHBOARD
# ============================================================
#
# NEW WORKFLOW
#
# WINDOW 1:
#   Upload Dataset + README
#
# WINDOW 2:
#   Top Navigation:
#       1. Overview
#       2. Customer Risk
#       3. SHAP Explainability
#       4. Business Impact Estimator
#
# ML WORKFLOW:
#   Uploaded CSV
#        ↓
#   Data Validation
#        ↓
#   Preprocessing
#        ↓
#   Train / Test Split
#        ↓
#   XGBoost Training
#        ↓
#   Evaluation internally
#        ↓
#   Customer Predictions
#        ↓
#   Risk Scoring
#        ↓
#   SHAP Explainability
#        ↓
#   Business Impact
#
# IMPORTANT:
#   No project dataset is loaded in the background.
#   The uploaded dataset is the ONLY dataset used.
#
# ============================================================

from __future__ import annotations

import hashlib
import warnings
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(textwrap.dedent(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background: #071A33;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }


    /* ======================================================
       LANDING PAGE
       ====================================================== */

    .hero-container {
        text-align: center;
        padding-top: 45px;
        padding-bottom: 20px;
    }

    .hero-icon {
        font-size: 60px;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 850;
        color: #f8fafc;
        letter-spacing: -1.5px;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #cbd5e1;
        max-width: 850px;
        margin: auto;
        line-height: 1.7;
    }


    /* ======================================================
       UPLOAD CARD
       ====================================================== */

    .upload-card {
        background: rgba(255,255,255,0.95);
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        padding: 35px;
        margin-top: 35px;
        box-shadow:
            0 15px 45px rgba(15,23,42,0.08);
    }


    /* ======================================================
       README CARDS
       ====================================================== */

    .readme-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 24px;
        min-height: 185px;
        box-shadow:
            0 5px 20px rgba(15,23,42,0.04);
    }

    .readme-icon {
        font-size: 30px;
        margin-bottom: 10px;
    }

    .readme-title {
        font-size: 17px;
        font-weight: 750;
        color: #f8fafc;
        margin-bottom: 8px;
    }

    .readme-text {
        font-size: 13px;
        color: #cbd5e1;
        line-height: 1.65;
    }


    /* ======================================================
       TOP NAVIGATION
       ====================================================== */

    .top-brand {
        font-size: 26px;
        font-weight: 850;
        color: #f8fafc;
        margin-bottom: 4px;
    }

    .top-description {
        color: #cbd5e1;
        font-size: 13px;
        margin-bottom: 12px;
    }


    /* ======================================================
       PAGE HEADER
       ====================================================== */

    .dashboard-title {
        font-size: 35px;
        font-weight: 850;
        color: #f8fafc;
        letter-spacing: -0.8px;
        margin-top: 18px;
        margin-bottom: 5px;
    }

    .dashboard-subtitle {
        color: #cbd5e1;
        font-size: 15px;
        margin-bottom: 25px;
    }


    /* ======================================================
       METRIC CARDS
       ====================================================== */

    .metric-card {
        background: #ffffff;
        border-radius: 17px;
        padding: 22px;
        border: 1px solid #e2e8f0;
        box-shadow:
            0 5px 18px rgba(15,23,42,0.05);
        min-height: 135px;
    }

    .metric-label {
        color: #cbd5e1;
        font-size: 12px;
        font-weight: 750;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 31px;
        font-weight: 850;
    }

    .metric-description {
        color: #a8b6c8;
        font-size: 12px;
        margin-top: 5px;
    }


    /* ======================================================
       SECTION
       ====================================================== */

    .section-title {
        color: #f8fafc;
        font-size: 22px;
        font-weight: 800;
        margin-top: 30px;
        margin-bottom: 12px;
    }

    .section-description {
        color: #cbd5e1;
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 15px;
    }


    /* ======================================================
       INFO BOX
       ====================================================== */

    .info-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 17px;
        padding: 22px;
        box-shadow:
            0 5px 18px rgba(15,23,42,0.04);
    }


    /* ======================================================
       CUSTOMER CARD
       ====================================================== */

    .customer-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 24px;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow:
            0 5px 18px rgba(15,23,42,0.05);
    }

    .customer-id {
        color: #f8fafc;
        font-size: 25px;
        font-weight: 850;
    }

    .customer-status {
        color: #cbd5e1;
        font-size: 14px;
        margin-top: 5px;
    }


    /* ======================================================
       RISK BOX
       ====================================================== */

    .risk-box {
        border-radius: 17px;
        padding: 22px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
    }

    .risk-label {
        color: #cbd5e1;
        font-size: 12px;
        font-weight: 700;
    }

    .risk-value {
        color: #f8fafc;
        font-size: 30px;
        font-weight: 850;
        margin-top: 5px;
    }


    /* ======================================================
       BUSINESS IMPACT
       ====================================================== */

    .impact-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow:
            0 5px 18px rgba(15,23,42,0.05);
        min-height: 150px;
    }

    .impact-label {
        color: #cbd5e1;
        font-size: 12px;
        font-weight: 750;
    }

    .impact-value {
        color: #f8fafc;
        font-size: 29px;
        font-weight: 850;
        margin-top: 7px;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        text-align: center;
        color: #a8b6c8;
        font-size: 12px;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #29405f;
    }


    /* Navy dashboard theme */
    .metric-card .metric-label, .impact-card .impact-label { color: #64748b; }
    .metric-card .metric-value, .impact-card .impact-value { color: #111827; }
    .metric-card .metric-description { color: #94a3b8; }
    .customer-card .customer-id { color: #111827; }
    .customer-card .customer-status { color: #64748b; }
    .readme-card .readme-title { color: #111827; }
    .readme-card .readme-text { color: #64748b; }
    .section-title { color: #f8fafc; }
    .dashboard-title { color: #f8fafc; }
    .dashboard-subtitle { color: #cbd5e1; }
    .top-brand { color: #f8fafc; }
    .top-description { color: #cbd5e1; }

    </style>
    """
    ),
        unsafe_allow_html=True,
    )


# ============================================================
# TERMINAL / AUDIT LOGGING
# ============================================================

def audit_log(message):
    """Print a timestamped ML workflow message to the Streamlit terminal."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[CCP-AUDIT {timestamp}] {message}", flush=True)


def audit_section(title):
    print("\n" + "=" * 78, flush=True)
    print(f"[CCP-AUDIT] {title}", flush=True)
    print("=" * 78, flush=True)


# ============================================================
# SESSION STATE
# ============================================================

if "dataset_hash" not in st.session_state:
    st.session_state.dataset_hash = None

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "predictions" not in st.session_state:
    st.session_state.predictions = None

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "metrics" not in st.session_state:
    st.session_state.metrics = None

if "feature_names" not in st.session_state:
    st.session_state.feature_names = []

if "numeric_features" not in st.session_state:
    st.session_state.numeric_features = []

if "categorical_features" not in st.session_state:
    st.session_state.categorical_features = []

if "customer_id_column" not in st.session_state:
    st.session_state.customer_id_column = None

if "customer_name_column" not in st.session_state:
    st.session_state.customer_name_column = None

if "target_column" not in st.session_state:
    st.session_state.target_column = None

if "test_features" not in st.session_state:
    st.session_state.test_features = None

if "test_target" not in st.session_state:
    st.session_state.test_target = None

if "trained" not in st.session_state:
    st.session_state.trained = False


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_column_name(column):
    """
    Internal comparison name only.
    Original dataframe column names are preserved.
    """
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def find_column(df, candidates):
    """
    Finds a column without changing the actual column name.
    """

    normalized = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for candidate in candidates:

        key = normalize_column_name(candidate)

        if key in normalized:
            return normalized[key]

    return None


def detect_columns(df):
    """
    Detect Customer ID, Customer Name and Churn columns.
    """

    customer_id = find_column(
        df,
        [
            "CustomerID",
            "Customer_ID",
            "Customer ID",
            "customerid",
            "customer_id",
            "id",
        ],
    )

    customer_name = find_column(
        df,
        [
            "CustomerName",
            "Customer_Name",
            "Customer Name",
            "Name",
            "customername",
        ],
    )

    target = find_column(
        df,
        [
            "Churn",
            "Churned",
            "Churn Status",
            "ChurnStatus",
            "Target",
        ],
    )

    return customer_id, customer_name, target


def normalize_target(series):
    """
    Converts common churn labels to 0 / 1.

    Returns:
        y
        positive_label
        negative_label
    """

    values = series.copy()

    # Numeric target
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    numeric_valid = numeric.notna().mean()

    if numeric_valid > 0.95:

        unique_values = sorted(
            numeric.dropna().unique()
        )

        if len(unique_values) != 2:
            raise ValueError(
                "The Churn column must contain exactly "
                "two classes."
            )

        mapping = {
            unique_values[0]: 0,
            unique_values[1]: 1,
        }

        return (
            numeric.map(mapping).astype(int),
            str(unique_values[1]),
            str(unique_values[0]),
        )

    # String target
    text = (
        values
        .astype(str)
        .str.strip()
        .str.lower()
    )

    positive_words = {
        "yes",
        "y",
        "true",
        "1",
        "churn",
        "churned",
        "left",
        "attrited",
        "attrition",
    }

    negative_words = {
        "no",
        "n",
        "false",
        "0",
        "no churn",
        "not churn",
        "not_churn",
        "retained",
        "active",
    }

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    result[text.isin(positive_words)] = 1
    result[text.isin(negative_words)] = 0

    unknown_mask = result.isna()

    if unknown_mask.any():

        unique = text[~unknown_mask].unique()

        # If no known labels were found,
        # try exactly two arbitrary classes.
        all_unique = text.unique()

        if len(all_unique) == 2:

            mapping = {
                all_unique[0]: 0,
                all_unique[1]: 1,
            }

            result = text.map(mapping)

            return (
                result.astype(int),
                str(all_unique[1]),
                str(all_unique[0]),
            )

    if result.isna().any():

        raise ValueError(
            "Unable to understand some values in the "
            "Churn column. Use values such as "
            "0/1, Yes/No, Churn/No Churn."
        )

    return (
        result.astype(int),
        "Churn",
        "No Churn",
    )


def make_one_hot_encoder():
    """
    Supports newer and older scikit-learn versions.
    """

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )


def risk_level(score):
    """
    Risk classification based on model probability.
    """

    if score >= 80:
        return "Critical"

    if score >= 60:
        return "High"

    if score >= 30:
        return "Medium"

    return "Low"


def calculate_file_hash(file_bytes):
    return hashlib.md5(
        file_bytes
    ).hexdigest()


# ============================================================
# TRAIN MODEL
# ============================================================

def train_uploaded_dataset(df):
    """
    Complete real ML workflow:

        uploaded data
            ↓
        target preparation
            ↓
        train/test split
            ↓
        preprocessing
            ↓
        XGBoost
            ↓
        evaluation
            ↓
        predictions
    """

    audit_section("STARTING REAL ML TRAINING PIPELINE")
    audit_log(f"Input dataframe received: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    audit_log(f"Input columns: {list(df.columns)}")
    audit_log(f"Missing values in input: {int(df.isna().sum().sum()):,}")

    if not XGBOOST_AVAILABLE:

        raise RuntimeError(
            "XGBoost is not installed.\n\n"
            "Install it using:\n"
            "pip install xgboost"
        )

    # --------------------------------------------------------
    # DETECT COLUMNS
    # --------------------------------------------------------

    customer_id_col, customer_name_col, target_col = (
        detect_columns(df)
    )

    audit_log(f"Detected Customer ID column: {customer_id_col!r}")
    audit_log(f"Detected Customer Name column: {customer_name_col!r}")
    audit_log(f"Detected target column: {target_col!r}")

    if target_col is None:

        raise ValueError(
            "No Churn target column was found.\n\n"
            "Your dataset must contain a column such as:\n"
            "Churn"
        )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    y, positive_label, negative_label = (
        normalize_target(
            df[target_col]
        )
    )

    audit_log(f"Target labels: positive={positive_label!r}, negative={negative_label!r}")
    audit_log(f"Raw target distribution: {df[target_col].value_counts(dropna=False).to_dict()}")

    valid_mask = y.notna()

    working_df = df.loc[
        valid_mask
    ].copy()

    y = y.loc[
        valid_mask
    ].astype(int)

    audit_log(f"Rows retained after target validation: {len(working_df):,}")
    audit_log(f"Encoded target distribution: {y.value_counts().sort_index().to_dict()}")

    if y.nunique() != 2:

        raise ValueError(
            "The uploaded dataset must contain "
            "both churn and non-churn customers."
        )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    excluded_columns = {
        target_col,
    }

    if customer_id_col is not None:
        excluded_columns.add(
            customer_id_col
        )

    if customer_name_col is not None:
        excluded_columns.add(
            customer_name_col
        )

    # Ignore common CSV index artifacts such as "Unnamed: 0".
    # These are row-number artifacts, not customer behavior features.
    index_artifact_columns = {
        col
        for col in working_df.columns
        if str(col).strip().lower().startswith("unnamed:")
    }
    excluded_columns.update(index_artifact_columns)

    feature_columns = [
        col
        for col in working_df.columns
        if col not in excluded_columns
    ]

    audit_log(f"Excluded columns from ML: {sorted(excluded_columns)}")
    audit_log(
        f"CSV index-artifact columns excluded: "
        f"{sorted(index_artifact_columns) if index_artifact_columns else 'None'}"
    )
    audit_log(f"ML feature columns ({len(feature_columns)}): {feature_columns}")

    if not feature_columns:

        raise ValueError(
            "No usable feature columns were found."
        )

    X = working_df[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # REMOVE COMPLETELY EMPTY COLUMNS
    # --------------------------------------------------------

    completely_empty = [
        col
        for col in X.columns
        if X[col].isna().all()
    ]

    if completely_empty:

        X = X.drop(
            columns=completely_empty
        )

        feature_columns = [
            col
            for col in feature_columns
            if col not in completely_empty
        ]

    audit_log(f"Completely empty feature columns removed: {completely_empty if completely_empty else 'None'}")

    # --------------------------------------------------------
    # DETECT NUMERIC / CATEGORICAL
    # --------------------------------------------------------

    numeric_features = (
        X.select_dtypes(
            include=[
                "number",
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    categorical_features = [
        col
        for col in X.columns
        if col not in numeric_features
    ]

    audit_log(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    audit_log(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    if not numeric_features and not categorical_features:

        raise ValueError(
            "No usable ML features were detected."
        )

    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    audit_log(f"Train/Test split complete: train={len(X_train):,}, test={len(X_test):,}")
    audit_log(f"Train target distribution: {y_train.value_counts().sort_index().to_dict()}")
    audit_log(f"Test target distribution: {y_test.value_counts().sort_index().to_dict()}")

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    transformers = []

    if numeric_features:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    make_one_hot_encoder(),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    audit_log("Preprocessing pipeline configured: numeric median imputation + categorical most-frequent imputation + OneHotEncoder")

    # --------------------------------------------------------
    # CLASS IMBALANCE
    # --------------------------------------------------------

    positive_count = int(
        (y_train == 1).sum()
    )

    negative_count = int(
        (y_train == 0).sum()
    )

    if positive_count > 0:

        scale_pos_weight = (
            negative_count /
            positive_count
        )

    else:

        scale_pos_weight = 1.0

    audit_log(f"Training class counts: non-churn={negative_count:,}, churn={positive_count:,}")
    audit_log(f"XGBoost scale_pos_weight={scale_pos_weight:.6f}")

    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                xgb_model,
            ),
        ]
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    audit_section("XGBOOST MODEL TRAINING")
    audit_log("Calling pipeline.fit(X_train, y_train) ...")
    pipeline.fit(
        X_train,
        y_train,
    )
    audit_log("XGBoost training completed successfully.")
    audit_log(f"Model class: {type(xgb_model).__name__}")
    audit_log(f"Model parameters: n_estimators={xgb_model.n_estimators}, max_depth={xgb_model.max_depth}, learning_rate={xgb_model.learning_rate}")

    try:
        transformed_train_shape = pipeline.named_steps["preprocessor"].transform(X_train).shape
        audit_log(f"Transformed training matrix shape: {transformed_train_shape}")
    except Exception as e:
        audit_log(f"Could not print transformed matrix shape: {e}")

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    test_prediction = (
        test_probability >= 0.50
    ).astype(int)

    audit_section("TEST-SET PREDICTION / EVALUATION")
    audit_log(f"Generated {len(test_probability):,} test-set probabilities using pipeline.predict_proba(X_test)")
    audit_log(f"Test probability range: min={test_probability.min():.6f}, max={test_probability.max():.6f}, mean={test_probability.mean():.6f}")
    audit_log(f"Test predicted class counts: {pd.Series(test_prediction).value_counts().sort_index().to_dict()}")

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        test_prediction,
    )

    precision = precision_score(
        y_test,
        test_prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        test_prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        test_prediction,
        zero_division=0,
    )

    try:

        roc_auc = roc_auc_score(
            y_test,
            test_probability,
        )

    except Exception:

        roc_auc = np.nan

    cm = confusion_matrix(
        y_test,
        test_prediction,
    )

    audit_log(f"Accuracy : {accuracy:.6f}")
    audit_log(f"Precision: {precision:.6f}")
    audit_log(f"Recall   : {recall:.6f}")
    audit_log(f"F1 Score : {f1:.6f}")
    audit_log(f"ROC-AUC  : {roc_auc:.6f}" if not pd.isna(roc_auc) else "ROC-AUC  : NaN")
    audit_log(f"Confusion matrix:\n{cm}")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": len(feature_columns),
        "positive_label": positive_label,
        "negative_label": negative_label,
    }

    audit_section("FULL DATASET MODEL PREDICTIONS")

    # --------------------------------------------------------
    # PREDICT ALL CUSTOMERS
    # --------------------------------------------------------

    all_probability = pipeline.predict_proba(
        X
    )[:, 1]

    all_prediction = (
        all_probability >= 0.50
    ).astype(int)

    audit_log("Generated customer probabilities with the TRAINED pipeline using predict_proba(X)")
    audit_log(f"Full-dataset probability range: min={all_probability.min():.6f}, max={all_probability.max():.6f}, mean={all_probability.mean():.6f}")
    audit_log(f"Full-dataset model prediction counts: {pd.Series(all_prediction).value_counts().sort_index().to_dict()}")

    # Direct audit comparison: model output vs original target
    audit_log(f"Original target counts: {y.value_counts().sort_index().to_dict()}")
    audit_log(f"Model prediction counts: {pd.Series(all_prediction).value_counts().sort_index().to_dict()}")

    result_df = working_df.copy()

    result_df["_model_churn_probability"] = (
        all_probability
    )

    result_df["_model_prediction"] = (
        all_prediction
    )

    result_df["_risk_score"] = (
        all_probability * 100
    )

    result_df["_risk_level"] = (
        result_df["_risk_score"]
        .apply(risk_level)
    )

    result_df["_prediction_label"] = np.where(
        all_prediction == 1,
        "Churn",
        "No Churn",
    )

    audit_log("Model-output columns created: _model_churn_probability, _model_prediction, _risk_score, _risk_level, _prediction_label")
    audit_log("Sample of MODEL-GENERATED outputs (not copied from target column):")
    audit_sample_cols = []
    if customer_id_col is not None:
        audit_sample_cols.append(customer_id_col)
    audit_sample_cols += [target_col, "_model_churn_probability", "_model_prediction", "_risk_score", "_risk_level", "_prediction_label"]
    audit_sample_cols = [c for c in audit_sample_cols if c in result_df.columns]
    print(result_df[audit_sample_cols].head(10).to_string(index=False), flush=True)
    audit_log(f"Predicted churn customers: {int((all_prediction == 1).sum()):,}")
    audit_log(f"Predicted no-churn customers: {int((all_prediction == 0).sum()):,}")
    audit_log("REAL ML PIPELINE COMPLETE: upload -> validation -> preprocessing -> split -> XGBoost -> evaluation -> model predictions -> risk scoring")

    return {
        "pipeline": pipeline,
        "predictions": result_df,
        "metrics": metrics,
        "customer_id_column": customer_id_col,
        "customer_name_column": customer_name_col,
        "target_column": target_col,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_columns": feature_columns,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# ============================================================
# SHAP ENGINE
# ============================================================

def get_shap_explanation(
    pipeline,
    original_df,
    customer_index,
    feature_columns,
):
    """
    Generates a SHAP explanation for one customer.

    SHAP is calculated on the actual trained model,
    not on fabricated values.
    """

    audit_section("SHAP EXPLANATION")
    audit_log(f"Generating SHAP explanation for customer index={customer_index}")
    audit_log("SHAP uses the trained pipeline/model stored in session state.")

    if not SHAP_AVAILABLE:

        raise RuntimeError(
            "SHAP is not installed.\n\n"
            "Install using:\n"
            "pip install shap"
        )

    preprocessor = (
        pipeline.named_steps[
            "preprocessor"
        ]
    )

    model = (
        pipeline.named_steps[
            "model"
        ]
    )

    # --------------------------------------------------------
    # CUSTOMER ROW
    # --------------------------------------------------------

    customer_row = (
        original_df.loc[
            [customer_index],
            feature_columns,
        ]
    )

    # --------------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------------

    transformed = (
        preprocessor.transform(
            customer_row
        )
    )

    # --------------------------------------------------------
    # FEATURE NAMES
    # --------------------------------------------------------

    try:

        transformed_names = (
            preprocessor
            .get_feature_names_out()
        )

    except Exception:

        transformed_names = np.array(
            feature_columns
        )

    # --------------------------------------------------------
    # TREE EXPLAINER
    # --------------------------------------------------------

    try:

        audit_log("Trying SHAP TreeExplainer on trained XGBoost model...")
        explainer = shap.TreeExplainer(
            model
        )

        explanation = explainer(
            transformed
        )

        values = explanation.values

        if isinstance(values, list):

            values = values[-1]

        values = np.asarray(
            values
        )

        if values.ndim == 3:

            values = values[
                0,
                :,
                -1,
            ]

        elif values.ndim == 2:

            values = values[0]

        else:

            values = values.flatten()

    except Exception as shap_tree_error:
        audit_log(f"TreeExplainer unavailable; using SHAP fallback. Reason: {shap_tree_error}")

        # ----------------------------------------------------
        # FALLBACK EXPLAINER
        # ----------------------------------------------------

        def predict_probability(data):

            return pipeline.predict_proba(
                data
            )[:, 1]

        # Small background sample
        background = original_df[
            feature_columns
        ].sample(
            min(
                50,
                len(original_df),
            ),
            random_state=42,
        )

        customer_original = original_df.loc[
            [customer_index],
            feature_columns,
        ]

        explainer = shap.Explainer(
            predict_probability,
            background,
        )

        explanation = explainer(
            customer_original
        )

        values = explanation.values

        if values.ndim == 2:

            values = values[0]

        values = np.asarray(
            values
        ).flatten()

        transformed_names = np.array(
            feature_columns
        )

    # --------------------------------------------------------
    # AGGREGATE ONE-HOT FEATURES
    # BACK TO ORIGINAL FEATURES
    # --------------------------------------------------------

    aggregated = []

    for original_feature in feature_columns:

        indices = []

        numeric_prefix = (
            f"numeric__{original_feature}"
        )

        categorical_prefix = (
            f"categorical__{original_feature}_"
        )

        for i, name in enumerate(
            transformed_names
        ):

            name = str(name)

            if (
                name == numeric_prefix
                or name.startswith(
                    categorical_prefix
                )
            ):
                indices.append(i)

        if indices:

            contribution = float(
                np.sum(
                    values[indices]
                )
            )

            aggregated.append(
                {
                    "Feature": original_feature,
                    "SHAP Value": contribution,
                    "Absolute Impact": abs(
                        contribution
                    ),
                }
            )

    if not aggregated:

        # Fallback
        for i, feature in enumerate(
            feature_columns
        ):

            if i < len(values):

                aggregated.append(
                    {
                        "Feature": feature,
                        "SHAP Value": float(
                            values[i]
                        ),
                        "Absolute Impact": abs(
                            float(values[i])
                        ),
                    }
                )

    shap_df = pd.DataFrame(
        aggregated
    )

    if shap_df.empty:

        raise RuntimeError(
            "No SHAP feature contributions "
            "could be generated."
        )

    shap_df["Impact"] = np.where(
        shap_df["SHAP Value"] > 0,
        "Increases Churn Risk",
        "Decreases Churn Risk",
    )

    shap_df = (
        shap_df
        .sort_values(
            "Absolute Impact",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return shap_df


# ============================================================
# LANDING PAGE
# ============================================================

def show_upload_page():

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    st.markdown(textwrap.dedent(
        """
        <div class="hero-container">

            <div class="hero-icon">
                📊
            </div>

            <div class="hero-title">
                Customer Churn Intelligence
            </div>

            <div class="hero-subtitle">
                Upload your customer dataset and transform
                it into a real machine-learning powered
                churn intelligence dashboard.
            </div>

        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------

    st.markdown(textwrap.dedent(
        """
        <div class="section-title">
            📖 How This Dashboard Works
        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(textwrap.dedent(
            """
            <div class="readme-card">

                <div class="readme-icon">
                    📁
                </div>

                <div class="readme-title">
                    1. Upload Dataset
                </div>

                <div class="readme-text">
                    Upload your customer CSV file.
                    The dashboard uses your uploaded
                    dataset as the source for the
                    complete ML workflow.
                </div>

            </div>
            """
            ),
                unsafe_allow_html=True,
            )

    with c2:

        st.markdown(textwrap.dedent(
            """
            <div class="readme-card">

                <div class="readme-icon">
                    🤖
                </div>

                <div class="readme-title">
                    2. Train Model
                </div>

                <div class="readme-text">
                    The system preprocesses the uploaded
                    data, creates a train/test split and
                    trains an XGBoost churn classifier.
                </div>

            </div>
            """
            ),
                unsafe_allow_html=True,
            )

    with c3:

        st.markdown(textwrap.dedent(
            """
            <div class="readme-card">

                <div class="readme-icon">
                    🧠
                </div>

                <div class="readme-title">
                    3. Explain Predictions
                </div>

                <div class="readme-text">
                    SHAP explains which real customer
                    features increase or decrease the
                    predicted churn risk.
                </div>

            </div>
            """
            ),
                unsafe_allow_html=True,
            )

    with c4:

        st.markdown(textwrap.dedent(
            """
            <div class="readme-card">

                <div class="readme-icon">
                    💰
                </div>

                <div class="readme-title">
                    4. Estimate Impact
                </div>

                <div class="readme-text">
                    Estimate potential revenue at risk
                    and the possible business value of
                    customer retention.
                </div>

            </div>
            """
            ),
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # DATASET REQUIREMENTS
    # --------------------------------------------------------

    st.markdown(textwrap.dedent(
        """
        <div class="section-title">
            📋 Dataset Requirements
        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    st.info(
        """
        **Required:** A CSV containing a target column
        representing customer churn, normally named
        `Churn`.

        **Recommended:** A `CustomerID` column.

        **Optional:** A `CustomerName` / `Customer Name`
        column. If your dataset does not contain a customer
        name, the dashboard will NOT invent one. It will use
        the actual Customer ID.

        Both numerical and categorical features are supported.
        """
    )

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    st.markdown(textwrap.dedent(
        """
        <div class="upload-card">

            <div class="section-title"
                 style="margin-top:0;">
                📤 Upload Your Customer Dataset
            </div>

            <div class="section-description">
                Upload one CSV file. Once uploaded,
                the dashboard will train the model
                using this dataset.
            </div>

        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"],
        help=(
            "Upload the customer churn CSV "
            "you want to train on."
        ),
    )

    return uploaded_file


# ============================================================
# TOP NAVIGATION
# ============================================================

def show_top_navigation():

    st.markdown(textwrap.dedent(
        """
        <div class="top-brand">
            📊 Customer Churn Intelligence
        </div>

        <div class="top-description">
            Machine Learning Powered Customer Retention Platform
        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    navigation = st.radio(
        "Dashboard Navigation",
        [
            "🏠 Overview",
            "👤 Customer Risk",
            "🧠 SHAP Explainability",
            "💰 Business Impact Estimator",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    return navigation


# ============================================================
# OVERVIEW PAGE
# ============================================================

def show_overview():

    df = st.session_state.dataset
    result = st.session_state.predictions
    metrics = st.session_state.metrics

    customer_id_col = (
        st.session_state.customer_id_column
    )

    customer_name_col = (
        st.session_state.customer_name_column
    )

    st.markdown(
        '<div class="dashboard-title">'
        'Customer Churn Overview'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        'Real predictions generated from the uploaded '
        'customer dataset.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    total_customers = len(result)

    predicted_churn = int(
        (
            result["_model_prediction"] == 1
        ).sum()
    )

    critical_customers = int(
        (
            result["_risk_level"]
            == "Critical"
        ).sum()
    )

    average_risk = float(
        result["_risk_score"].mean()
    )

    c1, c2, c3, c4 = st.columns(4)

    metric_data = [
        (
            "TOTAL CUSTOMERS",
            f"{total_customers:,}",
            "Customers in uploaded dataset",
        ),
        (
            "PREDICTED CHURN",
            f"{predicted_churn:,}",
            "Customers predicted to churn",
        ),
        (
            "CRITICAL RISK",
            f"{critical_customers:,}",
            "Customers above 80 risk score",
        ),
        (
            "AVERAGE RISK",
            f"{average_risk:.1f}%",
            "Average predicted churn probability",
        ),
    ]

    for col, item in zip(
        [c1, c2, c3, c4],
        metric_data,
    ):

        with col:

            st.markdown(textwrap.dedent(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        {item[0]}
                    </div>

                    <div class="metric-value">
                        {item[1]}
                    </div>

                    <div class="metric-description">
                        {item[2]}
                    </div>

                </div>
                """
                ),
                    unsafe_allow_html=True,
                )

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.markdown(textwrap.dedent(
        """
        <div class="section-title">
            📁 Uploaded Dataset
        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Rows",
            f"{len(df):,}",
        )

    with c2:
        st.metric(
            "Columns",
            f"{len(df.columns):,}",
        )

    with c3:
        st.metric(
            "ML Features",
            f"{metrics['feature_count']:,}",
        )

    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    st.markdown(textwrap.dedent(
        """
        <div class="section-title">
            🎯 Customer Risk Distribution
        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    risk_counts = (
        result["_risk_level"]
        .value_counts()
        .reindex(
            [
                "Low",
                "Medium",
                "High",
                "Critical",
            ],
            fill_value=0,
        )
        .reset_index()
    )

    risk_counts.columns = [
        "Risk Level",
        "Customers",
    ]

    fig = px.bar(
        risk_counts,
        x="Risk Level",
        y="Customers",
        text="Customers",
    )

    fig.update_layout(
        template="plotly_white",
        height=420,
        showlegend=False,
        xaxis_title="",
        yaxis_title="Customers",
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    # --------------------------------------------------------
    # HIGH RISK CUSTOMERS
    # --------------------------------------------------------

    st.markdown(textwrap.dedent(
        """
        <div class="section-title">
            🚨 Customers Requiring Attention
        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    high_risk = (
        result[
            result["_risk_score"] >= 60
        ]
        .sort_values(
            "_risk_score",
            ascending=False,
        )
        .head(15)
        .copy()
    )

    if high_risk.empty:

        st.success(
            "No customers currently have a risk score "
            "of 60 or above."
        )

    else:

        display_columns = []

        if customer_id_col is not None:

            display_columns.append(
                customer_id_col
            )

        if customer_name_col is not None:

            display_columns.append(
                customer_name_col
            )

        display_columns += [
            "_prediction_label",
            "_risk_score",
            "_risk_level",
        ]

        display_columns = [
            col
            for col in display_columns
            if col in high_risk.columns
        ]

        display = high_risk[
            display_columns
        ].copy()

        display = display.rename(
            columns={
                "_prediction_label":
                    "Prediction",
                "_risk_score":
                    "Risk Score",
                "_risk_level":
                    "Risk Level",
            }
        )

        display["Risk Score"] = (
            display["Risk Score"]
            .round(2)
        )

        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
        )

    # --------------------------------------------------------
    # DOWNLOAD PREDICTIONS
    # --------------------------------------------------------

    st.markdown(textwrap.dedent(
        """
        <div class="section-title">
            📥 Export Predictions
        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    export_df = result.copy()

    st.download_button(
        label="Download Customer Risk Predictions CSV",
        data=export_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="customer_churn_predictions.csv",
        mime="text/csv",
    )


# ============================================================
# CUSTOMER RISK PAGE
# ============================================================

def show_customer_risk():

    result = st.session_state.predictions

    customer_id_col = (
        st.session_state.customer_id_column
    )

    customer_name_col = (
        st.session_state.customer_name_column
    )

    feature_columns = (
        st.session_state.feature_names
    )

    st.markdown(textwrap.dedent(
        """
        <div class="dashboard-title">
            Customer Risk Analysis
        </div>

        <div class="dashboard-subtitle">
            Select a real customer from the uploaded
            dataset and inspect the model's churn prediction.
        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # CUSTOMER ID LIST
    # --------------------------------------------------------

    if customer_id_col is not None:

        customer_values = (
            result[
                customer_id_col
            ]
            .astype(str)
            .tolist()
        )

    else:

        customer_values = [
            str(index)
            for index in result.index
        ]

    selected_id = st.selectbox(
        "Select Customer",
        customer_values,
    )

    # --------------------------------------------------------
    # FIND CUSTOMER
    # --------------------------------------------------------

    if customer_id_col is not None:

        mask = (
            result[
                customer_id_col
            ]
            .astype(str)
            == selected_id
        )

        customer_index = result[
            mask
        ].index[0]

    else:

        customer_index = int(
            selected_id
        )

    customer = result.loc[
        customer_index
    ]

    probability = float(
        customer[
            "_model_churn_probability"
        ]
    )

    risk_score = float(
        customer[
            "_risk_score"
        ]
    )

    risk = customer[
        "_risk_level"
    ]

    prediction = customer[
        "_prediction_label"
    ]

    # --------------------------------------------------------
    # CUSTOMER HEADER
    # --------------------------------------------------------

    actual_name = None

    if (
        customer_name_col is not None
        and customer_name_col in customer.index
    ):

        actual_name = customer[
            customer_name_col
        ]

    if actual_name is not None:

        title = (
            f"Customer {selected_id} — "
            f"{actual_name}"
        )

    else:

        title = (
            f"Customer {selected_id}"
        )

    st.markdown(textwrap.dedent(
        f"""
        <div class="customer-card">

            <div class="customer-id">
                {title}
            </div>

            <div class="customer-status">
                Model Prediction: {prediction}
                &nbsp;&nbsp; | &nbsp;&nbsp;
                Risk Level: {risk}
            </div>

        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%",
        )

    with c2:

        st.metric(
            "Risk Score",
            f"{risk_score:.2f}/100",
        )

    with c3:

        st.metric(
            "Prediction",
            prediction,
        )

    # --------------------------------------------------------
    # GAUGE
    # --------------------------------------------------------

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={
                "text": "Customer Churn Risk Score"
            },
            gauge={
                "axis": {
                    "range": [
                        0,
                        100,
                    ]
                },
                "steps": [
                    {
                        "range": [
                            0,
                            30,
                        ]
                    },
                    {
                        "range": [
                            30,
                            60,
                        ]
                    },
                    {
                        "range": [
                            60,
                            80,
                        ]
                    },
                    {
                        "range": [
                            80,
                            100,
                        ]
                    },
                ],
                "threshold": {
                    "line": {
                        "width": 5
                    },
                    "value": risk_score,
                },
            },
        )
    )

    fig.update_layout(
        height=360,
        margin=dict(
            l=30,
            r=30,
            t=70,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    # --------------------------------------------------------
    # ACTUAL CUSTOMER FEATURES
    # --------------------------------------------------------

    st.markdown(textwrap.dedent(
        """
        <div class="section-title">
            📋 Customer Information
        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    customer_features = []

    for feature in feature_columns:

        if feature in customer.index:

            # Convert displayed values to strings so Streamlit/PyArrow
            # never has to infer one mixed dtype for numbers + text.
            value = customer[feature]
            if pd.isna(value):
                value = "Missing"
            else:
                value = str(value)

            customer_features.append(
                {
                    "Feature": str(feature),
                    "Value": value,
                }
            )

    if customer_features:

        st.dataframe(
            pd.DataFrame(
                customer_features
            ),
            width="stretch",
            hide_index=True,
        )


# ============================================================
# SHAP PAGE
# ============================================================

def show_shap():

    result = st.session_state.predictions
    pipeline = st.session_state.pipeline

    customer_id_col = (
        st.session_state.customer_id_column
    )

    feature_columns = (
        st.session_state.feature_names
    )

    st.markdown(textwrap.dedent(
        """
        <div class="dashboard-title">
            SHAP Explainability
        </div>

        <div class="dashboard-subtitle">
            Understand which actual customer features
            are driving the model's churn prediction.
        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    if not SHAP_AVAILABLE:

        st.error(
            """
            SHAP is not installed.

            Install it with:

            pip install shap
            """
        )

        return

    # --------------------------------------------------------
    # CUSTOMER SELECTOR
    # --------------------------------------------------------

    if customer_id_col is not None:

        customer_ids = (
            result[
                customer_id_col
            ]
            .astype(str)
            .tolist()
        )

    else:

        customer_ids = [
            str(i)
            for i in result.index
        ]

    selected_id = st.selectbox(
        "Select Customer for SHAP Explanation",
        customer_ids,
        key="shap_customer_selector",
    )

    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    if customer_id_col is not None:

        customer_index = result[
            result[
                customer_id_col
            ]
            .astype(str)
            == selected_id
        ].index[0]

    else:

        customer_index = int(
            selected_id
        )

    customer = result.loc[
        customer_index
    ]

    probability = float(
        customer[
            "_model_churn_probability"
        ]
    )

    risk_score = float(
        customer[
            "_risk_score"
        ]
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Customer",
            selected_id,
        )

    with c2:

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%",
        )

    with c3:

        st.metric(
            "Risk Score",
            f"{risk_score:.2f}",
        )

    # --------------------------------------------------------
    # GENERATE SHAP
    # --------------------------------------------------------

    with st.spinner(
        "Calculating SHAP explanation..."
    ):

        try:

            shap_df = get_shap_explanation(
                pipeline=pipeline,
                original_df=result,
                customer_index=customer_index,
                feature_columns=feature_columns,
            )

        except Exception as e:

            st.error(
                "SHAP explanation could not be generated."
            )

            st.code(
                str(e)
            )

            return

    # --------------------------------------------------------
    # MAIN CHART
    # --------------------------------------------------------

    st.markdown(textwrap.dedent(
        """
        <div class="section-title">
            🔍 What Is Driving This Prediction?
        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    chart_df = (
        shap_df
        .head(12)
        .sort_values(
            "SHAP Value"
        )
    )

    fig = px.bar(
        chart_df,
        x="SHAP Value",
        y="Feature",
        color="Impact",
        orientation="h",
        title=(
            "SHAP Feature Contributions"
        ),
    )

    fig.update_layout(
        template="plotly_white",
        height=550,
        yaxis_title="",
        xaxis_title="SHAP Contribution",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    # --------------------------------------------------------
    # POSITIVE / NEGATIVE FACTORS
    # --------------------------------------------------------

    positive = (
        shap_df[
            shap_df["SHAP Value"] > 0
        ]
        .head(5)
    )

    negative = (
        shap_df[
            shap_df["SHAP Value"] < 0
        ]
        .sort_values(
            "SHAP Value",
            ascending=True,
        )
        .head(5)
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(textwrap.dedent(
            """
            <div class="section-title">
                🔴 Increasing Churn Risk
            </div>
            """
            ),
                unsafe_allow_html=True,
            )

        if positive.empty:

            st.info(
                "No positive SHAP contributions."
            )

        else:

            display = positive[
                [
                    "Feature",
                    "SHAP Value",
                ]
            ].copy()

            display["SHAP Value"] = (
                display["SHAP Value"]
                .round(4)
            )

            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
            )

    with c2:

        st.markdown(textwrap.dedent(
            """
            <div class="section-title">
                🟢 Reducing Churn Risk
            </div>
            """
            ),
                unsafe_allow_html=True,
            )

        if negative.empty:

            st.info(
                "No negative SHAP contributions."
            )

        else:

            display = negative[
                [
                    "Feature",
                    "SHAP Value",
                ]
            ].copy()

            display["SHAP Value"] = (
                display["SHAP Value"]
                .round(4)
            )

            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
            )

    # --------------------------------------------------------
    # COMPLETE SHAP TABLE
    # --------------------------------------------------------

    st.markdown(textwrap.dedent(
        """
        <div class="section-title">
            📊 Detailed Feature Contributions
        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    display = shap_df[
        [
            "Feature",
            "SHAP Value",
            "Impact",
        ]
    ].copy()

    display["SHAP Value"] = (
        display["SHAP Value"]
        .round(5)
    )

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        """
        Positive SHAP values indicate that the feature
        pushes the model toward higher churn risk.
        Negative values indicate that the feature pushes
        the prediction toward lower churn risk.
        """
    )


# ============================================================
# BUSINESS IMPACT PAGE
# ============================================================

def show_business_impact():

    result = st.session_state.predictions

    customer_id_col = (
        st.session_state.customer_id_column
    )

    st.markdown(textwrap.dedent(
        """
        <div class="dashboard-title">
            Business Impact Estimator
        </div>

        <div class="dashboard-subtitle">
            Translate the model's churn predictions into
            an estimated customer-retention business impact.
        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    st.markdown(textwrap.dedent(
        """
        <div class="section-title">
            💰 Business Assumptions
        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        customer_value = st.number_input(
            "Estimated Customer Value (₹)",
            min_value=0.0,
            value=10000.0,
            step=500.0,
        )

    with c2:

        retention_cost = st.number_input(
            "Retention Cost per Customer (₹)",
            min_value=0.0,
            value=1000.0,
            step=100.0,
        )

    with c3:

        expected_success = st.slider(
            "Expected Retention Success (%)",
            min_value=0,
            max_value=100,
            value=40,
        )

    # --------------------------------------------------------
    # MODEL BASED COUNTS
    # --------------------------------------------------------

    churn_customers = result[
        result[
            "_model_prediction"
        ] == 1
    ].copy()

    high_risk_customers = result[
        result[
            "_risk_score"
        ] >= 60
    ].copy()

    critical_customers = result[
        result[
            "_risk_score"
        ] >= 80
    ].copy()

    churn_count = len(
        churn_customers
    )

    high_risk_count = len(
        high_risk_customers
    )

    critical_count = len(
        critical_customers
    )

    # --------------------------------------------------------
    # EXPECTED RETENTION
    # --------------------------------------------------------

    expected_retained = (
        high_risk_count
        * expected_success
        / 100
    )

    potential_value = (
        high_risk_count
        * customer_value
    )

    expected_recovered_value = (
        expected_retained
        * customer_value
    )

    retention_investment = (
        high_risk_count
        * retention_cost
    )

    net_impact = (
        expected_recovered_value
        - retention_investment
    )

    roi = (
        (
            expected_recovered_value
            - retention_investment
        )
        / retention_investment
        * 100
        if retention_investment > 0
        else 0
    )

    # --------------------------------------------------------
    # IMPACT CARDS
    # --------------------------------------------------------

    st.markdown(textwrap.dedent(
        """
        <div class="section-title">
            📊 Estimated Business Impact
        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        (
            "PREDICTED CHURN CUSTOMERS",
            f"{churn_count:,}",
        ),
        (
            "HIGH-RISK CUSTOMERS",
            f"{high_risk_count:,}",
        ),
        (
            "EXPECTED RETAINED",
            f"{expected_retained:,.0f}",
        ),
        (
            "ESTIMATED NET IMPACT",
            f"₹{net_impact:,.0f}",
        ),
    ]

    for col, card in zip(
        [c1, c2, c3, c4],
        cards,
    ):

        with col:

            st.markdown(textwrap.dedent(
                f"""
                <div class="impact-card">

                    <div class="impact-label">
                        {card[0]}
                    </div>

                    <div class="impact-value">
                        {card[1]}
                    </div>

                </div>
                """
                ),
                    unsafe_allow_html=True,
                )

    # --------------------------------------------------------
    # FINANCIAL BREAKDOWN
    # --------------------------------------------------------

    st.markdown(textwrap.dedent(
        """
        <div class="section-title">
            💼 Financial Breakdown
        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    financial_df = pd.DataFrame(
        {
            "Business Metric": [
                "Potential Value of High-Risk Customers",
                "Expected Recovered Customer Value",
                "Retention Investment",
                "Estimated Net Impact",
                "Estimated ROI",
            ],
            "Estimated Value": [
                f"₹{potential_value:,.2f}",
                f"₹{expected_recovered_value:,.2f}",
                f"₹{retention_investment:,.2f}",
                f"₹{net_impact:,.2f}",
                f"{roi:.2f}%",
            ],
        }
    )

    st.dataframe(
        financial_df,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # PRIORITY CUSTOMER LIST
    # --------------------------------------------------------

    st.markdown(textwrap.dedent(
        """
        <div class="section-title">
            🚨 Retention Priority Customers
        </div>
        """
        ),
            unsafe_allow_html=True,
        )

    priority = (
        high_risk_customers
        .sort_values(
            "_risk_score",
            ascending=False,
        )
        .head(20)
        .copy()
    )

    if priority.empty:

        st.info(
            "No high-risk customers were detected."
        )

    else:

        columns = []

        if customer_id_col is not None:

            columns.append(
                customer_id_col
            )

        columns += [
            "_risk_score",
            "_risk_level",
            "_prediction_label",
        ]

        columns = [
            col
            for col in columns
            if col in priority.columns
        ]

        priority_display = priority[
            columns
        ].copy()

        priority_display = (
            priority_display.rename(
                columns={
                    "_risk_score":
                        "Risk Score",
                    "_risk_level":
                        "Risk Level",
                    "_prediction_label":
                        "Prediction",
                }
            )
        )

        priority_display[
            "Risk Score"
        ] = (
            priority_display[
                "Risk Score"
            ]
            .round(2)
        )

        st.dataframe(
            priority_display,
            width="stretch",
            hide_index=True,
        )

    st.caption(
        """
        Business impact values are estimates based on the
        assumptions entered above. They are not direct
        financial forecasts.
        """
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # ========================================================
    # WINDOW 1
    # ========================================================

    if not st.session_state.trained:

        uploaded_file = (
            show_upload_page()
        )

        if uploaded_file is None:

            st.markdown(textwrap.dedent(
                """
                <div class="footer">

                    Upload a CSV dataset to begin
                    the customer churn ML workflow.

                </div>
                """
                ),
                    unsafe_allow_html=True,
                )

            return

        # ----------------------------------------------------
        # READ FILE
        # ----------------------------------------------------

        try:

            file_bytes = (
                uploaded_file.getvalue()
            )

            current_hash = (
                calculate_file_hash(
                    file_bytes
                )
            )
            audit_log(f"Uploaded file size: {len(file_bytes):,} bytes")
            audit_log(f"Dataset MD5 hash: {current_hash}")
            audit_log("This hash identifies the exact uploaded file used for this training run.")

        except Exception as e:

            st.error(
                f"Unable to read uploaded file: {e}"
            )

            return

        # ----------------------------------------------------
        # NEW FILE
        # ----------------------------------------------------

        if (
            st.session_state.dataset_hash
            != current_hash
        ):

            # Reset old model
            st.session_state.trained = False
            st.session_state.pipeline = None
            st.session_state.predictions = None
            st.session_state.metrics = None

            # ------------------------------------------------
            # LOAD DATA
            # ------------------------------------------------

            try:

                uploaded_df = pd.read_csv(
                    uploaded_file
                )
                audit_log(f"CSV loaded successfully: {uploaded_df.shape[0]:,} rows x {uploaded_df.shape[1]:,} columns")
                audit_log(f"CSV columns: {list(uploaded_df.columns)}")

            except Exception as e:

                st.error(
                    "The uploaded file could not "
                    "be read as a CSV."
                )

                st.code(
                    str(e)
                )

                return

            # ------------------------------------------------
            # BASIC VALIDATION
            # ------------------------------------------------

            if uploaded_df.empty:

                st.error(
                    "The uploaded dataset is empty."
                )

                return

            if len(uploaded_df.columns) < 2:

                st.error(
                    "The dataset must contain at least "
                    "one feature column and one target column."
                )

                return

            # ------------------------------------------------
            # TRAIN
            # ------------------------------------------------

            st.markdown(textwrap.dedent(
                """
                <div class="section-title">
                    🤖 Preparing Your Machine Learning Model
                </div>
                """
                ),
                    unsafe_allow_html=True,
                )

            progress = st.progress(
                0
            )

            status = st.empty()

            try:

                status.info(
                    "Reading and validating uploaded dataset..."
                )

                progress.progress(
                    15
                )

                status.info(
                    "Preparing numerical and categorical features..."
                )

                progress.progress(
                    30
                )

                status.info(
                    "Training XGBoost model..."
                )

                audit_log("Starting train_uploaded_dataset(uploaded_df) ...")
                result = train_uploaded_dataset(
                    uploaded_df
                )
                audit_log("train_uploaded_dataset() returned successfully.")

                progress.progress(
                    80
                )

                status.info(
                    "Generating real customer churn predictions..."
                )

                progress.progress(
                    90
                )

                # ------------------------------------------------
                # SAVE SESSION RESULTS
                # ------------------------------------------------

                st.session_state.dataset = (
                    uploaded_df
                )

                st.session_state.pipeline = (
                    result["pipeline"]
                )

                st.session_state.predictions = (
                    result["predictions"]
                )

                st.session_state.metrics = (
                    result["metrics"]
                )

                st.session_state.feature_names = (
                    result["feature_columns"]
                )

                st.session_state.numeric_features = (
                    result["numeric_features"]
                )

                st.session_state.categorical_features = (
                    result["categorical_features"]
                )

                st.session_state.customer_id_column = (
                    result["customer_id_column"]
                )

                st.session_state.customer_name_column = (
                    result["customer_name_column"]
                )

                st.session_state.target_column = (
                    result["target_column"]
                )

                st.session_state.test_features = (
                    result["X_test"]
                )

                st.session_state.test_target = (
                    result["y_test"]
                )

                st.session_state.dataset_hash = (
                    current_hash
                )

                st.session_state.trained = True
                audit_log("Session state updated: trained=True")
                audit_log("Dashboard will now display predictions generated by the trained XGBoost pipeline.")

                progress.progress(
                    100
                )

                status.success(
                    "Model trained successfully."
                )

                st.success(
                    f"""
                    Dataset uploaded successfully.

                    **{len(uploaded_df):,} customers**
                    were processed and the churn model
                    was trained on your uploaded dataset.
                    """
                )

                st.rerun()

            except Exception as e:

                audit_log(f"PIPELINE ERROR: {type(e).__name__}: {e}")
                progress.empty()

                status.empty()

                st.error(
                    "The machine-learning pipeline "
                    "could not process this dataset."
                )

                st.exception(
                    e
                )

                return

    # ========================================================
    # WINDOW 2
    # ========================================================

    if st.session_state.trained:

        navigation = (
            show_top_navigation()
        )

        # ----------------------------------------------------
        # UPLOADED FILE INFORMATION
        # ----------------------------------------------------

        col1, col2 = st.columns(
            [6, 1]
        )

        with col1:

            st.success(
                "✓ Uploaded dataset is active — "
                "all predictions are generated from this dataset."
            )

        with col2:

            if st.button(
                "🔄 New Dataset",
                width="stretch",
            ):

                st.session_state.clear()

                st.rerun()

        # ----------------------------------------------------
        # PAGE ROUTING
        # ----------------------------------------------------

        if navigation == "🏠 Overview":

            show_overview()

        elif navigation == "👤 Customer Risk":

            show_customer_risk()

        elif navigation == "🧠 SHAP Explainability":

            show_shap()

        elif navigation == "💰 Business Impact Estimator":

            show_business_impact()

        # ----------------------------------------------------
        # FOOTER
        # ----------------------------------------------------

        st.markdown(textwrap.dedent(
            """
            <div class="footer">

                Customer Churn Intelligence
                <br>
                Uploaded Dataset → XGBoost → Prediction
                → SHAP → Business Impact

            </div>
            """
            ),
                unsafe_allow_html=True,
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()'''






# ============================================================
# CUSTOMER CHURN INTELLIGENCE DASHBOARD
# ============================================================

from __future__ import annotations

import hashlib
import warnings
import textwrap
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# PROJECT ROOT & INGESTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingestion import DataIngestion

warnings.filterwarnings("ignore")

# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CSS
# ============================================================

st.markdown(
    """<style>
    .stApp { background: #071A33; }
    .main .block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 4rem; }
    .hero-container { text-align: center; padding-top: 45px; padding-bottom: 20px; }
    .hero-icon { font-size: 60px; margin-bottom: 10px; }
    .hero-title { font-size: 46px; font-weight: 850; color: #f8fafc; letter-spacing: -1.5px; margin-bottom: 10px; }
    .hero-subtitle { font-size: 18px; color: #cbd5e1; max-width: 850px; margin: auto; line-height: 1.7; }
    .upload-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 24px; padding: 35px; margin-top: 35px; box-shadow: 0 15px 45px rgba(15,23,42,0.08); }
    .readme-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 24px; min-height: 185px; box-shadow: 0 5px 20px rgba(15,23,42,0.04); }
    .readme-icon { font-size: 30px; margin-bottom: 10px; }
    .readme-title { font-size: 17px; font-weight: 750; color: #111827; margin-bottom: 8px; }
    .readme-text { font-size: 13px; color: #64748b; line-height: 1.65; }
    .top-brand { font-size: 26px; font-weight: 850; color: #f8fafc; margin-bottom: 4px; }
    .top-description { color: #cbd5e1; font-size: 13px; margin-bottom: 12px; }
    .dashboard-title { font-size: 35px; font-weight: 850; color: #f8fafc; letter-spacing: -0.8px; margin-top: 18px; margin-bottom: 5px; }
    .dashboard-subtitle { color: #cbd5e1; font-size: 15px; margin-bottom: 25px; }
    .section-title { color: #f8fafc; font-size: 22px; font-weight: 800; margin-top: 30px; margin-bottom: 12px; }
    .metric-card { background: #ffffff; border-radius: 17px; padding: 22px; border: 1px solid #e2e8f0; box-shadow: 0 5px 18px rgba(15,23,42,0.05); min-height: 135px; }
    .metric-label { color: #64748b; font-size: 12px; font-weight: 750; letter-spacing: 0.5px; margin-bottom: 8px; }
    .metric-value { color: #111827; font-size: 31px; font-weight: 850; }
    .metric-description { color: #94a3b8; font-size: 12px; margin-top: 5px; }
    .customer-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 24px; margin-top: 15px; margin-bottom: 20px; box-shadow: 0 5px 18px rgba(15,23,42,0.05); }
    .customer-id { color: #111827; font-size: 25px; font-weight: 850; }
    .customer-status { color: #64748b; font-size: 14px; margin-top: 5px; }
    .critical-card { background: #ffffff; border: 1px solid #fecaca; border-radius: 18px; padding: 20px; margin-bottom: 15px; box-shadow: 0 5px 18px rgba(15,23,42,0.06); }
    .critical-name { color: #111827; font-size: 18px; font-weight: 800; }
    .critical-id { color: #64748b; font-size: 12px; margin-top: 4px; }
    .critical-risk { color: #dc2626; font-size: 22px; font-weight: 850; }
    .risk-progress { width: 100%; height: 9px; background: #e5e7eb; border-radius: 10px; overflow: hidden; margin-top: 8px; }
    .risk-progress-inner { height: 100%; background: #dc2626; border-radius: 10px; }
    .critical-summary { background: #ffffff; border: 1px solid #fecaca; border-radius: 18px; padding: 24px; margin-bottom: 20px; }
    .critical-summary-number { font-size: 42px; font-weight: 850; color: #dc2626; }
    .critical-summary-text { color: #64748b; font-size: 14px; }
    .roi-positive { background: #ffffff; border: 1px solid #86efac; border-radius: 18px; padding: 25px; margin-top: 20px; }
    .roi-negative { background: #ffffff; border: 1px solid #fca5a5; border-radius: 18px; padding: 25px; margin-top: 20px; }
    .footer { text-align: center; color: #a8b6c8; font-size: 12px; margin-top: 50px; padding-top: 20px; border-top: 1px solid #29405f; }
    </style>""",
    unsafe_allow_html=True,
)

# ============================================================
# AUDIT LOG & UTILITIES
# ============================================================

def audit_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[CCP-AUDIT {timestamp}] {message}", flush=True)

def audit_section(title):
    print("\n" + "=" * 78, flush=True)
    print(f"[CCP-AUDIT] {title}", flush=True)
    print("=" * 78, flush=True)

DEFAULT_STATE = {
    "dataset_hash": None,
    "dataset": None,
    "predictions": None,
    "pipeline": None,
    "metrics": None,
    "feature_names": [],
    "numeric_features": [],
    "categorical_features": [],
    "customer_id_column": None,
    "customer_name_column": None,
    "target_column": None,
    "test_features": None,
    "test_target": None,
    "trained": False,
    "uploaded_filename": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

@st.cache_resource
def get_ingestion():
    return DataIngestion()

def normalize_column_name(column):
    return str(column).strip().lower().replace(" ", "").replace("_", "").replace("-", "")

def find_column(df, candidates):
    normalized = {normalize_column_name(col): col for col in df.columns}
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    return None

def detect_columns(df):
    customer_id = find_column(df, ["CustomerID", "Customer_ID", "Customer ID", "customerid", "customer_id", "id"])
    customer_name = find_column(df, ["CustomerName", "Customer_Name", "Customer Name", "customer_name", "Name", "customername"])
    target = find_column(df, ["Churn", "Churned", "Churn Status", "ChurnStatus", "Target"])
    return customer_id, customer_name, target

def get_customer_identity(row, customer_id_col, customer_name_col):
    customer_id = ""
    if customer_id_col is not None and customer_id_col in row.index:
        value = row[customer_id_col]
        if pd.notna(value):
            customer_id = str(value).strip()
    if customer_name_col is not None and customer_name_col in row.index:
        name_value = row[customer_name_col]
        if pd.notna(name_value):
            name_value = str(name_value).strip()
            if name_value:
                return name_value
    return customer_id

def normalize_target(series):
    values = series.copy()
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() > 0.95:
        unique_values = sorted(numeric.dropna().unique())
        if len(unique_values) != 2:
            raise ValueError("The Churn column must contain exactly two classes.")
        mapping = {unique_values[0]: 0, unique_values[1]: 1}
        return numeric.map(mapping).astype(int), str(unique_values[1]), str(unique_values[0])

    text = values.astype(str).str.strip().str.lower()
    positive_words = {"yes", "y", "true", "1", "churn", "churned", "left", "attrited", "attrition"}
    negative_words = {"no", "n", "false", "0", "no churn", "not churn", "not_churn", "retained", "active"}

    result = pd.Series(np.nan, index=series.index, dtype=float)
    result[text.isin(positive_words)] = 1
    result[text.isin(negative_words)] = 0

    if result.isna().any():
        all_unique = text.unique()
        if len(all_unique) == 2:
            mapping = {all_unique[0]: 0, all_unique[1]: 1}
            return text.map(mapping).astype(int), str(all_unique[1]), str(all_unique[0])
        raise ValueError("Unable to understand some values in the Churn column.")

    return result.astype(int), "Churn", "No Churn"

def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def risk_level(score):
    if score >= 80: return "Critical"
    if score >= 60: return "High"
    if score >= 30: return "Medium"
    return "Low"

def calculate_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def make_arrow_safe(df):
    output = df.copy()
    for column in output.columns:
        if pd.api.types.is_object_dtype(output[column]) or pd.api.types.is_string_dtype(output[column]):
            output[column] = output[column].astype("string").fillna("")
        else:
            output[column] = output[column].replace([np.inf, -np.inf], np.nan)
    return output

# ============================================================
# TRAIN MODEL
# ============================================================

def train_uploaded_dataset(df):
    audit_section("STARTING REAL ML TRAINING PIPELINE")
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("XGBoost is not installed.\n\nRun:\npip install xgboost")

    customer_id_col, customer_name_col, target_col = detect_columns(df)
    if target_col is None:
        raise ValueError("No Churn target column was found.")

    y, positive_label, negative_label = normalize_target(df[target_col])
    valid_mask = y.notna()
    working_df = df.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(int)

    if y.nunique() != 2:
        raise ValueError("The dataset must contain both churn and non-churn customers.")

    excluded_columns = {target_col}
    if customer_id_col: excluded_columns.add(customer_id_col)
    if customer_name_col: excluded_columns.add(customer_name_col)
    excluded_columns.update({col for col in working_df.columns if str(col).strip().lower().startswith("unnamed:")})

    feature_columns = [col for col in working_df.columns if col not in excluded_columns]
    if not feature_columns:
        raise ValueError("No usable ML feature columns found.")

    X = working_df[feature_columns].copy()
    completely_empty = [col for col in X.columns if X[col].isna().all()]
    if completely_empty:
        X = X.drop(columns=completely_empty)
        feature_columns = [col for col in feature_columns if col not in completely_empty]

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    transformers = []
    if numeric_features:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features))
    if categorical_features:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", make_one_hot_encoder())]), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = (int((y_train == 0).sum()) / positive_count) if positive_count > 0 else 1.0

    xgb_model = XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.08, subsample=0.85,
        colsample_bytree=0.85, min_child_weight=2, objective="binary:logistic",
        eval_metric="logloss", tree_method="hist", scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", xgb_model)])
    pipeline.fit(X_train, y_train)

    test_probability = pipeline.predict_proba(X_test)[:, 1]
    test_prediction = (test_probability >= 0.50).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, test_prediction),
        "precision": precision_score(y_test, test_prediction, zero_division=0),
        "recall": recall_score(y_test, test_prediction, zero_division=0),
        "f1": f1_score(y_test, test_prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_test, test_probability) if len(np.unique(y_test)) > 1 else np.nan,
        "confusion_matrix": confusion_matrix(y_test, test_prediction),
        "train_rows": len(X_train), "test_rows": len(X_test),
        "feature_count": len(feature_columns),
        "positive_label": positive_label, "negative_label": negative_label,
    }

    all_probability = pipeline.predict_proba(X)[:, 1]
    all_prediction = (all_probability >= 0.50).astype(int)

    result_df = working_df.copy()
    result_df["_model_churn_probability"] = all_probability
    result_df["_model_prediction"] = all_prediction
    result_df["_risk_score"] = all_probability * 100
    result_df["_risk_level"] = result_df["_risk_score"].apply(risk_level)
    result_df["_prediction_label"] = np.where(all_prediction == 1, "Churn", "No Churn")

    return {
        "pipeline": pipeline, "predictions": result_df, "metrics": metrics,
        "customer_id_column": customer_id_col, "customer_name_column": customer_name_col,
        "target_column": target_col, "numeric_features": numeric_features,
        "categorical_features": categorical_features, "feature_columns": feature_columns,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
    }

# ============================================================
# SHAP EXPLANATION
# ============================================================

def get_shap_explanation(pipeline, original_df, customer_index, feature_columns):
    if not SHAP_AVAILABLE:
        raise RuntimeError("SHAP is not installed.")

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    customer_row = original_df.loc[[customer_index], feature_columns]
    transformed = preprocessor.transform(customer_row)

    try:
        transformed_names = preprocessor.get_feature_names_out()
    except Exception:
        transformed_names = np.array(feature_columns)

    try:
        explainer = shap.TreeExplainer(model)
        explanation = explainer(transformed)
        values = explanation.values
        if isinstance(values, list): values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3: values = values[0, :, -1]
        elif values.ndim == 2: values = values[0]
        else: values = values.flatten()
    except Exception:
        def predict_probability(data):
            return pipeline.predict_proba(data)[:, 1]
        background = original_df[feature_columns].sample(min(50, len(original_df)), random_state=42)
        explainer = shap.Explainer(predict_probability, background)
        explanation = explainer(customer_row)
        values = np.asarray(explanation.values)
        if values.ndim == 2: values = values[0]
        values = values.flatten()
        transformed_names = np.array(feature_columns)

    aggregated = []
    for original_feature in feature_columns:
        indices = [i for i, name in enumerate(transformed_names) if str(name) == f"numeric__{original_feature}" or str(name).startswith(f"categorical__{original_feature}_")]
        if indices:
            contribution = float(np.sum(values[indices]))
            aggregated.append({"Feature": original_feature, "SHAP Value": contribution, "Absolute Impact": abs(contribution)})

    if not aggregated:
        for i, feature in enumerate(feature_columns):
            if i < len(values):
                aggregated.append({"Feature": feature, "SHAP Value": float(values[i]), "Absolute Impact": abs(float(values[i]))})

    shap_df = pd.DataFrame(aggregated)
    if shap_df.empty:
        raise RuntimeError("No SHAP contributions could be generated.")
    shap_df["Impact"] = np.where(shap_df["SHAP Value"] > 0, "Increases Churn Risk", "Decreases Churn Risk")
    return shap_df.sort_values("Absolute Impact", ascending=False).reset_index(drop=True)

# ============================================================
# UI RENDERING FUNCTIONS (WITH FIXED HTML BLOCKS)
# ============================================================

def show_upload_page():
    st.markdown('<div class="hero-container"><div class="hero-icon">📊</div><div class="hero-title">Customer Churn Intelligence</div><div class="hero-subtitle">Upload your real customer dataset and transform it into a machine-learning powered churn intelligence dashboard.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📖 How This Dashboard Works</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("📁", "1. Upload Dataset", "Upload your real customer CSV."),
        ("🤖", "2. Train Model", "Preprocess the data and train XGBoost."),
        ("🧠", "3. Explain Predictions", "Use SHAP to understand churn drivers."),
        ("💰", "4. Estimate Impact", "Estimate customer-retention business impact.")
    ]

    for col, card in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f'<div class="readme-card"><div class="readme-icon">{card[0]}</div><div class="readme-title">{card[1]}</div><div class="readme-text">{card[2]}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">📋 Dataset Requirements</div>', unsafe_allow_html=True)
    st.info("**Required:** CSV with a churn target such as `Churn`.\n\n**Recommended:** `CustomerID`.\n\n**Optional:** `CustomerName`, `Customer Name`, `customer_name`, or `Name`.\n\n**Important:** Customer names are NEVER generated.")
    st.markdown('<div class="upload-card"><div class="section-title" style="margin-top:0;color:#111827;">📤 Upload Your Customer Dataset</div><p style="color:#64748b;">The uploaded CSV will be passed directly to <b>src/data/ingestion.py</b>.</p></div>', unsafe_allow_html=True)

    return st.file_uploader("Choose CSV file", type=["csv"])

def show_top_navigation():
    st.markdown('<div class="top-brand">📊 Customer Churn Intelligence</div><div class="top-description">Machine Learning Powered Customer Retention Platform</div>', unsafe_allow_html=True)
    navigation = st.radio("Dashboard Navigation", ["🏠 Overview", "👤 Customer Risk", "🧠 SHAP Explainability", "💰 Business Impact Simulator"], horizontal=True, label_visibility="collapsed")
    st.divider()
    return navigation

def render_critical_customer_card(row, customer_id_col, customer_name_col):
    identity = get_customer_identity(row, customer_id_col, customer_name_col)
    customer_id = str(row[customer_id_col]).strip() if customer_id_col is not None and customer_id_col in row.index else ""
    score = float(row["_risk_score"])
    probability = float(row["_model_churn_probability"])

    card_html = f'''<div class="critical-card"><div class="critical-name">🔴 {identity}</div><div class="critical-id">Customer ID: {customer_id}</div><div style="margin-top:14px;"><div class="critical-risk">{score:.1f}/100</div><div style="color:#64748b;font-size:12px;">Risk Score</div><div class="risk-progress"><div class="risk-progress-inner" style="width:{min(score, 100):.1f}%;"></div></div></div><div style="display:flex;justify-content:space-between;margin-top:15px;color:#475569;font-size:13px;"><span><b>Churn Probability</b><br>{probability * 100:.2f}%</span><span><b>Risk Level</b><br>Critical</span><span><b>Prediction</b><br>{row["_prediction_label"]}</span></div></div>'''
    st.markdown(card_html, unsafe_allow_html=True)

def show_overview():
    result = st.session_state.predictions
    customer_id_col = st.session_state.customer_id_column
    customer_name_col = st.session_state.customer_name_column

    st.markdown('<div class="dashboard-title">Customer Churn Overview</div><div class="dashboard-subtitle">Real predictions generated from the uploaded customer dataset.</div>', unsafe_allow_html=True)

    total_customers = len(result)
    predicted_churn = int((result["_model_prediction"] == 1).sum())
    critical_customers = int((result["_risk_score"] > 80).sum())
    average_risk = float(result["_risk_score"].mean())

    metric_data = [
        ("TOTAL CUSTOMERS", f"{total_customers:,}", "Customers in uploaded dataset"),
        ("PREDICTED CHURN", f"{predicted_churn:,}", "Customers predicted to churn"),
        ("CRITICAL RISK", f"{critical_customers:,}", "Customers with risk score above 80"),
        ("AVERAGE RISK", f"{average_risk:.1f}%", "Average predicted churn risk")
    ]

    columns = st.columns(4)
    for col, item in zip(columns, metric_data):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{item[0]}</div><div class="metric-value">{item[1]}</div><div class="metric-description">{item[2]}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🚨 Critical Risk Customers</div>', unsafe_allow_html=True)
    critical = result[result["_risk_score"] > 80].sort_values("_risk_score", ascending=False).copy()
    cc1, cc2 = st.columns([1, 3])

    with cc1:
        st.markdown(f'<div class="critical-summary"><div class="critical-summary-number">{len(critical):,}</div><div class="critical-summary-text">Customers with a churn risk score above 80.</div></div>', unsafe_allow_html=True)

    with cc2:
        if critical.empty:
            st.success("No critical-risk customers were detected.")
        else:
            for _, row in critical.head(5).iterrows():
                render_critical_customer_card(row, customer_id_col, customer_name_col)

    st.markdown('<div class="section-title">🎯 Customer Risk Distribution</div>', unsafe_allow_html=True)
    risk_counts = result["_risk_level"].value_counts().reindex(["Low", "Medium", "High", "Critical"], fill_value=0).reset_index()
    risk_counts.columns = ["Risk Level", "Customers"]
    fig = px.bar(risk_counts, x="Risk Level", y="Customers", text="Customers")
    fig.update_layout(template="plotly_white", height=420, showlegend=False, xaxis_title="", yaxis_title="Customers")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="section-title">🚨 Customers Requiring Attention</div>', unsafe_allow_html=True)
    high_risk = result[result["_risk_score"] >= 60].sort_values("_risk_score", ascending=False).head(15).copy()
    if not high_risk.empty:
        display = pd.DataFrame()
        if customer_id_col: display["Customer ID"] = high_risk[customer_id_col].astype(str)
        if customer_name_col: display["Customer Name"] = high_risk[customer_name_col].astype("string").fillna("")
        display["Prediction"] = high_risk["_prediction_label"]
        display["Risk Score"] = high_risk["_risk_score"].round(2)
        display["Risk Level"] = high_risk["_risk_level"]
        st.dataframe(make_arrow_safe(display), width="stretch", hide_index=True)

    st.markdown('<div class="section-title">📥 Export Predictions</div>', unsafe_allow_html=True)
    st.download_button("Download Customer Risk Predictions CSV", data=result.to_csv(index=False).encode("utf-8"), file_name="customer_churn_predictions.csv", mime="text/csv")

def show_customer_risk():
    result = st.session_state.predictions
    customer_id_col = st.session_state.customer_id_column
    customer_name_col = st.session_state.customer_name_column
    feature_columns = st.session_state.feature_names

    st.markdown('<div class="dashboard-title">Customer Risk Analysis</div><div class="dashboard-subtitle">Select a real customer from the uploaded dataset and inspect the model prediction.</div>', unsafe_allow_html=True)
    ids = result[customer_id_col].astype(str).tolist() if customer_id_col else [str(i) for i in result.index]
    selected_id = st.selectbox("Select Customer", ids)

    if customer_id_col:
        customer_index = result[result[customer_id_col].astype(str) == selected_id].index[0]
    else:
        customer_index = int(selected_id)

    customer = result.loc[customer_index]
    probability = float(customer["_model_churn_probability"])
    risk_score = float(customer["_risk_score"])
    identity = get_customer_identity(customer, customer_id_col, customer_name_col)

    st.markdown(f'<div class="customer-card"><div class="customer-id">👤 {identity}</div><div class="customer-status">Customer ID: {selected_id} &nbsp;|&nbsp; Prediction: {customer["_prediction_label"]} &nbsp;|&nbsp; Risk: {customer["_risk_level"]}</div></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Churn Probability", f"{probability * 100:.2f}%")
    c2.metric("Risk Score", f"{risk_score:.2f}/100")
    c3.metric("Prediction", customer["_prediction_label"])

    fig = go.Figure(go.Indicator(mode="gauge+number", value=risk_score, title={"text": "Customer Churn Risk Score"}, gauge={"axis": {"range": [0, 100]}}))
    fig.update_layout(height=360, margin=dict(l=30, r=30, t=70, b=20))
    st.plotly_chart(fig, width="stretch")

def show_shap():
    result = st.session_state.predictions
    pipeline = st.session_state.pipeline
    customer_id_col = st.session_state.customer_id_column
    customer_name_col = st.session_state.customer_name_column
    feature_columns = st.session_state.feature_names

    st.markdown('<div class="dashboard-title">SHAP Explainability</div><div class="dashboard-subtitle">Understand which actual customer features are driving the churn prediction.</div>', unsafe_allow_html=True)
    if not SHAP_AVAILABLE:
        st.error("SHAP is not installed. Run: pip install shap")
        return

    ids = result[customer_id_col].astype(str).tolist() if customer_id_col else [str(i) for i in result.index]
    selected_id = st.selectbox("Select Customer for SHAP Explanation", ids, key="shap_customer_selector")

    customer_index = result[result[customer_id_col].astype(str) == selected_id].index[0] if customer_id_col else int(selected_id)
    customer = result.loc[customer_index]

    shap_df = get_shap_explanation(pipeline, result, customer_index, feature_columns)

    st.markdown('<div class="section-title">🔍 What Is Driving This Prediction?</div>', unsafe_allow_html=True)
    fig = px.bar(shap_df.head(12).sort_values("SHAP Value"), x="SHAP Value", y="Feature", color="Impact", orientation="h", title="SHAP Feature Contributions")
    fig.update_layout(template="plotly_white", height=550, yaxis_title="", xaxis_title="SHAP Contribution")
    st.plotly_chart(fig, width="stretch")

def business_impact_simulator(predictions):
    customer_id_col = st.session_state.customer_id_column
    customer_name_col = st.session_state.customer_name_column

    st.markdown('<div class="dashboard-title">💼 Business Impact Simulator</div><div class="dashboard-subtitle">Convert real churn predictions into a practical retention strategy and estimate financial impact.</div>', unsafe_allow_html=True)

    high_risk = predictions[predictions["_risk_score"] > 60].sort_values("_model_churn_probability", ascending=False).copy()
    high_risk_count = len(high_risk)

    st.markdown("### 🎯 Retention Opportunity")
    p1, p2, p3 = st.columns(3)
    p1.metric("High-Risk Customers", f"{high_risk_count:,}")
    p2.metric("High Risk", f"{int(((predictions['_risk_score'] > 60) & (predictions['_risk_score'] <= 80)).sum()):,}")
    p3.metric("Critical Risk", f"{int((predictions['_risk_score'] > 80).sum()):,}")

    st.markdown("### 🚨 Critical Risk Customers")
    critical_customers = predictions[predictions["_risk_score"] > 80].sort_values("_risk_score", ascending=False).copy()
    
    if not critical_customers.empty:
        st.markdown(f'<div class="critical-summary"><div class="critical-summary-number">{len(critical_customers):,}</div><div class="critical-summary-text">Customers with a churn risk score above 80.</div></div>', unsafe_allow_html=True)
        for _, row in critical_customers.head(10).iterrows():
            render_critical_customer_card(row, customer_id_col, customer_name_col)

    if high_risk_count == 0: return

    st.divider()
    st.markdown("### ⚙️ Campaign Assumptions")
    c1, c2 = st.columns(2)
    with c1:
        capacity = st.number_input("📞 Contact capacity", min_value=1, max_value=high_risk_count, value=min(2000, high_risk_count))
        save_rate_input = st.slider("🎯 Expected retention success rate", 1, 100, 15, 1, format="%d%%")
    with c2:
        customer_value = st.number_input("💰 Average customer lifetime value ($)", min_value=0.0, value=1000.0)
        campaign_cost = st.number_input("📣 Campaign cost per customer ($)", min_value=0.0, value=20.0)

    contacted = min(int(capacity), high_risk_count)
    revenue_saved = (contacted * (save_rate_input / 100)) * customer_value
    total_cost = contacted * campaign_cost
    net_impact = revenue_saved - total_cost

    st.markdown("### 📊 Estimated Business Impact")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Customers Contacted", f"{contacted:,}")
    r2.metric("Expected Saved", f"{contacted * (save_rate_input / 100):,.0f}")
    r3.metric("Revenue Saved", f"${revenue_saved:,.0f}")
    r4.metric("Campaign Cost", f"${total_cost:,.0f}")

    if net_impact >= 0:
        st.markdown(f'<div class="roi-positive"><div style="font-size:1.35rem;font-weight:750;color:#166534;">📈 Positive Business Impact</div><div style="font-size:2.2rem;font-weight:850;color:#111827;">${net_impact:,.0f}</div><p style="color:#64748b;">Estimated net value after campaign costs.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="roi-negative"><div style="font-size:1.35rem;font-weight:750;color:#b91c1c;">⚠️ Negative Business Impact</div><div style="font-size:2.2rem;font-weight:850;color:#111827;">${net_impact:,.0f}</div><p style="color:#64748b;">Campaign cost exceeds expected revenue saved.</p></div>', unsafe_allow_html=True)

def show_business_impact():
    business_impact_simulator(st.session_state.predictions)

# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    if not st.session_state.trained:
        uploaded_file = show_upload_page()
        if uploaded_file is None:
            st.markdown('<div class="footer">Upload a CSV dataset to begin the customer churn ML workflow.</div>', unsafe_allow_html=True)
            return

        file_bytes = uploaded_file.getvalue()
        current_hash = calculate_file_hash(file_bytes)

        if st.session_state.dataset_hash != current_hash:
            st.session_state.trained = False
            try:
                ingestion = get_ingestion()
                uploaded_file.seek(0)
                uploaded_df = ingestion.load_uploaded_data(uploaded_file)
                result = train_uploaded_dataset(uploaded_df)

                st.session_state.dataset = uploaded_df
                st.session_state.pipeline = result["pipeline"]
                st.session_state.predictions = result["predictions"]
                st.session_state.metrics = result["metrics"]
                st.session_state.feature_names = result["feature_columns"]
                st.session_state.customer_id_column = result["customer_id_column"]
                st.session_state.customer_name_column = result["customer_name_column"]
                st.session_state.dataset_hash = current_hash
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.trained = True
                st.rerun()

            except Exception as e:
                st.error("The machine-learning pipeline could not process this dataset.")
                st.exception(e)
                return

    if st.session_state.trained:
        navigation = show_top_navigation()
        col1, col2 = st.columns([6, 1])
        with col1:
            filename = st.session_state.uploaded_filename or "Uploaded dataset"
            st.success(f"✓ Active dataset: **{filename}**")
        with col2:
            if st.button("🔄 New Dataset", width="stretch"):
                st.session_state.clear()
                st.rerun()

        if navigation == "🏠 Overview": show_overview()
        elif navigation == "👤 Customer Risk": show_customer_risk()
        elif navigation == "🧠 SHAP Explainability": show_shap()
        elif navigation == "💰 Business Impact Simulator": show_business_impact()

        st.markdown('<div class="footer">Customer Churn Intelligence<br>Uploaded Dataset → Ingestion → Preprocessing → XGBoost → Prediction → SHAP → Business Impact</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
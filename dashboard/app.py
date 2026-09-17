# ============================================================
# CUSTOMER CHURN INTELLIGENCE DASHBOARD
# ============================================================

from __future__ import annotations
#use these libs
import hashlib
import warnings
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
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
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
    """
<style>

.stApp {
    background: #071A33;
}

.main .block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

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

.upload-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    padding: 35px;
    margin-top: 35px;
    box-shadow: 0 15px 45px rgba(15,23,42,0.08);
}

.readme-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 24px;
    min-height: 185px;
    box-shadow: 0 5px 20px rgba(15,23,42,0.04);
}

.readme-icon {
    font-size: 30px;
    margin-bottom: 10px;
}

.readme-title {
    font-size: 17px;
    font-weight: 750;
    color: #111827;
    margin-bottom: 8px;
}

.readme-text {
    font-size: 13px;
    color: #64748b;
    line-height: 1.65;
}

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

.section-title {
    color: #f8fafc;
    font-size: 22px;
    font-weight: 800;
    margin-top: 30px;
    margin-bottom: 12px;
}

.metric-card {
    background: #ffffff;
    border-radius: 17px;
    padding: 22px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 5px 18px rgba(15,23,42,0.05);
    min-height: 135px;
}

.metric-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.metric-value {
    color: #111827;
    font-size: 31px;
    font-weight: 850;
}

.metric-description {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 5px;
}

.customer-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 24px;
    margin-top: 15px;
    margin-bottom: 20px;
    box-shadow: 0 5px 18px rgba(15,23,42,0.05);
}

.customer-id {
    color: #111827;
    font-size: 25px;
    font-weight: 850;
}

.customer-status {
    color: #64748b;
    font-size: 14px;
    margin-top: 5px;
}

.critical-card {
    background: #ffffff;
    border: 1px solid #fecaca;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 5px 18px rgba(15,23,42,0.06);
}

.critical-name {
    color: #111827;
    font-size: 18px;
    font-weight: 800;
}

.critical-id {
    color: #64748b;
    font-size: 12px;
    margin-top: 4px;
}

.critical-risk {
    color: #dc2626;
    font-size: 22px;
    font-weight: 850;
}

.risk-progress {
    width: 100%;
    height: 9px;
    background: #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 8px;
}

.risk-progress-inner {
    height: 100%;
    background: #dc2626;
    border-radius: 10px;
}

.critical-summary {
    background: #ffffff;
    border: 1px solid #fecaca;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
}

.critical-summary-number {
    font-size: 42px;
    font-weight: 850;
    color: #dc2626;
}

.critical-summary-text {
    color: #64748b;
    font-size: 14px;
}

.roi-positive {
    background: #ffffff;
    border: 1px solid #86efac;
    border-radius: 18px;
    padding: 25px;
    margin-top: 20px;
}

.roi-negative {
    background: #ffffff;
    border: 1px solid #fca5a5;
    border-radius: 18px;
    padding: 25px;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: #a8b6c8;
    font-size: 12px;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #29405f;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# AUDIT LOG
# ============================================================

def audit_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[CCP-AUDIT {timestamp}] {message}",
        flush=True
    )


def audit_section(title):
    print("\n" + "=" * 78, flush=True)
    print(
        f"[CCP-AUDIT] {title}",
        flush=True
    )
    print("=" * 78, flush=True)


# ============================================================
# SESSION STATE
# ============================================================

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
    "trained": False,
    "uploaded_filename": None,
    "decision_threshold": 0.50,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# INGESTION
# ============================================================

@st.cache_resource
def get_ingestion():

    return DataIngestion()


# ============================================================
# COLUMN UTILITIES
# ============================================================

def normalize_column_name(column):

    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def find_column(df, candidates):

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
            "customer_name",
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


# ============================================================
# CUSTOMER ID / NAME
# ============================================================

def clean_identity_columns(
    df,
    customer_id_col,
    customer_name_col,
):

    output = df.copy()

    if customer_id_col is not None:

        output[customer_id_col] = (
            output[customer_id_col]
            .astype("string")
            .str.strip()
        )

        if output[customer_id_col].isna().any():

            raise ValueError(
                "Customer ID contains missing values."
            )

    if customer_name_col is not None:

        output[customer_name_col] = (
            output[customer_name_col]
            .astype("string")
            .str.strip()
        )

    return output


def get_customer_identity(
    row,
    customer_id_col,
    customer_name_col,
):

    customer_id = ""

    if (
        customer_id_col is not None
        and customer_id_col in row.index
    ):

        value = row[customer_id_col]

        if pd.notna(value):
            customer_id = str(value).strip()

    if (
        customer_name_col is not None
        and customer_name_col in row.index
    ):

        name_value = row[customer_name_col]

        if pd.notna(name_value):

            name_value = str(name_value).strip()

            if name_value:
                return name_value

    return customer_id


# ============================================================
# TARGET NORMALIZATION
# ============================================================

def normalize_target(series):

    values = series.copy()

    numeric = pd.to_numeric(
        values,
        errors="coerce"
    )

    if numeric.notna().mean() > 0.95:

        unique_values = sorted(
            numeric.dropna().unique()
        )

        if len(unique_values) != 2:

            raise ValueError(
                "The Churn column must contain exactly two classes."
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

    if result.isna().any():

        all_unique = text.unique()

        if len(all_unique) == 2:

            mapping = {
                all_unique[0]: 0,
                all_unique[1]: 1,
            }

            return (
                text.map(mapping).astype(int),
                str(all_unique[1]),
                str(all_unique[0]),
            )

        raise ValueError(
            "Unable to understand some values in the Churn column."
        )

    return (
        result.astype(int),
        "Churn",
        "No Churn",
    )


# ============================================================
# ONE HOT ENCODER
# ============================================================

def make_one_hot_encoder():

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


# ============================================================
# RISK
# ============================================================

def risk_level(score):

    if score >= 80:
        return "Critical"

    if score >= 60:
        return "High"

    if score >= 30:
        return "Medium"

    return "Low"


# ============================================================
# HASH
# ============================================================

def calculate_file_hash(file_bytes):

    return hashlib.md5(
        file_bytes
    ).hexdigest()


# ============================================================
# ARROW SAFE
# ============================================================

def make_arrow_safe(df):

    output = df.copy()

    for column in output.columns:

        if (
            pd.api.types.is_object_dtype(
                output[column]
            )
            or pd.api.types.is_string_dtype(
                output[column]
            )
        ):

            output[column] = (
                output[column]
                .astype("string")
                .fillna("")
            )

        else:

            output[column] = output[column].replace(
                [np.inf, -np.inf],
                np.nan,
            )

    return output


# ============================================================
# FIND BEST THRESHOLD
# ============================================================

def find_best_threshold(
    y_true,
    probabilities,
):

    thresholds = np.arange(
        0.20,
        0.81,
        0.01,
    )

    best_threshold = 0.50
    best_f1 = -1

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = float(
                threshold
            )

    return best_threshold, best_f1


# ============================================================
# TRAIN MODEL
# ============================================================

def train_uploaded_dataset(df):

    audit_section(
        "STARTING OPTIMIZED XGBOOST TRAINING PIPELINE"
    )

    if not XGBOOST_AVAILABLE:

        raise RuntimeError(
            "XGBoost is not installed.\n\n"
            "Run:\n"
            "pip install xgboost"
        )

    # --------------------------------------------------------
    # DETECT COLUMNS
    # --------------------------------------------------------

    (
        customer_id_col,
        customer_name_col,
        target_col,
    ) = detect_columns(df)

    if target_col is None:

        raise ValueError(
            "No Churn target column was found."
        )

    audit_log(
        f"Detected Target      : {target_col}"
    )

    audit_log(
        f"Detected Customer ID : {customer_id_col}"
    )

    audit_log(
        f"Detected Customer Name: {customer_name_col}"
    )

    # --------------------------------------------------------
    # CLEAN IDENTITY
    # --------------------------------------------------------

    df = clean_identity_columns(
        df,
        customer_id_col,
        customer_name_col,
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    y, positive_label, negative_label = (
        normalize_target(
            df[target_col]
        )
    )

    valid_mask = y.notna()

    working_df = (
        df.loc[valid_mask]
        .copy()
    )

    y = (
        y.loc[valid_mask]
        .astype(int)
    )

    if y.nunique() != 2:

        raise ValueError(
            "Dataset must contain both churn and non-churn customers."
        )

    # --------------------------------------------------------
    # REMOVE IDENTITY / TARGET COLUMNS
    # --------------------------------------------------------

    excluded_columns = {
        target_col
    }

    if customer_id_col:

        excluded_columns.add(
            customer_id_col
        )

    if customer_name_col:

        excluded_columns.add(
            customer_name_col
        )

    excluded_columns.update(
        {
            col
            for col in working_df.columns
            if str(col)
            .strip()
            .lower()
            .startswith("unnamed:")
        }
    )

    feature_columns = [
        col
        for col in working_df.columns
        if col not in excluded_columns
    ]

    if not feature_columns:

        raise ValueError(
            "No usable ML feature columns found."
        )

    X = working_df[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # REMOVE COMPLETELY EMPTY FEATURES
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

    # --------------------------------------------------------
    # FEATURE TYPES
    # --------------------------------------------------------

    numeric_features = (
        X
        .select_dtypes(
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

    audit_log(
        f"Total Features       : {len(feature_columns)}"
    )

    audit_log(
        f"Numeric Features     : {len(numeric_features)}"
    )

    audit_log(
        f"Categorical Features : {len(categorical_features)}"
    )

    audit_log(
        f"Dataset Rows         : {len(X)}"
    )

    # ========================================================
    # TRAIN / VALIDATION / TEST SPLIT
    # ========================================================

    X_train_full, X_test, y_train_full, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.20,
            random_state=42,
            stratify=y_train_full,
        )
    )

    # ========================================================
    # PREPROCESSOR
    # ========================================================

    transformers = []

    if numeric_features:

        transformers.append(
            (
                "numeric",

                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        )
                    ]
                ),

                numeric_features,
            )
        )

    if categorical_features:

        transformers.append(
            (
                "categorical",

                Pipeline(
                    [
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
                ),

                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    # ========================================================
    # CLASS BALANCE
    # ========================================================

    negative_count = int(
        (y_train == 0).sum()
    )

    positive_count = int(
        (y_train == 1).sum()
    )

    scale_pos_weight = (
        negative_count / positive_count
        if positive_count > 0
        else 1.0
    )

    audit_log(
        f"Non-Churn Training Rows : {negative_count}"
    )

    audit_log(
        f"Churn Training Rows     : {positive_count}"
    )

    audit_log(
        f"Scale Pos Weight        : {scale_pos_weight:.4f}"
    )

    # ========================================================
    # BASE XGBOOST
    # ========================================================

    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",

        tree_method="hist",

        random_state=42,

        n_jobs=1,

        scale_pos_weight=scale_pos_weight,

        verbosity=0,
    )

    pipeline = Pipeline(
        [
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

    # ========================================================
    # HYPERPARAMETER SEARCH
    # ========================================================

    param_distributions = {

        "model__n_estimators": [
            200,
            300,
            400,
            500,
        ],

        "model__max_depth": [
            3,
            4,
            5,
            6,
        ],

        "model__learning_rate": [
            0.02,
            0.03,
            0.05,
            0.08,
        ],

        "model__min_child_weight": [
            1,
            2,
            4,
            6,
        ],

        "model__subsample": [
            0.75,
            0.85,
            0.95,
            1.0,
        ],

        "model__colsample_bytree": [
            0.70,
            0.80,
            0.90,
            1.0,
        ],

        "model__gamma": [
            0,
            0.1,
            0.2,
            0.4,
        ],

        "model__reg_alpha": [
            0,
            0.01,
            0.1,
        ],

        "model__reg_lambda": [
            1,
            2,
            5,
        ],
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=8,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    audit_section(
        "STARTING XGBOOST HYPERPARAMETER SEARCH"
    )

    search.fit(
        X_train,
        y_train,
    )

    pipeline = search.best_estimator_

    best_params = search.best_params_

    audit_log(
        f"Best CV ROC-AUC: {search.best_score_:.4f}"
    )

    audit_log(
        f"Best Parameters: {best_params}"
    )

    # ========================================================
    # VALIDATION THRESHOLD
    # ========================================================

    validation_probability = (
        pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    decision_threshold, validation_f1 = (
        find_best_threshold(
            y_validation,
            validation_probability,
        )
    )

    audit_log(
        f"Optimized Decision Threshold: "
        f"{decision_threshold:.2f}"
    )

    audit_log(
        f"Validation F1 at Threshold: "
        f"{validation_f1:.4f}"
    )

    # ========================================================
    # TEST EVALUATION
    # ========================================================

    test_probability = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    test_prediction = (
        test_probability
        >= decision_threshold
    ).astype(int)

    metrics = {

        "accuracy": accuracy_score(
            y_test,
            test_prediction,
        ),

        "precision": precision_score(
            y_test,
            test_prediction,
            zero_division=0,
        ),

        "recall": recall_score(
            y_test,
            test_prediction,
            zero_division=0,
        ),

        "f1": f1_score(
            y_test,
            test_prediction,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_test,
            test_probability,
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            test_prediction,
        ),

        "train_rows": len(X_train),

        "validation_rows": len(X_validation),

        "test_rows": len(X_test),

        "feature_count": len(
            feature_columns
        ),

        "positive_label": positive_label,

        "negative_label": negative_label,

        "decision_threshold": decision_threshold,

        "validation_f1": validation_f1,

        "cv_roc_auc": search.best_score_,

        "best_params": best_params,
    }

    # ========================================================
    # METRICS LOG
    # ========================================================

    audit_section(
        "FINAL TEST MODEL PERFORMANCE"
    )

    audit_log(
        f"Accuracy       : {metrics['accuracy']:.4f}"
    )

    audit_log(
        f"Precision      : {metrics['precision']:.4f}"
    )

    audit_log(
        f"Recall         : {metrics['recall']:.4f}"
    )

    audit_log(
        f"F1 Score       : {metrics['f1']:.4f}"
    )

    audit_log(
        f"ROC AUC        : {metrics['roc_auc']:.4f}"
    )

    audit_log(
        f"CV ROC AUC     : {metrics['cv_roc_auc']:.4f}"
    )

    audit_log(
        f"Decision Thresh : {metrics['decision_threshold']:.2f}"
    )

    # ========================================================
    # PREDICT ALL CUSTOMERS
    # ========================================================

    all_probability = (
        pipeline.predict_proba(
            X
        )[:, 1]
    )

    all_prediction = (
        all_probability
        >= decision_threshold
    ).astype(int)

    result_df = working_df.copy()

    result_df[
        "_model_churn_probability"
    ] = all_probability

    result_df[
        "_model_prediction"
    ] = all_prediction

    result_df[
        "_risk_score"
    ] = all_probability * 100

    result_df[
        "_risk_level"
    ] = (
        result_df[
            "_risk_score"
        ]
        .apply(risk_level)
    )

    result_df[
        "_prediction_label"
    ] = np.where(
        all_prediction == 1,
        "Churn",
        "No Churn",
    )

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
    }


# ============================================================
# SHAP
# ============================================================

def get_shap_explanation(
    pipeline,
    original_df,
    customer_index,
    feature_columns,
):

    if not SHAP_AVAILABLE:

        raise RuntimeError(
            "SHAP is not installed."
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

    customer_row = (
        original_df
        .loc[
            [customer_index],
            feature_columns,
        ]
    )

    transformed = (
        preprocessor.transform(
            customer_row
        )
    )

    try:

        transformed_names = (
            preprocessor
            .get_feature_names_out()
        )

    except Exception:

        transformed_names = np.array(
            feature_columns
        )

    try:

        explainer = shap.TreeExplainer(
            model
        )

        explanation = explainer(
            transformed
        )

        values = explanation.values

        if isinstance(values, list):

            values = values[-1]

        values = np.asarray(values)

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

    except Exception as shap_error:

        audit_log(
            f"TreeExplainer fallback: {shap_error}"
        )

        def predict_probability(data):

            return pipeline.predict_proba(
                data
            )[:, 1]

        background = (
            original_df[
                feature_columns
            ]
            .sample(
                min(
                    50,
                    len(original_df),
                ),
                random_state=42,
            )
        )

        explainer = shap.Explainer(
            predict_probability,
            background,
        )

        explanation = explainer(
            customer_row
        )

        values = np.asarray(
            explanation.values
        )

        if values.ndim == 2:

            values = values[0]

        values = values.flatten()

        transformed_names = np.array(
            feature_columns
        )

    aggregated = []

    for original_feature in feature_columns:

        indices = [

            i

            for i, name
            in enumerate(
                transformed_names
            )

            if (
                str(name)
                == f"numeric__{original_feature}"
            )

            or str(name).startswith(
                f"categorical__{original_feature}_"
            )

        ]

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
            "No SHAP contributions could be generated."
        )

    shap_df["Impact"] = np.where(
        shap_df["SHAP Value"] > 0,
        "Increases Churn Risk",
        "Decreases Churn Risk",
    )

    return (
        shap_df
        .sort_values(
            "Absolute Impact",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# UPLOAD PAGE
# ============================================================

def show_upload_page():

    st.markdown(
        """
<div class="hero-container">
    <div class="hero-icon">📊</div>

    <div class="hero-title">
        Customer Churn Intelligence
    </div>

    <div class="hero-subtitle">
        Upload your real customer dataset and transform it into
        a machine-learning powered churn intelligence dashboard.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">📖 How This Dashboard Works</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    cards = [

        (
            "📁",
            "1. Upload Dataset",
            "Upload your real customer CSV.",
        ),

        (
            "🤖",
            "2. Optimize Model",
            "Tune XGBoost using cross-validation.",
        ),

        (
            "🧠",
            "3. Explain Predictions",
            "Use SHAP to understand churn drivers.",
        ),

        (
            "💰",
            "4. Estimate Impact",
            "Estimate customer-retention business impact.",
        ),
    ]

    for col, card in zip(
        [c1, c2, c3, c4],
        cards,
    ):

        with col:

            st.markdown(
                f"""
<div class="readme-card">

<div class="readme-icon">
{card[0]}
</div>

<div class="readme-title">
{card[1]}
</div>

<div class="readme-text">
{card[2]}
</div>

</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">📋 Dataset Requirements</div>',
        unsafe_allow_html=True,
    )

    st.info(
        """
**Required:** CSV with a churn target such as `Churn`.

**Recommended:** `CustomerID`.

**Optional:** `CustomerName`, `Customer Name`,
`customer_name`, or `Name`.

**Important:** Customer names are NEVER generated.
"""
    )

    st.markdown(
        """
<div class="upload-card">

<div class="section-title"
     style="margin-top:0;color:#111827;">

📤 Upload Your Customer Dataset

</div>

<p style="color:#64748b;">

The uploaded CSV will be passed directly to
<b>src/data/ingestion.py</b>.

</p>

</div>
""",
        unsafe_allow_html=True,
    )

    return st.file_uploader(
        "Choose CSV file",
        type=["csv"],
    )


# ============================================================
# NAVIGATION
# ============================================================

def show_top_navigation():

    st.markdown(
        """
<div class="top-brand">
📊 Customer Churn Intelligence
</div>

<div class="top-description">
Machine Learning Powered Customer Retention Platform
</div>
""",
        unsafe_allow_html=True,
    )

    navigation = st.radio(
        "Dashboard Navigation",
        [
            "🏠 Overview",
            "👤 Customer Risk",
            "🧠 SHAP Explainability",
            "💰 Business Impact Simulator",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    return navigation


# ============================================================
# CRITICAL CUSTOMER CARD
# ============================================================

def render_critical_customer_card(
    row,
    customer_id_col,
    customer_name_col,
):

    identity = get_customer_identity(
        row,
        customer_id_col,
        customer_name_col,
    )

    customer_id = (
        str(row[customer_id_col]).strip()
        if (
            customer_id_col is not None
            and customer_id_col in row.index
        )
        else ""
    )

    score = float(
        row["_risk_score"]
    )

    probability = float(
        row["_model_churn_probability"]
    )

    card_html = f"""
<div class="critical-card">

<div class="critical-name">
🔴 {identity}
</div>

<div class="critical-id">
Customer ID: {customer_id}
</div>

<div style="margin-top:14px;">

<div class="critical-risk">
{score:.1f}/100
</div>

<div style="color:#64748b;font-size:12px;">
Risk Score
</div>

<div class="risk-progress">

<div class="risk-progress-inner"
     style="width:{min(score,100):.1f}%;">
</div>

</div>

</div>

<div style="
display:flex;
justify-content:space-between;
margin-top:15px;
color:#475569;
font-size:13px;
">

<span>
<b>Churn Probability</b>
<br>
{probability * 100:.2f}%
</span>

<span>
<b>Risk Level</b>
<br>
Critical
</span>

<span>
<b>Prediction</b>
<br>
{row["_prediction_label"]}
</span>

</div>

</div>
"""

    st.markdown(
        card_html,
        unsafe_allow_html=True,
    )


# ============================================================
# OVERVIEW
# ============================================================

def show_overview():

    result = st.session_state.predictions

    customer_id_col = (
        st.session_state.customer_id_column
    )

    customer_name_col = (
        st.session_state.customer_name_column
    )

    metrics = st.session_state.metrics

    st.markdown(
        """
<div class="dashboard-title">
Customer Churn Overview
</div>

<div class="dashboard-subtitle">
Real predictions generated from the uploaded customer dataset.
</div>
""",
        unsafe_allow_html=True,
    )

    total_customers = len(result)

    predicted_churn = int(
        (
            result["_model_prediction"]
            == 1
        ).sum()
    )

    critical_customers = int(
        (
            result["_risk_score"]
            > 80
        ).sum()
    )

    average_risk = float(
        result["_risk_score"].mean()
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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
            "Risk score above 80",
        ),

        (
            "AVERAGE RISK",
            f"{average_risk:.1f}%",
            "Average predicted churn risk",
        ),
    ]

    columns = st.columns(4)

    for col, item in zip(
        columns,
        metric_data,
    ):

        with col:

            st.markdown(
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
""",
                unsafe_allow_html=True,
            )

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.markdown(
        '<div class="section-title">🤖 Model Performance</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%",
    )

    m2.metric(
        "Precision",
        f"{metrics['precision'] * 100:.2f}%",
    )

    m3.metric(
        "Recall",
        f"{metrics['recall'] * 100:.2f}%",
    )

    m4.metric(
        "F1 Score",
        f"{metrics['f1'] * 100:.2f}%",
    )

    m5.metric(
        "ROC AUC",
        f"{metrics['roc_auc']:.4f}",
    )

    st.caption(
        f"Decision threshold optimized on validation data: "
        f"{metrics['decision_threshold']:.2f} | "
        f"Cross-validation ROC-AUC: "
        f"{metrics['cv_roc_auc']:.4f}"
    )

    # ========================================================
    # CRITICAL CUSTOMERS
    # ========================================================

    st.markdown(
        '<div class="section-title">🚨 Critical Risk Customers</div>',
        unsafe_allow_html=True,
    )

    critical = (
        result[
            result["_risk_score"] > 80
        ]
        .sort_values(
            "_risk_score",
            ascending=False,
        )
        .copy()
    )

    cc1, cc2 = st.columns([1, 3])

    with cc1:

        st.markdown(
            f"""
<div class="critical-summary">

<div class="critical-summary-number">
{len(critical):,}
</div>

<div class="critical-summary-text">
Customers with a churn risk score above 80.
</div>

</div>
""",
            unsafe_allow_html=True,
        )

    with cc2:

        if critical.empty:

            st.success(
                "No critical-risk customers were detected."
            )

        else:

            for _, row in critical.head(5).iterrows():

                render_critical_customer_card(
                    row,
                    customer_id_col,
                    customer_name_col,
                )

    # ========================================================
    # RISK DISTRIBUTION
    # ========================================================

    st.markdown(
        '<div class="section-title">🎯 Customer Risk Distribution</div>',
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

    # ========================================================
    # ATTENTION TABLE
    # ========================================================

    st.markdown(
        '<div class="section-title">🚨 Customers Requiring Attention</div>',
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

    if not high_risk.empty:

        display = pd.DataFrame()

        if customer_id_col:

            display["Customer ID"] = (
                high_risk[
                    customer_id_col
                ]
                .astype(str)
            )

        if customer_name_col:

            display["Customer Name"] = (
                high_risk[
                    customer_name_col
                ]
                .astype("string")
                .fillna("")
            )

        display["Prediction"] = (
            high_risk[
                "_prediction_label"
            ]
        )

        display["Risk Score"] = (
            high_risk[
                "_risk_score"
            ]
            .round(2)
        )

        display["Risk Level"] = (
            high_risk[
                "_risk_level"
            ]
        )

        st.dataframe(
            make_arrow_safe(display),
            width="stretch",
            hide_index=True,
        )

    # ========================================================
    # EXPORT
    # ========================================================

    st.markdown(
        '<div class="section-title">📥 Export Predictions</div>',
        unsafe_allow_html=True,
    )

    st.download_button(
        "Download Customer Risk Predictions CSV",
        data=result.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="customer_churn_predictions.csv",
        mime="text/csv",
    )


# ============================================================
# CUSTOMER RISK
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

    st.markdown(
        """
<div class="dashboard-title">
Customer Risk Analysis
</div>

<div class="dashboard-subtitle">
Select a real customer from the uploaded dataset and inspect
the model prediction.
</div>
""",
        unsafe_allow_html=True,
    )

    if customer_id_col:

        ids = (
            result[
                customer_id_col
            ]
            .astype(str)
            .tolist()
        )

        labels = {
            str(row[customer_id_col]):
            get_customer_identity(
                row,
                customer_id_col,
                customer_name_col,
            )

            for _, row
            in result.iterrows()
        }

        selected_id = st.selectbox(
            "Select Customer",
            ids,
            format_func=lambda x:
                labels.get(x, x),
        )

        customer_index = (
            result[
                result[
                    customer_id_col
                ]
                .astype(str)
                == selected_id
            ]
            .index[0]
        )

    else:

        ids = [
            str(i)
            for i in result.index
        ]

        selected_id = st.selectbox(
            "Select Customer",
            ids,
        )

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

    identity = get_customer_identity(
        customer,
        customer_id_col,
        customer_name_col,
    )

    audit_section(
        f"CUSTOMER RISK DETAILS FOR ID {selected_id}"
    )

    audit_log(
        f"Identity: {identity}"
    )

    audit_log(
        f"Churn Probability: "
        f"{probability * 100:.2f}%"
    )

    audit_log(
        f"Risk Score: "
        f"{risk_score:.2f}/100"
    )

    audit_log(
        f"Prediction: "
        f"{customer['_prediction_label']}"
    )

    st.markdown(
        f"""
<div class="customer-card">

<div class="customer-id">
👤 {identity}
</div>

<div class="customer-status">
Customer ID: {selected_id}
&nbsp;|&nbsp;
Prediction: {customer["_prediction_label"]}
&nbsp;|&nbsp;
Risk: {customer["_risk_level"]}
</div>

</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%",
    )

    c2.metric(
        "Risk Score",
        f"{risk_score:.2f}/100",
    )

    c3.metric(
        "Prediction",
        customer[
            "_prediction_label"
        ],
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={
                "text":
                "Customer Churn Risk Score"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                }
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

    # ========================================================
    # CUSTOMER INFORMATION
    # ========================================================

    st.markdown(
        '<div class="section-title">📋 Customer Information</div>',
        unsafe_allow_html=True,
    )

    info_data = []

    for col in feature_columns:

        info_data.append(
            {
                "Feature": col,
                "Value": customer[col],
            }
        )

    info_df = pd.DataFrame(
        info_data
    )

    audit_log(
        "Customer Features:"
    )

    print(
        info_df.to_string(
            index=False
        ),
        flush=True,
    )

    st.dataframe(
        make_arrow_safe(info_df),
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

    customer_name_col = (
        st.session_state.customer_name_column
    )

    feature_columns = (
        st.session_state.feature_names
    )

    st.markdown(
        """
<div class="dashboard-title">
SHAP Explainability
</div>

<div class="dashboard-subtitle">
Understand which actual customer features are driving the
churn prediction.
</div>
""",
        unsafe_allow_html=True,
    )

    if not SHAP_AVAILABLE:

        st.error(
            "SHAP is not installed. Run: pip install shap"
        )

        return

    if customer_id_col:

        ids = (
            result[
                customer_id_col
            ]
            .astype(str)
            .tolist()
        )

        labels = {
            str(row[customer_id_col]):
            get_customer_identity(
                row,
                customer_id_col,
                customer_name_col,
            )

            for _, row
            in result.iterrows()
        }

        selected_id = st.selectbox(
            "Select Customer for SHAP Explanation",
            ids,
            format_func=lambda x:
                labels.get(x, x),
            key="shap_customer_selector",
        )

        customer_index = (
            result[
                result[
                    customer_id_col
                ]
                .astype(str)
                == selected_id
            ]
            .index[0]
        )

    else:

        ids = [
            str(i)
            for i in result.index
        ]

        selected_id = st.selectbox(
            "Select Customer for SHAP Explanation",
            ids,
            key="shap_customer_selector",
        )

        customer_index = int(
            selected_id
        )

    shap_df = get_shap_explanation(
        pipeline,
        result,
        customer_index,
        feature_columns,
    )

    increasing_df = (
        shap_df[
            shap_df[
                "SHAP Value"
            ] > 0
        ][
            [
                "Feature",
                "SHAP Value",
            ]
        ]
        .sort_values(
            "SHAP Value",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    reducing_df = (
        shap_df[
            shap_df[
                "SHAP Value"
            ] < 0
        ][
            [
                "Feature",
                "SHAP Value",
            ]
        ]
        .sort_values(
            "SHAP Value",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    audit_section(
        f"SHAP VALUES FOR CUSTOMER ID: {selected_id}"
    )

    audit_log(
        "Top Increasing Churn Risk Features:"
    )

    print(
        increasing_df.to_string(
            index=False
        ),
        flush=True,
    )

    audit_log(
        "Top Reducing Churn Risk Features:"
    )

    print(
        reducing_df.to_string(
            index=False
        ),
        flush=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
<div style="
font-size:18px;
font-weight:700;
color:#f8fafc;
margin-bottom:10px;">
🔴 Increasing Churn Risk
</div>
""",
            unsafe_allow_html=True,
        )

        if increasing_df.empty:

            st.info(
                "No features are increasing churn risk."
            )

        else:

            st.dataframe(
                make_arrow_safe(
                    increasing_df
                ),
                width="stretch",
                hide_index=True,
            )

    with col2:

        st.markdown(
            """
<div style="
font-size:18px;
font-weight:700;
color:#f8fafc;
margin-bottom:10px;">
🟢 Reducing Churn Risk
</div>
""",
            unsafe_allow_html=True,
        )

        if reducing_df.empty:

            st.info(
                "No features are reducing churn risk."
            )

        else:

            st.dataframe(
                make_arrow_safe(
                    reducing_df
                ),
                width="stretch",
                hide_index=True,
            )

    st.markdown(
        '<div class="section-title">📊 Detailed Feature Contributions</div>',
        unsafe_allow_html=True,
    )

    detailed_df = shap_df[
        [
            "Feature",
            "SHAP Value",
            "Impact",
        ]
    ].copy()

    st.dataframe(
        make_arrow_safe(
            detailed_df
        ),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# BUSINESS IMPACT
# ============================================================

def business_impact_simulator(
    predictions
):

    customer_id_col = (
        st.session_state.customer_id_column
    )

    customer_name_col = (
        st.session_state.customer_name_column
    )

    st.markdown(
        """
<div class="dashboard-title">
💼 Business Impact Simulator
</div>

<div class="dashboard-subtitle">
Convert real churn predictions into a practical retention
strategy and estimate financial impact.
</div>
""",
        unsafe_allow_html=True,
    )

    high_risk = (
        predictions[
            predictions[
                "_risk_score"
            ] > 60
        ]
        .sort_values(
            "_model_churn_probability",
            ascending=False,
        )
        .copy()
    )

    high_risk_count = len(
        high_risk
    )

    st.markdown(
        "### 🎯 Retention Opportunity"
    )

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "High-Risk Customers",
        f"{high_risk_count:,}",
    )

    p2.metric(
        "High Risk",
        f"{int(((predictions['_risk_score'] > 60) & (predictions['_risk_score'] <= 80)).sum()):,}",
    )

    p3.metric(
        "Critical Risk",
        f"{int((predictions['_risk_score'] > 80).sum()):,}",
    )

    # ========================================================
    # RETENTION LIST
    # ========================================================

    st.markdown(
        '<div class="section-title">🎯 Retention Customer List</div>',
        unsafe_allow_html=True,
    )

    retention_df = pd.DataFrame()

    if (
        customer_id_col
        and customer_id_col
        in predictions.columns
    ):

        retention_df[
            "Customer ID"
        ] = (
            predictions[
                customer_id_col
            ]
            .astype(str)
        )

    else:

        retention_df[
            "Customer ID"
        ] = predictions.index.astype(
            str
        )

    if (
        customer_name_col
        and customer_name_col
        in predictions.columns
    ):

        retention_df[
            "Customer Name"
        ] = (
            predictions[
                customer_name_col
            ]
            .astype("string")
            .fillna("")
        )

    retention_df[
        "Churn Probability (%)"
    ] = (
        predictions[
            "_model_churn_probability"
        ]
        * 100
    ).round(2)

    retention_df[
        "Risk Score"
    ] = (
        predictions[
            "_risk_score"
        ]
        .round(2)
    )

    retention_df[
        "Risk Level"
    ] = predictions[
        "_risk_level"
    ]

    retention_df[
        "Prediction"
    ] = predictions[
        "_prediction_label"
    ]

    retention_df = (
        retention_df
        .sort_values(
            "Risk Score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    audit_section(
        "RETENTION LIST"
    )

    print(
        retention_df.head(10).to_string(
            index=False
        ),
        flush=True,
    )

    st.dataframe(
        make_arrow_safe(
            retention_df
        ),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Customer identity shown above comes directly "
        "from the uploaded dataset. No customer names "
        "are generated or modified by the dashboard."
    )

    if high_risk_count == 0:

        return

    st.divider()

    st.markdown(
        "### ⚙️ Campaign Assumptions"
    )

    c1, c2 = st.columns(2)

    with c1:

        capacity = st.number_input(
            "📞 Contact capacity",
            min_value=1,
            max_value=high_risk_count,
            value=min(
                2000,
                high_risk_count,
            ),
        )

        save_rate_input = st.slider(
            "🎯 Expected retention success rate",
            1,
            100,
            15,
            1,
            format="%d%%",
        )

    with c2:

        customer_value = st.number_input(
            "💰 Average customer lifetime value ($)",
            min_value=0.0,
            value=1000.0,
        )

        campaign_cost = st.number_input(
            "📣 Campaign cost per customer ($)",
            min_value=0.0,
            value=20.0,
        )

    contacted = min(
        int(capacity),
        high_risk_count,
    )

    revenue_saved = (
        contacted
        * (save_rate_input / 100)
        * customer_value
    )

    total_cost = (
        contacted
        * campaign_cost
    )

    net_impact = (
        revenue_saved
        - total_cost
    )

    st.markdown(
        "### 📊 Estimated Business Impact"
    )

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Customers Contacted",
        f"{contacted:,}",
    )

    r2.metric(
        "Expected Saved",
        f"{contacted * (save_rate_input / 100):,.0f}",
    )

    r3.metric(
        "Revenue Saved",
        f"${revenue_saved:,.0f}",
    )

    r4.metric(
        "Campaign Cost",
        f"${total_cost:,.0f}",
    )

    if net_impact >= 0:

        st.markdown(
            f"""
<div class="roi-positive">

<div style="
font-size:1.35rem;
font-weight:750;
color:#166534;">

📈 Positive Business Impact

</div>

<div style="
font-size:2.2rem;
font-weight:850;
color:#111827;">

${net_impact:,.0f}

</div>

<p style="color:#64748b;">
Estimated net value after campaign costs.
</p>

</div>
""",
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
<div class="roi-negative">

<div style="
font-size:1.35rem;
font-weight:750;
color:#b91c1c;">

⚠️ Negative Business Impact

</div>

<div style="
font-size:2.2rem;
font-weight:850;
color:#111827;">

${net_impact:,.0f}

</div>

<p style="color:#64748b;">
Campaign cost exceeds expected revenue saved.
</p>

</div>
""",
            unsafe_allow_html=True,
        )


def show_business_impact():

    business_impact_simulator(
        st.session_state.predictions
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # ========================================================
    # UPLOAD / TRAINING
    # ========================================================

    if not st.session_state.trained:

        uploaded_file = (
            show_upload_page()
        )

        if uploaded_file is None:

            st.markdown(
                """
<div class="footer">
Upload a CSV dataset to begin the customer churn ML workflow.
</div>
""",
                unsafe_allow_html=True,
            )

            return

        file_bytes = (
            uploaded_file.getvalue()
        )

        current_hash = (
            calculate_file_hash(
                file_bytes
            )
        )

        if (
            st.session_state.dataset_hash
            != current_hash
        ):

            st.session_state.trained = False

            try:

                # ------------------------------------------------
                # IMPORTANT:
                # CSV IS READ BY ingestion.py
                # ------------------------------------------------

                ingestion = get_ingestion()

                uploaded_file.seek(0)

                uploaded_df = (
                    ingestion.load_uploaded_data(
                        uploaded_file
                    )
                )

                audit_log(
                    f"Uploaded file: "
                    f"{uploaded_file.name}"
                )

                audit_log(
                    f"Uploaded shape: "
                    f"{uploaded_df.shape}"
                )

                # ------------------------------------------------
                # TRAIN
                # ------------------------------------------------

                with st.spinner(
                    "Training and optimizing XGBoost model... "
                    "This may take a few minutes."
                ):

                    result = (
                        train_uploaded_dataset(
                            uploaded_df
                        )
                    )

                # ------------------------------------------------
                # SAVE STATE
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

                st.session_state.dataset_hash = (
                    current_hash
                )

                st.session_state.uploaded_filename = (
                    uploaded_file.name
                )

                st.session_state.decision_threshold = (
                    result[
                        "metrics"
                    ][
                        "decision_threshold"
                    ]
                )

                st.session_state.trained = True

                st.success(
                    "Model training completed successfully."
                )

                st.rerun()

            except Exception as e:

                st.error(
                    "The machine-learning pipeline "
                    "could not process this dataset."
                )

                st.exception(e)

                return

    # ========================================================
    # DASHBOARD
    # ========================================================

    if st.session_state.trained:

        navigation = (
            show_top_navigation()
        )

        col1, col2 = st.columns(
            [6, 1]
        )

        with col1:

            filename = (
                st.session_state.uploaded_filename
                or "Uploaded dataset"
            )

            st.success(
                f"✓ Active dataset: **{filename}**"
            )

        with col2:

            if st.button(
                "🔄 New Dataset",
                width="stretch",
            ):

                st.session_state.clear()

                st.rerun()

        # ----------------------------------------------------
        # NAVIGATION
        # ----------------------------------------------------

        if navigation == "🏠 Overview":

            show_overview()

        elif navigation == "👤 Customer Risk":

            show_customer_risk()

        elif navigation == "🧠 SHAP Explainability":

            show_shap()

        elif navigation == "💰 Business Impact Simulator":

            show_business_impact()

        # ----------------------------------------------------
        # FOOTER
        # ----------------------------------------------------

        st.markdown(
            """
<div class="footer">

Customer Churn Intelligence

<br><br>

Uploaded Dataset
→
Data Ingestion
→
Preprocessing
→
Hyperparameter Tuning
→
XGBoost
→
Prediction
→
SHAP
→
Business Impact

</div>
""",
            unsafe_allow_html=True,
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
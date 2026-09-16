from __future__ import annotations

import sys
import warnings
import hashlib
from pathlib import Path
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ChurnIQ — Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cell2celltrain.csv"

# Your existing model
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"

TARGET = "churn"
ID_COL = "customerid"
NAME_COL = "customer_name"


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Main page ---------- */

    .main {
        background: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: #f9fafb;
    }

    /* ---------- Header ---------- */

    .dashboard-title {
        font-size: 2.25rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.15rem;
    }

    .dashboard-subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    /* ---------- Section title ---------- */

    .section-title {
        font-size: 1.45rem;
        font-weight: 750;
        color: #111827;
        margin-top: 1.25rem;
        margin-bottom: 0.75rem;
    }

    /* ---------- Source badge ---------- */

    .source-badge {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    /* ---------- Info box ---------- */

    .info-box {
        padding: 1rem 1.15rem;
        border-radius: 12px;
        border: 1px solid #dbe4f0;
        background: #ffffff;
        margin-bottom: 1rem;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        padding: 2rem 0 0.5rem 0;
    }

    /* ---------- Tables ---------- */

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------- Metrics ---------- */

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.8rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SILENT MODEL OUTPUT
# ============================================================

@contextmanager
def suppress_model_output():
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        yield


# ============================================================
# COLUMN NAME HELPERS
# ============================================================

def canonicalize_column_name(name: object) -> str:
    """
    Converts a column name into a comparison key.

    Examples:

        Monthly Revenue
        monthly_revenue
        monthly-revenue
        MONTHLYREVENUE

    All become:

        monthlyrevenue
    """

    text = str(name).strip().lower()

    return "".join(
        character
        for character in text
        if character.isalnum()
    )


STRUCTURAL_ALIASES = {
    ID_COL: {
        "customerid",
        "customer_id",
        "customer id",
        "customer_no",
        "customer number",
        "customer_number",
        "id",
    },

    TARGET: {
        "churn",
        "churnflag",
        "churn_flag",
        "churned",
        "churn_status",
        "churn status",
    },

    NAME_COL: {
        "name",
        "customername",
        "customer_name",
        "customer name",
        "fullname",
        "full_name",
        "full name",
        "clientname",
        "client_name",
        "client name",
    },
}


def find_structural_column(
    df: pd.DataFrame,
    standard_name: str,
):
    """
    Finds ID/name/target columns using flexible matching.
    """

    aliases = STRUCTURAL_ALIASES.get(
        standard_name,
        set(),
    )

    alias_keys = {
        canonicalize_column_name(alias)
        for alias in aliases
    }

    for column in df.columns:

        if canonicalize_column_name(column) in alias_keys:
            return column

    return None


def build_canonical_column_map(df: pd.DataFrame):
    """
    Creates:

        canonical name -> actual source column
    """

    column_map = {}
    duplicates = {}

    for column in df.columns:

        canonical = canonicalize_column_name(column)

        if not canonical:
            continue

        if canonical in column_map:

            duplicates.setdefault(
                canonical,
                [column_map[canonical]],
            )

            duplicates[canonical].append(column)

        else:

            column_map[canonical] = column

    return column_map, duplicates


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    try:

        with suppress_model_output():

            model = joblib.load(MODEL_PATH)

        return model

    except Exception:
        return None


model = load_model()


# ============================================================
# ORIGINAL DATASET
# ============================================================

@st.cache_data
def load_original_dataset():

    if not DATA_PATH.exists():
        return None

    try:

        df = pd.read_csv(DATA_PATH)

        return df

    except Exception:
        return None


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_customer_dataframe(
    df: pd.DataFrame,
    require_target: bool = False,
):
    """
    Prepares a dataset without destroying the user's
    original feature names.

    Customer names are NEVER generated.
    """

    df = df.copy()

    # Clean only header whitespace.
    # Actual customer values remain unchanged.
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # --------------------------------------------------------
    # CUSTOMER ID
    # --------------------------------------------------------

    id_source = find_structural_column(
        df,
        ID_COL,
    )

    if id_source is None:

        raise ValueError(
            "No Customer ID column was found.\n\n"
            "Expected something like:\n"
            "CustomerID\n"
            "Customer ID\n"
            "customer_id"
        )

    if id_source != ID_COL:

        df[ID_COL] = df[id_source]

        df.drop(
            columns=[id_source],
            inplace=True,
        )

    # Preserve ID values as strings
    df[ID_COL] = df[ID_COL].astype(str)

    # --------------------------------------------------------
    # CUSTOMER NAME
    # --------------------------------------------------------

    name_source = find_structural_column(
        df,
        NAME_COL,
    )

    has_names = False

    if name_source is not None:

        if name_source != NAME_COL:

            df[NAME_COL] = df[name_source]

            df.drop(
                columns=[name_source],
                inplace=True,
            )

        has_names = (
            df[NAME_COL]
            .notna()
            .any()
        )

    # --------------------------------------------------------
    # CHURN TARGET
    # --------------------------------------------------------

    target_source = find_structural_column(
        df,
        TARGET,
    )

    if target_source is not None:

        if target_source != TARGET:

            df[TARGET] = df[target_source]

            df.drop(
                columns=[target_source],
                inplace=True,
            )

    elif require_target:

        raise ValueError(
            "The uploaded dataset does not contain a Churn column."
        )

    return df, has_names


# ============================================================
# READ UPLOADED CSV
# ============================================================

def read_uploaded_csv(uploaded_file):

    """
    IMPORTANT:

    Read the uploaded CSV directly.

    We intentionally do NOT use the project's
    DataIngestion loader here because we want to
    preserve the actual uploaded feature names.
    """

    uploaded_file.seek(0)

    try:

        df = pd.read_csv(
            uploaded_file,
            low_memory=False,
        )

    except UnicodeDecodeError:

        uploaded_file.seek(0)

        df = pd.read_csv(
            uploaded_file,
            encoding="latin1",
            low_memory=False,
        )

    if df.empty:

        raise ValueError(
            "The uploaded CSV is empty."
        )

    return df


# ============================================================
# DATASET FEATURE LIST
# ============================================================

def get_dataset_feature_columns(
    df: pd.DataFrame,
):

    structural = {
        ID_COL,
        NAME_COL,
        TARGET,
    }

    return [
        column
        for column in df.columns
        if column not in structural
    ]


def build_feature_schema(
    df: pd.DataFrame,
):

    feature_columns = get_dataset_feature_columns(df)

    rows = []

    for column in feature_columns:

        rows.append(
            {
                "Feature Name": column,
                "Data Type": str(df[column].dtype),
                "Non-Null Values": int(
                    df[column].notna().sum()
                ),
                "Missing Values": int(
                    df[column].isna().sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MODEL FEATURE DISCOVERY
# ============================================================

def get_pipeline_steps(model):

    if model is None:
        return []

    if hasattr(model, "named_steps"):

        return list(
            model.named_steps.items()
        )

    return []


def get_preprocessor(model):

    if model is None:
        return None

    steps = get_pipeline_steps(model)

    for _, step in steps:

        if hasattr(
            step,
            "transformers_",
        ) and hasattr(
            step,
            "get_feature_names_out",
        ):

            return step

    for _, step in steps:

        if hasattr(
            step,
            "get_feature_names_out",
        ) and hasattr(
            step,
            "transform",
        ):

            return step

    return None


def get_final_estimator(model):

    if model is None:
        return None

    if hasattr(
        model,
        "named_steps",
    ):

        steps = list(
            model.named_steps.values()
        )

        if steps:
            return steps[-1]

    return model


def get_model_input_columns(
    model,
    reference_df=None,
):

    if model is None:

        return []

    candidates = []

    # --------------------------------------------------------
    # Pipeline feature_names_in_
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_names_in_",
    ):

        candidates.extend(
            list(
                model.feature_names_in_
            )
        )

    # --------------------------------------------------------
    # Search pipeline steps
    # --------------------------------------------------------

    if not candidates:

        for _, step in get_pipeline_steps(model):

            if hasattr(
                step,
                "feature_names_in_",
            ):

                candidates.extend(
                    list(
                        step.feature_names_in_
                    )
                )

                break

    # --------------------------------------------------------
    # Remove structural columns
    # --------------------------------------------------------

    result = []

    for column in candidates:

        canonical = canonicalize_column_name(
            column
        )

        if canonical in {
            canonicalize_column_name(ID_COL),
            canonicalize_column_name(NAME_COL),
            canonicalize_column_name(TARGET),
        }:

            continue

        if column not in result:

            result.append(column)

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if not result and reference_df is not None:

        result = get_dataset_feature_columns(
            reference_df
        )

    return result


# ============================================================
# MODEL FEATURE MATCHING
# ============================================================

def build_model_feature_mapping(
    uploaded_df: pd.DataFrame,
    model_input_columns,
):
    """
    Matches model feature names against uploaded columns.

    This is ONLY used for determining whether a dataset
    is compatible with the existing model.

    It does NOT rename the user's visible columns.
    """

    column_map, duplicates = (
        build_canonical_column_map(
            uploaded_df
        )
    )

    feature_mapping = {}

    missing = []

    ambiguous = []

    for model_column in model_input_columns:

        model_key = canonicalize_column_name(
            model_column
        )

        if model_key not in column_map:

            missing.append(
                model_column
            )

            continue

        actual_column = column_map[
            model_key
        ]

        duplicate_columns = duplicates.get(
            model_key
        )

        if duplicate_columns:

            ambiguous.append(
                f"{model_column}: "
                f"{duplicate_columns}"
            )

            continue

        feature_mapping[
            model_column
        ] = actual_column

    return (
        feature_mapping,
        missing,
        ambiguous,
    )


# ============================================================
# MODEL INPUT CREATION
# ============================================================

def create_model_input_dataframe(
    customers: pd.DataFrame,
    model_input_columns,
    feature_mapping=None,
):
    """
    Creates exactly the dataframe expected by the model.

    Example:

        Uploaded:
            Monthly Revenue

        Model:
            monthlyrevenue

        Model input:
            monthlyrevenue
    """

    X = pd.DataFrame(
        index=customers.index
    )

    for model_column in model_input_columns:

        source_column = (
            feature_mapping.get(
                model_column
            )
            if feature_mapping
            else model_column
        )

        if source_column is None:

            raise ValueError(
                f"Could not map model feature "
                f"'{model_column}'."
            )

        if source_column not in customers.columns:

            raise ValueError(
                f"Uploaded dataset does not contain "
                f"'{source_column}'."
            )

        X[model_column] = customers[
            source_column
        ]

    return X


# ============================================================
# PREDICTION
# ============================================================

def silent_predict_proba(
    model,
    X,
):

    with suppress_model_output():

        probabilities = model.predict_proba(X)

    probabilities = np.asarray(
        probabilities
    )

    if probabilities.ndim == 2:

        probabilities = probabilities[:, -1]

    return probabilities.astype(float)


def get_risk_level(score):

    if score <= 30:
        return "LOW"

    if score <= 60:
        return "MEDIUM"

    if score <= 80:
        return "HIGH"

    return "CRITICAL"


def recommendation(risk):

    if risk == "CRITICAL":

        return "Immediate retention action"

    if risk == "HIGH":

        return "Prioritize retention outreach"

    if risk == "MEDIUM":

        return "Monitor customer"

    return "Maintain engagement"


def create_predictions(
    customers,
    model,
    model_input_columns,
    feature_mapping=None,
):

    X = create_model_input_dataframe(
        customers,
        model_input_columns,
        feature_mapping,
    )

    probabilities = silent_predict_proba(
        model,
        X,
    )

    result = pd.DataFrame(
        index=customers.index
    )

    result["Customer ID"] = (
        customers[ID_COL]
        .astype(str)
    )

    if NAME_COL in customers.columns:

        result["Customer Name"] = (
            customers[NAME_COL]
        )

    result["Churn Probability"] = (
        probabilities
    )

    result["Risk Score"] = (
        probabilities * 100
    )

    result["Risk Level"] = (
        result["Risk Score"]
        .apply(get_risk_level)
    )

    result["Prediction"] = np.where(
        probabilities >= 0.5,
        "Churn",
        "No Churn",
    )

    result["Recommendation"] = (
        result["Risk Level"]
        .apply(recommendation)
    )

    return result


# ============================================================
# SAFE DATAFRAME DISPLAY
# ============================================================

def make_arrow_safe(df):

    result = df.copy()

    for column in result.columns:

        if (
            result[column].dtype == "object"
            or pd.api.types.is_string_dtype(
                result[column]
            )
        ):

            result[column] = (
                result[column]
                .where(
                    result[column].notna(),
                    "",
                )
                .astype(str)
            )

    return result


# ============================================================
# CUSTOMER TABLE
# ============================================================

def build_customer_table(
    predictions,
    top_n=10,
):

    table = predictions.copy()

    table = table.sort_values(
        "Risk Score",
        ascending=False,
    ).head(top_n)

    columns = [
        "Customer ID",
    ]

    if "Customer Name" in table.columns:

        columns.append(
            "Customer Name"
        )

    columns.extend(
        [
            "Churn Probability",
            "Risk Score",
            "Risk Level",
        ]
    )

    return make_arrow_safe(
        table[columns]
    )


# ============================================================
# DATASET CHURN ANALYSIS
# ============================================================

def normalize_churn_value(value):

    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if text in {
        "yes",
        "y",
        "1",
        "true",
        "churn",
        "churned",
    }:

        return 1

    if text in {
        "no",
        "n",
        "0",
        "false",
        "not churn",
        "no churn",
        "not_churn",
    }:

        return 0

    return None


def calculate_dataset_churn_rate(
    df,
):

    if TARGET not in df.columns:

        return None

    normalized = (
        df[TARGET]
        .apply(normalize_churn_value)
    )

    valid = normalized.dropna()

    if len(valid) == 0:

        return None

    return float(
        valid.mean() * 100
    )


def get_churn_distribution(
    df,
):

    if TARGET not in df.columns:

        return None

    counts = (
        df[TARGET]
        .fillna("Missing")
        .astype(str)
        .value_counts()
    )

    return counts


# ============================================================
# DATASET RISK SCORE
# ============================================================

def calculate_dataset_risk_scores(
    customers,
):
    """
    Dataset-only fallback.

    IMPORTANT:
    This does NOT pretend to be the ML model.

    If the uploaded dataset has a Churn column,
    actual churn labels are used only for analytics.

    We do NOT call these model probabilities.
    """

    if TARGET not in customers.columns:

        return None

    normalized = (
        customers[TARGET]
        .apply(normalize_churn_value)
    )

    if normalized.notna().sum() == 0:

        return None

    result = pd.DataFrame(
        index=customers.index
    )

    result["Customer ID"] = (
        customers[ID_COL]
        .astype(str)
    )

    if NAME_COL in customers.columns:

        result["Customer Name"] = (
            customers[NAME_COL]
        )

    result["Actual Churn"] = (
        customers[TARGET]
    )

    result["Risk Score"] = (
        normalized.fillna(0) * 100
    )

    result["Risk Level"] = (
        result["Risk Score"]
        .apply(get_risk_level)
    )

    return result


# ============================================================
# CUSTOMER PROFILE
# ============================================================

def build_customer_profile(
    customer_row,
    feature_columns,
):

    rows = []

    for feature in feature_columns:

        value = customer_row.get(
            feature
        )

        rows.append(
            {
                "Feature": feature,
                "Value": value,
            }
        )

    return make_arrow_safe(
        pd.DataFrame(rows)
    )


# ============================================================
# BUSINESS IMPACT
# ============================================================

def calculate_business_impact(
    predictions,
    retention_cost,
    customer_value,
):

    critical_high = predictions[
        predictions["Risk Score"] > 60
    ]

    count = len(
        critical_high
    )

    potential_value = (
        count * customer_value
    )

    retention_cost_total = (
        count * retention_cost
    )

    potential_net = (
        potential_value
        - retention_cost_total
    )

    return {
        "priority_customers": count,
        "potential_value": potential_value,
        "retention_cost": retention_cost_total,
        "potential_net": potential_net,
    }


# ============================================================
# MODEL COMPATIBILITY CHECK
# ============================================================

def check_model_compatibility(
    customers,
    model,
    model_input_columns,
):

    if model is None:

        return {
            "compatible": False,
            "reason": "Saved model was not found.",
            "mapping": {},
            "missing": [],
            "ambiguous": [],
        }

    if not model_input_columns:

        return {
            "compatible": False,
            "reason": "Could not determine model feature schema.",
            "mapping": {},
            "missing": [],
            "ambiguous": [],
        }

    (
        mapping,
        missing,
        ambiguous,
    ) = build_model_feature_mapping(
        customers,
        model_input_columns,
    )

    if ambiguous:

        return {
            "compatible": False,
            "reason": "Some model features match multiple uploaded columns.",
            "mapping": mapping,
            "missing": missing,
            "ambiguous": ambiguous,
        }

    if missing:

        return {
            "compatible": False,
            "reason": "Uploaded dataset uses a different feature schema.",
            "mapping": mapping,
            "missing": missing,
            "ambiguous": [],
        }

    return {
        "compatible": True,
        "reason": "All model features matched.",
        "mapping": mapping,
        "missing": [],
        "ambiguous": [],
    }


# ============================================================
# SESSION STATE
# ============================================================

if "uploaded_customers" not in st.session_state:
    st.session_state.uploaded_customers = None

if "uploaded_has_names" not in st.session_state:
    st.session_state.uploaded_has_names = False

if "uploaded_file_hash" not in st.session_state:
    st.session_state.uploaded_file_hash = None

if "uploaded_feature_mapping" not in st.session_state:
    st.session_state.uploaded_feature_mapping = None

if "uploaded_model_compatible" not in st.session_state:
    st.session_state.uploaded_model_compatible = False

if "uploaded_model_missing" not in st.session_state:
    st.session_state.uploaded_model_missing = []

if "uploaded_model_ambiguous" not in st.session_state:
    st.session_state.uploaded_model_ambiguous = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 📊 ChurnIQ"
    )

    st.caption(
        "Customer Churn Intelligence Dashboard"
    )

    st.divider()

    st.markdown(
        "### 📁 Upload Customer Dataset"
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help=(
            "Upload your customer dataset. "
            "When a file is uploaded, the dashboard "
            "uses ONLY that uploaded dataset."
        ),
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.md5(
            file_bytes
        ).hexdigest()

        # Process only if this is a new file
        if (
            st.session_state.uploaded_file_hash
            != file_hash
        ):

            try:

                raw_uploaded_df = (
                    read_uploaded_csv(
                        uploaded_file
                    )
                )

                prepared_uploaded_df, has_names = (
                    prepare_customer_dataframe(
                        raw_uploaded_df,
                        require_target=False,
                    )
                )

                # ------------------------------------------------
                # Determine model compatibility.
                #
                # IMPORTANT:
                # This does NOT replace the uploaded dataset.
                # It only checks whether the existing model can
                # legitimately operate on it.
                # ------------------------------------------------

                model_reference_df = (
                    load_original_dataset()
                )

                model_input_columns = (
                    get_model_input_columns(
                        model,
                        model_reference_df,
                    )
                )

                compatibility = (
                    check_model_compatibility(
                        prepared_uploaded_df,
                        model,
                        model_input_columns,
                    )
                )

                st.session_state.uploaded_customers = (
                    prepared_uploaded_df
                )

                st.session_state.uploaded_has_names = (
                    has_names
                )

                st.session_state.uploaded_file_hash = (
                    file_hash
                )

                st.session_state.uploaded_feature_mapping = (
                    compatibility["mapping"]
                )

                st.session_state.uploaded_model_compatible = (
                    compatibility["compatible"]
                )

                st.session_state.uploaded_model_missing = (
                    compatibility["missing"]
                )

                st.session_state.uploaded_model_ambiguous = (
                    compatibility["ambiguous"]
                )

                st.rerun()

            except Exception as error:

                st.session_state.uploaded_customers = None
                st.session_state.uploaded_has_names = False
                st.session_state.uploaded_file_hash = None
                st.session_state.uploaded_feature_mapping = None
                st.session_state.uploaded_model_compatible = False

                st.error(
                    f"Could not load uploaded dataset:\n\n{error}"
                )

    # ------------------------------------------------------------
    # Dataset source
    # ------------------------------------------------------------

    st.divider()

    if (
        uploaded_file is not None
        and st.session_state.uploaded_customers is not None
    ):

        st.success(
            "Uploaded dataset is active"
        )

        st.caption(
            "The dashboard is using ONLY the uploaded CSV."
        )

    else:

        st.info(
            "No CSV uploaded. "
            "The original project dataset is active."
        )

    # ------------------------------------------------------------
    # Model status
    # ------------------------------------------------------------

    st.divider()

    st.markdown(
        "### 🤖 Model Status"
    )

    if model is not None:

        st.success(
            "Saved model loaded"
        )

    else:

        st.warning(
            "Saved model not available"
        )


# ============================================================
# ACTIVE DATASET
# ============================================================

if (
    uploaded_file is not None
    and st.session_state.uploaded_customers is not None
):

    active_customers = (
        st.session_state.uploaded_customers
    )

    active_source = "Uploaded Dataset"

    active_mapping = (
        st.session_state.uploaded_feature_mapping
    )

    model_compatible = (
        st.session_state.uploaded_model_compatible
    )

else:

    original_df = (
        load_original_dataset()
    )

    if original_df is None:

        st.error(
            "No dataset is available."
        )

        st.stop()

    try:

        active_customers, _ = (
            prepare_customer_dataframe(
                original_df,
                require_target=False,
            )
        )

    except Exception as error:

        st.error(
            f"Could not prepare original dataset: {error}"
        )

        st.stop()

    active_source = "Original Project Dataset"

    model_reference_df = original_df

    model_input_columns = (
        get_model_input_columns(
            model,
            model_reference_df,
        )
    )

    (
        active_mapping,
        active_missing,
        active_ambiguous,
    ) = build_model_feature_mapping(
        active_customers,
        model_input_columns,
    )

    model_compatible = (
        model is not None
        and not active_missing
        and not active_ambiguous
        and len(model_input_columns) > 0
    )


# ============================================================
# MODEL INPUT COLUMNS
# ============================================================

reference_df = (
    load_original_dataset()
)

model_input_columns = (
    get_model_input_columns(
        model,
        reference_df,
    )
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">'
    'Customer Churn Intelligence'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Analyze customer behavior, churn patterns, risk signals, '
    'and retention opportunities.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<span class="source-badge">'
    f'📁 Active Source: {active_source}'
    f'</span>',
    unsafe_allow_html=True,
)


# ============================================================
# DATASET SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📊 Dataset Overview'
    '</div>',
    unsafe_allow_html=True,
)

feature_columns = (
    get_dataset_feature_columns(
        active_customers
    )
)

total_customers = len(
    active_customers
)

total_features = len(
    feature_columns
)

churn_rate = (
    calculate_dataset_churn_rate(
        active_customers
    )
)

missing_cells = int(
    active_customers.isna()
    .sum()
    .sum()
)


kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:

    st.metric(
        "Total Customers",
        f"{total_customers:,}",
    )

with kpi2:

    st.metric(
        "Features",
        f"{total_features:,}",
    )

with kpi3:

    if churn_rate is not None:

        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%",
        )

    else:

        st.metric(
            "Churn Rate",
            "N/A",
        )

with kpi4:

    st.metric(
        "Missing Values",
        f"{missing_cells:,}",
    )


# ============================================================
# UPLOADED DATASET NOTICE
# ============================================================

if active_source == "Uploaded Dataset":

    if model_compatible:

        st.success(
            f"✅ Uploaded dataset is compatible with the saved model. "
            f"{len(model_input_columns):,} model features matched."
        )

    else:

        st.warning(
            "⚠️ This uploaded dataset uses a different feature schema "
            "from the currently saved Cell2Cell model. "
            "Dataset analytics are available, but ML predictions are "
            "disabled for this file. No fake or substitute features "
            "will be created."
        )

        missing = (
            st.session_state.uploaded_model_missing
        )

        ambiguous = (
            st.session_state.uploaded_model_ambiguous
        )

        if missing:

            with st.expander(
                f"Model compatibility details — "
                f"{len(missing)} required features are not present"
            ):

                st.write(
                    "The current saved model expects features that "
                    "do not exist in this uploaded dataset."
                )

                st.write(
                    "Missing model features:"
                )

                st.code(
                    "\n".join(
                        str(x)
                        for x in missing
                    )
                )

        if ambiguous:

            with st.expander(
                "Ambiguous columns"
            ):

                st.write(
                    "\n".join(
                        ambiguous
                    )
                )


# ============================================================
# CHURN DISTRIBUTION
# ============================================================

left, right = st.columns(
    [1, 1]
)

with left:

    st.markdown(
        '<div class="section-title">'
        '📈 Churn Distribution'
        '</div>',
        unsafe_allow_html=True,
    )

    distribution = (
        get_churn_distribution(
            active_customers
        )
    )

    if distribution is not None:

        chart_df = pd.DataFrame(
            {
                "Customers": distribution
            }
        )

        st.bar_chart(
            chart_df
        )

    else:

        st.info(
            "This dataset does not contain a usable Churn column."
        )


with right:

    st.markdown(
        '<div class="section-title">'
        '🧩 Dataset Features'
        '</div>',
        unsafe_allow_html=True,
    )

    schema = build_feature_schema(
        active_customers
    )

    if not schema.empty:

        st.dataframe(
            make_arrow_safe(schema),
            width="stretch",
            height=360,
            hide_index=True,
        )

    else:

        st.info(
            "No feature columns found."
        )


# ============================================================
# CUSTOMER DATA PREVIEW
# ============================================================

st.markdown(
    '<div class="section-title">'
    '👥 Customer Data'
    '</div>',
    unsafe_allow_html=True,
)

display_columns = [
    ID_COL
]

if NAME_COL in active_customers.columns:

    display_columns.append(
        NAME_COL
    )

display_columns.extend(
    feature_columns
)

if TARGET in active_customers.columns:

    display_columns.append(
        TARGET
    )

preview_df = (
    active_customers[
        display_columns
    ]
    .head(100)
    .copy()
)

# Rename only internal ID/name columns for UI.
# Actual feature names remain untouched.
preview_df = preview_df.rename(
    columns={
        ID_COL: "Customer ID",
        NAME_COL: "Customer Name",
        TARGET: "Churn",
    }
)

st.dataframe(
    make_arrow_safe(preview_df),
    width="stretch",
    height=420,
    hide_index=True,
)


# ============================================================
# CUSTOMER RISK
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🎯 Customer Risk Analysis'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# MODEL-BASED RISK
# ============================================================

if model_compatible:

    try:

        predictions = create_predictions(
            active_customers,
            model,
            model_input_columns,
            active_mapping,
        )

        # --------------------------------------------------------
        # Risk KPIs
        # --------------------------------------------------------

        low_count = int(
            (
                predictions["Risk Level"]
                == "LOW"
            ).sum()
        )

        medium_count = int(
            (
                predictions["Risk Level"]
                == "MEDIUM"
            ).sum()
        )

        high_count = int(
            (
                predictions["Risk Level"]
                == "HIGH"
            ).sum()
        )

        critical_count = int(
            (
                predictions["Risk Level"]
                == "CRITICAL"
            ).sum()
        )

        r1, r2, r3, r4 = st.columns(4)

        with r1:

            st.metric(
                "Low Risk",
                f"{low_count:,}",
            )

        with r2:

            st.metric(
                "Medium Risk",
                f"{medium_count:,}",
            )

        with r3:

            st.metric(
                "High Risk",
                f"{high_count:,}",
            )

        with r4:

            st.metric(
                "Critical Risk",
                f"{critical_count:,}",
            )

        # --------------------------------------------------------
        # Critical Risk Customers
        # --------------------------------------------------------

        st.markdown(
            "#### 🔴 Critical Risk Customers"
        )

        critical = predictions[
            predictions["Risk Level"]
            == "CRITICAL"
        ].sort_values(
            "Risk Score",
            ascending=False,
        )

        if not critical.empty:

            st.caption(
                "Customers with model-predicted churn risk above 80."
            )

            critical_table = build_customer_table(
                critical,
                top_n=20,
            )

            st.dataframe(
                critical_table,
                width="stretch",
                height=400,
                hide_index=True,
            )

        else:

            st.success(
                "No customers currently fall into the critical-risk category."
            )

        # --------------------------------------------------------
        # Highest Risk Customers
        # --------------------------------------------------------

        st.markdown(
            "#### 🔥 Highest Risk Customers"
        )

        highest_risk = build_customer_table(
            predictions,
            top_n=10,
        )

        st.dataframe(
            highest_risk,
            width="stretch",
            height=350,
            hide_index=True,
        )

        # --------------------------------------------------------
        # Customer selector
        # --------------------------------------------------------

        selector_options = list(
            active_customers.index
        )

        def customer_label(index):

            row = active_customers.loc[
                index
            ]

            customer_id = str(
                row[ID_COL]
            )

            if (
                NAME_COL in active_customers.columns
                and pd.notna(
                    row[NAME_COL]
                )
            ):

                return (
                    f"{row[NAME_COL]} "
                    f"({customer_id})"
                )

            return customer_id

        selected_index = st.selectbox(
            "Select a customer",
            selector_options,
            format_func=customer_label,
        )

        selected_prediction = predictions.loc[
            predictions.index
            == selected_index
        ].iloc[0]

        selected_customer = (
            active_customers.loc[
                selected_index
            ]
        )

        st.markdown(
            "#### 👤 Selected Customer"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Customer ID",
                str(
                    selected_customer[
                        ID_COL
                    ]
                ),
            )

        with c2:

            if NAME_COL in active_customers.columns:

                st.metric(
                    "Customer Name",
                    str(
                        selected_customer[
                            NAME_COL
                        ]
                    ),
                )

            else:

                st.metric(
                    "Customer Name",
                    "Not available",
                )

        with c3:

            st.metric(
                "Churn Probability",
                f"{selected_prediction['Churn Probability'] * 100:.2f}%",
            )

        with c4:

            st.metric(
                "Risk Level",
                selected_prediction[
                    "Risk Level"
                ],
            )

        # --------------------------------------------------------
        # Customer profile
        # --------------------------------------------------------

        st.markdown(
            "#### 🧾 Customer Feature Profile"
        )

        profile = build_customer_profile(
            selected_customer,
            feature_columns,
        )

        st.dataframe(
            profile,
            width="stretch",
            height=400,
            hide_index=True,
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )

else:

    # ============================================================
    # DATASET-ONLY ANALYTICS
    # ============================================================

    st.info(
        "ML prediction is not available for this dataset because "
        "the uploaded feature schema does not match the saved model. "
        "The dashboard will continue using the actual uploaded data."
    )

    dataset_risk = (
        calculate_dataset_risk_scores(
            active_customers
        )
    )

    if dataset_risk is not None:

        actual_churn = dataset_risk[
            dataset_risk["Risk Level"]
            == "CRITICAL"
        ]

        st.markdown(
            "#### 🔴 Actual Churn Customers"
        )

        st.caption(
            "These customers are marked as churned in the uploaded dataset. "
            "This is actual dataset information, not an ML prediction."
        )

        critical_columns = [
            "Customer ID"
        ]

        if "Customer Name" in dataset_risk.columns:

            critical_columns.append(
                "Customer Name"
            )

        critical_columns.extend(
            [
                "Actual Churn",
                "Risk Score",
                "Risk Level",
            ]
        )

        if not actual_churn.empty:

            st.dataframe(
                make_arrow_safe(
                    actual_churn[
                        critical_columns
                    ]
                    .head(50)
                ),
                width="stretch",
                height=400,
                hide_index=True,
            )

        else:

            st.info(
                "No churned customers were found in the uploaded dataset."
            )

        # --------------------------------------------------------
        # Actual churn customers
        # --------------------------------------------------------

        st.markdown(
            "#### 👥 Customer Analysis"
        )

        customer_options = list(
            active_customers.index
        )

        def dataset_customer_label(index):

            row = active_customers.loc[
                index
            ]

            customer_id = str(
                row[ID_COL]
            )

            if (
                NAME_COL in active_customers.columns
                and pd.notna(
                    row[NAME_COL]
                )
            ):

                return (
                    f"{row[NAME_COL]} "
                    f"({customer_id})"
                )

            return customer_id

        selected_index = st.selectbox(
            "Select a customer",
            customer_options,
            format_func=dataset_customer_label,
            key="dataset_customer_selector",
        )

        selected_customer = (
            active_customers.loc[
                selected_index
            ]
        )

        selected_risk = (
            dataset_risk.loc[
                selected_index
            ]
        )

        d1, d2, d3, d4 = st.columns(4)

        with d1:

            st.metric(
                "Customer ID",
                str(
                    selected_customer[
                        ID_COL
                    ]
                ),
            )

        with d2:

            if NAME_COL in active_customers.columns:

                st.metric(
                    "Customer Name",
                    str(
                        selected_customer[
                            NAME_COL
                        ]
                    ),
                )

            else:

                st.metric(
                    "Customer Name",
                    "Not available",
                )

        with d3:

            st.metric(
                "Actual Churn",
                str(
                    selected_customer[
                        TARGET
                    ]
                ),
            )

        with d4:

            st.metric(
                "Dataset Risk",
                str(
                    selected_risk[
                        "Risk Level"
                    ]
                ),
            )

        st.markdown(
            "#### 🧾 Customer Feature Profile"
        )

        profile = build_customer_profile(
            selected_customer,
            feature_columns,
        )

        st.dataframe(
            profile,
            width="stretch",
            height=400,
            hide_index=True,
        )

    else:

        st.warning(
            "The uploaded dataset does not contain a usable Churn "
            "column, so customer churn analytics cannot be calculated."
        )


# ============================================================
# WHAT DRIVES CHURN
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔎 What Drives Churn'
    '</div>',
    unsafe_allow_html=True,
)

if TARGET in active_customers.columns:

    churn_numeric = (
        active_customers[
            feature_columns
        ]
        .select_dtypes(
            include=np.number
        )
        .copy()
    )

    churn_target = (
        active_customers[TARGET]
        .apply(normalize_churn_value)
    )

    valid_mask = (
        churn_target.notna()
    )

    if (
        not churn_numeric.empty
        and valid_mask.sum() > 1
    ):

        correlations = []

        for feature in churn_numeric.columns:

            try:

                corr = (
                    churn_numeric.loc[
                        valid_mask,
                        feature
                    ]
                    .corr(
                        churn_target.loc[
                            valid_mask
                        ]
                    )
                )

                if pd.notna(corr):

                    correlations.append(
                        {
                            "Feature": feature,
                            "Correlation": corr,
                            "Absolute Correlation": abs(corr),
                        }
                    )

            except Exception:
                continue

        if correlations:

            correlation_df = (
                pd.DataFrame(
                    correlations
                )
                .sort_values(
                    "Absolute Correlation",
                    ascending=False,
                )
                .head(15)
            )

            display_corr = (
                correlation_df[
                    [
                        "Feature",
                        "Correlation",
                    ]
                ]
                .copy()
            )

            st.bar_chart(
                display_corr.set_index(
                    "Feature"
                )
            )

            st.caption(
                "Correlation is calculated from the uploaded dataset's "
                "numeric features and actual Churn labels. "
                "Correlation is not model SHAP importance."
            )

        else:

            st.info(
                "Not enough numeric information to calculate "
                "feature correlations."
            )

    else:

        st.info(
            "The dataset does not contain enough numeric features "
            "for churn-driver analysis."
        )

else:

    st.info(
        "Upload a dataset containing a Churn column to analyze "
        "which numeric features are associated with churn."
    )


# ============================================================
# BUSINESS IMPACT
# ============================================================

st.markdown(
    '<div class="section-title">'
    '💼 Business Impact'
    '</div>',
    unsafe_allow_html=True,
)

if model_compatible:

    st.caption(
        "Business impact estimates use model-predicted high/critical "
        "risk customers."
    )

    b1, b2 = st.columns(2)

    with b1:

        retention_cost = st.number_input(
            "Retention Cost / Customer",
            min_value=0.0,
            value=100.0,
            step=10.0,
        )

    with b2:

        customer_value = st.number_input(
            "Estimated Customer Value",
            min_value=0.0,
            value=500.0,
            step=50.0,
        )

    impact = calculate_business_impact(
        predictions,
        retention_cost,
        customer_value,
    )

    i1, i2, i3 = st.columns(3)

    with i1:

        st.metric(
            "Priority Customers",
            f"{impact['priority_customers']:,}",
        )

    with i2:

        st.metric(
            "Potential Customer Value",
            f"₹{impact['potential_value']:,.0f}",
        )

    with i3:

        st.metric(
            "Potential Net Value",
            f"₹{impact['potential_net']:,.0f}",
        )

    st.markdown(
        "#### 🎯 Recommended Customers"
    )

    recommended = (
        predictions[
            predictions["Risk Score"] > 60
        ]
        .sort_values(
            "Risk Score",
            ascending=False,
        )
    )

    if not recommended.empty:

        st.dataframe(
            build_customer_table(
                recommended,
                top_n=25,
            ),
            width="stretch",
            height=400,
            hide_index=True,
        )

    else:

        st.success(
            "No high-priority customers were identified."
        )

else:

    st.info(
        "Business-impact prediction is disabled because the "
        "currently uploaded dataset is not compatible with the "
        "saved ML model."
    )


# ============================================================
# SIDEBAR DATASET DETAILS
# ============================================================

with st.sidebar:

    st.divider()

    st.markdown(
        "### 📋 Active Dataset"
    )

    st.write(
        f"**Rows:** {len(active_customers):,}"
    )

    st.write(
        f"**Features:** {len(feature_columns):,}"
    )

    st.write(
        f"**Columns:** {len(active_customers.columns):,}"
    )

    if ID_COL in active_customers.columns:

        st.write(
            "**Customer ID:** `Customer ID`"
        )

    if NAME_COL in active_customers.columns:

        st.write(
            "**Customer Name:** Available"
        )

    else:

        st.write(
            "**Customer Name:** Not available"
        )

    if TARGET in active_customers.columns:

        st.write(
            "**Target:** `Churn`"
        )

    st.divider()

    st.caption(
        "Uploaded data is displayed using its actual "
        "customer IDs and feature names."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ChurnIQ • Customer Churn Intelligence Dashboard
        <br>
        Built for data-driven customer retention analysis
    </div>
    """,
    unsafe_allow_html=True,
)
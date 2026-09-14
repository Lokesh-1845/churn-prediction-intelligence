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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ChurnIQ — Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cell2celltrain.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"

TARGET = "churn"
ID_COL = "customerid"
NAME_COL = "customer_name"


# ============================================================
# REAL FEATURE NAMES
# ============================================================

REAL_FEATURE_NAMES = [
    "MonthlyRevenue",
    "MonthlyMinutes",
    "TotalRecurringCharge",
    "DirectorAssistedCalls",
    "OverageMinutes",
    "RoamingCalls",
    "PercChangeMinutes",
    "PercChangeRevenues",
    "DroppedCalls",
    "BlockedCalls",
    "UnansweredCalls",
    "CustomerCareCalls",
    "ThreeWayCalls",
    "ReceivedCalls",
    "OutgoingCalls",
    "InboundCalls",
    "PeakCallsInOut",
    "OffPeakCallsInOut",
    "DroppedBlockedCalls",
    "CallForwardingCalls",
    "CallWaitingCalls",
    "MonthsInService",
    "UniqueSubs",
    "ActiveSubs",
    "ServiceArea",
    "Handsets",
    "HandsetModels",
    "CurrentEquipmentDays",
    "AgeHH1",
    "AgeHH2",
    "ChildrenInHH",
    "HandsetRefurbished",
    "HandsetWebCapable",
    "TruckOwner",
    "RVOwner",
    "Homeownership",
    "BuysViaMailOrder",
    "RespondsToMailOffers",
    "OptOutMailings",
    "NonOptimalMonths3",
    "CreditRating",
    "FieldOpsWorking",
    "MultipleLines",
    "HandsetPrice",
    "MadeCallToRetentionTeam",
    "CreditCardHolder",
    "IncomeGroup",
    "OwnsMotorcycle",
    "AdjustmentsToCreditRating",
    "HandsetTech",
    "PrizmCode",
    "Occupation",
    "MaritalStatus",
]


# ============================================================
# SYNTHETIC CUSTOMER NAME DATA
# ============================================================

FIRST_NAMES = [
    "Lokesh", "Ruthvik", "Nihansh", "Rishik", "Satvik",
    "Komal", "Shivansh", "Aarav", "Vivaan", "Aditya",
    "Ananya", "Diya", "Isha", "Kavya", "Meera",
    "Pranav", "Rahul", "Sai", "Tanvi", "Varun",
    "Yash", "Zoya", "Neelaveni", "Arjun", "Kabir",
    "Krishna", "Riya", "Rohan", "Sunita", "Vikram",
    "Aditi", "Amit", "Dev", "Ganesh", "Jyoti",
    "Karan", "Neha", "Pooja", "Raj", "Sanjay",
    "Aaron", "Abigail", "Benjamin", "Chloe", "Daniel",
    "Elijah", "Grace", "Hannah", "Isaac", "Jacob",
    "John", "Joseph", "Luke", "Matthew", "Mary",
    "Noah", "Peter", "Rachel", "Samuel", "Sarah",
    "Thomas", "David", "Elizabeth", "Paul", "Rebecca",
    "Aisha", "Ali", "Amira", "Bilal", "Fatima",
    "Hamza", "Hassan", "Ibrahim", "Imran", "Layla",
    "Maryam", "Mohammed", "Mustafa", "Omar", "Sana",
    "Yousef", "Zainab", "Zara", "Ahmed", "Farhan",
    "Khadija", "Nadia", "Rayyan", "Tariq", "Yasmin",
    "Arjan", "Gurpreet", "Jaspreet", "Manpreet",
    "Navjot", "Prabhjot", "Rajdeep", "Ananda",
    "Asoka", "Gautam", "Siddharth", "Tenzin",
    "Mayadevi", "Anshul", "Jinendra", "Mahavir",
    "Rishabh", "Shrenik", "Asher", "Eli", "Levi",
    "Miriam", "Nathan", "Ezra",
]


LAST_NAMES = [
    "Kumar", "Sharma", "Reddy", "Patel", "Rao",
    "Verma", "Singh", "Gupta", "Mehta", "Nair",
    "Iyer", "Joshi", "Kapoor", "Malhotra", "Shah",
    "Mishra", "Agarwal", "Chowdhury", "Das", "Roy",
    "Morgan", "Johnson", "Williams", "Brown", "Davis",
    "Miller", "Wilson", "Moore", "Taylor", "Anderson",
    "Thomas", "Jackson", "White", "Harris", "Martin",
    "Thompson", "Garcia", "Martinez", "Robinson", "Clark",
    "Lewis", "Lee", "Walker", "Hall", "Allen",
    "Young", "King", "Wright", "Scott", "Green",
]


# ============================================================
# DETERMINISTIC NAME GENERATOR
# ============================================================

@st.cache_data
def generate_customer_names(customer_ids):

    customer_ids = [
        str(customer_id)
        for customer_id in customer_ids
    ]

    names = []

    total_first = len(FIRST_NAMES)
    total_last = len(LAST_NAMES)

    for customer_id in customer_ids:

        numeric_hash = int(
            hashlib.md5(
                customer_id.encode("utf-8")
            ).hexdigest()[:12],
            16
        )

        first_index = (
            numeric_hash
            % total_first
        )

        last_index = (
            (numeric_hash // total_first)
            % total_last
        )

        first_name = FIRST_NAMES[first_index]
        last_name = LAST_NAMES[last_index]

        names.append(
            f"{first_name} {last_name}"
        )

    return names


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GENERAL
       ======================================================== */

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }


    /* ========================================================
       MAIN TITLE
       ======================================================== */

    .main-title {
        font-size: 2.35rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-top: 4px;
        margin-bottom: 25px;
    }


    /* ========================================================
       KPI CARD
       ======================================================== */

    .kpi-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 20px;
        min-height: 120px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        color: #111827;
    }

    .kpi-label {
        color: #6b7280;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .kpi-value {
        color: #111827;
        font-size: 2rem;
        font-weight: 800;
    }


    /* ========================================================
       RISK CARDS
       ======================================================== */

    .risk-critical {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        border-left: 6px solid #e11d48;
        border-radius: 14px;
        padding: 18px;
        color: #9f1239 !important;
    }

    .risk-high {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-left: 6px solid #f97316;
        border-radius: 14px;
        padding: 18px;
        color: #9a3412 !important;
    }

    .risk-medium {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 6px solid #3b82f6;
        border-radius: 14px;
        padding: 18px;
        color: #1e40af !important;
    }

    .risk-low {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-left: 6px solid #10b981;
        border-radius: 14px;
        padding: 18px;
        color: #065f46 !important;
    }

    .risk-number {
        font-size: 3rem;
        font-weight: 900;
        margin: 0;
        line-height: 1;
    }

    .risk-label {
        font-size: 1.1rem;
        font-weight: 800;
        margin-bottom: 8px;
    }


    /* ========================================================
       LIGHT SECTION HEADERS
       ======================================================== */

    .section-title {
        font-size: 1.25rem;
        font-weight: 800;
        margin-top: 25px;
        margin-bottom: 14px;
        color: #1f2937;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #6366f1;
        border-radius: 10px;
        padding: 10px 15px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }

    /* Customer Risk */
    .section-risk {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 5px solid #3b82f6;
        color: #1e3a8a;
    }

    /* Customer Information */
    .section-information {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-left: 5px solid #0284c7;
        color: #075985;
    }

    /* Recommended Action */
    .section-action {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 5px solid #f59e0b;
        color: #92400e;
    }

    /* Why Churn */
    .section-why {
        background: #f5f3ff;
        border: 1px solid #ddd6fe;
        border-left: 5px solid #7c3aed;
        color: #5b21b6;
    }

    /* Risk Distribution */
    .section-distribution {
        background: #f0fdfa;
        border: 1px solid #99f6e4;
        border-left: 5px solid #0d9488;
        color: #115e59;
    }

    /* Highest Risk */
    .section-highest-risk {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        border-left: 5px solid #e11d48;
        color: #9f1239;
    }

    /* Customer Churn Intelligence */
    .section-intelligence {
        background: linear-gradient(
            135deg,
            #eef2ff,
            #f0f9ff
        );
        border: 1px solid #c7d2fe;
        border-left: 6px solid #4f46e5;
        color: #3730a3;
        font-size: 1.45rem;
        padding: 12px 17px;
    }


    /* ========================================================
       SHAP
       ======================================================== */

    .shap-positive {
        background: #fff1f2;
        border-left: 4px solid #ef4444;
        padding: 10px 14px;
        margin-bottom: 7px;
        border-radius: 8px;
        color: #9f1239 !important;
    }

    .shap-negative {
        background: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 10px 14px;
        margin-bottom: 7px;
        border-radius: 8px;
        color: #065f46 !important;
    }


    /* ========================================================
       ROI
       ======================================================== */

    .roi-positive {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-left: 6px solid #10b981;
        border-radius: 14px;
        padding: 20px;
        color: #065f46 !important;
    }

    .roi-negative {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        border-left: 6px solid #e11d48;
        border-radius: 14px;
        padding: 20px;
        color: #9f1239 !important;
    }


    /* ========================================================
       SIMULATOR
       ======================================================== */

    .simulator-header {
        background: linear-gradient(
            135deg,
            #111827,
            #1f2937
        );
        border-radius: 18px;
        padding: 28px;
        color: white !important;
        margin-bottom: 25px;
    }

    .simulator-header h1 {
        color: white !important;
        margin-bottom: 5px;
        font-size: 2.2rem;
        font-weight: 800;
    }

    .simulator-header p {
        color: #d1d5db !important;
        margin-bottom: 0;
    }


    /* ========================================================
       CRITICAL SUMMARY
       ======================================================== */

    .critical-summary {
        background: linear-gradient(
            135deg,
            #fff1f2,
            #fff7ed
        );
        border: 1px solid #fecdd3;
        border-left: 6px solid #e11d48;
        border-radius: 16px;
        padding: 22px;
        color: #111827;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }

    .critical-summary-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #9f1239;
        margin-bottom: 5px;
    }

    .critical-summary-number {
        font-size: 2.5rem;
        font-weight: 900;
        color: #9f1239;
        line-height: 1.1;
    }

    .critical-summary-text {
        margin-top: 7px;
        color: #6b7280;
    }


    /* ========================================================
       CUSTOMER ID
       ======================================================== */

    .customer-id {
        color: #6b7280;
        font-family: monospace;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        color: #9ca3af;
        margin-top: 50px;
        font-size: 0.8rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SUPPRESS MODEL OUTPUT
# ============================================================

@contextmanager
def suppress_model_output():

    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    with redirect_stdout(
        stdout_buffer
    ), redirect_stderr(
        stderr_buffer
    ):
        yield


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(
    show_spinner="Loading ChurnIQ model..."
)
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = joblib.load(
            MODEL_PATH
        )

    return model


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource(
    show_spinner="Loading customer data..."
)
def load_customers():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if ID_COL not in df.columns:
        raise ValueError(
            f"Column '{ID_COL}' not found."
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Column '{TARGET}' not found."
        )

    df[ID_COL] = (
        df[ID_COL]
        .astype(str)
        .str.strip()
    )

    unique_ids = (
        df[ID_COL]
        .drop_duplicates()
        .tolist()
    )

    generated_names = (
        generate_customer_names(
            unique_ids
        )
    )

    name_map = dict(
        zip(
            unique_ids,
            generated_names
        )
    )

    df[NAME_COL] = (
        df[ID_COL]
        .map(name_map)
    )

    return df


# ============================================================
# PREDICTION
# ============================================================

def silent_predict_proba(
    model,
    X
):

    with suppress_model_output():

        probabilities = (
            model.predict_proba(X)
        )

    probabilities = np.asarray(
        probabilities
    )

    if probabilities.ndim == 2:

        probabilities = (
            probabilities[:, 1]
        )

    return probabilities.astype(
        float
    )


@st.cache_resource(
    show_spinner="Running churn predictions..."
)
def create_predictions(
    _model,
    customers
):

    X = customers.drop(
        columns=[
            TARGET,
            ID_COL,
            NAME_COL,
        ],
        errors="ignore"
    )

    probabilities = (
        silent_predict_proba(
            _model,
            X
        )
    )

    result = pd.DataFrame(
        {
            "customerid":
                customers[ID_COL]
                .astype(str),

            "customer_name":
                customers[NAME_COL]
                .astype(str),

            "probability":
                probabilities,
        }
    )

    result["risk_score"] = (
        result["probability"]
        * 100
    ).round().astype(int)

    result["risk_level"] = (
        result["risk_score"]
        .apply(
            calculate_risk_level
        )
    )

    result["prediction"] = (
        result["probability"] >= 0.50
    ).astype(int)

    return result


# ============================================================
# RISK
# ============================================================

def calculate_risk_level(
    score
):

    if score <= 30:
        return "LOW"

    if score <= 60:
        return "MEDIUM"

    if score <= 80:
        return "HIGH"

    return "CRITICAL"


def recommendation(
    level
):

    if level == "CRITICAL":

        return (
            "Immediate retention intervention recommended. "
            "Contact the customer and review service, billing "
            "and retention opportunities."
        )

    if level == "HIGH":

        return (
            "Prioritize this customer for retention outreach "
            "and investigate the major churn drivers."
        )

    if level == "MEDIUM":

        return (
            "Monitor customer behaviour and consider targeted "
            "engagement or retention communication."
        )

    return (
        "No immediate retention action required. "
        "Continue normal customer engagement."
    )


# ============================================================
# ARROW SAFE
# ============================================================

def make_arrow_safe(
    df
):

    safe_df = df.copy()

    for column in safe_df.columns:

        if safe_df[column].dtype == "object":

            safe_df[column] = (
                safe_df[column]
                .map(
                    lambda value:
                    ""
                    if pd.isna(value)
                    else str(value)
                )
            )

    return safe_df


# ============================================================
# PIPELINE HELPERS
# ============================================================

def get_pipeline_steps(
    model
):

    if not hasattr(
        model,
        "named_steps"
    ):

        raise TypeError(
            "Saved model is not an sklearn Pipeline."
        )

    return model.named_steps


def get_preprocessor(
    model
):

    steps = get_pipeline_steps(
        model
    )

    if "preprocessor" in steps:

        return steps[
            "preprocessor"
        ]

    for name, step in steps.items():

        if (
            hasattr(
                step,
                "transform"
            )
            and name not in [
                "classifier",
                "model",
                "estimator",
            ]
        ):

            return step

    raise ValueError(
        "Could not find preprocessing step."
    )


def get_final_estimator(
    model
):

    if not hasattr(
        model,
        "steps"
    ):

        return model

    return model.steps[-1][1]


# ============================================================
# SHAP TRANSFORMATION
# ============================================================

def transform_for_shap(
    model,
    X
):

    preprocessor = (
        get_preprocessor(
            model
        )
    )

    with suppress_model_output():

        transformed = (
            preprocessor.transform(X)
        )

    if hasattr(
        transformed,
        "toarray"
    ):

        transformed = (
            transformed.toarray()
        )

    return np.asarray(
        transformed,
        dtype=float
    )


def get_feature_names(
    model,
    X
):

    preprocessor = (
        get_preprocessor(
            model
        )
    )

    try:

        raw_names = (
            preprocessor
            .get_feature_names_out()
        )

        cleaned = []

        for name in raw_names:

            clean = (
                str(name)
                .replace(
                    "num__",
                    ""
                )
                .replace(
                    "cat__",
                    ""
                )
                .strip()
            )

            cleaned.append(
                clean
            )

        return cleaned

    except Exception:

        transformed = (
            transform_for_shap(
                model,
                X
            )
        )

        count = (
            transformed.shape[1]
        )

        if count <= len(
            REAL_FEATURE_NAMES
        ):

            return (
                REAL_FEATURE_NAMES[
                    :count
                ]
            )

        return (
            REAL_FEATURE_NAMES
            + [
                f"Feature_{i}"
                for i in range(
                    len(
                        REAL_FEATURE_NAMES
                    ),
                    count
                )
            ]
        )


# ============================================================
# SHAP EXPLAINER
# ============================================================

@st.cache_resource(
    show_spinner="Preparing SHAP explainer..."
)
def create_shap_explainer(
    _model
):

    import shap

    final_model = (
        get_final_estimator(
            _model
        )
    )

    model_name = (
        final_model
        .__class__
        .__name__
    )

    tree_models = {
        "RandomForestClassifier",
        "RandomForestRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "XGBClassifier",
        "XGBRegressor",
        "LGBMClassifier",
        "LGBMRegressor",
        "CatBoostClassifier",
        "CatBoostRegressor",
    }

    if model_name in tree_models:

        try:

            explainer = (
                shap.TreeExplainer(
                    final_model
                )
            )

            return {
                "explainer":
                    explainer,

                "type":
                    "tree",

                "model":
                    final_model,

                "error":
                    None,
            }

        except Exception as error:

            return {
                "explainer":
                    None,

                "type":
                    "error",

                "model":
                    final_model,

                "error":
                    str(error),
            }

    return {
        "explainer":
            None,

        "type":
            "generic",

        "model":
            final_model,

        "error":
            None,
    }


# ============================================================
# SHAP NORMALIZATION
# ============================================================

def normalize_shap_values(
    values,
    expected_rows,
    expected_features
):

    if hasattr(
        values,
        "values"
    ):

        values = values.values

    if isinstance(
        values,
        list
    ):

        if len(values) > 1:

            values = values[1]

        else:

            values = values[0]

    values = np.asarray(
        values
    )

    if values.ndim == 2:

        return values

    if values.ndim == 3:

        if (
            values.shape[0]
            == expected_rows
            and
            values.shape[1]
            == expected_features
        ):

            return values[:, :, -1]

    raise ValueError(
        "Unexpected SHAP output shape: "
        f"{values.shape}"
    )


# ============================================================
# LOCAL SHAP
# ============================================================

@st.cache_data(
    show_spinner="Calculating customer explanation..."
)
def get_local_shap_cached(
    _model,
    customer_row,
    top_n=8
):

    import shap

    transformed = (
        transform_for_shap(
            _model,
            customer_row
        )
    )

    feature_names = (
        get_feature_names(
            _model,
            customer_row
        )
    )

    shap_info = (
        create_shap_explainer(
            _model
        )
    )

    if shap_info["error"]:

        raise RuntimeError(
            shap_info["error"]
        )

    if (
        shap_info["type"]
        == "tree"
    ):

        with suppress_model_output():

            values = (
                shap_info[
                    "explainer"
                ]
                .shap_values(
                    transformed
                )
            )

        values = (
            normalize_shap_values(
                values,
                transformed.shape[0],
                transformed.shape[1]
            )
        )

    else:

        final_model = (
            shap_info["model"]
        )

        def prediction_function(
            transformed_data
        ):

            with suppress_model_output():

                return (
                    final_model
                    .predict_proba(
                        transformed_data
                    )[:, 1]
                )

        background = (
            transformed[:20]
        )

        generic_explainer = (
            shap.Explainer(
                prediction_function,
                background
            )
        )

        with suppress_model_output():

            explanation = (
                generic_explainer(
                    transformed
                )
            )

        values = (
            normalize_shap_values(
                explanation,
                transformed.shape[0],
                transformed.shape[1]
            )
        )

    values = values[0]

    result = pd.DataFrame(
        {
            "feature":
                feature_names,

            "shap":
                values,
        }
    )

    result["impact"] = (
        result["shap"].abs()
    )

    result["direction"] = np.where(
        result["shap"] > 0,
        "Increases churn",
        "Reduces churn"
    )

    total = (
        result["impact"].sum()
    )

    if total > 0:

        result["impact_pct"] = (
            result["impact"]
            / total
        ) * 100

    else:

        result["impact_pct"] = 0

    return (
        result
        .sort_values(
            "impact",
            ascending=False
        )
        .head(top_n)
        .reset_index(
            drop=True
        )
    )


# ============================================================
# GLOBAL SHAP
# ============================================================

@st.cache_data(
    show_spinner="Calculating global churn drivers..."
)
def get_global_shap_cached(
    _model,
    customers,
    sample_size=100
):

    X = customers.drop(
        columns=[
            TARGET,
            ID_COL,
            NAME_COL,
        ],
        errors="ignore"
    )

    sample = X.sample(
        n=min(
            sample_size,
            len(X)
        ),
        random_state=42
    )

    transformed = (
        transform_for_shap(
            _model,
            sample
        )
    )

    feature_names = (
        get_feature_names(
            _model,
            sample
        )
    )

    shap_info = (
        create_shap_explainer(
            _model
        )
    )

    if shap_info["error"]:

        raise RuntimeError(
            shap_info["error"]
        )

    if (
        shap_info["type"]
        != "tree"
    ):

        raise RuntimeError(
            "Global SHAP is available "
            "only for the tree-based model."
        )

    with suppress_model_output():

        values = (
            shap_info["explainer"]
            .shap_values(
                transformed
            )
        )

    values = (
        normalize_shap_values(
            values,
            transformed.shape[0],
            transformed.shape[1]
        )
    )

    importance = (
        np.abs(values)
        .mean(axis=0)
    )

    result = pd.DataFrame(
        {
            "feature":
                feature_names,

            "importance":
                importance,
        }
    )

    return (
        result
        .sort_values(
            "importance",
            ascending=False
        )
        .head(15)
        .reset_index(
            drop=True
        )
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

@st.cache_data(
    show_spinner="Calculating model performance..."
)
def calculate_model_performance(
    _model,
    customers
):

    from sklearn.model_selection import (
        train_test_split
    )

    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_score,
        recall_score,
        f1_score,
        brier_score_loss,
        confusion_matrix,
    )

    X = customers.drop(
        columns=[
            TARGET,
            ID_COL,
            NAME_COL,
        ],
        errors="ignore"
    )

    target = (
        customers[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    y = target.map(
        {
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
        }
    )

    if y.isna().any():

        raise ValueError(
            "Target labels could not be converted."
        )

    y = y.astype(int)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    probability = (
        silent_predict_proba(
            _model,
            X_test
        )
    )

    prediction = (
        probability >= 0.50
    ).astype(int)

    auc = roc_auc_score(
        y_test,
        probability
    )

    pr_auc = (
        average_precision_score(
            y_test,
            probability
        )
    )

    precision = precision_score(
        y_test,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0
    )

    brier = brier_score_loss(
        y_test,
        probability
    )

    cm = confusion_matrix(
        y_test,
        prediction,
        labels=[0, 1]
    )

    evaluation = pd.DataFrame(
        {
            "actual":
                y_test.to_numpy(),

            "probability":
                probability,
        }
    ).sort_values(
        "probability",
        ascending=False
    )

    top_n = max(
        1,
        int(
            len(evaluation)
            * 0.10
        )
    )

    top_10 = (
        evaluation.head(
            top_n
        )
    )

    actual_churners = (
        evaluation["actual"]
        .sum()
    )

    recall_at_top10 = (
        top_10["actual"].sum()
        / actual_churners
        if actual_churners > 0
        else 0
    )

    return {
        "roc_auc":
            auc,

        "pr_auc":
            pr_auc,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "brier":
            brier,

        "confusion_matrix":
            cm,

        "recall_at_top10":
            recall_at_top10,

        "test_size":
            len(y_test),
    }


# ============================================================
# BUSINESS IMPACT SIMULATOR
# ============================================================

def business_impact_simulator(
    predictions
):

    st.markdown(
        """
        <div class="simulator-header">
            <h1>💼 Business Impact Simulator</h1>
            <p>
                Convert real churn predictions into a practical
                retention strategy and estimate financial impact.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    high_risk = (
        predictions[
            predictions["risk_score"] > 60
        ]
        .sort_values(
            "probability",
            ascending=False
        )
        .copy()
    )

    high_risk_count = (
        len(high_risk)
    )

    critical_count = int(
        (
            predictions["risk_score"]
            > 80
        ).sum()
    )

    high_count = int(
        (
            (predictions["risk_score"] > 60)
            &
            (predictions["risk_score"] <= 80)
        ).sum()
    )

    st.markdown(
        "### 🎯 Retention Opportunity"
    )

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "High-Risk Customers",
        f"{high_risk_count:,}"
    )

    p2.metric(
        "High Risk",
        f"{high_count:,}"
    )

    p3.metric(
        "Critical Risk",
        f"{critical_count:,}"
    )

    if high_risk_count == 0:

        st.warning(
            "No customers currently have a risk score above 60."
        )

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
                high_risk_count
            ),
            step=(
                100
                if high_risk_count >= 100
                else 1
            )
        )

        save_rate_input = (
            st.slider(
                "🎯 Expected retention success rate",
                1,
                100,
                15,
                1,
                format="%d%%"
            )
        )

    with c2:

        customer_value = (
            st.number_input(
                "💰 Average customer lifetime value ($)",
                min_value=0.0,
                value=1000.0,
                step=100.0
            )
        )

        campaign_cost = (
            st.number_input(
                "📣 Campaign cost per customer ($)",
                min_value=0.0,
                value=20.0,
                step=5.0
            )
        )

    contacted = min(
        int(capacity),
        high_risk_count
    )

    save_rate = (
        save_rate_input / 100
    )

    expected_saved = (
        contacted
        * save_rate
    )

    revenue_saved = (
        expected_saved
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

    roi = (
        net_impact
        / total_cost
        * 100
        if total_cost > 0
        else 0
    )

    revenue_cost_ratio = (
        revenue_saved
        / total_cost
        if total_cost > 0
        else 0
    )

    st.markdown(
        "### 📊 Estimated Business Impact"
    )

    r1, r2, r3, r4 = (
        st.columns(4)
    )

    r1.metric(
        "Customers Contacted",
        f"{contacted:,}"
    )

    r2.metric(
        "Expected Saved",
        f"{expected_saved:,.0f}"
    )

    r3.metric(
        "Revenue Saved",
        f"${revenue_saved:,.0f}"
    )

    r4.metric(
        "Campaign Cost",
        f"${total_cost:,.0f}"
    )

    if net_impact >= 0:

        roi_col1, roi_col2 = st.columns(
            [2, 1]
        )

        with roi_col1:

            st.success(
                f"📈 Positive Business Impact\n\n"
                f"Estimated Net Impact: **${net_impact:,.0f}**"
            )

        with roi_col2:

            st.metric(
                "Estimated ROI",
                f"{roi:,.1f}%"
            )

        st.info(
            f"Revenue / Campaign Cost: **{revenue_cost_ratio:,.2f}x**"
        )

    else:

        roi_col1, roi_col2 = st.columns(
            [2, 1]
        )

        with roi_col1:

            st.error(
                f"⚠️ Negative Business Impact\n\n"
                f"Estimated Net Impact: **${net_impact:,.0f}**"
            )

        with roi_col2:

            st.metric(
                "Estimated ROI",
                f"{roi:,.1f}%"
            )

        st.warning(
            "Campaign cost exceeds the estimated revenue saved."
        )

    # ========================================================
    # RECOMMENDED CUSTOMERS
    # ========================================================

    st.markdown(
        "### 🏆 Recommended Customers"
    )

    st.caption(
        "Ranked using actual predicted churn probability."
    )

    selected = (
        high_risk
        .head(contacted)
        .copy()
    )

    selected[
        "Churn Probability (%)"
    ] = (
        selected["probability"]
        * 100
    ).round(1)

    # Customer ID FIRST, Customer Name SECOND
    selected = selected[
        [
            "customerid",
            "customer_name",
            "Churn Probability (%)",
            "risk_score",
            "risk_level",
        ]
    ]

    selected.columns = [
        "Customer ID",
        "Customer Name",
        "Churn Probability (%)",
        "Risk Score",
        "Risk Level",
    ]

    st.dataframe(
        make_arrow_safe(
            selected
        ),
        width="stretch",
        height=450,
        hide_index=True
    )


# ============================================================
# LOAD CORE RESOURCES
# ============================================================

try:

    model = load_model()

    customers = load_customers()

    predictions = create_predictions(
        model,
        customers
    )

except Exception as error:

    st.error(
        "Dashboard initialization failed:\n\n"
        f"{error}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h1 style="font-size:26px;color:#ffffff;">
            📊 ChurnIQ
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Customer Churn Intelligence"
    )

    st.divider()

    page = st.radio(
        "Dashboard",
        [
            "🏠 Overview",
            "👤 Customer Risk",
            "🔍 What Drives Churn",
            "💼 Business Impact Simulator",
        ]
    )

    st.divider()

    st.markdown(
        "**Data Source**"
    )

    st.caption(
        f"Real customers: {len(customers):,}"
    )

    st.caption(
        "Model: Trained Churn Pipeline"
    )

    st.divider()

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.markdown(
        "### 📈 Model Performance"
    )

    if st.button(
        "Load Model Performance",
        use_container_width=True
    ):

        try:

            performance = (
                calculate_model_performance(
                    model,
                    customers
                )
            )

            st.metric(
                "ROC-AUC",
                f"{performance['roc_auc']:.3f}"
            )

            st.metric(
                "PR-AUC",
                f"{performance['pr_auc']:.3f}"
            )

            st.metric(
                "Precision",
                f"{performance['precision']:.3f}"
            )

            st.metric(
                "Recall",
                f"{performance['recall']:.3f}"
            )

            st.metric(
                "F1 Score",
                f"{performance['f1']:.3f}"
            )

            st.metric(
                "Brier Score",
                f"{performance['brier']:.3f}"
            )

            st.metric(
                "Recall @ Top 10%",
                f"{performance['recall_at_top10']:.1%}"
            )

            st.caption(
                f"Test customers: "
                f"{performance['test_size']:,}"
            )

            st.markdown(
                "#### Confusion Matrix"
            )

            cm = performance[
                "confusion_matrix"
            ]

            cm_df = pd.DataFrame(
                cm,
                index=[
                    "Actual No Churn",
                    "Actual Churn",
                ],
                columns=[
                    "Predicted No Churn",
                    "Predicted Churn",
                ]
            )

            st.dataframe(
                make_arrow_safe(
                    cm_df
                ),
                width="stretch"
            )

        except Exception as error:

            st.warning(
                "Performance metrics unavailable:\n"
                f"{error}"
            )


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    # ========================================================
    # CUSTOMER CHURN INTELLIGENCE
    # ========================================================

    st.markdown(
        '<div class="section-title section-intelligence">'
        '📊 Customer Churn Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Real-time customer risk intelligence powered by the trained churn model.'
        '</div>',
        unsafe_allow_html=True
    )

    total = len(
        predictions
    )

    predicted_churn = int(
        (
            predictions["probability"]
            >= 0.50
        ).sum()
    )

    high_risk = int(
        (
            predictions["risk_score"]
            > 60
        ).sum()
    )

    critical = int(
        (
            predictions["risk_score"]
            > 80
        ).sum()
    )

    avg_probability = (
        predictions["probability"]
        .mean()
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">
                    TOTAL CUSTOMERS
                </div>
                <div class="kpi-value">
                    {total:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">
                    PREDICTED CHURN
                </div>
                <div class="kpi-value">
                    {predicted_churn:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">
                    HIGH / CRITICAL RISK
                </div>
                <div class="kpi-value">
                    {high_risk:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">
                    AVG CHURN PROBABILITY
                </div>
                <div class="kpi-value">
                    {avg_probability:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # RISK DISTRIBUTION
    # ========================================================

    st.markdown(
        '<div class="section-title section-distribution">'
        '📊 Risk Distribution'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1.1, 1]
    )

    with left:

        distribution = (
            predictions["risk_level"]
            .value_counts()
            .reindex(
                [
                    "LOW",
                    "MEDIUM",
                    "HIGH",
                    "CRITICAL",
                ]
            )
            .fillna(0)
        )

        st.bar_chart(
            distribution
        )

    with right:

        st.markdown(
            f"""<div class="critical-summary">
<div class="critical-summary-title">🚨 Critical Risk Customers</div>
<div class="critical-summary-number">{critical:,}</div>
<div class="critical-summary-text">Customers with a churn risk score above 80.</div>
</div>""",
            unsafe_allow_html=True
        )


    # ========================================================
    # HIGHEST RISK CUSTOMERS
    # ========================================================

    st.markdown(
        '<div class="section-title section-highest-risk">'
        '🚨 Highest Risk Customers'
        '</div>',
        unsafe_allow_html=True
    )

    top = (
        predictions
        .sort_values(
            "probability",
            ascending=False
        )
        .head(20)
        .copy()
    )

    top[
        "Churn Probability (%)"
    ] = (
        top["probability"]
        * 100
    ).round(1)

    # Customer ID FIRST, Customer Name SECOND
    top = top[
        [
            "customerid",
            "customer_name",
            "Churn Probability (%)",
            "risk_score",
            "risk_level",
        ]
    ]

    top.columns = [
        "Customer ID",
        "Customer Name",
        "Churn Probability (%)",
        "Risk Score",
        "Risk Level",
    ]

    st.dataframe(
        make_arrow_safe(
            top
        ),
        width="stretch",
        hide_index=True
    )


# ============================================================
# CUSTOMER RISK
# ============================================================

elif page == "👤 Customer Risk":

    st.markdown(
        '<div class="section-title section-risk">'
        '👤 Customer Risk'
        '</div>',
        unsafe_allow_html=True
    )

    customer_options = (
        predictions[
            [
                "customer_name",
                "customerid",
            ]
        ]
        .drop_duplicates()
    )

    customer_labels = {
        f"{row['customer_name']} — ID {row['customerid']}":
            row["customerid"]
        for _, row in customer_options.iterrows()
    }

    selected_label = st.selectbox(
        "Select Customer",
        list(
            customer_labels.keys()
        )
    )

    selected_customer_id = (
        customer_labels[
            selected_label
        ]
    )

    selected_prediction = (
        predictions[
            predictions["customerid"]
            == selected_customer_id
        ]
        .iloc[0]
    )

    customer_row = (
        customers[
            customers[ID_COL]
            .astype(str)
            == str(
                selected_customer_id
            )
        ]
        .head(1)
    )

    probability = float(
        selected_prediction[
            "probability"
        ]
    )

    risk_score = int(
        selected_prediction[
            "risk_score"
        ]
    )

    risk_level = (
        selected_prediction[
            "risk_level"
        ]
    )

    customer_name = (
        selected_prediction[
            "customer_name"
        ]
    )

    left, right = st.columns(
        [1, 2]
    )

    # ========================================================
    # RISK SCORE UI
    # ========================================================

    with left:

        if risk_level == "CRITICAL":

            st.error(
                f"🚨 CRITICAL RISK\n\n"
                f"Risk Score: **{risk_score}/100**"
            )

        elif risk_level == "HIGH":

            st.warning(
                f"🔴 HIGH RISK\n\n"
                f"Risk Score: **{risk_score}/100**"
            )

        elif risk_level == "MEDIUM":

            st.info(
                f"🟡 MEDIUM RISK\n\n"
                f"Risk Score: **{risk_score}/100**"
            )

        else:

            st.success(
                f"🟢 LOW RISK\n\n"
                f"Risk Score: **{risk_score}/100**"
            )

    with right:

        m1, m2 = st.columns(2)

        m1.metric(
            "Churn Probability",
            f"{probability:.1%}"
        )

        m2.metric(
            "Customer ID",
            str(
                selected_customer_id
            )
        )

        st.progress(
            probability,
            text=(
                f"{customer_name} • "
                f"Risk Score: "
                f"{risk_score}/100"
            )
        )


    # ========================================================
    # CUSTOMER INFORMATION
    # ========================================================

    st.markdown(
        '<div class="section-title section-information">'
        '👤 Customer Information'
        '</div>',
        unsafe_allow_html=True
    )

    info1, info2 = st.columns(2)

    info1.metric(
        "Customer Name",
        customer_name
    )

    info2.metric(
        "Customer ID",
        str(
            selected_customer_id
        )
    )


    # ========================================================
    # RECOMMENDED ACTION
    # ========================================================

    st.markdown(
        '<div class="section-title section-action">'
        '🎯 Recommended Action'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        recommendation(
            risk_level
        )
    )


    # ========================================================
    # WHY MIGHT CUSTOMER CHURN
    # ========================================================

    st.markdown(
        '<div class="section-title section-why">'
        '🔍 Why might this customer churn?'
        '</div>',
        unsafe_allow_html=True
    )

    try:

        shap_df = (
            get_local_shap_cached(
                model,
                customer_row,
                top_n=8
            )
        )

        left, right = st.columns(
            [1.25, 1]
        )

        with left:

            chart_data = (
                shap_df
                .set_index(
                    "feature"
                )["shap"]
                .sort_values()
            )

            st.bar_chart(
                chart_data
            )

        with right:

            for _, row in (
                shap_df.iterrows()
            ):

                feature = (
                    str(
                        row["feature"]
                    )
                )

                impact = float(
                    row["shap"]
                )

                percentage = float(
                    row["impact_pct"]
                )

                if impact > 0:

                    st.error(
                        f"↑ **{feature}**\n\n"
                        f"Increases churn\n\n"
                        f"SHAP impact: **{percentage:.1f}%**"
                    )

                else:

                    st.success(
                        f"↓ **{feature}**\n\n"
                        f"Reduces churn\n\n"
                        f"SHAP impact: **{percentage:.1f}%**"
                    )

    except Exception as error:

        st.warning(
            "SHAP explanation unavailable: "
            f"{error}"
        )


    # ========================================================
    # CUSTOMER PROFILE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📋 Customer Profile'
        '</div>',
        unsafe_allow_html=True
    )

    profile = (
        customer_row
        .drop(
            columns=[
                TARGET,
                ID_COL,
                NAME_COL,
            ],
            errors="ignore"
        )
        .T
        .reset_index()
    )

    profile.columns = [
        "Feature",
        "Actual Value",
    ]

    st.dataframe(
        make_arrow_safe(
            profile
        ),
        width="stretch",
        hide_index=True
    )


# ============================================================
# WHAT DRIVES CHURN
# ============================================================

elif page == "🔍 What Drives Churn":

    st.markdown(
        '<div class="section-title section-why">'
        '🔍 What Drives Churn?'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Global SHAP analysis is calculated only when this page is opened."
    )

    try:

        importance = (
            get_global_shap_cached(
                model,
                customers,
                sample_size=100
            )
        )

        chart = (
            importance
            .set_index(
                "feature"
            )["importance"]
            .sort_values()
        )

        st.bar_chart(
            chart
        )

        st.markdown(
            '<div class="section-title section-why">'
            '📌 Top Churn Drivers'
            '</div>',
            unsafe_allow_html=True
        )

        display = (
            importance.copy()
        )

        display["importance"] = (
            display["importance"]
            .round(5)
        )

        display.columns = [
            "Feature",
            "Mean |SHAP Value|",
        ]

        st.dataframe(
            make_arrow_safe(
                display
            ),
            width="stretch",
            hide_index=True
        )

    except Exception as error:

        st.warning(
            "Global SHAP unavailable: "
            f"{error}"
        )


# ============================================================
# BUSINESS IMPACT
# ============================================================

elif page == "💼 Business Impact Simulator":

    business_impact_simulator(
        predictions
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ChurnIQ • Customer Churn Prediction & Explainability
    </div>
    """,
    unsafe_allow_html=True
)
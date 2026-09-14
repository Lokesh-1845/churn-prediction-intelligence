import os
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from fastapi.responses import Response
from src.explainability.shap_engine import ChurnExplainer


# ============================================================
# PROJECT PATH
# ============================================================

# api/main.py
#   parent      -> api
#   parent.parent -> CCP

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"


# ============================================================
# FASTAPI CONFIGURATION
# ============================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="XGBoost + SHAP based Customer Churn Prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================================
# GLOBAL MODEL VARIABLES
# ============================================================

pipeline = None
explainer = None


# ============================================================
# LOAD MODEL + SHAP EXPLAINER
# ============================================================

print("\n" + "=" * 60)
print("       CUSTOMER CHURN PREDICTION API")
print("=" * 60)

print(f"\nProject root:")
print(PROJECT_ROOT)

print(f"\nModel path:")
print(MODEL_PATH)


if not MODEL_PATH.exists():

    print("\n❌ MODEL NOT FOUND")
    print(f"Expected model at:")
    print(MODEL_PATH)

else:

    try:

        # ----------------------------------------------------
        # Load trained pipeline
        # ----------------------------------------------------

        pipeline = joblib.load(MODEL_PATH)

        print("\n✓ Model pipeline loaded successfully.")

        # ----------------------------------------------------
        # Initialize SHAP
        # ----------------------------------------------------

        explainer = ChurnExplainer(pipeline)

        print("✓ SHAP Explainer initialized successfully.")

    except Exception as e:

        print("\n❌ Failed to initialize model/SHAP.")

        print(
            f"Reason: {type(e).__name__}: {e}"
        )

        pipeline = None
        explainer = None


print("\n" + "=" * 60)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class CustomerData(BaseModel):

    tenure: float = Field(
        ...,
        example=4.0,
        description="Customer tenure in months"
    )

    monthly_charges: float = Field(
        ...,
        example=79.5,
        description="Monthly customer charges"
    )

    support_tickets: float = Field(
        ...,
        example=6.0,
        description="Number of support tickets"
    )

    days_since_last_login: float = Field(
        ...,
        example=21.0,
        description="Days since customer's last login"
    )

    contract_type: str = Field(
        ...,
        example="monthly",
        description="Customer contract type"
    )


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class PredictionResponse(BaseModel):

    churn_probability: float
    risk_score: int
    risk_level: str
    recommendation: str


# ============================================================
# ROOT / HEALTH CHECK
# ============================================================

@app.get("/")
def health_check():

    return {
        "status": "healthy",
        "service": "churn-prediction-api",
        "model_loaded": pipeline is not None,
        "shap_loaded": explainer is not None
    }


# ============================================================
# FAVICON
# ============================================================

@app.get(
    "/favicon.ico",
    include_in_schema=False
)
def favicon():

    # Browser asks for favicon.
    # We intentionally return 204 instead of 404.

    return Response(status_code=204)


# ============================================================
# PREDICT ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_churn(data: CustomerData):

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if pipeline is None:

        raise HTTPException(
            status_code=500,
            detail="Model pipeline is not loaded."
        )

    if explainer is None:

        raise HTTPException(
            status_code=500,
            detail="SHAP explainer is not loaded."
        )

    try:

        # ----------------------------------------------------
        # Convert request to DataFrame
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [data.model_dump()]
        )

        # ----------------------------------------------------
        # Predict churn probability
        # ----------------------------------------------------

        probabilities = pipeline.predict_proba(
            input_df
        )

        prob = float(
            probabilities[0][1]
        )

        # ----------------------------------------------------
        # Risk tier
        # ----------------------------------------------------

        risk_info = explainer.get_risk_tier(
            prob
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "churn_probability": round(
                prob,
                4
            ),

            "risk_score": int(
                risk_info["score"]
            ),

            "risk_level": risk_info["level"],

            "recommendation": risk_info["action"]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: "
                f"{type(e).__name__}: {e}"
            )
        )


# ============================================================
# EXPLAIN ENDPOINT
# ============================================================

@app.post("/explain")
def explain_churn(data: CustomerData):

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if pipeline is None:

        raise HTTPException(
            status_code=500,
            detail="Model pipeline is not loaded."
        )

    # --------------------------------------------------------
    # Check SHAP
    # --------------------------------------------------------

    if explainer is None:

        raise HTTPException(
            status_code=500,
            detail="SHAP explainer is not loaded."
        )

    try:

        # ----------------------------------------------------
        # Convert request to DataFrame
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [data.model_dump()]
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        probabilities = pipeline.predict_proba(
            input_df
        )

        prob = float(
            probabilities[0][1]
        )

        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        risk_info = explainer.get_risk_tier(
            prob
        )

        # ----------------------------------------------------
        # SHAP explanation
        # ----------------------------------------------------

        top_reasons = explainer.explain_instance(
            input_df
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {

            "churn_probability": round(
                prob,
                4
            ),

            "risk_score": int(
                risk_info["score"]
            ),

            "risk_level": risk_info["level"],

            "recommendation": risk_info["action"],

            "top_reasons": top_reasons
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Explanation failed: "
                f"{type(e).__name__}: {e}"
            )
        )
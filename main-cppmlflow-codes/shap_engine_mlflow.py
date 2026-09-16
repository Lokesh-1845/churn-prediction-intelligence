import joblib
import mlflow
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

X_PATH = ROOT / "src" / "models" / "transformed_features.npy"
Y_PATH = ROOT / "src" / "models" / "target.npy"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"

SHAP_IMAGE = ROOT / "shap_summary.png"

MLFLOW_URI = "http://127.0.0.1:5050"
EXPERIMENT = "Customer Churn Prediction"


# ============================================================
# MLFLOW SETUP
# ============================================================

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")

X = np.load(X_PATH)
y = np.load(Y_PATH, allow_pickle=True).ravel()

X = np.asarray(X, dtype=np.float32)

print(f"Features : {X.shape}")
print(f"Target   : {y.shape}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# SAMPLE DATA
# ============================================================

SAMPLE_SIZE = min(100, len(X))

X_sample = X[:SAMPLE_SIZE]

feature_names = [
    f"Feature_{i + 1}"
    for i in range(X.shape[1])
]

X_display = pd.DataFrame(
    X_sample,
    columns=feature_names
)


# ============================================================
# SHAP CALCULATION
# ============================================================

print("\nGenerating SHAP values...")

try:

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_sample
    )

    method = "TreeExplainer"

except Exception as e:

    print("\nTreeExplainer failed.")
    print("Using PermutationExplainer...")

    background = shap.sample(
        X_sample,
        min(20, len(X_sample))
    )

    explainer = shap.PermutationExplainer(
        model.predict_proba,
        background
    )

    shap_values = explainer(
        X_sample
    )

    method = "PermutationExplainer"


# ============================================================
# FORMAT SHAP VALUES
# ============================================================

if isinstance(shap_values, list):

    shap_values = shap_values[-1]


if hasattr(shap_values, "values"):

    shap_values = shap_values.values


if shap_values.ndim == 3:

    shap_values = shap_values[:, :, -1]


print(f"SHAP method: {method}")
print(f"SHAP shape : {shap_values.shape}")


# ============================================================
# SHAP IMPORTANCE
# ============================================================

importance = np.mean(
    np.abs(shap_values),
    axis=0
)

top_indices = np.argsort(
    importance
)[::-1][:10]


print("\nTop SHAP Features")
print("-" * 40)

for rank, index in enumerate(
    top_indices,
    start=1
):

    print(
        f"{rank}. "
        f"{feature_names[index]} : "
        f"{importance[index]:.4f}"
    )


# ============================================================
# CREATE SHAP IMAGE
# ============================================================

print("\nCreating SHAP image...")

plt.figure(
    figsize=(12, 8)
)

shap.summary_plot(
    shap_values,
    X_display,
    max_display=20,
    show=False
)

plt.title(
    "Customer Churn - SHAP Feature Importance"
)

plt.tight_layout()

plt.savefig(
    SHAP_IMAGE,
    dpi=150,
    bbox_inches="tight"
)

# ============================================================
# DISPLAY SHAP IMAGE
# ============================================================

print(f"\nSHAP image saved:")
print(SHAP_IMAGE)

plt.show()

plt.close()


# ============================================================
# MLFLOW LOGGING
# ============================================================

print("\nLogging SHAP results to MLflow...")

with mlflow.start_run(
    run_name="SHAP_Explainability"
) as run:

    mlflow.log_params({

        "model": "XGBoost",
        "explainer": method,
        "sample_size": SAMPLE_SIZE,
        "features": X.shape[1]

    })

    mlflow.log_metrics({

        "average_shap_importance":
            float(np.mean(importance)),

        "maximum_shap_importance":
            float(np.max(importance)),

        "minimum_shap_importance":
            float(np.min(importance))

    })

    mlflow.log_artifact(
        str(SHAP_IMAGE),
        artifact_path="shap"
    )

    mlflow.log_dict(
        {
            "explainer": method,
            "sample_size": SAMPLE_SIZE,
            "top_features": [
                {
                    "feature":
                        feature_names[i],

                    "importance":
                        float(importance[i])
                }
                for i in top_indices
            ]
        },
        "shap/shap_summary.json"
    )

    mlflow.set_tags({

        "task": "customer_churn",
        "explainability": "SHAP",
        "status": "completed"

    })

    print("\n" + "=" * 45)
    print("SHAP + MLFLOW COMPLETED")
    print("=" * 45)

    print(f"Run ID : {run.info.run_id}")

    print(
        f"MLflow : {MLFLOW_URI}"
    )

    print(
        f"Image  : {SHAP_IMAGE}"
    )

    print("=" * 45)
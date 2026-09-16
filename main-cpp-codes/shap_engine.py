import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# Paths
ROOT = Path(__file__).resolve().parents[2]

X_PATH = ROOT / "src" / "models" / "transformed_features.npy"
Y_PATH = ROOT / "src" / "models" / "target.npy"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"


class ChurnExplainer:

    def __init__(self, model):
        self.model = model

    def get_risk_tier(self, probability):
        score = int(round(probability * 100))

        if score <= 30:
            level = "LOW"
            action = "Standard customer engagement."
        elif score <= 60:
            level = "MEDIUM"
            action = "Monitor customer engagement."
        elif score <= 80:
            level = "HIGH"
            action = "Offer retention benefits."
        else:
            level = "CRITICAL"
            action = "Contact customer immediately."

        return {
            "risk_score": score,
            "risk_level": level,
            "recommended_action": action
        }

    def explain(self, X, top_n=10):

        X_sample = X[:min(100, len(X))]

        try:
            explainer = shap.TreeExplainer(self.model)
            values = explainer.shap_values(X_sample)
            method = "TreeExplainer"

        except Exception:
            print("TreeExplainer failed. Using PermutationExplainer.")

            background = shap.sample(
                X_sample,
                min(20, len(X_sample))
            )

            explainer = shap.PermutationExplainer(
                self.model.predict_proba,
                background
            )

            values = explainer(X_sample)
            method = "PermutationExplainer"

        if isinstance(values, list):
            values = values[-1]

        if hasattr(values, "values"):
            values = values.values

        if values.ndim == 3:
            values = values[:, :, -1]

        feature_names = [
            f"Feature_{i + 1}"
            for i in range(X.shape[1])
        ]

        shap_df = pd.DataFrame(
            X_sample,
            columns=feature_names
        )

        # SHAP image
        image_path = ROOT / "shap_summary.png"

        plt.figure(figsize=(12, 8))

        shap.summary_plot(
            values,
            shap_df,
            max_display=20,
            show=False
        )

        plt.title("Customer Churn - SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(
            image_path,
            dpi=150,
            bbox_inches="tight"
        )

        # Display image
        plt.show()
        plt.close()

        # Top features
        importance = np.mean(
            np.abs(values),
            axis=0
        )

        top_indices = np.argsort(
            importance
        )[::-1][:top_n]

        results = []

        for i in top_indices:
            results.append({
                "feature": feature_names[i],
                "shap_value": float(importance[i])
            })

        return results, method, image_path


def main():

    print("\n" + "=" * 50)
    print("       CUSTOMER CHURN - SHAP EXPLANATION")
    print("=" * 50)

    # Load data
    X = np.load(X_PATH)
    y = np.load(
        Y_PATH,
        allow_pickle=True
    ).ravel()

    X = np.asarray(
        X,
        dtype=np.float32
    )

    print(f"\nFeature shape : {X.shape}")
    print(f"Target shape  : {y.shape}")

    # Load model
    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")

    # Select customer
    customer = X[:1]

    probability = float(
        model.predict_proba(customer)[0, 1]
    )

    prediction = int(
        probability >= 0.5
    )

    print("\nPrediction")
    print("-" * 50)
    print(
        f"Churn Probability : "
        f"{probability:.4f} "
        f"({probability * 100:.2f}%)"
    )

    print(
        f"Prediction        : "
        f"{'CHURN' if prediction else 'NO CHURN'}"
    )

    # Risk
    explainer = ChurnExplainer(model)

    risk = explainer.get_risk_tier(
        probability
    )

    print("\nRisk Assessment")
    print("-" * 50)
    print(f"Risk Score        : {risk['risk_score']}")
    print(f"Risk Level        : {risk['risk_level']}")
    print(f"Recommended Action: {risk['recommended_action']}")

    # SHAP
    print("\nGenerating SHAP explanation...")

    results, method, image_path = explainer.explain(
        X,
        top_n=10
    )

    print("\nTop SHAP Features")
    print("-" * 50)

    for i, item in enumerate(results, 1):
        print(
            f"{i}. {item['feature']} "
            f"-> {item['shap_value']:.4f}"
        )

    print("\n" + "=" * 50)
    print("SHAP COMPLETED")
    print("=" * 50)
    print(f"Explainer : {method}")
    print(f"SHAP Image: {image_path}")


if __name__ == "__main__":
    main()
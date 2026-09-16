import pandas as pd
import numpy as np
import shap
from xgboost import XGBClassifier

class ChurnExplainer:
    def __init__(self, pipeline):
        """
        Extracts model and preprocessor from the production pipeline to initialize SHAP.
        """
        self.pipeline = pipeline
        self.model = pipeline.named_steps['classifier'] if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps else pipeline
        self.preprocessor = pipeline.named_steps['preprocessor'] if hasattr(pipeline, 'named_steps') and 'preprocessor' in pipeline.named_steps else None

        # Use raw booster to avoid string parsing issues with SHAP
        if hasattr(self.model, "get_booster"):
            booster = self.model.get_booster()

            # Fix base_score issue by converting it to a float
            learner_params = booster.attributes()
            if "base_score" in learner_params:
                try:
                    base_score = learner_params["base_score"]
                    booster.set_attr(base_score=str(float(base_score.strip("[]"))))
                except ValueError:
                    print(f"Warning: Unable to convert base_score '{base_score}' to float.")

            self.explainer = shap.TreeExplainer(booster)
        else:
            self.explainer = shap.TreeExplainer(self.model)

    def get_risk_tier(self, churn_prob: float) -> dict:
        """
        Converts probability (0.0 to 1.0) into a structured Risk Score and Level.
        """
        score = int(np.round(churn_prob * 100))
        
        if score <= 30:
            level = "LOW"
            action = "No immediate action needed. Include in standard engagement campaigns."
        elif score <= 60:
            level = "MEDIUM"
            action = "Monitor engagement. Send targeted feature tips or data upgrade bundles."
        elif score <= 80:
            level = "HIGH"
            action = "High churn risk. Offer 15% contract renewal discount within 7 days."
        else:
            level = "CRITICAL"
            action = "Contact customer within 48 hours. Schedule account consultation & offer annual plan discount."

        return {
            "risk_score": score,
            "risk_level": level,
            "recommended_action": action
        }

    def explain_instance(self, input_df: pd.DataFrame, top_n: int = 5) -> list:
        """
        Calculates local SHAP feature contributions for a single customer input.
        """
        # Preprocess input if preprocessor step exists
        if self.preprocessor is not None:
            processed_data = self.preprocessor.transform(input_df)
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                feature_names = self.preprocessor.get_feature_names_out()
            else:
                feature_names = input_df.columns
        else:
            processed_data = input_df
            feature_names = input_df.columns

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(processed_data)
        
        # Handle binary classification SHAP dimensions
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 2:
            sv = shap_values[0]
        else:
            sv = shap_values

        # Map features to contribution percentages
        total_impact = np.sum(np.abs(sv)) + 1e-9
        contributions = []
        
        for feat, val in zip(feature_names, sv):
            impact_pct = int(np.round((abs(val) / total_impact) * 100))
            direction = "↑" if val > 0 else "↓"
            contributions.append({
                "feature": feat,
                "impact_pct": impact_pct,
                "direction": direction,
                "shap_value": float(val),
                "summary": f"{direction} {impact_pct}% — {feat}"
            })

        # Sort by impact severity
        contributions = sorted(contributions, key=lambda x: abs(x["shap_value"]), reverse=True)
        return contributions[:top_n]


# ----------------------------------------------------------------------------- #
# DIRECT EXECUTION TEST WITH PRINT STATEMENTS                                   #
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("\n--- 1. Setting up Mock Dataset & Model ---")
    mock_data = pd.DataFrame({
        "tenure": [1, 2, 3, 4, 5],
        "monthly_charges": [50, 60, 70, 80, 90],
        "support_tickets": [1, 2, 3, 4, 5],
        "days_since_last_login": [10, 20, 30, 40, 50]
    })
    mock_labels = [0, 1, 0, 1, 0]

    model = XGBClassifier()
    model.fit(mock_data, mock_labels)
    print("Model successfully trained on sample data.")

    print("\n--- 2. Initializing ChurnExplainer ---")
    explainer = ChurnExplainer(pipeline=model)

    test_customer = pd.DataFrame([{
        "tenure": 4,
        "monthly_charges": 79.5,
        "support_tickets": 6,
        "days_since_last_login": 21
    }])
    
    churn_prob = float(model.predict_proba(test_customer)[:, 1][0])
    print(f"Predicted Churn Probability: {churn_prob:.4f} ({churn_prob * 100:.1f}%)")

    print("\n--- 3. Testing Risk Tier Output ---")
    risk_info = explainer.get_risk_tier(churn_prob)
    print(f"Risk Score          : {risk_info['risk_score']}")
    print(f"Risk Level          : {risk_info['risk_level']}")
    print(f"Recommended Action  : {risk_info['recommended_action']}")

    print("\n--- 4. Testing Local SHAP Explanations ---")
    reasons = explainer.explain_instance(test_customer, top_n=5)
    print("Top Feature Drivers:")
    for idx, item in enumerate(reasons, start=1):
        print(f"  {idx}. {item['summary']} (SHAP value: {item['shap_value']:+.4f})")
    print("\nExecution completed successfully!")
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from pathlib import Path


class ChurnExplainer:

    def __init__(
        self,
        model,
        preprocessor
    ):

        self.model = model
        self.preprocessor = preprocessor

    # ========================================================
    # RISK
    # ========================================================

    @staticmethod
    def get_risk_tier(
        probability
    ):

        score = int(
            round(
                probability * 100
            )
        )

        if score <= 30:

            return {
                "risk_score": score,
                "risk_level": "LOW",
                "recommended_action":
                    "Standard customer engagement."
            }

        if score <= 60:

            return {
                "risk_score": score,
                "risk_level": "MEDIUM",
                "recommended_action":
                    "Monitor customer engagement."
            }

        if score <= 80:

            return {
                "risk_score": score,
                "risk_level": "HIGH",
                "recommended_action":
                    "Offer retention benefits."
            }

        return {
            "risk_score": score,
            "risk_level": "CRITICAL",
            "recommended_action":
                "Contact customer immediately."
        }

    # ========================================================
    # FEATURE NAMES
    # ========================================================

    def get_feature_names(self):

        try:

            return list(
                self.preprocessor
                .get_feature_names_out()
            )

        except Exception:

            transformed_size = (
                self.preprocessor
                .transform(
                    pd.DataFrame(
                        [
                            {
                                col: 0
                                for col
                                in self.preprocessor
                                .feature_names_in_
                            }
                        ]
                    )
                )
                .shape[1]
            )

            return [
                f"Feature_{i + 1}"
                for i in range(
                    transformed_size
                )
            ]

    # ========================================================
    # EXPLAIN
    # ========================================================

    def explain(
        self,
        X_raw: pd.DataFrame,
        top_n=10
    ):

        X_transformed = (
            self.preprocessor.transform(
                X_raw
            )
        )

        feature_names = (
            self.get_feature_names()
        )

        X_transformed = np.asarray(
            X_transformed,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Try TreeExplainer
        # ----------------------------------------------------

        try:

            explainer = shap.TreeExplainer(
                self.model
            )

            values = explainer.shap_values(
                X_transformed
            )

            method = "TreeExplainer"

        except Exception:

            background = shap.sample(
                X_transformed,
                min(
                    20,
                    len(X_transformed)
                )
            )

            explainer = (
                shap.PermutationExplainer(
                    self.model.predict_proba,
                    background
                )
            )

            explanation = explainer(
                X_transformed
            )

            values = explanation.values

            method = "PermutationExplainer"

        if isinstance(
            values,
            list
        ):

            values = values[-1]

        if hasattr(
            values,
            "values"
        ):

            values = values.values

        values = np.asarray(
            values
        )

        if values.ndim == 3:

            values = values[:, :, -1]

        # ----------------------------------------------------
        # One customer
        # ----------------------------------------------------

        customer_values = values[0]

        importance = np.abs(
            customer_values
        )

        top_indices = np.argsort(
            importance
        )[::-1][:top_n]

        results = []

        for index in top_indices:

            results.append(
                {
                    "feature":
                        feature_names[index],

                    "shap_value":
                        float(
                            customer_values[index]
                        ),

                    "impact":
                        (
                            "Increases churn risk"
                            if customer_values[index] > 0
                            else
                            "Decreases churn risk"
                        )
                }
            )

        return results, method
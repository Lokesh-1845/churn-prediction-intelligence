from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "models"
    / "churn_pipeline.joblib"
)


class ChurnPredictor:

    def __init__(
        self,
        model_path=MODEL_PATH
    ):

        artifact = joblib.load(
            model_path
        )

        self.model = artifact["model"]
        self.preprocessor = artifact[
            "preprocessor"
        ]

        self.model_input_columns = (
            artifact["model_input_columns"]
        )

        self.metrics = artifact.get(
            "metrics",
            {}
        )

    # ========================================================
    # TRANSFORM
    # ========================================================

    def transform(
        self,
        X: pd.DataFrame
    ):

        X = X[
            self.model_input_columns
        ].copy()

        return self.preprocessor.transform(
            X
        )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        X: pd.DataFrame
    ):

        transformed = self.transform(
            X
        )

        probability = (
            self.model
            .predict_proba(
                transformed
            )[:, 1]
        )

        prediction = (
            probability >= 0.5
        ).astype(int)

        return pd.DataFrame(
            {
                "churn_probability":
                    probability,

                "churn_prediction":
                    prediction
            },
            index=X.index
        )
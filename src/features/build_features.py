from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer


class UnknownMarkerCleaner(
    BaseEstimator,
    TransformerMixin
):

    def __init__(self, markers=None):

        self.markers = markers or [
            "Unknown",
            "unknown",
            "?",
            "NA",
            "na",
            "null",
            "NULL",
            "none",
            "None",
        ]

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X_out = pd.DataFrame(
            X
        ).copy()

        for column in X_out.columns:

            if (
                X_out[column].dtype == "object"
                or pd.api.types.is_string_dtype(
                    X_out[column]
                )
            ):

                X_out[column] = (
                    X_out[column]
                    .replace(
                        self.markers,
                        np.nan
                    )
                )

        return X_out


def build_preprocessing_pipeline(
    num_cols,
    cat_cols
):

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "cleaner",
                UnknownMarkerCleaner()
            ),
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                num_cols
            ),
            (
                "cat",
                categorical_pipeline,
                cat_cols
            ),
        ],
        remainder="drop"
    )

    return preprocessor
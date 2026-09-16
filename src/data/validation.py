from __future__ import annotations

from typing import List, Tuple

import pandas as pd


class DataValidation:

    def __init__(
        self,
        required_columns: List[str]
    ):

        self.required_columns = [
            str(col).strip().lower()
            for col in required_columns
        ]

    # ========================================================
    # SCHEMA VALIDATION
    # ========================================================

    def validate_schema(
        self,
        df: pd.DataFrame
    ) -> Tuple[bool, List[str]]:

        df_columns = {
            str(col).strip().lower()
            for col in df.columns
        }

        missing_columns = [
            col
            for col in self.required_columns
            if col not in df_columns
        ]

        return (
            len(missing_columns) == 0,
            missing_columns
        )

    # ========================================================
    # BASIC DATA VALIDATION
    # ========================================================

    def validate_data(
        self,
        df: pd.DataFrame
    ) -> dict:

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "duplicate_rows": int(
                df.duplicated().sum()
            ),
            "missing_cells": int(
                df.isna().sum().sum()
            ),
            "empty_columns": [
                col
                for col in df.columns
                if df[col].isna().all()
            ],
        }
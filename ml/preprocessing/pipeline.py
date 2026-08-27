"""
ML Preprocessing Pipeline Module
Constructs feature engineering and encoding pipelines for tabular vehicle data.
"""
from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

CATEGORICAL_FEATURES = ["make", "model", "fuel_type", "transmission", "condition", "district"]
NUMERICAL_FEATURES = ["yom", "mileage_km", "vehicle_age", "log_mileage", "engine_cc"]

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Calculates derived features such as vehicle age and log mileage."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        # Vehicle Age
        if "yom" in X_df.columns:
            X_df["vehicle_age"] = 2024 - pd.to_numeric(X_df["yom"], errors="coerce").fillna(2015)
        else:
            X_df["vehicle_age"] = 9.0

        # Log Mileage
        if "mileage_km" in X_df.columns:
            m = pd.to_numeric(X_df["mileage_km"], errors="coerce").fillna(60000.0)
            X_df["mileage_km"] = m
            X_df["log_mileage"] = np.log1p(np.maximum(0, m))
        else:
            X_df["mileage_km"] = 60000.0
            X_df["log_mileage"] = np.log1p(60000.0)

        # Engine cc
        if "engine_cc" in X_df.columns:
            X_df["engine_cc"] = pd.to_numeric(X_df["engine_cc"], errors="coerce").fillna(1500.0)
        else:
            X_df["engine_cc"] = 1500.0

        for col in CATEGORICAL_FEATURES:
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna("Unknown").astype(str)
            else:
                X_df[col] = "Unknown"

        return X_df

class VehiclePreprocessor:
    @staticmethod
    def build_pipeline() -> Tuple[Pipeline, ColumnTransformer]:
        """Constructs a scikit-learn ColumnTransformer pipeline for vehicle data."""
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        num_transformer = StandardScaler()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, NUMERICAL_FEATURES),
                ("cat", cat_transformer, CATEGORICAL_FEATURES)
            ],
            remainder="drop"
        )

        full_pipeline = Pipeline([
            ("feature_engineer", FeatureEngineer()),
            ("column_transform", preprocessor)
        ])

        return full_pipeline, preprocessor

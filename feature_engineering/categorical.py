"""
Categorical Feature Engineering and Rare Category Handling.
Provides scikit-learn compatible transformers for grouping low-frequency
categories (brand, model) and engineering domain-justified interaction features.
"""

import logging
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

DEFAULT_CATEGORICAL_COLUMNS = [
    "category",
    "brand",
    "model",
    "fuel_type",
    "transmission",
    "district",
    "condition",
]

DEFAULT_RARE_COLUMNS = ["brand", "model"]
DEFAULT_MIN_FREQUENCY = 5
OTHER_LABEL = "Other"


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that replaces low-frequency category
    values with a designated 'Other' label.
    
    Guarantees:
    1. Fits ONLY on training data to prevent test/validation leakage.
    2. Does not drop rare luxury or exotic records, only collapses rare levels.
    3. Handles unknown categories encountered at inference time gracefully.
    4. Deterministic sorting and frequency calculation.
    """

    def __init__(
        self,
        min_frequency: int = DEFAULT_MIN_FREQUENCY,
        columns: Optional[List[str]] = None,
        other_label: str = OTHER_LABEL,
    ):
        self.min_frequency = min_frequency
        self.columns = columns or DEFAULT_RARE_COLUMNS
        self.other_label = other_label
        self.frequent_categories_: Dict[str, Set[str]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learns frequent categories with count >= min_frequency on training split.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.frequent_categories_ = {}

        for col in self.columns:
            if col in X.columns:
                value_counts = X[col].dropna().astype(str).value_counts()
                frequent = set(value_counts[value_counts >= self.min_frequency].index)
                self.frequent_categories_[col] = frequent
                logger.debug(
                    f"RareCategoryGrouper: column '{col}' retained {len(frequent)} frequent levels "
                    f"(min_frequency={self.min_frequency})."
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Maps categories not in frequent_categories_ to other_label.
        Returns a modified copy of X without mutating the input.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()

        for col, frequent in self.frequent_categories_.items():
            if col in X_out.columns:
                # Map non-null, non-frequent values to other_label
                series = X_out[col].astype(str)
                # Keep original nulls if any, replace others not frequent
                is_null = X_out[col].isna()
                masked = series.apply(lambda val: val if val in frequent else self.other_label)
                if is_null.any():
                    masked[is_null] = np.nan
                X_out[col] = masked

        return X_out


class CategoricalFeatureEngineer:
    """
    Manages categorical feature generation and domain interactions.
    
    Justified Interactions:
    - brand_model: 'brand' + '_' + 'model'
      Sri Lankan vehicle valuations are heavily stratified by brand and model
      (e.g., Toyota Corolla vs Toyota Land Cruiser).
    """

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        include_interactions: bool = True,
        min_frequency: int = DEFAULT_MIN_FREQUENCY,
    ):
        self.categorical_columns = categorical_columns or list(DEFAULT_CATEGORICAL_COLUMNS)
        self.include_interactions = include_interactions
        self.min_frequency = min_frequency

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives interaction features on a copy of DataFrame.
        """
        df_out = df.copy()

        if self.include_interactions:
            if "brand" in df_out.columns and "model" in df_out.columns:
                brand_s = df_out["brand"].fillna("Unknown").astype(str)
                model_s = df_out["model"].fillna("Unknown").astype(str)
                df_out["brand_model"] = brand_s + "_" + model_s

        return df_out

    def get_active_categorical_features(self) -> List[str]:
        """
        Returns the active list of categorical features for ML preprocessors.
        """
        cols = list(self.categorical_columns)
        if self.include_interactions and "brand_model" not in cols:
            cols.append("brand_model")
        return cols

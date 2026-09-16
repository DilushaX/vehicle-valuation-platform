"""
Target Variable Preparation and Transformation.
Provides target extraction, validation, and transformations (raw vs log1p)
for seller asking prices in the Sri Lankan vehicle market.

IMPORTANT:
The target variable is ALWAYS the observed seller asking/advertised price,
NOT the confirmed transaction or final sold price.
"""

import logging
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TargetTransformType = Literal["raw", "log1p"]


class TargetTransformer:
    """
    Target transformer and validator for vehicle asking prices.
    
    Guarantees:
    1. Rejects or excludes non-positive, NaN, infinite, or non-numeric prices.
    2. Never converts missing price to zero.
    3. Supports invertible transformations: 'raw' and 'log1p'.
    4. Explicitly documents the target as seller asking price.
    """

    def __init__(self, transform: TargetTransformType = "raw"):
        if transform not in ("raw", "log1p"):
            raise ValueError(f"Unsupported target transform '{transform}'. Must be 'raw' or 'log1p'.")
        self.transform_type = transform

    def transform(self, y: Union[pd.Series, np.ndarray, list]) -> np.ndarray:
        """
        Transforms asking price series or array according to configured policy.
        
        Args:
            y: Numeric target array or Series.
            
        Returns:
            Transformed numpy array.
        """
        y_arr = np.asarray(y, dtype=np.float64)

        if np.any(np.isnan(y_arr)):
            raise ValueError("Target contains NaN values. Cannot transform target with missing values.")
        if np.any(y_arr <= 0):
            raise ValueError("Target contains non-positive asking prices (<= 0). Prices must be strictly positive.")

        if self.transform_type == "raw":
            return y_arr.copy()
        elif self.transform_type == "log1p":
            return np.log1p(y_arr)
        else:
            raise ValueError(f"Unknown transform type '{self.transform_type}'")

    def inverse_transform(self, y_transformed: Union[pd.Series, np.ndarray, list]) -> np.ndarray:
        """
        Inverts transformed values back to the original asking price scale (LKR).
        
        Args:
            y_transformed: Array or Series of transformed predictions or targets.
            
        Returns:
            Numpy array on original raw asking price scale.
        """
        arr = np.asarray(y_transformed, dtype=np.float64)

        if self.transform_type == "raw":
            return arr.copy()
        elif self.transform_type == "log1p":
            return np.expm1(arr)
        else:
            raise ValueError(f"Unknown transform type '{self.transform_type}'")

    @staticmethod
    def extract_valid_target(
        df: pd.DataFrame,
        target_col: str = "asking_price",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Safely extracts valid target records from DataFrame without mutating original DataFrame.
        Rows with missing, non-numeric, or non-positive asking prices are dropped.
        
        Args:
            df: Input DataFrame.
            target_col: Name of target column (default 'asking_price').
            
        Returns:
            Tuple of (df_valid, y_series)
        """
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

        df_copy = df.copy()

        # Convert to numeric, coercing errors to NaN
        prices = pd.to_numeric(df_copy[target_col], errors="coerce")

        # Valid mask: not null, finite, strictly > 0
        valid_mask = prices.notna() & np.isfinite(prices) & (prices > 0)
        invalid_count = int((~valid_mask).sum())

        if invalid_count > 0:
            logger.warning(
                f"Dropping {invalid_count} records with invalid/missing asking_price. "
                "Missing prices are NEVER imputed to zero."
            )

        df_valid = df_copy[valid_mask].copy()
        y_valid = prices[valid_mask].astype(np.float64)

        return df_valid, y_valid

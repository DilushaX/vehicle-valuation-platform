"""
Feature Leakage Protection and Validation.
Provides comprehensive validation checks to strictly prevent target leakage,
lifecycle outcome leakage, future information leakage, and seller contact leakage
from entering the ML feature matrix.
"""

import logging
import re
from typing import List, Optional, Sequence, Set

import pandas as pd

logger = logging.getLogger(__name__)


class DataLeakageError(ValueError):
    """Raised when data leakage or forbidden columns are detected in feature matrix."""
    pass


# Explicit blacklist of column names and patterns forbidden from entering ML feature matrix X
FORBIDDEN_EXACT_COLUMNS: Set[str] = {
    # Target and price representations
    "asking_price",
    "price",
    "target",
    "y",
    "observed_price",
    "log_asking_price",
    "asking_price_log1p",
    "log_price",
    # Lifecycle status and post-event outcomes
    "current_status",
    "status",
    "sold",
    "is_sold",
    "sold_at",
    "delisted_at",
    # Observation aggregations and counts (historical leakage)
    "observation_count",
    "price_history_count",
    "price_history",
    "observations",
    # Data quality / pipeline metadata
    "quality_score",
    "validation_issues",
    "ml_eligible",
    # Seller private contact information
    "phone",
    "telephone",
    "mobile",
    "seller_phone",
    "email",
    "seller_email",
    "contact",
    "contact_number",
    # Raw descriptions and free text that may leak price or contact info
    "description",
    "title",
}

FORBIDDEN_PATTERNS = [
    re.compile(r"^price", re.IGNORECASE),
    re.compile(r"_price$", re.IGNORECASE),
    re.compile(r"target", re.IGNORECASE),
    re.compile(r"^phone", re.IGNORECASE),
    re.compile(r"^email", re.IGNORECASE),
    re.compile(r"status$", re.IGNORECASE),
]


class LeakageValidator:
    """
    Validates that feature sets and matrices are completely free of data leakage.
    
    Guarantees:
    1. Target variable (asking_price) cannot appear in X.
    2. Lifecycle outcome fields (current_status) cannot appear in X.
    3. Future observation info cannot appear in X.
    4. Private seller contact information cannot appear in X.
    5. Derived price features cannot appear in X.
    """

    def __init__(self, additional_forbidden: Optional[Sequence[str]] = None):
        self.forbidden_columns = set(FORBIDDEN_EXACT_COLUMNS)
        if additional_forbidden:
            self.forbidden_columns.update(additional_forbidden)

    def validate_features(self, df_or_columns: Sequence[str] | pd.DataFrame) -> None:
        """
        Validates feature columns against forbidden leakage patterns.
        
        Args:
            df_or_columns: DataFrame or list of column names to validate.
            
        Raises:
            DataLeakageError: If any forbidden column or pattern is detected.
        """
        if isinstance(df_or_columns, pd.DataFrame):
            columns = list(df_or_columns.columns)
        else:
            columns = list(df_or_columns)

        leaked_columns: List[str] = []

        for col in columns:
            col_clean = str(col).strip().lower()

            # Exact match check
            if col_clean in self.forbidden_columns:
                leaked_columns.append(col)
                continue

            # Pattern check
            for pattern in FORBIDDEN_PATTERNS:
                if pattern.search(col_clean):
                    leaked_columns.append(col)
                    break

        if leaked_columns:
            raise DataLeakageError(
                f"Data leakage detected! Forbidden columns found in feature set: {leaked_columns}. "
                "Target variables, seller contact info, and lifecycle outcomes must NEVER enter ML features."
            )

    def validate_matrix_against_target(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_name: str = "asking_price",
    ) -> None:
        """
        Validates that X and y are strictly decoupled and no column in X is an exact clone of y.
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            target_name: Name of target column.
            
        Raises:
            DataLeakageError: If leakage or direct target duplication is detected.
        """
        self.validate_features(X)

        if target_name in X.columns:
            raise DataLeakageError(f"Target '{target_name}' is present inside feature matrix X!")

        # Check for numeric clone of target in X
        for col in X.select_dtypes(include=["number"]).columns:
            if len(X[col]) == len(y):
                # If non-null values are identical
                valid_both = X[col].notna() & y.notna()
                if valid_both.sum() > 5 and (X.loc[valid_both, col] == y[valid_both]).all():
                    raise DataLeakageError(
                        f"Numeric feature '{col}' is an exact copy of target variable y! "
                        "Direct target leakage is prohibited."
                    )

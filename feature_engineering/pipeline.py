"""
Train/Test Preprocessor Architecture and Feature Pipeline.
Provides an unfitted, reusable, scikit-learn compatible ColumnTransformer
and feature preparation coordinator that guarantees zero leakage between
dataset construction and future model training.
"""

from dataclasses import dataclass, field
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_engineering.categorical import (
    CategoricalFeatureEngineer,
    DEFAULT_CATEGORICAL_COLUMNS,
    DEFAULT_MIN_FREQUENCY,
    RareCategoryGrouper,
)
from feature_engineering.numerical import (
    DEFAULT_REFERENCE_YEAR,
    AgeRepresentationType,
    MileageTransformType,
    NumericalFeatureEngineer,
)
from feature_engineering.target import TargetTransformer, TargetTransformType
from feature_engineering.validation import LeakageValidator

logger = logging.getLogger(__name__)


@dataclass
class FeaturePipelineConfig:
    """
    Configuration parameters for feature engineering and preprocessor pipelines.
    """
    target_transform: TargetTransformType = "raw"
    age_representation: AgeRepresentationType = "vehicle_age"
    reference_year: int = DEFAULT_REFERENCE_YEAR
    include_registration_year: bool = False
    mileage_transform: MileageTransformType = "none"
    include_interactions: bool = True
    min_frequency: int = DEFAULT_MIN_FREQUENCY
    categorical_columns: List[str] = field(default_factory=lambda: list(DEFAULT_CATEGORICAL_COLUMNS))


class FeaturePipeline:
    """
    Coordinates dataset validation, feature derivation, and preprocessor construction.
    
    Guarantees:
    1. The ColumnTransformer is returned UNFITTED; it is fitted strictly on training data
       during future ML training steps.
    2. Input DataFrames are never mutated.
    3. Target is strictly separated from X.
    4. Full leakage validation runs on both candidate and engineered features.
    5. Traceability columns (listing_id, vehicle_id, manufacture_year) are preserved
       in a metadata artifact.
    """

    def __init__(self, config: Optional[FeaturePipelineConfig] = None):
        self.config = config or FeaturePipelineConfig()
        self.numerical_engineer = NumericalFeatureEngineer(
            reference_year=self.config.reference_year,
            age_representation=self.config.age_representation,
            include_registration_year=self.config.include_registration_year,
            mileage_transform=self.config.mileage_transform,
        )
        self.categorical_engineer = CategoricalFeatureEngineer(
            categorical_columns=self.config.categorical_columns,
            include_interactions=self.config.include_interactions,
            min_frequency=self.config.min_frequency,
        )
        self.target_transformer = TargetTransformer(transform=self.config.target_transform)
        self.leakage_validator = LeakageValidator()

    def build_preprocessor(self) -> ColumnTransformer:
        """
        Builds an UNFITTED scikit-learn ColumnTransformer.
        
        Structure:
        - Numerical: SimpleImputer(strategy='median')
        - Categorical: SimpleImputer(strategy='constant', fill_value='missing')
                     -> RareCategoryGrouper(min_frequency=config.min_frequency)
                     -> OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        """
        num_features = self.numerical_engineer.get_active_numerical_features()
        cat_features = self.categorical_engineer.get_active_categorical_features()

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "rare_grouper",
                    RareCategoryGrouper(
                        min_frequency=self.config.min_frequency,
                        columns=["brand", "model", "brand_model"] if self.config.include_interactions else ["brand", "model"],
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, num_features),
                ("cat", cat_pipeline, cat_features),
            ],
            remainder="drop",
        )

        return preprocessor

    def prepare_features(
        self,
        raw_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Prepares raw dataset into:
        1. Feature DataFrame X (with derived features, verified against leakage)
        2. Target Series y (raw or log1p per config)
        3. Traceability DataFrame meta (listing_id, vehicle_id, raw manufacture_year, raw asking_price)
        
        Args:
            raw_df: Raw DataFrame loaded from PostgreSQL.
            
        Returns:
            Tuple of (X, y, meta)
        """
        if raw_df.empty:
            raise ValueError("Input raw_df is empty. Cannot prepare features from empty dataset.")

        # 1. Target extraction and validation (non-mutating, drops invalid prices)
        df_valid, y_raw = TargetTransformer.extract_valid_target(raw_df, target_col="asking_price")

        if df_valid.empty:
            raise ValueError("No valid records remain after target validation. All asking prices were missing or invalid.")

        # Transform target per configuration (raw vs log1p)
        y_trans = pd.Series(
            self.target_transformer.transform(y_raw),
            index=df_valid.index,
            name="asking_price" if self.config.target_transform == "raw" else "asking_price_log1p",
        )

        # 2. Preserve traceability metadata
        trace_cols = [c for c in ["listing_id", "vehicle_id", "manufacture_year", "first_seen_at"] if c in df_valid.columns]
        meta = df_valid[trace_cols].copy()
        meta["raw_asking_price"] = y_raw

        # 3. Apply numerical feature engineering
        df_engineered = self.numerical_engineer.engineer_features(df_valid)

        # 4. Apply categorical feature engineering (interactions)
        df_engineered = self.categorical_engineer.engineer_features(df_engineered)

        # 5. Extract active feature columns for X
        active_num = self.numerical_engineer.get_active_numerical_features()
        active_cat = self.categorical_engineer.get_active_categorical_features()
        active_cols = active_num + active_cat

        missing_cols = [c for c in active_cols if c not in df_engineered.columns]
        if missing_cols:
            raise KeyError(f"Expected engineered columns not found in DataFrame: {missing_cols}")

        X = df_engineered[active_cols].copy()

        # 6. Comprehensive Leakage Validation
        self.leakage_validator.validate_matrix_against_target(X, y_raw)

        return X, y_trans, meta

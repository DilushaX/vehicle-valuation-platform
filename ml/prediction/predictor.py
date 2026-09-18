"""
Vehicle Price Predictor.
Loads trained valuation model pipelines and provides safe, schema-validated
inference on vehicle listings in original Sri Lankan Rupees (LKR).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from data_pipeline.cleaning.cleaners import VehicleCleaner
from feature_engineering.numerical import DEFAULT_REFERENCE_YEAR
from feature_engineering.validation import DataLeakageError, LeakageValidator

CANONICAL_CATEGORIES = set(VehicleCleaner.CANONICAL_CATEGORIES.values())

logger = logging.getLogger(__name__)

REQUIRED_CATEGORICAL_COLUMNS = [
    "category",
    "brand",
    "model",
    "fuel_type",
    "transmission",
    "district",
    "condition",
]

NUMERICAL_FEATURE_COLUMNS = [
    "vehicle_age",
    "mileage",
    "engine_cc",
]


class ValidationError(ValueError):
    """Raised when input features fail schema, range, or categorical validation."""
    pass


class VehiclePricePredictor:
    """
    Inference interface for vehicle valuation models.
    
    Guarantees:
    1. Validates required input features against expected schema.
    2. Enforces zero target leakage (no asking_price or price features in input).
    3. Handles unknown categories gracefully without pipeline crashes.
    4. Handles missing numerical values (mileage, engine CC) via training imputation.
    5. Derives vehicle_age from manufacture_year if vehicle_age is omitted.
    6. Outputs strictly positive prices in original LKR scale.
    """

    def __init__(
        self,
        model: BaseEstimator,
        metadata: Optional[Dict[str, Any]] = None,
        reference_year: int = DEFAULT_REFERENCE_YEAR,
    ):
        self.model = model
        self.metadata = metadata or {}
        self.reference_year = reference_year
        self.leakage_validator = LeakageValidator()

    @classmethod
    def load(
        cls,
        model_path: Union[str, Path] = "data/analysis/ml/models/model.joblib",
        metadata_path: Optional[Union[str, Path]] = "data/analysis/ml/models/model_metadata.json",
        reference_year: int = DEFAULT_REFERENCE_YEAR,
    ) -> "VehiclePricePredictor":
        """Loads serialized model artifact and optional metadata."""
        mp = Path(model_path)
        if not mp.exists():
            raise FileNotFoundError(f"Model artifact not found at: {mp}")

        model = joblib.load(mp)

        metadata: Dict[str, Any] = {}
        if metadata_path:
            meta_p = Path(metadata_path)
            if meta_p.exists():
                metadata = json.loads(meta_p.read_text(encoding="utf-8"))

        return cls(model=model, metadata=metadata, reference_year=reference_year)

    def validate_and_format_input(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], Sequence[Dict[str, Any]]],
    ) -> pd.DataFrame:
        """
        Validates input structure and formats DataFrame with all required pipeline columns.
        
        Args:
            data: DataFrame, dictionary of feature values, or list of feature dicts.
            
        Returns:
            pd.DataFrame ready for feature pipeline transformation.
            
        Raises:
            DataLeakageError: If forbidden target or outcome columns exist.
            ValidationError: If required features are missing or values are out of bounds.
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValidationError(f"Unsupported input type: {type(data)}. Must be DataFrame, dict, or list of dicts.")

        if df.empty:
            return df

        # 1. Target and Private Info Leakage validation
        self.leakage_validator.validate_features(df)

        # 2. Derive vehicle_age from manufacture_year if vehicle_age is absent
        if "vehicle_age" not in df.columns:
            if "manufacture_year" in df.columns:
                mfg = pd.to_numeric(df["manufacture_year"], errors="coerce")
                if mfg.isna().any():
                    raise ValidationError("manufacture_year contains null or non-numeric values.")
                df["vehicle_age"] = self.reference_year - mfg
            else:
                raise ValidationError("Missing required feature: 'vehicle_age' (or 'manufacture_year').")

        # 3. Check for required categorical features
        missing_cats = [col for col in REQUIRED_CATEGORICAL_COLUMNS if col not in df.columns]
        if missing_cats:
            raise ValidationError(f"Missing required categorical features: {missing_cats}")

        # 4. Canonicalize and validate category against canonical categories
        df["category"] = df["category"].apply(
            lambda c: VehicleCleaner.canonicalize_category(str(c).strip()) if pd.notna(c) else None
        )
        invalid_mask = ~df["category"].isin(CANONICAL_CATEGORIES)
        if invalid_mask.any():
            invalid_vals = list(df.loc[invalid_mask, "category"].unique())
            raise ValidationError(
                f"Invalid vehicle category: {invalid_vals}. "
                f"Must be one of: {sorted(list(CANONICAL_CATEGORIES))}"
            )

        # 5. Check numerical columns and bounds
        for num_col in NUMERICAL_FEATURE_COLUMNS:
            if num_col not in df.columns:
                df[num_col] = np.nan
            else:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

        # Bounds checks
        if (df["vehicle_age"].dropna() < 0).any():
            raise ValidationError("vehicle_age cannot be negative.")
        if (df["vehicle_age"].dropna() > 100).any():
            raise ValidationError("vehicle_age exceeds realistic range (> 100 years).")
        if (df["mileage"].dropna() < 0).any():
            raise ValidationError("mileage cannot be negative.")
        if (df["engine_cc"].dropna() < 0).any():
            raise ValidationError("engine_cc cannot be negative.")

        # 6. Ensure optional registration_year exists
        if "registration_year" not in df.columns:
            df["registration_year"] = np.nan
        else:
            df["registration_year"] = pd.to_numeric(df["registration_year"], errors="coerce")

        # 7. Construct brand_model interaction feature
        brand_s = df["brand"].fillna("Unknown").astype(str).str.strip()
        model_s = df["model"].fillna("Unknown").astype(str).str.strip()
        df["brand_model"] = brand_s + "_" + model_s

        # Fill missing categoricals with 'Unknown' string
        for cat_col in REQUIRED_CATEGORICAL_COLUMNS:
            df[cat_col] = df[cat_col].fillna("Unknown").astype(str).str.strip()

        return df

    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], Sequence[Dict[str, Any]]],
    ) -> np.ndarray:
        """
        Generates predicted seller asking prices in LKR for input vehicle features.
        
        Args:
            data: DataFrame, single dict, or list of dicts containing vehicle features.
                 
        Returns:
            1D numpy array of predicted asking prices in LKR (strictly > 0).
        """
        df_formatted = self.validate_and_format_input(data)
        if df_formatted.empty:
            return np.array([], dtype=np.float64)

        # Run pipeline predict (TransformedTargetRegressor will automatically invert log1p if used)
        preds = self.model.predict(df_formatted)
        preds_arr = np.asarray(preds, dtype=np.float64)

        # Enforce physical positivity constraint (asking price >= 10,000 LKR)
        preds_clipped = np.maximum(preds_arr, 10_000.0)

        return preds_clipped

    def predict_single(
        self,
        vehicle_data: Dict[str, Any],
    ) -> float:
        """
        Convenience method to predict asking price for a single vehicle record.
        
        Args:
            vehicle_data: Dictionary of vehicle features.
            
        Returns:
            Predicted seller asking price in LKR as a float.
        """
        preds = self.predict(vehicle_data)
        if len(preds) == 0:
            raise ValidationError("No prediction generated for empty input.")
        return float(preds[0])

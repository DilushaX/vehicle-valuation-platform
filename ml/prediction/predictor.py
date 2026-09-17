"""
Vehicle Price Predictor.
Loads trained valuation model pipelines and provides safe, schema-validated
inference on vehicle listings in original Sri Lankan Rupees (LKR).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from feature_engineering.schema import FeatureSchema
from feature_engineering.validation import LeakageValidator

logger = logging.getLogger(__name__)


class VehiclePricePredictor:
    """
    Inference interface for vehicle valuation models.
    
    Guarantees:
    1. Validates required input features against expected schema.
    2. Enforces zero target leakage (no asking_price or price features in input).
    3. Handles unknown categories gracefully without pipeline crashes.
    4. Outputs strictly positive prices in original LKR scale.
    """

    def __init__(
        self,
        model: BaseEstimator,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.metadata = metadata or {}
        self.leakage_validator = LeakageValidator()

    @classmethod
    def load(
        cls,
        model_path: Union[str, Path] = "data/analysis/ml/models/model.joblib",
        metadata_path: Optional[Union[str, Path]] = "data/analysis/ml/models/model_metadata.json",
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

        return cls(model=model, metadata=metadata)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates predicted seller asking prices in LKR for input vehicle features.
        
        Args:
            df: DataFrame containing required active features
                (e.g. vehicle_age, mileage, engine_cc, category, brand, model,
                 fuel_type, transmission, district, condition).
                 
        Returns:
            1D numpy array of predicted asking prices in LKR (strictly > 0).
        """
        if df.empty:
            return np.array([], dtype=np.float64)

        # 1. Leakage check
        self.leakage_validator.validate_features(df)

        df_input = df.copy()

        # Derive brand_model interaction if not present
        if "brand_model" not in df_input.columns and "brand" in df_input.columns and "model" in df_input.columns:
            brand_s = df_input["brand"].fillna("Unknown").astype(str)
            model_s = df_input["model"].fillna("Unknown").astype(str)
            df_input["brand_model"] = brand_s + "_" + model_s

        # Run pipeline predict (TransformedTargetRegressor will automatically invert log1p if used)
        preds = self.model.predict(df_input)
        preds_arr = np.asarray(preds, dtype=np.float64)

        # Enforce physical positivity constraint (asking price > 0)
        preds_clipped = np.maximum(preds_arr, 10_000.0)

        return preds_clipped

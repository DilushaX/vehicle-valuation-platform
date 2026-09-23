"""
Valuation Input Data Quality Assessment Subsystem.

Evaluates the completeness of input vehicle attributes provided for
model-based valuation, distinguishing between complete and partial inputs.

Important:
This module measures input data completeness and availability only.
It does NOT compute model confidence, prediction probability, or accuracy percentages.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical 11 features expected by the valuation model
VALUATION_EXPECTED_FEATURES: List[str] = [
    "vehicle_age",
    "mileage",
    "engine_cc",
    "category",
    "brand",
    "model",
    "brand_model",
    "fuel_type",
    "transmission",
    "district",
    "condition",
]


@dataclass
class ValuationDataQuality:
    """
    Structured outcome of valuation input data completeness evaluation.
    
    Attributes:
        status: Completeness state ('COMPLETE' or 'PARTIAL').
        provided_features: Count of valid features provided in the input.
        expected_features: Total count of features expected by the model (11).
        missing_features: List of expected feature names that were omitted/unspecified.
        comparable_count: Number of comparable market listings retrieved.
    """
    status: str
    provided_features: int
    expected_features: int
    missing_features: List[str]
    comparable_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "provided_features": int(self.provided_features),
            "expected_features": int(self.expected_features),
            "missing_features": list(self.missing_features),
            "comparable_count": int(self.comparable_count),
        }


def assess_valuation_data_quality(
    formatted_df: pd.DataFrame,
    comparable_count: int = 0,
) -> ValuationDataQuality:
    """
    Evaluates input data completeness on a validated vehicle DataFrame.
    
    Args:
        formatted_df: Single-row DataFrame processed by VehiclePricePredictor.validate_and_format_input.
        comparable_count: Count of comparable listings retrieved for this vehicle.
        
    Returns:
        ValuationDataQuality structured indicator.
    """
    expected_total = len(VALUATION_EXPECTED_FEATURES)
    comp_count = max(0, int(comparable_count))

    if formatted_df.empty:
        return ValuationDataQuality(
            status="PARTIAL",
            provided_features=0,
            expected_features=expected_total,
            missing_features=list(VALUATION_EXPECTED_FEATURES),
            comparable_count=comp_count,
        )

    row = formatted_df.iloc[0]
    missing_features: List[str] = []

    for feat in VALUATION_EXPECTED_FEATURES:
        if feat not in row.index:
            missing_features.append(feat)
            continue

        val = row[feat]

        # Numerical features
        if feat in ("vehicle_age", "mileage", "engine_cc"):
            if pd.isna(val) or val is None:
                missing_features.append(feat)
            elif not isinstance(val, (int, float, np.number)):
                missing_features.append(feat)

        # Categorical features
        elif feat in ("category", "brand", "model", "fuel_type", "transmission", "district", "condition"):
            if pd.isna(val) or val is None:
                missing_features.append(feat)
            else:
                val_str = str(val).strip()
                if not val_str or val_str.lower() in ("unknown", "none", "nan", "null", "missing"):
                    missing_features.append(feat)

        # Interaction feature brand_model
        elif feat == "brand_model":
            if pd.isna(val) or val is None:
                missing_features.append(feat)
            else:
                val_str = str(val).strip()
                if (
                    not val_str
                    or val_str.lower() in ("unknown_unknown", "unknown", "none", "nan", "null")
                    or "brand" in missing_features
                    or "model" in missing_features
                ):
                    missing_features.append(feat)

    provided_total = expected_total - len(missing_features)
    status = "COMPLETE" if len(missing_features) == 0 else "PARTIAL"

    return ValuationDataQuality(
        status=status,
        provided_features=provided_total,
        expected_features=expected_total,
        missing_features=missing_features,
        comparable_count=comp_count,
    )

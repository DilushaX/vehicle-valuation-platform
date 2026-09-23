"""
Valuation Audit and Reproducibility Subsystem.

Provides transparent audit metadata and deterministic reproducibility fingerprints
for vehicle valuation predictions without exposing private seller or security data.

Important Methodological Notice:
- The reproducibility fingerprint identifies equivalent valuation configurations and inputs.
- It is NOT a cryptographic security signature or guarantee of market price accuracy.
- Valuations reflect experimental research benchmark models predicting asking prices on Riyasewana.
- Prediction ranges reflect ensemble decision tree dispersion, NOT statistical confidence intervals.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_SCHEMA_VERSION = "1.0"
VALUATION_METHOD = "RandomForest asking-price regression with Tree SHAP explainability"
COMPARABLE_METHOD = "Multi-attribute weighted similarity scoring (Category, Brand, Model, Age, Mileage, Specs)"
PREDICTION_RANGE_METHOD = "individual_tree_percentiles"


@dataclass
class ValuationAuditMetadata:
    """
    Structured audit metadata documenting how a valuation was computed.
    """
    model_name: str
    model_version: Optional[str]
    model_status: str
    target: str
    generated_at: str
    feature_schema_version: str
    valuation_method: str
    comparable_method: str
    prediction_range_method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_status": self.model_status,
            "target": self.target,
            "generated_at": self.generated_at,
            "feature_schema_version": self.feature_schema_version,
            "valuation_method": self.valuation_method,
            "comparable_method": self.comparable_method,
            "prediction_range_method": self.prediction_range_method,
        }


@dataclass
class ValuationReproducibility:
    """
    Deterministic reproducibility identifier for identical valuation inputs and configurations.
    """
    fingerprint: str
    algorithm: str = "SHA-256"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "algorithm": self.algorithm,
        }


def normalize_vehicle_input(
    vehicle_input: Union[Dict[str, Any], pd.DataFrame, pd.Series],
) -> Dict[str, Any]:
    """
    Normalizes vehicle input attributes into a canonical, sorted dictionary
    for consistent deterministic serialization.
    """
    if isinstance(vehicle_input, pd.DataFrame):
        raw = vehicle_input.iloc[0].to_dict() if not vehicle_input.empty else {}
    elif isinstance(vehicle_input, pd.Series):
        raw = vehicle_input.to_dict()
    elif isinstance(vehicle_input, dict):
        raw = vehicle_input.copy()
    else:
        raw = {}

    # Transient/ephemeral request parameters excluded from physical vehicle identity
    transient_fields = {
        "top_k_factors",
        "top_k_comparables",
        "percentile_lower",
        "percentile_upper",
        "exclude_listing_id",
        "generated_at",
    }

    normalized: Dict[str, Any] = {}
    for k in sorted(raw.keys()):
        clean_key = str(k).lower().strip()
        if clean_key in transient_fields:
            continue

        val = raw[k]
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            normalized[clean_key] = None
        elif isinstance(val, (int, np.integer)):
            normalized[clean_key] = int(val)
        elif isinstance(val, (float, np.floating)):
            normalized[clean_key] = round(float(val), 4)
        elif isinstance(val, str):
            clean_str = val.strip()
            normalized[clean_key] = clean_str if clean_str else None
        else:
            normalized[clean_key] = str(val)

    return normalized


def compute_reproducibility_fingerprint(
    vehicle_input: Union[Dict[str, Any], pd.DataFrame, pd.Series],
    model_metadata: Optional[Dict[str, Any]] = None,
    percentile_lower: int = 10,
    percentile_upper: int = 90,
) -> str:
    """
    Generates a deterministic SHA-256 fingerprint for identical vehicle inputs,
    model artifact version, and valuation configuration.
    
    Excludes transient timestamps (`generated_at`) so identical evaluations produce
    identical fingerprints.
    """
    meta = model_metadata or {}
    norm_vehicle = normalize_vehicle_input(vehicle_input)

    canonical_payload = {
        "vehicle_input": norm_vehicle,
        "model": {
            "model_name": meta.get("model_name"),
            "model_version": meta.get("trained_at") or meta.get("model_version") or meta.get("sklearn_version"),
            "target_variable": meta.get("target_variable", "asking_price"),
            "target_transform": meta.get("target_transform", "log1p"),
        },
        "configuration": {
            "percentile_lower": int(percentile_lower),
            "percentile_upper": int(percentile_upper),
        },
    }

    canonical_json = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def create_valuation_audit(
    vehicle_input: Union[Dict[str, Any], pd.DataFrame, pd.Series],
    model_metadata: Optional[Dict[str, Any]] = None,
    percentile_lower: int = 10,
    percentile_upper: int = 90,
    generated_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Constructs both audit metadata and reproducibility fingerprint.
    
    Returns:
        Dictionary containing 'audit' and 'reproducibility' structures.
    """
    meta = model_metadata or {}
    ts = generated_at or datetime.now(timezone.utc)
    ts_str = ts.isoformat()

    model_name = meta.get("model_name", "RandomForestRegressor")
    model_ver = meta.get("trained_at") or meta.get("model_version") or meta.get("sklearn_version")
    model_status = meta.get("status", "EXPERIMENTAL_RESEARCH_BENCHMARK")
    target = meta.get("target_variable", "asking_price")

    audit = ValuationAuditMetadata(
        model_name=model_name,
        model_version=str(model_ver) if model_ver else None,
        model_status=model_status,
        target=target,
        generated_at=ts_str,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        valuation_method=VALUATION_METHOD,
        comparable_method=COMPARABLE_METHOD,
        prediction_range_method=PREDICTION_RANGE_METHOD,
    )

    fingerprint = compute_reproducibility_fingerprint(
        vehicle_input=vehicle_input,
        model_metadata=meta,
        percentile_lower=percentile_lower,
        percentile_upper=percentile_upper,
    )

    reproducibility = ValuationReproducibility(
        fingerprint=fingerprint,
        algorithm="SHA-256",
    )

    return {
        "audit": audit.to_dict(),
        "reproducibility": reproducibility.to_dict(),
    }

"""
Tests for Valuation Input Data Quality Indicators (Step 9.12).

Validates:
1. Complete input returns COMPLETE status.
2. Missing optional/non-critical information returns PARTIAL status.
3. Invalid required input still raises the existing ValidationError.
4. provided_features is deterministic and accurate.
5. expected_features is deterministic (strictly 11).
6. missing_features accurately lists omitted/unspecified attributes.
7. comparable_count is correctly included.
8. No confidence percentage or accuracy metric is generated.
9. Existing valuation prediction still works.
10. Existing valuation range still works.
11. Existing SHAP explanation still works.
12. Existing comparable retrieval still works.
"""

from pathlib import Path
from typing import Any, Dict

import pytest

from ml.prediction.predictor import ValidationError, VehiclePricePredictor
from ml.valuation.data_quality import (
    VALUATION_EXPECTED_FEATURES,
    ValuationDataQuality,
    assess_valuation_data_quality,
)
from ml.valuation.valuation_service import ValuationResult, VehicleValuationService


@pytest.fixture
def complete_car_spec() -> Dict[str, Any]:
    """A valid, complete vehicle specification."""
    return {
        "category": "Cars",
        "brand": "Toyota",
        "model": "Premio",
        "manufacture_year": 2016,
        "mileage": 85000.0,
        "engine_cc": 1500.0,
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "district": "Colombo",
        "condition": "Registered (Used)",
    }


@pytest.fixture
def valuation_service() -> VehicleValuationService:
    model_path = Path("data/analysis/ml/models/model.joblib")
    meta_path = Path("data/analysis/ml/models/model_metadata.json")
    if not model_path.exists():
        pytest.skip(f"Model artifact not found at {model_path}")
    return VehicleValuationService(model_path=model_path, metadata_path=meta_path)


def test_complete_input_returns_complete(valuation_service, complete_car_spec):
    """1. Complete input returns COMPLETE."""
    result = valuation_service.valuate(complete_car_spec)
    dq = result.data_quality

    assert dq is not None
    assert dq["status"] == "COMPLETE"
    assert dq["provided_features"] == 11
    assert dq["expected_features"] == 11
    assert dq["missing_features"] == []


def test_missing_optional_information_returns_partial(valuation_service, complete_car_spec):
    """2. Missing optional/non-critical information returns PARTIAL where applicable."""
    car_partial = complete_car_spec.copy()
    car_partial["mileage"] = None

    result = valuation_service.valuate(car_partial)
    dq = result.data_quality

    assert dq is not None
    assert dq["status"] == "PARTIAL"
    assert dq["provided_features"] == 10
    assert dq["expected_features"] == 11
    assert "mileage" in dq["missing_features"]


def test_invalid_required_input_raises_validation_error(valuation_service, complete_car_spec):
    """3. Invalid required input still raises the existing validation error."""
    # Negative mileage
    invalid_mileage = complete_car_spec.copy()
    invalid_mileage["mileage"] = -5000.0
    with pytest.raises(ValidationError, match="mileage cannot be negative"):
        valuation_service.valuate(invalid_mileage)

    # Negative engine CC
    invalid_cc = complete_car_spec.copy()
    invalid_cc["engine_cc"] = -1500.0
    with pytest.raises(ValidationError, match="engine_cc cannot be negative"):
        valuation_service.valuate(invalid_cc)

    # Invalid category
    invalid_cat = complete_car_spec.copy()
    invalid_cat["category"] = "Hovercrafts"
    with pytest.raises(ValidationError, match="Invalid vehicle category"):
        valuation_service.valuate(invalid_cat)


def test_provided_features_is_deterministic(valuation_service, complete_car_spec):
    """4. provided_features is deterministic."""
    # Complete spec
    res1 = valuation_service.valuate(complete_car_spec)
    dq1 = res1.data_quality
    assert dq1["provided_features"] == 11
    assert isinstance(dq1["provided_features"], int)

    # Missing mileage
    spec2 = complete_car_spec.copy()
    spec2["mileage"] = None
    res2 = valuation_service.valuate(spec2)
    assert res2.data_quality["provided_features"] == 10

    # Missing both mileage and engine_cc
    spec3 = complete_car_spec.copy()
    spec3["mileage"] = None
    spec3["engine_cc"] = None
    res3 = valuation_service.valuate(spec3)
    assert res3.data_quality["provided_features"] == 9
    assert res3.data_quality["provided_features"] == res3.data_quality["expected_features"] - len(res3.data_quality["missing_features"])


def test_expected_features_is_deterministic(valuation_service, complete_car_spec):
    """5. expected_features is deterministic (always 11)."""
    res = valuation_service.valuate(complete_car_spec)
    dq = res.data_quality
    assert dq["expected_features"] == 11
    assert len(VALUATION_EXPECTED_FEATURES) == 11


def test_missing_features_is_correct(valuation_service, complete_car_spec):
    """6. missing_features is correct."""
    spec = complete_car_spec.copy()
    spec["mileage"] = None
    spec["engine_cc"] = None

    res = valuation_service.valuate(spec)
    missing = res.data_quality["missing_features"]

    assert "mileage" in missing
    assert "engine_cc" in missing
    assert len(missing) == 2


def test_comparable_count_is_correctly_included(valuation_service, complete_car_spec):
    """7. comparable_count is correctly included."""
    res = valuation_service.valuate(complete_car_spec, top_k_comparables=4)
    dq = res.data_quality

    assert "comparable_count" in dq
    assert isinstance(dq["comparable_count"], int)
    assert dq["comparable_count"] == len(res.comparables)
    assert dq["comparable_count"] <= 4


def test_no_confidence_percentage_generated(valuation_service, complete_car_spec):
    """8. No confidence percentage is generated."""
    res = valuation_service.valuate(complete_car_spec)
    dq = res.data_quality
    d = res.to_dict()

    forbidden_terms = ["confidence", "probability", "accuracy", "high_confidence", "low_confidence"]

    # Check data_quality keys and values
    for k, v in dq.items():
        k_lower = str(k).lower()
        v_lower = str(v).lower()
        for term in forbidden_terms:
            assert term not in k_lower, f"Forbidden term '{term}' found in key '{k}'"
            assert term not in v_lower, f"Forbidden term '{term}' found in value '{v}'"

    # Status must be strictly COMPLETE or PARTIAL
    assert dq["status"] in ["COMPLETE", "PARTIAL"]
    assert "%" not in str(dq["status"])


def test_existing_valuation_prediction_still_works(valuation_service, complete_car_spec):
    """9. Existing valuation prediction still works."""
    res = valuation_service.valuate(complete_car_spec)
    assert res.estimated_asking_price_lkr > 1_000_000.0
    assert isinstance(res.estimated_asking_price_lkr, float)


def test_existing_valuation_range_still_works(valuation_service, complete_car_spec):
    """10. Existing valuation range still works."""
    res = valuation_service.valuate(complete_car_spec)
    r = res.prediction_range_lkr
    assert r["lower"] <= res.estimated_asking_price_lkr <= r["upper"]
    assert r["lower"] > 0
    assert "spread" in r
    assert "method" in r


def test_existing_shap_explanation_still_works(valuation_service, complete_car_spec):
    """11. Existing SHAP explanation still works."""
    res = valuation_service.valuate(complete_car_spec, top_k_factors=5)
    assert len(res.explanation) == 5
    for item in res.explanation:
        assert "feature" in item
        assert "contribution" in item
        assert "direction" in item


def test_existing_comparable_retrieval_still_works(valuation_service, complete_car_spec):
    """12. Existing comparable retrieval still works."""
    res = valuation_service.valuate(complete_car_spec, top_k_comparables=3)
    assert len(res.comparables) > 0
    for comp in res.comparables:
        assert comp["category"] == "Cars"
        assert comp["asking_price"] > 0
        assert "similarity_percentage" in comp

"""
Tests for Unified Vehicle Valuation Service (Step 9 - Part 5).

Validates:
1. Complete valuation request returns all structured components (estimate, range, explanation, comparables, limitations).
2. Missing required fields raise ValidationError.
3. Invalid category raises ValidationError with allowed list.
4. Negative mileage raises ValidationError.
5. Unrealistic / negative vehicle age raises ValidationError.
6. Negative engine CC raises ValidationError.
7. Unknown categorical values are handled gracefully without pipeline failure.
8. Point estimate and range are strictly positive LKR values.
9. Explanation list contains structured feature contribution records.
10. Comparable vehicles are retrieved from PostgreSQL.
11. Standard legal and methodological limitations are present in every response.
"""

from pathlib import Path
from typing import Any, Dict

import pytest

from ml.prediction.predictor import ValidationError
from ml.valuation.valuation_service import (
    STANDARD_VALUATION_LIMITATIONS,
    ValuationResult,
    VehicleValuationService,
)


@pytest.fixture
def valuation_service() -> VehicleValuationService:
    """Initializes the valuation service with the trained model."""
    model_path = Path("data/analysis/ml/models/model.joblib")
    meta_path = Path("data/analysis/ml/models/model_metadata.json")
    if not model_path.exists():
        pytest.skip(f"Model artifact not found at {model_path}")
    return VehicleValuationService(model_path=model_path, metadata_path=meta_path)


@pytest.fixture
def valid_car_request() -> Dict[str, Any]:
    return {
        "category": "Cars",
        "brand": "Toyota",
        "model": "Premio",
        "vehicle_age": 10,
        "mileage": 85000.0,
        "engine_cc": 1500.0,
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "district": "Colombo",
        "condition": "Registered (Used)",
    }


def test_complete_valuation_request(valuation_service, valid_car_request):
    result = valuation_service.valuate(valid_car_request, top_k_factors=3, top_k_comparables=3)

    assert isinstance(result, ValuationResult)
    d = result.to_dict()

    # 1. Price estimate
    assert d["estimated_asking_price_lkr"] > 1_000_000.0
    assert d["currency"] == "LKR"

    # 2. Prediction range
    r = d["prediction_range_lkr"]
    assert r["lower"] <= d["estimated_asking_price_lkr"] <= r["upper"]
    assert r["lower"] > 0
    assert "method" in r

    # 3. Model metadata
    m = d["model"]
    assert m["name"] == "RandomForestRegressor"
    assert m["target_variable"] == "asking_price"
    assert m["target_transform"] == "log1p"

    # 4. Explanation
    expl = d["explanation"]
    assert len(expl) == 3
    for factor in expl:
        assert "feature" in factor
        assert "contribution" in factor
        assert "direction" in factor

    # 5. Comparables
    comps = d["comparables"]
    assert len(comps) > 0
    assert len(comps) <= 3
    for comp in comps:
        assert comp["category"] == "Cars"
        assert comp["asking_price"] > 0

    # 6. Limitations
    assert len(d["limitations"]) == len(STANDARD_VALUATION_LIMITATIONS)
    assert any("not a confirmed transaction" in lim.lower() or "not the confirmed transaction" in lim.lower() for lim in d["limitations"])
    assert any("physical condition" in lim.lower() for lim in d["limitations"])

    # 7. Data Quality
    assert "data_quality" in d
    assert d["data_quality"]["status"] == "COMPLETE"
    assert d["data_quality"]["provided_features"] == 11

    # 8. Comparable Market Summary
    assert "comparable_market_summary" in d
    cms = d["comparable_market_summary"]
    assert cms["comparable_count"] == len(comps)
    assert cms["min_asking_price"] is not None
    assert cms["max_asking_price"] is not None
    assert cms["min_asking_price"] <= cms["max_asking_price"]
    assert cms["price_spread"] == round(cms["max_asking_price"] - cms["min_asking_price"], 2)

    # 9. Audit and Reproducibility
    assert "audit" in d
    assert d["audit"]["model_name"] == "RandomForestRegressor"
    assert d["audit"]["target"] == "asking_price"
    assert "reproducibility" in d
    assert len(d["reproducibility"]["fingerprint"]) == 64





def test_missing_fields_validation(valuation_service, valid_car_request):
    bad = valid_car_request.copy()
    del bad["brand"]
    with pytest.raises(ValidationError, match="Missing required categorical features"):
        valuation_service.valuate(bad)


def test_invalid_category_validation(valuation_service, valid_car_request):
    bad = valid_car_request.copy()
    bad["category"] = "Hovercraft"
    with pytest.raises(ValidationError, match="Invalid vehicle category"):
        valuation_service.valuate(bad)


def test_invalid_mileage_validation(valuation_service, valid_car_request):
    bad = valid_car_request.copy()
    bad["mileage"] = -25000.0
    with pytest.raises(ValidationError, match="mileage cannot be negative"):
        valuation_service.valuate(bad)


def test_invalid_year_validation(valuation_service, valid_car_request):
    bad = valid_car_request.copy()
    del bad["vehicle_age"]
    bad["manufacture_year"] = 2030  # Future year relative to reference year 2026 -> age -4
    with pytest.raises(ValidationError, match="vehicle_age cannot be negative"):
        valuation_service.valuate(bad)


def test_invalid_engine_cc_validation(valuation_service, valid_car_request):
    bad = valid_car_request.copy()
    bad["engine_cc"] = -1500.0
    with pytest.raises(ValidationError, match="engine_cc cannot be negative"):
        valuation_service.valuate(bad)


def test_unknown_categorical_values_safe(valuation_service, valid_car_request):
    novel = valid_car_request.copy()
    novel["brand"] = "FuturisticBrand"
    novel["model"] = "CyberSpeedster"
    novel["district"] = "NonExistentDistrict"

    result = valuation_service.valuate(novel)
    assert result.estimated_asking_price_lkr > 10_000.0
    assert result.prediction_range_lkr["lower"] > 0
    assert len(result.explanation) > 0

"""
Tests for Valuation Audit and Reproducibility Subsystem (Step 9.14).

Validates:
1. Audit metadata is present in valuation result.
2. Model name and version come from existing model metadata.
3. Target variable is asking_price.
4. Model status reflects EXPERIMENTAL_RESEARCH_BENCHMARK.
5. Generated timestamp is present, valid ISO-8601, and UTC-compatible.
6. Reproducibility fingerprint exists and uses SHA-256 algorithm.
7. Same identical input produces the exact same fingerprint.
8. Changing a relevant input feature changes the fingerprint.
9. Changing generated_at alone does NOT change the fingerprint.
10. No secrets, credentials, or private seller details appear in the audit structure.
11. Existing valuation output fields (estimate, range, explanation, comparables,
    market summary, data quality, limitations) remain fully intact.
12. API response validates audit and reproducibility metadata.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from api.main import app
from ml.valuation.audit import (
    compute_reproducibility_fingerprint,
    create_valuation_audit,
    normalize_vehicle_input,
)
from ml.valuation.valuation_service import ValuationResult, VehicleValuationService

client = TestClient(app)


@pytest.fixture
def sample_car() -> Dict[str, Any]:
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


def test_audit_metadata_is_present(valuation_service, sample_car):
    """1. Audit metadata is present."""
    result = valuation_service.valuate(sample_car)
    d = result.to_dict()

    assert "audit" in d
    assert isinstance(d["audit"], dict)
    assert "model_name" in d["audit"]
    assert "target" in d["audit"]
    assert "generated_at" in d["audit"]
    assert "valuation_method" in d["audit"]
    assert "comparable_method" in d["audit"]
    assert "prediction_range_method" in d["audit"]


def test_model_name_and_version_from_existing_metadata(valuation_service, sample_car):
    """2. Model name/version comes from existing metadata rather than fabricated values."""
    result = valuation_service.valuate(sample_car)
    audit = result.to_dict()["audit"]
    meta = valuation_service.predictor.metadata or {}

    expected_model_name = meta.get("model_name", "RandomForestRegressor")
    assert audit["model_name"] == expected_model_name

    expected_version = meta.get("trained_at") or meta.get("model_version") or meta.get("sklearn_version")
    assert audit["model_version"] == str(expected_version)


def test_target_is_asking_price(valuation_service, sample_car):
    """3. Target is asking_price."""
    result = valuation_service.valuate(sample_car)
    audit = result.to_dict()["audit"]
    assert audit["target"] == "asking_price"


def test_model_status_correctly_represented(valuation_service, sample_car):
    """4. Model status is correctly represented."""
    result = valuation_service.valuate(sample_car)
    audit = result.to_dict()["audit"]
    assert audit["model_status"] == "EXPERIMENTAL_RESEARCH_BENCHMARK"


def test_generated_timestamp_is_utc_compatible(valuation_service, sample_car):
    """5. Generated timestamp is present and UTC-compatible."""
    result = valuation_service.valuate(sample_car)
    audit = result.to_dict()["audit"]
    ts_str = audit["generated_at"]

    assert ts_str is not None
    # Parse ISO-8601 timestamp
    parsed = datetime.fromisoformat(ts_str)
    assert parsed.tzinfo is not None  # Must be timezone-aware (UTC)


def test_reproducibility_fingerprint_exists(valuation_service, sample_car):
    """6. Reproducibility fingerprint exists."""
    result = valuation_service.valuate(sample_car)
    d = result.to_dict()

    assert "reproducibility" in d
    rep = d["reproducibility"]
    assert "fingerprint" in rep
    assert isinstance(rep["fingerprint"], str)
    assert len(rep["fingerprint"]) == 64  # SHA-256 produces 64 hex characters
    assert rep["algorithm"] == "SHA-256"


def test_same_identical_input_produces_same_fingerprint(valuation_service, sample_car):
    """7. Same identical input produces the same fingerprint."""
    # Test identical calls through service
    res1 = valuation_service.valuate(sample_car)
    res2 = valuation_service.valuate(sample_car)

    fp1 = res1.to_dict()["reproducibility"]["fingerprint"]
    fp2 = res2.to_dict()["reproducibility"]["fingerprint"]
    assert fp1 == fp2

    # Test identical dict with different key ordering
    reordered_car = {k: sample_car[k] for k in reversed(list(sample_car.keys()))}
    res3 = valuation_service.valuate(reordered_car)
    fp3 = res3.to_dict()["reproducibility"]["fingerprint"]
    assert fp1 == fp3


def test_changing_relevant_input_changes_fingerprint(valuation_service, sample_car):
    """8. Changing a relevant input changes the fingerprint."""
    res_base = valuation_service.valuate(sample_car)
    fp_base = res_base.to_dict()["reproducibility"]["fingerprint"]

    # Change mileage
    car_diff_mileage = sample_car.copy()
    car_diff_mileage["mileage"] = 120000.0
    res_diff_mileage = valuation_service.valuate(car_diff_mileage)
    fp_diff_mileage = res_diff_mileage.to_dict()["reproducibility"]["fingerprint"]
    assert fp_diff_mileage != fp_base

    # Change district
    car_diff_district = sample_car.copy()
    car_diff_district["district"] = "Kandy"
    res_diff_district = valuation_service.valuate(car_diff_district)
    fp_diff_district = res_diff_district.to_dict()["reproducibility"]["fingerprint"]
    assert fp_diff_district != fp_base


def test_changing_generated_at_does_not_change_fingerprint(sample_car):
    """9. Changing generated_at alone does NOT change the fingerprint."""
    meta = {"model_name": "RandomForestRegressor", "target_variable": "asking_price"}
    t1 = datetime(2026, 9, 23, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 23, 18, 30, 0, tzinfo=timezone.utc)

    audit_1 = create_valuation_audit(sample_car, model_metadata=meta, generated_at=t1)
    audit_2 = create_valuation_audit(sample_car, model_metadata=meta, generated_at=t2)

    assert audit_1["audit"]["generated_at"] != audit_2["audit"]["generated_at"]
    assert audit_1["reproducibility"]["fingerprint"] == audit_2["reproducibility"]["fingerprint"]


def test_no_secrets_in_audit_object(valuation_service, sample_car):
    """10. No secrets/private seller data appear in the audit object."""
    result = valuation_service.valuate(sample_car)
    d = result.to_dict()

    audit_json = json.dumps(d["audit"]).lower()
    rep_json = json.dumps(d["reproducibility"]).lower()

    forbidden = ["password", "secret", "token", "phone", "email", "seller", "credential", "conn_str"]
    for word in forbidden:
        assert word not in audit_json, f"Found private word '{word}' in audit metadata"
        assert word not in rep_json, f"Found private word '{word}' in reproducibility data"


def test_existing_valuation_output_still_contains_all_components(valuation_service, sample_car):
    """11. Existing valuation output still contains all previously implemented components."""
    result = valuation_service.valuate(sample_car)
    d = result.to_dict()

    # 1. Estimate
    assert "estimated_asking_price_lkr" in d
    assert d["estimated_asking_price_lkr"] > 0

    # 2. Prediction Range
    assert "prediction_range_lkr" in d
    assert d["prediction_range_lkr"]["lower"] <= d["estimated_asking_price_lkr"] <= d["prediction_range_lkr"]["upper"]

    # 3. Explanation
    assert "explanation" in d
    assert len(d["explanation"]) > 0

    # 4. Comparables
    assert "comparables" in d
    assert len(d["comparables"]) > 0

    # 5. Comparable Market Summary
    assert "comparable_market_summary" in d
    assert "min_asking_price" in d["comparable_market_summary"]

    # 6. Data Quality
    assert "data_quality" in d
    assert d["data_quality"]["status"] == "COMPLETE"

    # 7. Limitations
    assert "limitations" in d
    assert len(d["limitations"]) > 0

    # 8. Audit and Reproducibility
    assert "audit" in d
    assert "reproducibility" in d


def test_api_endpoint_returns_audit_and_reproducibility(sample_car):
    """12 & 13. API valuation endpoint returns validated audit and reproducibility metadata."""
    payload = sample_car.copy()
    payload["top_k_factors"] = 3
    payload["top_k_comparables"] = 3

    response = client.post("/api/valuation/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert "audit" in data
    assert data["audit"] is not None
    assert data["audit"]["model_name"] == "RandomForestRegressor"
    assert data["audit"]["target"] == "asking_price"
    assert "generated_at" in data["audit"]

    assert "reproducibility" in data
    assert data["reproducibility"] is not None
    assert len(data["reproducibility"]["fingerprint"]) == 64
    assert data["reproducibility"]["algorithm"] == "SHA-256"

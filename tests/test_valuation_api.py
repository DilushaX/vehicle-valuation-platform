"""
Tests for Vehicle Valuation REST API (Step 9 - Part 6 & Part 7).

Validates:
1. POST /api/valuation/predict returns 200 with complete valuation response.
2. GET /health returns 200 healthy.
3. 422 Unprocessable Entity for malformed schema (missing required fields).
4. 400 Bad Request for domain validation errors (e.g. invalid category, negative values).
5. 404 Not Found when model artifact is missing.
6. 500 Internal Server Error handles unexpected failures safely without leaking stack traces.
"""

from unittest.mock import patch

from fastapi import status
from fastapi.testclient import TestClient
import pytest

from api.main import app

client = TestClient(app)


@pytest.fixture
def valid_car_payload():
    return {
        "category": "Cars",
        "brand": "Toyota",
        "model": "Premio",
        "vehicle_age": 10,
        "mileage": 85000,
        "engine_cc": 1500,
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "district": "Colombo",
        "condition": "Registered (Used)",
        "top_k_factors": 3,
        "top_k_comparables": 3,
    }


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"


def test_successful_valuation_api(valid_car_payload):
    response = client.post("/api/valuation/predict", json=valid_car_payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    # Price and range
    assert data["estimated_asking_price_lkr"] > 1_000_000.0
    assert data["currency"] == "LKR"
    assert data["prediction_range_lkr"]["lower"] <= data["estimated_asking_price_lkr"]
    assert data["prediction_range_lkr"]["upper"] >= data["estimated_asking_price_lkr"]

    # Model metadata
    assert data["model"]["name"] == "RandomForestRegressor"
    assert data["model"]["target_variable"] == "asking_price"

    # Explanations
    assert len(data["explanation"]) == 3
    for f in data["explanation"]:
        assert "feature" in f
        assert "contribution" in f
        assert "direction" in f

    # Comparables
    assert len(data["comparables"]) > 0
    assert len(data["comparables"]) <= 3

    # Limitations
    assert len(data["limitations"]) > 0

    # Data Quality
    assert "data_quality" in data
    dq = data["data_quality"]
    assert dq["status"] == "COMPLETE"
    assert dq["provided_features"] == 11
    assert dq["expected_features"] == 11
    assert dq["missing_features"] == []
    assert isinstance(dq["comparable_count"], int)



def test_manufacture_year_payload(valid_car_payload):
    payload = valid_car_payload.copy()
    del payload["vehicle_age"]
    payload["manufacture_year"] = 2016

    response = client.post("/api/valuation/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["estimated_asking_price_lkr"] > 0


def test_422_missing_required_fields():
    # Empty payload
    response = client.post("/api/valuation/predict", json={})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_400_invalid_category(valid_car_payload):
    payload = valid_car_payload.copy()
    payload["category"] = "Helicopters"
    response = client.post("/api/valuation/predict", json=payload)
    assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY]
    detail = response.json()["detail"]
    assert "category" in str(detail).lower()


def test_422_negative_mileage(valid_car_payload):
    payload = valid_car_payload.copy()
    payload["mileage"] = -50000
    response = client.post("/api/valuation/predict", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_404_model_artifact_missing(valid_car_payload):
    with patch("api.routes.valuation.get_valuation_service", side_effect=FileNotFoundError("model.joblib missing")):
        response = client.post("/api/valuation/predict", json=valid_car_payload)
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "unavailable" in response.json()["detail"].lower()


def test_500_internal_error_safe_response(valid_car_payload):
    with patch("api.routes.valuation.get_valuation_service") as mock_service_fn:
        mock_service = mock_service_fn.return_value
        mock_service.valuate.side_effect = RuntimeError("Fatal internal bug at /secret/path/error")

        response = client.post("/api/valuation/predict", json=valid_car_payload)
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        # Internal path and exception details must NOT be leaked
        assert "/secret/path" not in str(data)
        assert "unexpected error" in data["detail"].lower()

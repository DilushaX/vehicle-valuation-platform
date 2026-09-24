"""
Tests for Hardened Vehicle Valuation REST API (Phase 10.1).

Validates:
1. Valid valuation request -> HTTP 200.
2. Missing required field -> HTTP 4xx.
3. Invalid numeric value -> HTTP 4xx (negative mileage, negative engine_cc, impossible year, malformed).
4. Invalid categorical value -> HTTP 4xx where schema requires it (invalid category, empty strings).
5. Internal valuation error -> safe HTTP 500 response without leaking traces or secrets.
6. Response contains existing valuation fields (estimate, range, explanation, comparables, limitations).
7. Audit metadata is present (audit and reproducibility fingerprint).
8. Comparable market summary is present (count, min, max, median, average, spread).
9. Data-quality indicator is present (completeness, counts).
10. /health works (HTTP 200, status "ok").
11. /ready works when the model artifact exists (HTTP 200, status "ready") and fails safely when unavailable.
12. API response does not expose secrets or private contact fields.
13. CORS headers are configured for local development.
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


# ==============================================================================
# 1. HEALTH AND READINESS ENDPOINTS (Requirement 4 & 10 & 11)
# ==============================================================================

def test_health_endpoint():
    """GET /health returns HTTP 200 with status 'ok'."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] in ["ok", "healthy"]


def test_ready_endpoint_when_model_artifact_exists():
    """GET /ready returns HTTP 200 with status 'ready' when model is available."""
    response = client.get("/ready")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ready"


def test_ready_endpoint_model_unavailable():
    """GET /ready returns HTTP 503 when model artifact cannot be loaded."""
    with patch("api.main.get_valuation_service", side_effect=FileNotFoundError("model.joblib missing")):
        response = client.get("/ready")
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert data["status"] == "unavailable"
        assert "unavailable" in data["detail"].lower()


# ==============================================================================
# 2. SUCCESSFUL PREDICTION & RESPONSE CONTRACT (Requirements 1, 3, 6, 7, 8, 9)
# ==============================================================================

def test_valid_valuation_request_returns_200(valid_car_payload):
    """POST /api/valuation/predict returns HTTP 200 for valid vehicle specification."""
    response = client.post("/api/valuation/predict", json=valid_car_payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    # Core estimate and range
    assert data["estimated_asking_price_lkr"] > 1_000_000.0
    assert data["currency"] == "LKR"
    assert data["prediction_range_lkr"]["lower"] <= data["estimated_asking_price_lkr"]
    assert data["prediction_range_lkr"]["upper"] >= data["estimated_asking_price_lkr"]
    assert data["prediction_range_lkr"]["lower"] > 0
    assert "method" in data["prediction_range_lkr"]

    # Model metadata
    assert data["model"]["name"] == "RandomForestRegressor"
    assert data["model"]["target_variable"] == "asking_price"

    # Explanation
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
    assert any("not a confirmed transaction" in lim.lower() for lim in data["limitations"])

    # Data Quality Indicator
    assert "data_quality" in data
    dq = data["data_quality"]
    assert dq["status"] == "COMPLETE"
    assert dq["provided_features"] == 11
    assert dq["expected_features"] == 11
    assert dq["missing_features"] == []
    assert isinstance(dq["comparable_count"], int)

    # Comparable Market Summary
    assert "comparable_market_summary" in data
    cms = data["comparable_market_summary"]
    assert cms is not None
    assert cms["comparable_count"] == len(data["comparables"])
    assert cms["min_asking_price"] is not None
    assert cms["max_asking_price"] is not None
    assert cms["min_asking_price"] <= cms["max_asking_price"]

    # Audit & Reproducibility Metadata
    assert "audit" in data
    assert data["audit"]["model_name"] == "RandomForestRegressor"
    assert data["audit"]["target"] == "asking_price"
    assert "reproducibility" in data
    assert len(data["reproducibility"]["fingerprint"]) == 64


def test_manufacture_year_payload_derivation(valid_car_payload):
    """POST /api/valuation/predict successfully derives vehicle age from manufacture_year."""
    payload = valid_car_payload.copy()
    del payload["vehicle_age"]
    payload["manufacture_year"] = 2016

    response = client.post("/api/valuation/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["estimated_asking_price_lkr"] > 0


# ==============================================================================
# 3. REQUEST VALIDATION FAILURES (Requirement 1, 2, 3, 4)
# ==============================================================================

def test_missing_required_fields_return_4xx():
    """Empty payload or missing required fields return HTTP 422."""
    response = client.post("/api/valuation/predict", json={})
    assert response.status_code == 422

    # Missing age AND manufacture_year
    incomplete = {
        "category": "Cars",
        "brand": "Toyota",
        "model": "Premio",
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "district": "Colombo",
        "condition": "Registered (Used)",
    }
    resp = client.post("/api/valuation/predict", json=incomplete)
    assert resp.status_code == 422


def test_invalid_numeric_values_return_4xx(valid_car_payload):
    """Negative or physically impossible numbers return HTTP 422."""
    # Negative mileage
    p1 = valid_car_payload.copy()
    p1["mileage"] = -50000
    assert client.post("/api/valuation/predict", json=p1).status_code == 422

    # Negative engine_cc
    p2 = valid_car_payload.copy()
    p2["engine_cc"] = -1500
    assert client.post("/api/valuation/predict", json=p2).status_code == 422

    # Impossible manufacture year (ancient)
    p3 = valid_car_payload.copy()
    del p3["vehicle_age"]
    p3["manufacture_year"] = 1850
    assert client.post("/api/valuation/predict", json=p3).status_code == 422

    # Impossible manufacture year (future)
    p4 = valid_car_payload.copy()
    del p4["vehicle_age"]
    p4["manufacture_year"] = 2050
    assert client.post("/api/valuation/predict", json=p4).status_code == 422

    # Malformed numeric value
    p5 = valid_car_payload.copy()
    p5["mileage"] = "not_a_number"
    assert client.post("/api/valuation/predict", json=p5).status_code == 422


def test_invalid_categorical_values_return_4xx(valid_car_payload):
    """Invalid category or empty required string fields return HTTP 4xx."""
    # Invalid category
    p1 = valid_car_payload.copy()
    p1["category"] = "Helicopters"
    r1 = client.post("/api/valuation/predict", json=p1)
    assert r1.status_code in [status.HTTP_400_BAD_REQUEST, 422]
    assert "category" in str(r1.json()).lower()

    # Empty required string
    p2 = valid_car_payload.copy()
    p2["brand"] = "   "
    r2 = client.post("/api/valuation/predict", json=p2)
    assert r2.status_code == 422


# ==============================================================================
# 4. ERROR HANDLING & SECURITY (Requirements 2, 6, 12)
# ==============================================================================

def test_model_artifact_missing_returns_404(valid_car_payload):
    """POST /api/valuation/predict returns HTTP 404 if model file is missing."""
    with patch("api.routes.valuation.get_valuation_service", side_effect=FileNotFoundError("model.joblib missing")):
        response = client.post("/api/valuation/predict", json=valid_car_payload)
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "unavailable" in response.json()["detail"].lower()


def test_internal_error_safe_500_response(valid_car_payload):
    """Unexpected internal errors return safe HTTP 500 without leaking stack traces or paths."""
    with patch("api.routes.valuation.get_valuation_service") as mock_service_fn:
        mock_service = mock_service_fn.return_value
        mock_service.valuate.side_effect = RuntimeError("Fatal DB leak at /internal/secret/password/db.py")

        response = client.post("/api/valuation/predict", json=valid_car_payload)
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        # Internal path and exception details must NOT be leaked
        assert "/internal/secret" not in str(data)
        assert "password" not in str(data).lower()
        assert "unexpected error" in data["detail"].lower()


def test_api_response_does_not_expose_secrets_or_private_contact(valid_car_payload):
    """Verifies that API response does NOT contain phone numbers, emails, passwords, or credentials."""
    response = client.post("/api/valuation/predict", json=valid_car_payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    forbidden_terms = [
        "phone",
        "email",
        "seller_phone",
        "seller_email",
        "contact_number",
        "password",
        "secret",
        "database_url",
        "db_url",
        "postgres://",
        "postgresql://",
    ]

    def scan_for_secrets(obj, parent_key=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                lower_k = k.lower()
                for term in forbidden_terms:
                    assert term not in lower_k, f"Forbidden term '{term}' found in key: {k}"
                scan_for_secrets(v, parent_key=k)
        elif isinstance(obj, list):
            for item in obj:
                scan_for_secrets(item, parent_key=parent_key)
        elif isinstance(obj, str):
            lower_v = obj.lower()
            for term in ["password", "postgres://", "postgresql://"]:
                assert term not in lower_v, f"Forbidden term '{term}' found in value: {obj}"

    scan_for_secrets(data)


# ==============================================================================
# 5. CORS CONFIGURATION (Requirement 7)
# ==============================================================================

def test_cors_configuration():
    """Verifies CORS headers for local development origins."""
    headers = {
        "Origin": "http://localhost:8501",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/api/valuation/predict", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    assert response.headers.get("access-control-allow-origin") == "http://localhost:8501"

"""
Tests for dashboard.api_client (Phase 10.2).

These tests mock httpx at the transport layer so no live API server is needed.
They verify:
  1. Successful response parsing → returns dict.
  2. Connection error → raises APIClientError with human-readable message.
  3. 422 validation failure → raises APIClientError, extracts first error loc/msg.
  4. 400 bad request → raises APIClientError with "Invalid input" prefix.
  5. 404 model unavailable → raises APIClientError with model-specific message.
  6. 503 service not ready → raises APIClientError.
  7. 500 server error → raises APIClientError with status code.
  8. Timeout → raises APIClientError with timeout message.
  9. predict() method assembles full URL and passes JSON payload.
  10. health() and ready() methods call correct endpoints.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from dashboard.api_client import APIClientError, ValuationAPIClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> ValuationAPIClient:
    return ValuationAPIClient(base_url="http://localhost:8000", timeout=10.0)


def _mock_response(status_code: int, json_body: Any) -> MagicMock:
    """Build a fake httpx.Response with ``status_code`` and ``.json()``."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    return resp


# ---------------------------------------------------------------------------
# 1. Successful response
# ---------------------------------------------------------------------------

def test_predict_success_returns_dict(client: ValuationAPIClient) -> None:
    """A 200 response from POST /api/valuation/predict returns a plain dict."""
    payload: Dict[str, Any] = {
        "category": "Cars",
        "brand": "Toyota",
        "model": "Premio",
        "manufacture_year": 2015,
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "district": "Colombo",
        "condition": "Registered (Used)",
        "top_k_factors": 5,
        "top_k_comparables": 5,
        "percentile_lower": 10,
        "percentile_upper": 90,
    }
    expected = {"estimated_asking_price_lkr": 5_000_000.0, "currency": "LKR"}

    mock_resp = _mock_response(200, expected)
    with patch("httpx.post", return_value=mock_resp) as mock_post:
        result = client.predict(payload)

    mock_post.assert_called_once()
    assert result == expected


# ---------------------------------------------------------------------------
# 2. Health endpoint
# ---------------------------------------------------------------------------

def test_health_success(client: ValuationAPIClient) -> None:
    """GET /health with 200 returns {'status': 'ok'}."""
    mock_resp = _mock_response(200, {"status": "ok"})
    with patch("httpx.get", return_value=mock_resp):
        result = client.health()
    assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# 3. Ready endpoint
# ---------------------------------------------------------------------------

def test_ready_success(client: ValuationAPIClient) -> None:
    """GET /ready with 200 returns {'status': 'ready'}."""
    mock_resp = _mock_response(200, {"status": "ready"})
    with patch("httpx.get", return_value=mock_resp):
        result = client.ready()
    assert result["status"] == "ready"


# ---------------------------------------------------------------------------
# 4. Connection error
# ---------------------------------------------------------------------------

def test_predict_connection_error_raises_api_client_error(
    client: ValuationAPIClient,
) -> None:
    """A ConnectError raises APIClientError with a helpful message."""
    import httpx

    with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(APIClientError) as exc_info:
            client.predict({"category": "Cars"})

    assert "Cannot reach the valuation API" in str(exc_info.value)
    assert exc_info.value.status_code is None


def test_health_connection_error_raises_api_client_error(
    client: ValuationAPIClient,
) -> None:
    import httpx

    with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(APIClientError) as exc_info:
            client.health()

    assert "Cannot reach the valuation API" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 5. Timeout
# ---------------------------------------------------------------------------

def test_predict_timeout_raises_api_client_error(client: ValuationAPIClient) -> None:
    """A TimeoutException raises APIClientError mentioning timeout."""
    import httpx

    with patch("httpx.post", side_effect=httpx.TimeoutException("timeout")):
        with pytest.raises(APIClientError) as exc_info:
            client.predict({})

    assert "timed out" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 6. 422 Validation failure
# ---------------------------------------------------------------------------

def test_predict_422_raises_api_client_error_with_detail(
    client: ValuationAPIClient,
) -> None:
    """A 422 response extracts the first validation error location and message."""
    body = {
        "detail": [
            {
                "loc": ["body", "category"],
                "msg": "Invalid vehicle category 'Truck'",
                "type": "value_error",
            }
        ]
    }
    mock_resp = _mock_response(422, body)
    with patch("httpx.post", return_value=mock_resp):
        with pytest.raises(APIClientError) as exc_info:
            client.predict({"category": "Truck"})

    err = exc_info.value
    assert err.status_code == 422
    assert "body" in str(err) or "category" in str(err)
    assert "Invalid vehicle category" in str(err)


# ---------------------------------------------------------------------------
# 7. 400 Bad request
# ---------------------------------------------------------------------------

def test_predict_400_raises_api_client_error(client: ValuationAPIClient) -> None:
    """A 400 response raises APIClientError with 'Invalid input' prefix."""
    body = {"detail": "Either vehicle_age or manufacture_year must be provided."}
    mock_resp = _mock_response(400, body)
    with patch("httpx.post", return_value=mock_resp):
        with pytest.raises(APIClientError) as exc_info:
            client.predict({})

    assert exc_info.value.status_code == 400
    assert "Invalid input" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 8. 404 Model unavailable
# ---------------------------------------------------------------------------

def test_predict_404_raises_model_unavailable_error(
    client: ValuationAPIClient,
) -> None:
    """A 404 response raises APIClientError mentioning model artifact."""
    mock_resp = _mock_response(404, {"detail": "Model not found"})
    with patch("httpx.post", return_value=mock_resp):
        with pytest.raises(APIClientError) as exc_info:
            client.predict({})

    assert exc_info.value.status_code == 404
    assert "model artifact" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 9. 503 Service not ready
# ---------------------------------------------------------------------------

def test_ready_503_raises_api_client_error(client: ValuationAPIClient) -> None:
    """A 503 response from /ready raises APIClientError."""
    mock_resp = _mock_response(503, {"status": "unavailable", "detail": "Not loaded"})
    with patch("httpx.get", return_value=mock_resp):
        with pytest.raises(APIClientError) as exc_info:
            client.ready()

    assert exc_info.value.status_code == 503
    assert "not ready" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 10. 500 Server error
# ---------------------------------------------------------------------------

def test_predict_500_raises_api_client_error(client: ValuationAPIClient) -> None:
    """A 500 response raises APIClientError with status code in message."""
    mock_resp = _mock_response(500, {"detail": "An unexpected internal server error occurred."})
    with patch("httpx.post", return_value=mock_resp):
        with pytest.raises(APIClientError) as exc_info:
            client.predict({})

    assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# 11. URL construction
# ---------------------------------------------------------------------------

def test_predict_posts_to_correct_url(client: ValuationAPIClient) -> None:
    """predict() sends a POST to /api/valuation/predict."""
    mock_resp = _mock_response(200, {"estimated_asking_price_lkr": 1_000_000.0})
    with patch("httpx.post", return_value=mock_resp) as mock_post:
        client.predict({"category": "Cars"})

    call_args = mock_post.call_args
    url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
    assert "/api/valuation/predict" in url


def test_health_gets_correct_url(client: ValuationAPIClient) -> None:
    """health() sends a GET to /health."""
    mock_resp = _mock_response(200, {"status": "ok"})
    with patch("httpx.get", return_value=mock_resp) as mock_get:
        client.health()

    call_args = mock_get.call_args
    url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
    assert "/health" in url


# ---------------------------------------------------------------------------
# 12. Custom base URL respected
# ---------------------------------------------------------------------------

def test_custom_base_url_used() -> None:
    """Client strips trailing slash and uses custom base URL."""
    custom_client = ValuationAPIClient(base_url="http://myserver:9999/")
    assert custom_client.base_url == "http://myserver:9999"

"""
Dashboard API Client for Vehicle Valuation Platform.

Provides a clean, error-handling wrapper around the FastAPI backend endpoints.
All machine-learning logic stays server-side; this module only handles HTTP
transport, response parsing, and user-facing error formatting.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# Default base URL for local development
DEFAULT_API_BASE_URL = "http://localhost:8000"


class APIClientError(Exception):
    """Raised when the API returns a non-success response or is unreachable."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.user_message = message


class ValuationAPIClient:
    """
    HTTP client for the Vehicle Valuation FastAPI backend.

    Usage::

        client = ValuationAPIClient()
        result = client.predict(payload_dict)
    """

    def __init__(
        self,
        base_url: str = DEFAULT_API_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """
        Call GET /health.  Returns dict with ``status`` key.

        Raises:
            APIClientError: If the endpoint is unreachable or returns non-200.
        """
        return self._get("/health")

    def ready(self) -> Dict[str, Any]:
        """
        Call GET /ready.  Returns dict with ``status`` key.

        Raises:
            APIClientError: If the model artifact is unavailable or server error.
        """
        return self._get("/ready")

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call POST /api/valuation/predict.

        Args:
            payload: Dictionary matching ``VehicleValuationRequest`` schema.

        Returns:
            Parsed ``VehicleValuationResponse`` as a plain dict.

        Raises:
            APIClientError: On validation failure (422/400), missing model (404),
                            or server error (500).
        """
        return self._post("/api/valuation/predict", payload)

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = httpx.get(url, timeout=self.timeout)
        except httpx.ConnectError:
            raise APIClientError(
                "Cannot reach the valuation API. Please ensure the FastAPI server is "
                f"running on {self.base_url}."
            )
        except httpx.TimeoutException:
            raise APIClientError(
                f"Request to {url} timed out after {self.timeout}s. "
                "The API server may be overloaded."
            )
        except httpx.RequestError as exc:
            raise APIClientError(f"Network error communicating with API: {exc}")

        return self._parse_response(response)

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = httpx.post(url, json=payload, timeout=self.timeout)
        except httpx.ConnectError:
            raise APIClientError(
                "Cannot reach the valuation API. Please ensure the FastAPI server is "
                f"running on {self.base_url}."
            )
        except httpx.TimeoutException:
            raise APIClientError(
                f"Request to {url} timed out after {self.timeout}s. "
                "The API server may be overloaded."
            )
        except httpx.RequestError as exc:
            raise APIClientError(f"Network error communicating with API: {exc}")

        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: httpx.Response) -> Dict[str, Any]:
        """
        Parses an httpx response into a dict, raising ``APIClientError`` on failure.
        """
        try:
            data = response.json()
        except Exception:
            data = {}

        if response.status_code == 200:
            return data

        # Extract a human-readable detail message
        detail = (
            data.get("detail")
            or data.get("message")
            or f"HTTP {response.status_code}"
        )

        if response.status_code == 422:
            # FastAPI validation error — extract first validation message
            errors = data.get("detail", [])
            if isinstance(errors, list) and errors:
                first = errors[0]
                loc = " → ".join(str(x) for x in first.get("loc", []))
                msg = first.get("msg", "Validation error")
                detail = f"Validation error at [{loc}]: {msg}"
            raise APIClientError(detail, status_code=422)

        if response.status_code == 400:
            raise APIClientError(f"Invalid input: {detail}", status_code=400)

        if response.status_code == 404:
            raise APIClientError(
                "Valuation model artifact is unavailable. "
                "Ensure model training has been completed.",
                status_code=404,
            )

        if response.status_code == 503:
            raise APIClientError(
                "Valuation service is not ready. "
                "The model artifact may not be loaded yet.",
                status_code=503,
            )

        raise APIClientError(
            f"API returned an unexpected error ({response.status_code}): {detail}",
            status_code=response.status_code,
        )

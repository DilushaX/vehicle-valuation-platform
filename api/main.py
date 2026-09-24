"""
FastAPI Application Entrypoint for Vehicle Market Intelligence & Valuation Platform.
"""

import logging
import os
from typing import List

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.valuation import get_valuation_service, router as valuation_router
from api.schemas.schemas import ErrorResponse, HealthResponse, ReadyResponse

logger = logging.getLogger("api")

app = FastAPI(
    title="Sri Lankan Vehicle Market Valuation API",
    description=(
        "Explainable machine learning valuation and market intelligence for Sri Lankan used vehicles.\n\n"
        "**Important Operational Disclaimers**:\n"
        "- **Asking Price vs. Transaction Price**: The model estimates advertised asking prices observed on "
        "online vehicle listings (Riyasewana). It does NOT predict verified transaction or final settlement prices.\n"
        "- **Experimental Research Benchmark**: The valuation model is an experimental research benchmark "
        "trained on an initial verified dataset across 8 vehicle categories.\n"
        "- **Unobserved Physical Factors**: Actual vehicle value may differ due to physical condition, "
        "mechanical state, accident history, and paperwork not captured in listing records."
    ),
    version="1.0.0",
)

# Explicitly configure CORS for local development environments
# Configurable via CORS_ALLOWED_ORIGINS environment variable (comma-separated),
# with secure local development defaults preserving existing Streamlit and local client workflows.
default_origins: List[str] = [
    "http://localhost:8501",  # Streamlit dashboard local
    "http://127.0.0.1:8501",  # Streamlit dashboard loopback
    "http://localhost:8000",  # FastAPI swagger / local client
    "http://127.0.0.1:8000",  # FastAPI swagger / local client loopback
    "http://localhost:3000",  # Standard local frontend development port
    "http://127.0.0.1:3000",  # Standard local frontend development loopback
]
env_origins = os.getenv("CORS_ALLOWED_ORIGINS")
allowed_origins = [o.strip() for o in env_origins.split(",") if o.strip()] if env_origins else default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler ensuring unexpected errors return a safe HTTP 500 response
    without leaking internal stack traces, paths, or secrets.
    """
    logger.error(
        f"Unhandled internal server error during {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected internal server error occurred.",
            "error_type": "InternalServerError",
        },
    )


app.include_router(valuation_router)


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health & Readiness"],
    summary="Service Liveness Check",
    description="Minimal health endpoint verifying the API service is up and running.",
)
def health_check() -> HealthResponse:
    """Liveness check endpoint."""
    return HealthResponse(status="ok")


@app.get(
    "/ready",
    response_model=ReadyResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health & Readiness"],
    summary="Model Readiness Check",
    description=(
        "Verifies that the valuation machine learning model artifact is available and ready for inference. "
        "Returns HTTP 200 with status 'ready' when model is loaded, or HTTP 503 if unavailable."
    ),
    responses={
        200: {"model": ReadyResponse, "description": "Valuation model artifact is available and loaded."},
        503: {"model": ErrorResponse, "description": "Valuation model artifact is unavailable."},
    },
)
def readiness_check():
    """Readiness check endpoint verifying valuation model availability."""
    try:
        service = get_valuation_service()
        if service is None or getattr(service, "predictor", None) is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "unavailable",
                    "detail": "Valuation model is not loaded.",
                    "error_type": "ServiceUnavailable",
                },
            )
        return ReadyResponse(status="ready")
    except Exception as e:
        logger.warning(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unavailable",
                "detail": "Valuation model artifact is unavailable.",
                "error_type": "ServiceUnavailable",
            },
        )

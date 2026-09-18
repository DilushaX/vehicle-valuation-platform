"""
FastAPI Application Entrypoint for Vehicle Market Intelligence & Valuation Platform.
"""

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from api.routes.valuation import router as valuation_router

app = FastAPI(
    title="Sri Lankan Vehicle Market Valuation API",
    description="Explainable machine learning valuation and market intelligence for Sri Lankan vehicles.",
    version="1.0.0",
)

app.include_router(valuation_router)


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
def health_check():
    """Liveness check endpoint."""
    return {"status": "healthy", "service": "vehicle-valuation-api"}

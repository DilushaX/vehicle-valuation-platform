"""
FastAPI Application Entrypoint
Sri Lankan Vehicle Market Intelligence & ML Valuation Platform Backend API
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database.connection import init_db
from api.routes import valuation, comparables, analytics, quality
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes database schema on startup."""
    logger.info("Initializing database schema...")
    init_db()
    logger.info(f"API Server started successfully in {settings.ENVIRONMENT} mode.")
    yield
    logger.info("Shutting down API server...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description=(
        "Production-ready backend API for Sri Lankan Vehicle Market Intelligence, "
        "Historical Price Tracking, Comparable Matching, and ML Valuation with SHAP Explainability."
    ),
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(valuation.router)
app.include_router(comparables.router)
app.include_router(analytics.router)
app.include_router(quality.router)

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "environment": settings.ENVIRONMENT,
        "documentation": "/docs"
    }

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )

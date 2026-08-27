"""
Application Configuration Module
Centralized settings for Database, Scraper, ML Pipeline, API, and Dashboard.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    # App General
    PROJECT_NAME: str = "Sri Lankan Vehicle Market Intelligence & ML Valuation Platform"
    PROJECT_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Database
    # Default to local SQLite database in workspace if PostgreSQL is not specified
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{BASE_DIR}/vehicle_market.db"
    )
    DB_ECHO: bool = False

    # Scraper Settings
    SCRAPER_BASE_URL: str = "https://riyasewana.com"
    SCRAPER_MIN_DELAY_SECONDS: float = 1.5
    SCRAPER_MAX_DELAY_SECONDS: float = 3.0
    SCRAPER_MAX_PAGES_PER_RUN: int = 10
    SCRAPER_TIMEOUT_SECONDS: int = 15
    SCRAPER_USER_AGENT: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )

    # Multi-Category Support
    SUPPORTED_CATEGORIES: list[str] = [
        "cars",
        "vans",
        "suvs",
        "motorbikes",
        "three-wheel",
        "lorries",
        "buses"
    ]

    # Data Quality Thresholds
    MIN_VALID_YEAR: int = 1970
    MAX_VALID_YEAR: int = 2026
    MIN_VALID_MILEAGE: int = 100
    MAX_VALID_MILEAGE: int = 1_000_000
    MIN_VALID_PRICE_LKR: float = 100_000.0  # Rs. 100,000 min
    MAX_VALID_PRICE_LKR: float = 250_000_000.0  # Rs. 250 Million max

    # ML & Valuation Settings
    ML_MODELS_DIR: Path = BASE_DIR / "models"
    MIN_TRAINING_SAMPLES_PER_CATEGORY: int = 15
    MIN_TRAINING_SAMPLES_PER_MODEL: int = 5
    RANDOM_STATE: int = 42

    # API Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Dashboard Server
    DASHBOARD_PORT: int = 8501

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()

# Ensure model and data storage directories exist
os.makedirs(settings.ML_MODELS_DIR, exist_ok=True)
os.makedirs(BASE_DIR / "data" / "raw", exist_ok=True)
os.makedirs(BASE_DIR / "data" / "processed", exist_ok=True)
os.makedirs(BASE_DIR / "data" / "samples", exist_ok=True)

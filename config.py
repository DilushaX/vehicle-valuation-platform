import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration settings.
    Values can be overridden via environment variables or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATABASE_URL: str = (
        "postgresql+psycopg2://postgres:postgres@localhost:5432/vehicle_valuation"
    )
    REQUEST_TIMEOUT: float = 20.0
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    RETRY_BACKOFF: float = 2.0
    REQUEST_DELAY: float = 1.5
    RAW_DATA_DIR: Path = Path("data/raw")
    SOURCE_NAME: str = "riyasewana"


settings = Settings()

"""
Database Connection & Session Management
Supports both PostgreSQL (Production) and SQLite (Development/Test) seamlessly.
"""
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config import settings

connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=settings.DB_ECHO,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

def init_db() -> None:
    """Initialize database tables."""
    import database.models  # Ensure models are imported
    Base.metadata.create_all(bind=engine)

def get_db() -> Generator:
    """Dependency for obtaining database session in API routes or scripts."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

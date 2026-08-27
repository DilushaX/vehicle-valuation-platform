"""
Database Models Module
Defines SQLAlchemy ORM entities supporting historical tracking, delta observations,
data quality audit, price movements, scrape runs, and ML model registry.
"""
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    Index
)
from sqlalchemy.orm import relationship
from database.connection import Base

def utc_now():
    return datetime.now(timezone.utc)

class Vehicle(Base):
    """Canonical vehicle entity grouping make, model, year, category and core specifications."""
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    category = Column(String(50), nullable=False, index=True)
    make = Column(String(100), nullable=False, index=True)
    model = Column(String(100), nullable=False, index=True)
    yom = Column(Integer, nullable=False, index=True)
    fuel_type = Column(String(50), nullable=True, index=True)
    transmission = Column(String(50), nullable=True, index=True)
    engine_cc = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=utc_now)

    listings = relationship("Listing", back_populates="vehicle")

    __table_args__ = (
        Index("ix_vehicles_canonical", "category", "make", "model", "yom", "fuel_type", "transmission"),
    )


class Listing(Base):
    """
    Current state of a vehicle listing.
    Listings are never hard-deleted when they disappear from the website;
    their status is transitioned to NO_LONGER_OBSERVED.
    """
    __tablename__ = "listings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    listing_id = Column(String(100), unique=True, nullable=False, index=True)
    source = Column(String(50), default="riyasewana", nullable=False)
    url = Column(String(500), nullable=True)
    title = Column(String(255), nullable=True)
    
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=True)
    vehicle = relationship("Vehicle", back_populates="listings")

    # Structured Fields
    category = Column(String(50), nullable=False, index=True)
    make = Column(String(100), nullable=False, index=True)
    model = Column(String(100), nullable=False, index=True)
    yom = Column(Integer, nullable=False, index=True)
    condition = Column(String(50), nullable=True)  # Brand New, Reconditioned, Used
    mileage_km = Column(Integer, nullable=True, index=True)
    fuel_type = Column(String(50), nullable=True, index=True)
    transmission = Column(String(50), nullable=True, index=True)
    engine_cc = Column(Integer, nullable=True)
    district = Column(String(100), nullable=True, index=True)
    
    # Financial & State
    current_price = Column(Float, nullable=False, index=True)
    current_status = Column(
        String(50),
        default="ACTIVE",
        nullable=False,
        index=True
    )  # ACTIVE, NO_LONGER_OBSERVED, UPDATED, PRICE_CHANGED, DUPLICATE, FAILED
    
    data_quality_status = Column(
        String(50),
        default="VALID",
        nullable=False,
        index=True
    )  # VALID, SUSPICIOUS, INVALID, MISSING
    quality_score = Column(Float, default=100.0)

    # Observation Timestamps
    first_seen_date = Column(DateTime, default=utc_now, nullable=False, index=True)
    last_seen_date = Column(DateTime, default=utc_now, nullable=False, index=True)
    posted_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Relationships
    price_history = relationship("PriceHistory", back_populates="listing", cascade="all, delete-orphan")
    status_history = relationship("ListingStatusHistory", back_populates="listing", cascade="all, delete-orphan")
    quality_records = relationship("DataQualityRecord", back_populates="listing", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_listings_search", "category", "make", "model", "yom", "current_status"),
    )


class PriceHistory(Base):
    """Audit log for every observed price change of a listing over time."""
    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    listing_id = Column(Integer, ForeignKey("listings.id"), nullable=False, index=True)
    price = Column(Float, nullable=False)
    recorded_at = Column(DateTime, default=utc_now, nullable=False, index=True)
    change_amount = Column(Float, nullable=True)
    change_percent = Column(Float, nullable=True)

    listing = relationship("Listing", back_populates="price_history")


class ListingStatusHistory(Base):
    """Tracks state transitions of listings (e.g. ACTIVE -> NO_LONGER_OBSERVED)."""
    __tablename__ = "listing_status_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    listing_id = Column(Integer, ForeignKey("listings.id"), nullable=False, index=True)
    old_status = Column(String(50), nullable=True)
    new_status = Column(String(50), nullable=False)
    changed_at = Column(DateTime, default=utc_now, nullable=False)
    notes = Column(Text, nullable=True)

    listing = relationship("Listing", back_populates="status_history")


class ScrapeRun(Base):
    """Tracks metadata and telemetry for each scrape job execution."""
    __tablename__ = "scrape_runs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_id = Column(String(100), unique=True, nullable=False, index=True)
    category = Column(String(50), nullable=True, index=True)
    started_at = Column(DateTime, default=utc_now, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    total_scraped = Column(Integer, default=0)
    new_count = Column(Integer, default=0)
    updated_count = Column(Integer, default=0)
    price_change_count = Column(Integer, default=0)
    no_longer_observed_count = Column(Integer, default=0)
    duplicate_count = Column(Integer, default=0)
    invalid_count = Column(Integer, default=0)
    status = Column(String(50), default="RUNNING", nullable=False)  # RUNNING, COMPLETED, FAILED
    error_message = Column(Text, nullable=True)


class DataQualityRecord(Base):
    """Detailed quality validation results per listing."""
    __tablename__ = "data_quality_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    listing_id = Column(Integer, ForeignKey("listings.id"), nullable=False, index=True)
    status = Column(String(50), nullable=False)  # VALID, SUSPICIOUS, INVALID, MISSING
    issues_json = Column(Text, nullable=True)  # JSON string of detected issues
    is_outlier = Column(Boolean, default=False)
    score = Column(Float, default=100.0)
    evaluated_at = Column(DateTime, default=utc_now, nullable=False)

    listing = relationship("Listing", back_populates="quality_records")


class ModelVersion(Base):
    """Registry of trained machine learning models with evaluation metrics."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_id = Column(String(100), unique=True, nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    algorithm = Column(String(100), nullable=False)
    metrics_json = Column(Text, nullable=False)  # MAE, RMSE, R2, MAPE
    feature_names_json = Column(Text, nullable=False)
    artifact_path = Column(String(500), nullable=False)
    is_active = Column(Boolean, default=True)
    trained_at = Column(DateTime, default=utc_now, nullable=False)


class PredictionLog(Base):
    """Log of ML valuation inferences and SHAP explainability summaries."""
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    category = Column(String(50), nullable=False)
    make = Column(String(100), nullable=False)
    model = Column(String(100), nullable=False)
    yom = Column(Integer, nullable=False)
    mileage_km = Column(Integer, nullable=False)
    fuel_type = Column(String(50), nullable=True)
    transmission = Column(String(50), nullable=True)
    district = Column(String(100), nullable=True)
    asking_price = Column(Float, nullable=True)
    
    predicted_value = Column(Float, nullable=False)
    range_low = Column(Float, nullable=False)
    range_high = Column(Float, nullable=False)
    confidence = Column(String(20), nullable=False)  # High, Medium, Low
    assessment = Column(String(50), nullable=True)  # BELOW_MARKET, WITHIN_MARKET, ABOVE_MARKET
    shap_values_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

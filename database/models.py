import json
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.connection import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Vehicle(Base):
    __tablename__ = "vehicles"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    category: Mapped[str | None] = mapped_column(String(50))
    brand: Mapped[str | None] = mapped_column(String(100))
    model: Mapped[str | None] = mapped_column(String(150))

    manufacture_year: Mapped[int | None] = mapped_column(Integer)
    registration_year: Mapped[int | None] = mapped_column(Integer)

    fuel_type: Mapped[str | None] = mapped_column(String(50))
    transmission: Mapped[str | None] = mapped_column(String(50))
    engine_cc: Mapped[int | None] = mapped_column(Integer)
    condition: Mapped[str | None] = mapped_column(String(50))

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        onupdate=utc_now,
    )

    listings: Mapped[list["Listing"]] = relationship(
        back_populates="vehicle",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_vehicle_category", "category"),
        Index("idx_vehicle_brand_model", "brand", "model"),
    )


class Listing(Base):
    __tablename__ = "listings"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    listing_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    vehicle_id: Mapped[int] = mapped_column(
        ForeignKey("vehicles.id"),
        nullable=False,
    )

    listing_url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    title: Mapped[str | None] = mapped_column(Text)

    description: Mapped[str | None] = mapped_column(Text)

    location: Mapped[str | None] = mapped_column(String(150))
    district: Mapped[str | None] = mapped_column(String(100))
    ad_date: Mapped[str | None] = mapped_column(String(100))

    source: Mapped[str] = mapped_column(
        String(50),
        default="riyasewana",
    )

    current_status: Mapped[str] = mapped_column(
        String(50),
        default="ACTIVE",
    )

    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
    )

    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
    )

    validation_issues: Mapped[str | None] = mapped_column(Text)

    quality_score: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 2)
    )

    ml_eligible: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )

    vehicle: Mapped["Vehicle"] = relationship(
        back_populates="listings"
    )

    price_history: Mapped[list["PriceHistory"]] = relationship(
        back_populates="listing",
        cascade="all, delete-orphan",
    )

    observations: Mapped[list["ListingObservation"]] = relationship(
        back_populates="listing",
        cascade="all, delete-orphan",
    )

    @property
    def condition(self) -> str | None:
        return self.vehicle.condition if self.vehicle else None

    def get_validation_issues(self) -> list:
        if not self.validation_issues:
            return []
        try:
            return json.loads(self.validation_issues)
        except (json.JSONDecodeError, TypeError):
            return []

    __table_args__ = (
        UniqueConstraint("source", "listing_id", name="uq_source_listing_id"),
        Index("idx_listing_status", "current_status"),
        Index("idx_listing_source", "source"),
        Index("idx_listing_vehicle", "vehicle_id"),
    )


class PriceHistory(Base):
    __tablename__ = "price_history"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    listing_id: Mapped[int] = mapped_column(
        ForeignKey("listings.id"),
        nullable=False,
    )

    price: Mapped[int | None] = mapped_column(Integer)

    observed_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
    )

    listing: Mapped["Listing"] = relationship(
        back_populates="price_history"
    )

    __table_args__ = (
        Index("idx_price_history_listing", "listing_id"),
        Index("idx_price_history_date", "observed_at"),
    )


class ScrapeRun(Base):
    __tablename__ = "scrape_runs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    source: Mapped[str] = mapped_column(
        String(50),
        default="riyasewana",
    )

    category: Mapped[str | None] = mapped_column(String(100))

    started_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime
    )

    pages_requested: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    pages_scraped: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    failed_pages: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    listings_found: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    new_listings: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    updated_listings: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    errors: Mapped[str | None] = mapped_column(Text)

    status: Mapped[str] = mapped_column(
        String(30),
        default="RUNNING",
    )

    observations: Mapped[list["ListingObservation"]] = relationship(
        back_populates="scrape_run",
        cascade="all, delete-orphan",
    )


class ListingObservation(Base):
    __tablename__ = "listing_observations"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    listing_id: Mapped[int] = mapped_column(
        ForeignKey("listings.id"),
        nullable=False,
    )

    scrape_run_id: Mapped[int] = mapped_column(
        ForeignKey("scrape_runs.id"),
        nullable=False,
    )

    observed_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
    )

    observed_price: Mapped[int | None] = mapped_column(Integer)

    observed_mileage: Mapped[int | None] = mapped_column(Integer)

    availability: Mapped[str] = mapped_column(
        String(30),
        default="AVAILABLE",
    )

    listing: Mapped["Listing"] = relationship(
        back_populates="observations"
    )

    scrape_run: Mapped["ScrapeRun"] = relationship(
        back_populates="observations"
    )

    __table_args__ = (
        Index("idx_observation_listing", "listing_id"),
        Index("idx_observation_scrape", "scrape_run_id"),
        Index("idx_observation_date", "observed_at"),
    )

"""
EDA Dataset Loader.
Provides read-only retrieval and DataFrame construction for vehicle market analysis
from PostgreSQL or test database sessions.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from data_pipeline.cleaning.cleaners import VehicleCleaner
from database.connection import get_sessionmaker
from database.models import (
    Listing,
    ListingObservation,
    PriceHistory,
    Vehicle,
)

logger = logging.getLogger(__name__)

LISTING_COLUMNS = [
    "listing_id",
    "category",
    "canonical_category",
    "brand",
    "model",
    "manufacture_year",
    "registration_year",
    "vehicle_age",
    "asking_price",
    "mileage",
    "fuel_type",
    "transmission",
    "engine_cc",
    "condition",
    "location",
    "district",
    "ad_date",
    "current_status",
    "ml_eligible",
    "quality_score",
    "validation_issues",
    "first_seen_at",
    "last_seen_at",
    "observation_count",
    "price_history_count",
]


class EDADatasetLoader:
    """
    Reusable, read-only dataset loader for Exploratory Data Analysis.
    Extracts structured market datasets into pandas DataFrames without mutating records.
    """

    def __init__(self, session_factory=None):
        self._session_factory = session_factory or get_sessionmaker()

    @contextmanager
    def _get_session(self, session: Optional[Session] = None) -> Iterator[Session]:
        """Provides a database session, reusing an existing one or opening a new one."""
        if session is not None:
            yield session
        else:
            new_session = self._session_factory()
            try:
                yield new_session
            finally:
                new_session.close()

    def load_listings(
        self,
        session: Optional[Session] = None,
        category: Optional[str] = None,
        ml_eligible_only: bool = False,
        active_only: bool = False,
    ) -> pd.DataFrame:
        """
        Loads listings joined with vehicles, latest price history, and latest observation.
        Computes dynamic vehicle age and normalized attributes.
        Returns a read-only pandas DataFrame.
        """
        with self._get_session(session) as s:
            stmt = select(Listing).join(Vehicle)

            if ml_eligible_only:
                stmt = stmt.where(Listing.ml_eligible.is_(True))

            if active_only:
                stmt = stmt.where(Listing.current_status == "ACTIVE")

            if category:
                can_cat = VehicleCleaner.canonicalize_category(category) or category
                singular = can_cat.rstrip("s")
                stmt = stmt.where(
                    or_(
                        Vehicle.category == can_cat,
                        Vehicle.category == singular,
                        Vehicle.category.ilike(f"%{can_cat}%"),
                    )
                )

            listings = list(s.scalars(stmt).all())

            if not listings:
                return pd.DataFrame(columns=LISTING_COLUMNS)

            current_year = datetime.now(timezone.utc).year
            records: List[Dict[str, Any]] = []

            for l in listings:
                v = l.vehicle
                cat_raw = v.category if v else None
                can_cat = VehicleCleaner.canonicalize_category(cat_raw) or cat_raw or "Unknown"

                # Extract latest asking price from price history or observations
                latest_price = None
                if l.price_history:
                    sorted_ph = sorted(
                        l.price_history,
                        key=lambda p: (p.observed_at or datetime.min, p.id or 0),
                        reverse=True,
                    )
                    latest_price = sorted_ph[0].price

                # Extract latest mileage and fallback price from observations
                latest_mileage = None
                if l.observations:
                    sorted_obs = sorted(
                        l.observations,
                        key=lambda o: (o.observed_at or datetime.min, o.id or 0),
                        reverse=True,
                    )
                    latest_mileage = sorted_obs[0].observed_mileage
                    if latest_price is None:
                        latest_price = sorted_obs[0].observed_price

                # Derive vehicle age dynamically (not hard-coded)
                manuf_year = v.manufacture_year if v else None
                vehicle_age = None
                if manuf_year and isinstance(manuf_year, int) and 1900 <= manuf_year <= current_year:
                    vehicle_age = current_year - manuf_year

                # Parse validation issues list
                issues_raw = l.get_validation_issues()

                records.append(
                    {
                        "listing_id": l.listing_id,
                        "category": cat_raw,
                        "canonical_category": can_cat,
                        "brand": v.brand if v else None,
                        "model": v.model if v else None,
                        "manufacture_year": manuf_year,
                        "registration_year": v.registration_year if v else None,
                        "vehicle_age": vehicle_age,
                        "asking_price": latest_price,
                        "mileage": latest_mileage,
                        "fuel_type": VehicleCleaner.normalize_fuel_type(v.fuel_type) if v else None,
                        "transmission": VehicleCleaner.normalize_transmission(v.transmission) if v else None,
                        "engine_cc": v.engine_cc if v else None,
                        "condition": v.condition if v else None,
                        "location": l.location,
                        "district": l.district,
                        "ad_date": l.ad_date,
                        "current_status": l.current_status,
                        "ml_eligible": bool(l.ml_eligible),
                        "quality_score": float(l.quality_score) if l.quality_score is not None else None,
                        "validation_issues": issues_raw,
                        "first_seen_at": l.first_seen_at,
                        "last_seen_at": l.last_seen_at,
                        "observation_count": len(l.observations),
                        "price_history_count": len(l.price_history),
                    }
                )

            df = pd.DataFrame(records)
            return df

    def load_eligible_listings(
        self,
        session: Optional[Session] = None,
        category: Optional[str] = None,
    ) -> pd.DataFrame:
        """Loads only ML-eligible listings."""
        return self.load_listings(
            session=session,
            category=category,
            ml_eligible_only=True,
            active_only=False,
        )

    def load_category(
        self,
        category: str,
        session: Optional[Session] = None,
        ml_eligible_only: bool = False,
    ) -> pd.DataFrame:
        """Loads listings for a specific category."""
        return self.load_listings(
            session=session,
            category=category,
            ml_eligible_only=ml_eligible_only,
            active_only=False,
        )

    def load_price_history(
        self,
        session: Optional[Session] = None,
        listing_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Loads price history records for market dynamics and historical analysis."""
        with self._get_session(session) as s:
            stmt = select(PriceHistory, Listing.listing_id).join(Listing)
            if listing_id:
                stmt = stmt.where(Listing.listing_id == str(listing_id).strip())

            stmt = stmt.order_by(PriceHistory.observed_at.asc())
            rows = s.execute(stmt).all()

            if not rows:
                return pd.DataFrame(
                    columns=["id", "listing_id", "price", "observed_at"]
                )

            records = [
                {
                    "id": ph.id,
                    "listing_id": lid,
                    "price": ph.price,
                    "observed_at": ph.observed_at,
                }
                for ph, lid in rows
            ]
            return pd.DataFrame(records)

    def load_observations(
        self,
        session: Optional[Session] = None,
        listing_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Loads daily scrape observation records for lifecycle and availability analysis."""
        with self._get_session(session) as s:
            stmt = select(ListingObservation, Listing.listing_id).join(Listing)
            if listing_id:
                stmt = stmt.where(Listing.listing_id == str(listing_id).strip())

            stmt = stmt.order_by(ListingObservation.observed_at.asc())
            rows = s.execute(stmt).all()

            if not rows:
                return pd.DataFrame(
                    columns=[
                        "id",
                        "listing_id",
                        "scrape_run_id",
                        "observed_price",
                        "observed_mileage",
                        "availability",
                        "observed_at",
                    ]
                )

            records = [
                {
                    "id": obs.id,
                    "listing_id": lid,
                    "scrape_run_id": obs.scrape_run_id,
                    "observed_price": obs.observed_price,
                    "observed_mileage": obs.observed_mileage,
                    "availability": obs.availability,
                    "observed_at": obs.observed_at,
                }
                for obs, lid in rows
            ]
            return pd.DataFrame(records)

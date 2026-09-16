"""
ML Dataset Loader.
Provides read-only extraction of ML-eligible listing records from PostgreSQL
for feature engineering and ML dataset preparation.
"""

from contextlib import contextmanager
from datetime import datetime
import logging
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
from sqlalchemy import or_, select
from sqlalchemy.orm import Session, joinedload

from data_pipeline.cleaning.cleaners import VehicleCleaner
from database.connection import get_sessionmaker
from database.models import Listing, Vehicle

logger = logging.getLogger(__name__)

RAW_FEATURE_COLUMNS = [
    "listing_id",
    "vehicle_id",
    "category",
    "brand",
    "model",
    "manufacture_year",
    "registration_year",
    "mileage",
    "engine_cc",
    "fuel_type",
    "transmission",
    "district",
    "condition",
    "asking_price",
    "ml_eligible",
    "first_seen_at",
]


class MLDatasetLoader:
    """
    Read-only dataset loader for machine learning feature preparation.
    Extracts structured listing and vehicle records from PostgreSQL.
    
    Guarantees:
    1. Read-only operation (no writes or commits to database).
    2. Default filter to ml_eligible = True.
    3. Excludes seller phone, email, and raw contact data.
    4. Deterministic ordering by listing primary key.
    5. Target variable is explicitly seller asking price, not transaction price.
    """

    def __init__(self, session_factory=None):
        self._session_factory = session_factory or get_sessionmaker()

    @contextmanager
    def _get_session(self, session: Optional[Session] = None) -> Iterator[Session]:
        """Provides a database session, reusing an existing one or creating a new read-only one."""
        if session is not None:
            yield session
        else:
            new_session = self._session_factory()
            try:
                yield new_session
            finally:
                new_session.close()

    def load_raw_dataset(
        self,
        session: Optional[Session] = None,
        ml_eligible_only: bool = True,
        category: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Loads listings joined with vehicles, latest price history, and latest observation.
        
        Args:
            session: Optional existing SQLAlchemy session.
            ml_eligible_only: If True (default), filters for ml_eligible == True.
            category: Optional category filter.
            
        Returns:
            pandas DataFrame containing raw candidate features and traceability IDs.
        """
        with self._get_session(session) as s:
            stmt = (
                select(Listing)
                .join(Vehicle)
                .options(
                    joinedload(Listing.vehicle),
                    joinedload(Listing.price_history),
                    joinedload(Listing.observations),
                )
                .order_by(Listing.id.asc())
            )

            if ml_eligible_only:
                stmt = stmt.where(Listing.ml_eligible.is_(True))

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

            listings = list(s.scalars(stmt).unique().all())

            if not listings:
                return pd.DataFrame(columns=RAW_FEATURE_COLUMNS)

            records: List[Dict[str, Any]] = []

            for l in listings:
                v = l.vehicle
                cat_raw = v.category if v else None
                can_cat = VehicleCleaner.canonicalize_category(cat_raw) or cat_raw or "Unknown"

                # Extract latest asking price from price history or fallback to observations
                latest_price: Optional[float] = None
                if l.price_history:
                    sorted_ph = sorted(
                        l.price_history,
                        key=lambda p: (p.observed_at or datetime.min, p.id or 0),
                        reverse=True,
                    )
                    latest_price = float(sorted_ph[0].price) if sorted_ph[0].price is not None else None

                # Extract latest mileage and fallback price from observations
                latest_mileage: Optional[float] = None
                if l.observations:
                    sorted_obs = sorted(
                        l.observations,
                        key=lambda o: (o.observed_at or datetime.min, o.id or 0),
                        reverse=True,
                    )
                    latest_mileage = float(sorted_obs[0].observed_mileage) if sorted_obs[0].observed_mileage is not None else None
                    if latest_price is None and sorted_obs[0].observed_price is not None:
                        latest_price = float(sorted_obs[0].observed_price)

                # Clean and normalize categorical fields
                fuel_clean = VehicleCleaner.normalize_fuel_type(v.fuel_type) if v else None
                trans_clean = VehicleCleaner.normalize_transmission(v.transmission) if v else None

                records.append(
                    {
                        "listing_id": l.listing_id,
                        "vehicle_id": v.id if v else None,
                        "category": can_cat,
                        "brand": v.brand if v else None,
                        "model": v.model if v else None,
                        "manufacture_year": v.manufacture_year if v else None,
                        "registration_year": v.registration_year if v else None,
                        "mileage": latest_mileage,
                        "engine_cc": float(v.engine_cc) if v and v.engine_cc is not None else None,
                        "fuel_type": fuel_clean,
                        "transmission": trans_clean,
                        "district": l.district,
                        "condition": v.condition if v else None,
                        "asking_price": latest_price,
                        "ml_eligible": bool(l.ml_eligible),
                        "first_seen_at": l.first_seen_at,
                    }
                )

            df = pd.DataFrame(records)
            return df

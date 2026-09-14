"""
Tests for Phase 4 — Step 6: Exploratory Data Analysis (EDA).
Covers:
1. Dataset loading, category filtering, ML-eligibility filtering, and active-status filtering.
2. Empty dataset handling without errors.
3. Read-only safety guarantees (no database mutations).
4. Derived analytical fields (vehicle_age, canonical_category).
5. PriceHistory and Observation loader integration.
"""

from datetime import datetime, timezone
import pytest
import pandas as pd
from sqlalchemy import select, func

from database.models import Listing, ListingObservation, PriceHistory, ScrapeRun, Vehicle
from database.repository import VehicleRepository
from eda.dataset import EDADatasetLoader, LISTING_COLUMNS


# ==============================================================================
# 1. DATASET LOADER TESTS
# ==============================================================================

def test_load_empty_dataset(db_session):
    loader = EDADatasetLoader()
    df = loader.load_listings(session=db_session)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == LISTING_COLUMNS

    df_ph = loader.load_price_history(session=db_session)
    assert isinstance(df_ph, pd.DataFrame)
    assert len(df_ph) == 0

    df_obs = loader.load_observations(session=db_session)
    assert isinstance(df_obs, pd.DataFrame)
    assert len(df_obs) == 0


def test_load_listings_basic(db_session, vehicle_repo):
    # Setup test vehicles & listings
    v1 = vehicle_repo.create_vehicle({
        "category": "Cars",
        "brand": "Toyota",
        "model": "Corolla",
        "manufacture_year": 2018,
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "engine_cc": 1500,
    })
    l1 = vehicle_repo.create_listing(v1, {
        "listing_id": "1001",
        "listing_url": "https://riyasewana.com/buy/1001",
        "title": "Toyota Corolla 2018",
        "district": "Colombo",
        "price": 8500000,
        "ml_eligible": True,
    })
    vehicle_repo.add_price_history(l1, 8500000)

    sr = vehicle_repo.create_scrape_run(category="Cars")
    vehicle_repo.add_observation(l1, sr, price=8500000, mileage=45000)

    vehicle_repo.commit()

    loader = EDADatasetLoader()
    df = loader.load_listings(session=db_session)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["listing_id"] == "1001"
    assert row["brand"] == "Toyota"
    assert row["model"] == "Corolla"
    assert row["canonical_category"] == "Cars"
    assert row["asking_price"] == 8500000
    assert row["mileage"] == 45000
    assert row["ml_eligible"] == True
    current_year = datetime.now(timezone.utc).year
    assert row["vehicle_age"] == current_year - 2018


def test_load_listings_filters(db_session, vehicle_repo):
    sr = vehicle_repo.create_scrape_run(category="Test")

    # Listing 1: Active, ML-eligible, Car
    v1 = vehicle_repo.create_vehicle({"category": "Car", "brand": "Toyota", "manufacture_year": 2015})
    l1 = vehicle_repo.create_listing(v1, {"listing_id": "L1", "listing_url": "u1", "price": 5000000, "ml_eligible": True})
    vehicle_repo.add_price_history(l1, 5000000)
    vehicle_repo.add_observation(l1, sr, price=5000000, mileage=60000)

    # Listing 2: Active, ML-ineligible, Van
    v2 = vehicle_repo.create_vehicle({"category": "Van", "brand": "Nissan", "manufacture_year": 2010})
    l2 = vehicle_repo.create_listing(v2, {"listing_id": "L2", "listing_url": "u2", "price": 4000000, "ml_eligible": False})
    vehicle_repo.add_price_history(l2, 4000000)
    vehicle_repo.add_observation(l2, sr, price=4000000, mileage=120000)

    # Listing 3: NO_LONGER_OBSERVED, ML-eligible, Car
    v3 = vehicle_repo.create_vehicle({"category": "Car", "brand": "Honda", "manufacture_year": 2019})
    l3 = vehicle_repo.create_listing(v3, {"listing_id": "L3", "listing_url": "u3", "price": 7000000, "ml_eligible": True})
    l3.current_status = "NO_LONGER_OBSERVED"
    vehicle_repo.add_price_history(l3, 7000000)
    vehicle_repo.add_observation(l3, sr, price=7000000, mileage=30000)

    vehicle_repo.commit()

    loader = EDADatasetLoader()

    # All listings
    df_all = loader.load_listings(session=db_session)
    assert len(df_all) == 3

    # ML-eligible only
    df_eligible = loader.load_eligible_listings(session=db_session)
    assert len(df_eligible) == 2
    assert set(df_eligible["listing_id"]) == {"L1", "L3"}

    # Category filter (Cars)
    df_cars = loader.load_category("Cars", session=db_session)
    assert len(df_cars) == 2
    assert set(df_cars["listing_id"]) == {"L1", "L3"}

    # Category filter (Vans)
    df_vans = loader.load_category("Vans", session=db_session)
    assert len(df_vans) == 1
    assert df_vans.iloc[0]["listing_id"] == "L2"

    # Active only
    df_active = loader.load_listings(session=db_session, active_only=True)
    assert len(df_active) == 2
    assert set(df_active["listing_id"]) == {"L1", "L2"}


def test_read_only_safety(db_session, vehicle_repo):
    """Verify that dataset loader queries do not modify database rows or counts."""
    v = vehicle_repo.create_vehicle({"category": "Motorbike", "brand": "Bajaj", "manufacture_year": 2020})
    l = vehicle_repo.create_listing(v, {"listing_id": "MB1", "listing_url": "u_mb", "price": 350000, "ml_eligible": True})
    vehicle_repo.add_price_history(l, 350000)
    vehicle_repo.commit()

    before_v = db_session.scalar(select(func.count(Vehicle.id)))
    before_l = db_session.scalar(select(func.count(Listing.id)))
    before_ph = db_session.scalar(select(func.count(PriceHistory.id)))

    loader = EDADatasetLoader()
    df = loader.load_listings(session=db_session)
    df_ph = loader.load_price_history(session=db_session)
    df_obs = loader.load_observations(session=db_session)

    after_v = db_session.scalar(select(func.count(Vehicle.id)))
    after_l = db_session.scalar(select(func.count(Listing.id)))
    after_ph = db_session.scalar(select(func.count(PriceHistory.id)))

    assert before_v == after_v == 1
    assert before_l == after_l == 1
    assert before_ph == after_ph == 1


def test_load_price_history_and_observations(db_session, vehicle_repo):
    sr = vehicle_repo.create_scrape_run(category="Cars")
    v = vehicle_repo.create_vehicle({"category": "Car", "brand": "Toyota"})
    l = vehicle_repo.create_listing(v, {"listing_id": "PH_OBS_1", "listing_url": "url"})

    vehicle_repo.add_price_history(l, 5000000, observed_at=datetime(2026, 9, 8, tzinfo=timezone.utc))
    vehicle_repo.add_price_history(l, 4800000, observed_at=datetime(2026, 9, 9, tzinfo=timezone.utc))

    vehicle_repo.add_observation(l, sr, price=5000000, mileage=50000, observed_at=datetime(2026, 9, 8, tzinfo=timezone.utc))
    vehicle_repo.add_observation(l, sr, price=4800000, mileage=50050, observed_at=datetime(2026, 9, 9, tzinfo=timezone.utc))
    vehicle_repo.commit()

    loader = EDADatasetLoader()
    df_ph = loader.load_price_history(session=db_session, listing_id="PH_OBS_1")
    assert len(df_ph) == 2
    assert list(df_ph["price"]) == [5000000, 4800000]

    df_obs = loader.load_observations(session=db_session, listing_id="PH_OBS_1")
    assert len(df_obs) == 2
    assert list(df_obs["observed_mileage"]) == [50000, 50050]

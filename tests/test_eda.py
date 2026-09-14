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


# ==============================================================================
# 2. DESCRIPTIVE & CATEGORY ANALYSIS TESTS
# ==============================================================================

def test_descriptive_and_categorical_empty():
    from eda.descriptive import DescriptiveAnalyzer
    from eda.categorical import CategoricalAnalyzer

    empty_df = pd.DataFrame()
    overview = DescriptiveAnalyzer.compute_dataset_overview(empty_df)
    assert overview["total_listings"] == 0
    assert overview["ml_eligible_listings"] == 0

    num_summary = DescriptiveAnalyzer.compute_numerical_summary(empty_df)
    assert len(num_summary) == 0

    cat_summary = CategoricalAnalyzer.analyze_categories(empty_df)
    assert len(cat_summary) == 0


def test_compute_dataset_overview_and_numerical_summary():
    from eda.descriptive import DescriptiveAnalyzer

    data = {
        "listing_id": ["1", "2", "3", "4"],
        "canonical_category": ["Cars", "Cars", "Vans", "Motorbikes"],
        "category": ["Car", "Car", "Van", "Motorbike"],
        "brand": ["Toyota", "Toyota", "Nissan", "Bajaj"],
        "model": ["Corolla", "Axio", "Caravan", "Pulsar"],
        "asking_price": [5000000, 6000000, 4500000, 500000],
        "mileage": [50000, 60000, 100000, 20000],
        "manufacture_year": [2015, 2017, 2012, 2020],
        "registration_year": [2016, 2018, 2014, 2021],
        "vehicle_age": [11, 9, 14, 6],
        "engine_cc": [1500, 1500, 2500, 150],
        "current_status": ["ACTIVE", "ACTIVE", "ACTIVE", "NO_LONGER_OBSERVED"],
        "ml_eligible": [True, True, True, False],
        "district": ["Colombo", "Gampaha", "Colombo", "Kandy"],
        "fuel_type": ["Petrol", "Hybrid", "Diesel", "Petrol"],
        "transmission": ["Automatic", "Automatic", "Manual", "Manual"],
        "observation_count": [1, 2, 1, 1],
        "price_history_count": [1, 1, 1, 1],
    }
    df = pd.DataFrame(data)

    overview = DescriptiveAnalyzer.compute_dataset_overview(df)
    assert overview["total_listings"] == 4
    assert overview["ml_eligible_listings"] == 3
    assert overview["ml_ineligible_listings"] == 1
    assert overview["ml_eligibility_rate_pct"] == 75.0
    assert overview["active_listings"] == 3
    assert overview["no_longer_observed_listings"] == 1
    assert overview["distinct_categories"] == 3
    assert overview["distinct_brands"] == 3
    assert overview["distinct_districts"] == 3

    num_summary = DescriptiveAnalyzer.compute_numerical_summary(df)
    assert len(num_summary) == 6
    price_row = num_summary[num_summary["variable"] == "asking_price"].iloc[0]
    assert price_row["count"] == 4
    assert price_row["min"] == 500000
    assert price_row["max"] == 6000000
    assert price_row["median"] == 4750000.0
    assert price_row["iqr"] == price_row["q3"] - price_row["q1"]


def test_analyze_categories():
    from eda.categorical import CategoricalAnalyzer

    data = {
        "listing_id": ["1", "2", "3", "4", "5"],
        "canonical_category": ["Cars", "Cars", "Vans", "Three Wheelers", "Cars"],
        "asking_price": [5000000, 7000000, 4000000, 800000, 6000000],
        "mileage": [50000, 40000, 120000, 30000, 60000],
        "manufacture_year": [2015, 2018, 2010, 2012, 2016],
        "ml_eligible": [True, True, False, True, True],
    }
    df = pd.DataFrame(data)

    cat_df = CategoricalAnalyzer.analyze_categories(df)
    assert len(cat_df) == 3
    # Check Cars row
    cars_row = cat_df[cat_df["category"] == "Cars"].iloc[0]
    assert cars_row["listing_count"] == 3
    assert cars_row["pct_of_total"] == 60.0
    assert cars_row["ml_eligible_count"] == 3
    assert cars_row["ml_eligible_pct"] == 100.0
    assert cars_row["median_asking_price"] == 6000000.0
    assert cars_row["median_yom"] == 2016

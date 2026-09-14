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


# ==============================================================================
# 3. BRAND & MODEL ANALYSIS TESTS
# ==============================================================================

def test_analyze_brands_with_sample_threshold():
    from eda.categorical import CategoricalAnalyzer

    # 5 Toyota, 2 Honda, 1 Suzuki -> Toyota sufficient (>=5), Honda/Suzuki low-sample (<5)
    brands = ["Toyota"] * 5 + ["Honda"] * 2 + ["Suzuki"] * 1
    prices = [5000000, 5200000, 4800000, 5500000, 6000000] + [4000000, 4200000] + [2500000]
    data = {
        "listing_id": [str(i) for i in range(len(brands))],
        "brand": brands,
        "asking_price": prices,
        "ml_eligible": [True] * len(brands),
    }
    df = pd.DataFrame(data)

    res = CategoricalAnalyzer.analyze_brands(df, min_sample=5)
    assert len(res) == 3
    # Top brand should be Toyota
    toyota = res.iloc[0]
    assert toyota["brand"] == "Toyota"
    assert toyota["listing_count"] == 5
    assert toyota["pct_of_total"] == 62.5
    assert toyota["is_low_sample"] == False
    assert toyota["sample_flag"] == "Sufficient Sample"
    assert toyota["median_asking_price"] == 5200000.0

    # Honda is low sample
    honda = res[res["brand"] == "Honda"].iloc[0]
    assert honda["listing_count"] == 2
    assert honda["is_low_sample"] == True
    assert "Low Sample" in honda["sample_flag"]


def test_analyze_models_with_sample_threshold():
    from eda.categorical import CategoricalAnalyzer

    data = {
        "listing_id": [str(i) for i in range(6)],
        "brand": ["Toyota", "Toyota", "Toyota", "Toyota", "Nissan", "Nissan"],
        "model": ["Axio", "Axio", "Axio", "Corolla", "Sunny", "Caravan"],
        "asking_price": [6000000, 6200000, 5800000, 5000000, 2000000, 3500000],
        "mileage": [60000, 65000, 55000, 90000, 150000, 120000],
        "manufacture_year": [2015, 2016, 2014, 2010, 2000, 2005],
    }
    df = pd.DataFrame(data)

    res = CategoricalAnalyzer.analyze_models(df, min_sample=3)
    assert len(res) == 4
    # Top model: Toyota Axio
    axio = res.iloc[0]
    assert axio["brand"] == "Toyota"
    assert axio["model"] == "Axio"
    assert axio["listing_count"] == 3
    assert axio["is_low_sample"] == False
    assert axio["median_asking_price"] == 6000000.0
    assert axio["median_mileage"] == 60000.0
    assert axio["median_yom"] == 2015
    assert axio["min_yom"] == 2014
    assert axio["max_yom"] == 2016

    # Corolla has 1 count -> low sample
    corolla = res[res["model"] == "Corolla"].iloc[0]
    assert corolla["listing_count"] == 1
    assert corolla["is_low_sample"] == True

    # Filter by brand
    toyota_only = CategoricalAnalyzer.analyze_models(df, brand="Toyota")
    assert len(toyota_only) == 2
    assert all(toyota_only["brand"] == "Toyota")


def test_brand_and_model_empty():
    from eda.categorical import CategoricalAnalyzer

    empty = pd.DataFrame()
    assert len(CategoricalAnalyzer.analyze_brands(empty)) == 0
    assert len(CategoricalAnalyzer.analyze_models(empty)) == 0


# ==============================================================================
# 4. DISTRIBUTION ANALYSIS & VISUALIZATION TESTS
# ==============================================================================

def test_analyze_distributions():
    from eda.distributions import DistributionAnalyzer

    data = {
        "listing_id": ["1", "2", "3", "4", "5"],
        "canonical_category": ["Cars", "Cars", "Motorbikes", "Motorbikes", "Vans"],
        "asking_price": [4000000, 6000000, 300000, 500000, None],
        "mileage": [30000, 60000, 15000, 25000, 123456],
        "manufacture_year": [2018, 2015, 2020, 2019, 2005],
        "vehicle_age": [8, 11, 6, 7, 21],
        "engine_cc": [1500, 1800, 150, 200, 2500],
        "validation_issues": [[], [], [], [], ["suspicious_mileage_pattern"]],
    }
    df = pd.DataFrame(data)

    # Price distribution
    p_dist = DistributionAnalyzer.analyze_price_distribution(df)
    assert p_dist["count"] == 4
    assert p_dist["missing_count"] == 1
    assert p_dist["min"] == 300000.0
    assert p_dist["max"] == 6000000.0
    assert p_dist["median"] == 2250000.0
    assert p_dist["iqr"] == p_dist["q3"] - p_dist["q1"]

    # YOM and age distribution
    yom_dist = DistributionAnalyzer.analyze_yom_and_age_distribution(df)
    assert yom_dist["count"] == 5
    assert yom_dist["yom_median"] == 2018
    assert yom_dist["age_median"] == 8.0

    # Mileage distribution
    m_dist = DistributionAnalyzer.analyze_mileage_distribution(df)
    assert m_dist["count"] == 5
    assert m_dist["suspicious_count"] == 1
    assert m_dist["min"] == 15000.0
    assert m_dist["brackets"]["< 25,000 km"] == 1
    assert m_dist["brackets"]["25,000 - 50,000 km"] == 2

    # Engine CC distribution
    cc_dist = DistributionAnalyzer.analyze_engine_cc_distribution(df)
    assert cc_dist["count"] == 5
    assert cc_dist["min"] == 150.0
    assert cc_dist["max"] == 2500.0
    assert "Motorbikes" in cc_dist["by_category"]
    assert cc_dist["by_category"]["Motorbikes"]["median"] == 175.0


def test_plot_generators(tmp_path):
    from eda.distributions import DistributionAnalyzer

    data = {
        "listing_id": ["1", "2", "3"],
        "canonical_category": ["Cars", "Cars", "Vans"],
        "asking_price": [5000000, 6000000, 4500000],
        "mileage": [50000, 60000, 100000],
        "manufacture_year": [2015, 2017, 2012],
        "vehicle_age": [11, 9, 14],
        "engine_cc": [1500, 1500, 2500],
    }
    df = pd.DataFrame(data)

    p1 = tmp_path / "price.png"
    p2 = tmp_path / "cat_price.png"
    p3 = tmp_path / "yom.png"
    p4 = tmp_path / "mileage.png"

    fig1 = DistributionAnalyzer.plot_price_distribution(df, output_path=p1)
    fig2 = DistributionAnalyzer.plot_category_price_comparison(df, output_path=p2)
    fig3 = DistributionAnalyzer.plot_yom_and_age_distribution(df, output_path=p3)
    fig4 = DistributionAnalyzer.plot_mileage_distribution(df, output_path=p4)

    assert p1.exists() and p1.stat().st_size > 0
    assert p2.exists() and p2.stat().st_size > 0
    assert p3.exists() and p3.stat().st_size > 0
    assert p4.exists() and p4.stat().st_size > 0


# ==============================================================================
# 5. FUEL, TRANSMISSION & DISTRICT ANALYSIS TESTS
# ==============================================================================

def test_analyze_fuel_types():
    from eda.categorical import CategoricalAnalyzer

    data = {
        "listing_id": ["1", "2", "3", "4", "5"],
        "fuel_type": ["Petrol", "Petrol", "Petrol", "Diesel", "Hybrid"],
        "asking_price": [5000000, 5200000, 4800000, 7500000, 8500000],
    }
    df = pd.DataFrame(data)

    res = CategoricalAnalyzer.analyze_fuel_types(df, min_sample=3)
    assert len(res) == 3
    # Petrol has 3 listings -> sufficient sample
    petrol = res[res["fuel_type"] == "Petrol"].iloc[0]
    assert petrol["listing_count"] == 3
    assert petrol["pct_of_total"] == 60.0
    assert petrol["is_low_sample"] == False
    assert petrol["median_asking_price"] == 5000000.0

    # Diesel has 1 listing -> low sample
    diesel = res[res["fuel_type"] == "Diesel"].iloc[0]
    assert diesel["listing_count"] == 1
    assert diesel["is_low_sample"] == True


def test_analyze_transmissions():
    from eda.categorical import CategoricalAnalyzer

    data = {
        "listing_id": ["1", "2", "3", "4"],
        "transmission": ["Automatic", "Automatic", "Automatic", "Manual"],
        "asking_price": [6000000, 6500000, 7000000, 2500000],
    }
    df = pd.DataFrame(data)

    res = CategoricalAnalyzer.analyze_transmissions(df, min_sample=3)
    assert len(res) == 2
    auto = res[res["transmission"] == "Automatic"].iloc[0]
    assert auto["listing_count"] == 3
    assert auto["pct_of_total"] == 75.0
    assert auto["is_low_sample"] == False
    assert auto["median_asking_price"] == 6500000.0


def test_analyze_districts():
    from eda.categorical import CategoricalAnalyzer

    data = {
        "listing_id": ["1", "2", "3", "4", "5", "6"],
        "district": ["Colombo", "Colombo", "Colombo", "Gampaha", "Kandy", None],
        "asking_price": [8000000, 8500000, 9000000, 5000000, 4000000, 3000000],
    }
    df = pd.DataFrame(data)

    res = CategoricalAnalyzer.analyze_districts(df, min_sample=3)
    assert len(res) == 3  # None is excluded from district grouping
    colombo = res[res["district"] == "Colombo"].iloc[0]
    assert colombo["listing_count"] == 3
    assert colombo["pct_of_total"] == 50.0  # 3 / 6 total listings
    assert colombo["is_low_sample"] == False
    assert colombo["median_asking_price"] == 8500000.0

    gampaha = res[res["district"] == "Gampaha"].iloc[0]
    assert gampaha["listing_count"] == 1
    assert gampaha["is_low_sample"] == True


def test_fuel_trans_district_plots(tmp_path):
    from eda.distributions import DistributionAnalyzer

    data = {
        "listing_id": ["1", "2", "3"],
        "fuel_type": ["Petrol", "Diesel", "Hybrid"],
        "transmission": ["Automatic", "Manual", "Automatic"],
        "district": ["Colombo", "Gampaha", "Colombo"],
        "asking_price": [5000000, 6000000, 7000000],
    }
    df = pd.DataFrame(data)

    p1 = tmp_path / "fuel_trans.png"
    p2 = tmp_path / "district.png"

    fig1 = DistributionAnalyzer.plot_fuel_and_transmission_distribution(df, output_path=p1)
    fig2 = DistributionAnalyzer.plot_district_distribution(df, output_path=p2)

    assert p1.exists() and p1.stat().st_size > 0
    assert p2.exists() and p2.stat().st_size > 0

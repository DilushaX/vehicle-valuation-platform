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


# ==============================================================================
# 6. RELATIONSHIP & CORRELATION TESTS
# ==============================================================================

def test_compute_correlations():
    from eda.relationships import RelationshipAnalyzer

    # Create synthetic dataset with known inverse relationship between age/mileage and price
    data = {
        "asking_price": [10000000, 8000000, 6000000, 4000000, 2000000],
        "mileage": [10000, 30000, 60000, 90000, 150000],
        "manufacture_year": [2022, 2020, 2018, 2015, 2010],
        "vehicle_age": [4, 6, 8, 11, 16],
        "engine_cc": [2000, 1800, 1500, 1300, 1000],
    }
    df = pd.DataFrame(data)

    corrs = RelationshipAnalyzer.compute_correlations(df)
    assert "pearson" in corrs and "spearman" in corrs

    p_mat = corrs["pearson"]
    s_mat = corrs["spearman"]

    # Price vs Mileage should be strongly negative
    assert p_mat.loc["asking_price", "mileage"] < -0.9
    assert s_mat.loc["asking_price", "mileage"] == -1.0

    # Price vs Age should be strongly negative
    assert s_mat.loc["asking_price", "vehicle_age"] == -1.0

    # Price vs YOM should be strongly positive
    assert s_mat.loc["asking_price", "manufacture_year"] == 1.0


def test_compute_bivariate_relationships():
    from eda.relationships import RelationshipAnalyzer

    data = {
        "asking_price": [8000000, 6000000, 4000000, 2000000],
        "mileage": [20000, 50000, 80000, 120000],
        "manufacture_year": [2020, 2018, 2016, 2012],
        "vehicle_age": [6, 8, 10, 14],
        "engine_cc": [1800, 1500, 1300, 1000],
    }
    df = pd.DataFrame(data)

    rel = RelationshipAnalyzer.compute_bivariate_relationships(df)
    assert "asking_price_vs_mileage" in rel
    assert "asking_price_vs_vehicle_age" in rel
    assert rel["asking_price_vs_mileage"]["sample_size"] == 4
    assert rel["asking_price_vs_mileage"]["spearman_rho"] == -1.0
    assert "STATISTICAL NOTICE" in rel["asking_price_vs_mileage"]["disclaimer"]


def test_compute_category_correlations():
    from eda.relationships import RelationshipAnalyzer

    # 5 Cars, 2 Vans
    cars_data = {
        "canonical_category": ["Cars"] * 5,
        "asking_price": [9000000, 8000000, 7000000, 6000000, 5000000],
        "mileage": [20000, 40000, 60000, 80000, 100000],
        "manufacture_year": [2021, 2019, 2017, 2015, 2013],
        "vehicle_age": [5, 7, 9, 11, 13],
        "engine_cc": [1500, 1500, 1500, 1500, 1500],
    }
    vans_data = {
        "canonical_category": ["Vans"] * 2,
        "asking_price": [4000000, 3500000],
        "mileage": [120000, 150000],
        "manufacture_year": [2010, 2008],
        "vehicle_age": [16, 18],
        "engine_cc": [2500, 2500],
    }
    df = pd.concat([pd.DataFrame(cars_data), pd.DataFrame(vans_data)], ignore_index=True)

    cat_corrs = RelationshipAnalyzer.compute_category_correlations(df, min_sample=5)
    assert "Cars" in cat_corrs
    assert cat_corrs["Cars"]["sample_size"] == 5
    assert cat_corrs["Cars"]["spearman_price_mileage"] == -1.0

    # Vans should be tagged as below threshold (<5)
    assert "Vans" in cat_corrs
    assert "below threshold" in cat_corrs["Vans"]["status"].lower()


def test_relationship_plots(tmp_path):
    from eda.relationships import RelationshipAnalyzer

    data = {
        "asking_price": [8000000, 6000000, 4000000, 2000000, 5000000],
        "mileage": [20000, 50000, 80000, 120000, 60000],
        "manufacture_year": [2020, 2018, 2016, 2012, 2017],
        "vehicle_age": [6, 8, 10, 14, 9],
        "engine_cc": [1800, 1500, 1300, 1000, 1500],
    }
    df = pd.DataFrame(data)

    p1 = tmp_path / "corr_matrix.png"
    p2 = tmp_path / "price_mileage.png"
    p3 = tmp_path / "price_age.png"

    fig1 = RelationshipAnalyzer.plot_correlation_matrix(df, output_path=p1)
    fig2 = RelationshipAnalyzer.plot_price_vs_mileage(df, output_path=p2)
    fig3 = RelationshipAnalyzer.plot_price_vs_age(df, output_path=p3)

    assert p1.exists() and p1.stat().st_size > 0
    assert p2.exists() and p2.stat().st_size > 0
    assert p3.exists() and p3.stat().st_size > 0


# ==============================================================================
# 7. OUTLIER & HISTORICAL ANALYSIS TESTS
# ==============================================================================

def test_detect_iqr_and_percentile_outliers():
    from eda.outliers import OutlierAnalyzer

    # 10 values with 1 high outlier (100) and 1 low outlier (-50)
    vals = [10, 11, 12, 12, 13, 13, 14, 15, 100, -50]
    series = pd.Series(vals)

    iqr_res = OutlierAnalyzer.detect_iqr_outliers(series)
    assert iqr_res["count"] == 10
    assert iqr_res["low_outlier_count"] >= 1
    assert iqr_res["high_outlier_count"] >= 1
    assert iqr_res["total_outliers"] >= 2

    pct_res = OutlierAnalyzer.detect_percentile_outliers(series, low_p=0.05, high_p=0.95)
    assert pct_res["count"] == 10
    assert pct_res["total_outliers"] >= 2


def test_analyze_outliers_classification():
    from eda.outliers import OutlierAnalyzer

    # Create dataset with 1 suspicious outlier (123456 dummy mileage),
    # 1 genuine luxury market observation (high price, ML eligible, no issues),
    # and normal points
    data = {
        "listing_id": ["L1", "L2", "L3", "L4", "L5", "L6"],
        "canonical_category": ["Cars"] * 6,
        "asking_price": [5000000, 5200000, 4800000, 5500000, 5100000, 35000000],  # 35M is high outlier
        "mileage": [50000, 60000, 55000, 65000, 123456, 10000],  # 123456 is suspicious outlier
        "engine_cc": [1500, 1500, 1500, 1500, 1500, 3000],
        "ml_eligible": [True, True, True, True, False, True],
        "validation_issues": [
            [],
            [],
            [],
            [],
            ["suspicious_mileage_pattern"],
            [],
        ],
    }
    df = pd.DataFrame(data)

    outliers = OutlierAnalyzer.analyze_outliers(df)
    assert outliers["flagged_outliers_count"] >= 2

    records = {r["variable"]: r for r in outliers["flagged_records"]}

    # Mileage outlier should be classified as suspicious_data
    m_out = [r for r in outliers["flagged_records"] if r["variable"] == "mileage" and r["value"] == 123456][0]
    assert m_out["classification"] == "suspicious_data"

    # Luxury car price should be classified as possible_genuine_market_observation
    p_out = [r for r in outliers["flagged_records"] if r["variable"] == "asking_price" and r["value"] == 35000000][0]
    assert p_out["classification"] == "possible_genuine_market_observation"


def test_analyze_price_history():
    from eda.historical import HistoricalAnalyzer

    # Listing 1: reduced from 5M to 4.8M
    # Listing 2: raised from 3M to 3.2M
    # Listing 3: unchanged single price of 6M
    data = {
        "listing_id": ["L1", "L1", "L2", "L2", "L3"],
        "price": [5000000, 4800000, 3000000, 3200000, 6000000],
        "observed_at": [
            datetime(2026, 9, 8, tzinfo=timezone.utc),
            datetime(2026, 9, 9, tzinfo=timezone.utc),
            datetime(2026, 9, 8, tzinfo=timezone.utc),
            datetime(2026, 9, 9, tzinfo=timezone.utc),
            datetime(2026, 9, 8, tzinfo=timezone.utc),
        ],
    }
    df_ph = pd.DataFrame(data)

    res = HistoricalAnalyzer.analyze_price_history(df_ph)
    assert res["total_price_records"] == 5
    assert res["unique_listings_tracked"] == 3
    assert res["listings_with_price_changes"] == 2
    assert res["price_reductions"] == 1
    assert res["price_increases"] == 1
    assert res["unchanged_listings"] == 1
    # Average change: (-200,000 + 200,000) / 2 = 0.0
    assert res["avg_price_change_lkr"] == 0.0


def test_analyze_observations_and_historical_depth():
    from eda.historical import HistoricalAnalyzer

    # Short observation window: 3 days (Sept 8 to Sept 10)
    data_obs = {
        "listing_id": ["L1", "L1", "L2", "L3"],
        "availability": ["AVAILABLE", "AVAILABLE", "AVAILABLE", "AVAILABLE"],
        "observed_at": [
            datetime(2026, 9, 8, tzinfo=timezone.utc),
            datetime(2026, 9, 10, tzinfo=timezone.utc),
            datetime(2026, 9, 8, tzinfo=timezone.utc),
            datetime(2026, 9, 9, tzinfo=timezone.utc),
        ],
    }
    df_obs = pd.DataFrame(data_obs)

    obs_summary = HistoricalAnalyzer.analyze_observations(df_obs)
    assert obs_summary["total_observations"] == 4
    assert obs_summary["unique_listings_observed"] == 3
    assert obs_summary["avg_observations_per_listing"] == round(4 / 3, 2)

    # Historical depth evaluation on short dataset
    empty_ph = pd.DataFrame(columns=["listing_id", "price", "observed_at"])
    depth_res = HistoricalAnalyzer.evaluate_historical_depth(df_obs, empty_ph, min_days_for_monthly_trends=60)
    assert depth_res["has_sufficient_depth"] is False
    assert "Insufficient historical depth for reliable monthly market trend inference." in depth_res["message"]
    assert depth_res["total_days_span"] == 2.0

    # Test with deep dataset (e.g. 100 days)
    deep_obs = pd.DataFrame({
        "observed_at": [
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 4, 15, tzinfo=timezone.utc),
        ]
    })
    deep_res = HistoricalAnalyzer.evaluate_historical_depth(deep_obs, empty_ph, min_days_for_monthly_trends=60)
    assert deep_res["has_sufficient_depth"] is True
    assert "Historical depth sufficient" in deep_res["message"]


# ==============================================================================
# 8. REPRODUCIBLE REPORT & EXPORT TESTS
# ==============================================================================

def test_eda_reporter_end_to_end(tmp_path, db_session, vehicle_repo):
    from eda.report import EDAReporter

    # Populate 4 vehicles with various categories and status
    sr = vehicle_repo.create_scrape_run(category="Cars")

    v1 = vehicle_repo.create_vehicle({
        "category": "Cars", "brand": "Toyota", "model": "Axio",
        "manufacture_year": 2018, "fuel_type": "Hybrid", "transmission": "Automatic", "engine_cc": 1500
    })
    l1 = vehicle_repo.create_listing(v1, {
        "listing_id": "REP_1", "listing_url": "u1", "price": 8500000, "district": "Colombo", "ml_eligible": True
    })
    vehicle_repo.add_price_history(l1, 8500000)
    vehicle_repo.add_observation(l1, sr, price=8500000, mileage=45000)

    v2 = vehicle_repo.create_vehicle({
        "category": "Motorbikes", "brand": "Bajaj", "model": "Pulsar",
        "manufacture_year": 2020, "fuel_type": "Petrol", "transmission": "Manual", "engine_cc": 150
    })
    l2 = vehicle_repo.create_listing(v2, {
        "listing_id": "REP_2", "listing_url": "u2", "price": 450000, "district": "Gampaha", "ml_eligible": True
    })
    vehicle_repo.add_price_history(l2, 450000)
    vehicle_repo.add_observation(l2, sr, price=450000, mileage=20000)

    vehicle_repo.commit()

    reporter = EDAReporter(output_dir=tmp_path)
    res = reporter.run(session=db_session)

    assert res["total_listings_analyzed"] == 2
    assert res["figures_generated"] >= 7
    assert res["tables_generated"] >= 8

    # Verify figures exist
    for fig_name in [
        "price_distribution.png",
        "category_price_comparison.png",
        "yom_age_distribution.png",
        "mileage_distribution.png",
        "fuel_transmission_distribution.png",
        "district_distribution.png",
        "correlation_matrix.png",
    ]:
        fpath = tmp_path / "figures" / fig_name
        assert fpath.exists() and fpath.stat().st_size > 0

    # Verify tables exist
    for table_name in [
        "category_summary.csv",
        "category_summary.json",
        "brand_summary.csv",
        "brand_summary.json",
        "model_summary.csv",
        "numerical_summary.csv",
        "dataset_overview.json",
        "correlations.json",
    ]:
        tpath = tmp_path / "tables" / table_name
        assert tpath.exists() and tpath.stat().st_size > 0

    # Verify report markdown exists and contains critical disclaimers
    report_file = tmp_path / "reports" / "eda_market_report.md"
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "Sri Lankan Vehicle Market" in content
    assert "IMPORTANT DISCLAIMER" in content
    assert "STATISTICAL NOTE" in content
    assert "Insufficient historical depth" in content or "Historical depth" in content

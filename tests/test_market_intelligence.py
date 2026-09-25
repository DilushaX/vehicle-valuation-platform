"""
Tests for Market Intelligence Dashboard & Analytics (Phase 10.3).
"""

import pandas as pd
import pytest

from analytics.market.market_analytics import (
    apply_filters,
    compute_market_overview,
    get_category_summary,
    get_filter_options,
)


@pytest.fixture
def sample_market_df() -> pd.DataFrame:
    """Provides a controlled synthetic DataFrame mimicking EDADatasetLoader schema."""
    return pd.DataFrame([
        {
            "listing_id": "1001",
            "category": "Car",
            "canonical_category": "Cars",
            "brand": "Toyota",
            "model": "Corolla",
            "manufacture_year": 2018,
            "vehicle_age": 8,
            "asking_price": 9_500_000,
            "mileage": 65_000,
            "fuel_type": "Petrol",
            "transmission": "Automatic",
            "condition": "Registered (Used)",
            "district": "Colombo",
            "current_status": "ACTIVE",
            "ml_eligible": True,
        },
        {
            "listing_id": "1002",
            "category": "Car",
            "canonical_category": "Cars",
            "brand": "Honda",
            "model": "Civic",
            "manufacture_year": 2020,
            "vehicle_age": 6,
            "asking_price": 14_200_000,
            "mileage": 35_000,
            "fuel_type": "Petrol",
            "transmission": "Automatic",
            "condition": "Registered (Used)",
            "district": "Gampaha",
            "current_status": "ACTIVE",
            "ml_eligible": True,
        },
        {
            "listing_id": "1003",
            "category": "SUV",
            "canonical_category": "SUVs",
            "brand": "Toyota",
            "model": "Prado",
            "manufacture_year": 2015,
            "vehicle_age": 11,
            "asking_price": 28_000_000,
            "mileage": 110_000,
            "fuel_type": "Diesel",
            "transmission": "Automatic",
            "condition": "Registered (Used)",
            "district": "Colombo",
            "current_status": "ACTIVE",
            "ml_eligible": True,
        },
        {
            "listing_id": "1004",
            "category": "Van",
            "canonical_category": "Vans",
            "brand": "Nissan",
            "model": "Caravan",
            "manufacture_year": 2012,
            "vehicle_age": 14,
            "asking_price": 8_200_000,
            "mileage": 180_000,
            "fuel_type": "Diesel",
            "transmission": "Manual",
            "condition": "Registered (Used)",
            "district": "Kandy",
            "current_status": "NO_LONGER_OBSERVED",
            "ml_eligible": False,
        },
        {
            "listing_id": "1005",
            "category": "Motorbike",
            "canonical_category": "Motorbikes",
            "brand": "Yamaha",
            "model": "FZ",
            "manufacture_year": 2021,
            "vehicle_age": 5,
            "asking_price": 750_000,
            "mileage": 15_000,
            "fuel_type": "Petrol",
            "transmission": "Manual",
            "condition": "Registered (Used)",
            "district": "Kurunegala",
            "current_status": "ACTIVE",
            "ml_eligible": True,
        },
    ])


def test_get_filter_options(sample_market_df):
    opts = get_filter_options(sample_market_df)
    assert "Cars" in opts["categories"]
    assert "SUVs" in opts["categories"]
    assert "Toyota" in opts["brands"]
    assert "Honda" in opts["brands"]
    assert "Corolla" in opts["models"]
    assert "Colombo" in opts["districts"]
    assert "Petrol" in opts["fuel_types"]
    assert "Automatic" in opts["transmissions"]
    assert opts["year_min"] == 2012
    assert opts["year_max"] == 2021
    assert opts["price_min"] == 750_000
    assert opts["price_max"] == 28_000_000


def test_get_filter_options_empty():
    opts = get_filter_options(pd.DataFrame())
    assert opts["categories"] == []
    assert opts["brands"] == []
    assert opts["year_min"] <= opts["year_max"]


def test_apply_filters_by_category(sample_market_df):
    filtered = apply_filters(sample_market_df, categories=["Cars"])
    assert len(filtered) == 2
    assert all(filtered["canonical_category"] == "Cars")


def test_apply_filters_by_brand(sample_market_df):
    filtered = apply_filters(sample_market_df, brands=["Toyota"])
    assert len(filtered) == 2
    assert set(filtered["model"]) == {"Corolla", "Prado"}


def test_apply_filters_by_model(sample_market_df):
    filtered = apply_filters(sample_market_df, models=["Civic"])
    assert len(filtered) == 1
    assert filtered.iloc[0]["brand"] == "Honda"


def test_apply_filters_by_district(sample_market_df):
    filtered = apply_filters(sample_market_df, districts=["Colombo"])
    assert len(filtered) == 2
    assert set(filtered["listing_id"]) == {"1001", "1003"}


def test_apply_filters_by_fuel_and_transmission(sample_market_df):
    filtered = apply_filters(sample_market_df, fuel_types=["Diesel"], transmissions=["Manual"])
    assert len(filtered) == 1
    assert filtered.iloc[0]["listing_id"] == "1004"


def test_apply_filters_by_year_range(sample_market_df):
    filtered = apply_filters(sample_market_df, year_range=(2018, 2021))
    assert len(filtered) == 3
    assert set(filtered["listing_id"]) == {"1001", "1002", "1005"}


def test_apply_filters_by_price_range(sample_market_df):
    filtered = apply_filters(sample_market_df, price_range=(5_000_000, 15_000_000))
    assert len(filtered) == 3
    assert set(filtered["listing_id"]) == {"1001", "1002", "1004"}


def test_apply_filters_no_matches(sample_market_df):
    filtered = apply_filters(sample_market_df, brands=["Ferrari"])
    assert len(filtered) == 0
    assert isinstance(filtered, pd.DataFrame)


def test_compute_market_overview(sample_market_df):
    overview = compute_market_overview(sample_market_df)
    assert overview["total_listings"] == 5
    assert overview["total_vehicles"] == 5
    assert overview["distinct_categories"] == 4
    assert overview["distinct_brands"] == 4
    assert overview["distinct_models"] == 5
    assert overview["median_asking_price"] == 9_500_000
    assert overview["average_asking_price"] == (9_500_000 + 14_200_000 + 28_000_000 + 8_200_000 + 750_000) / 5
    assert overview["active_listings"] == 4
    assert overview["ml_eligible_listings"] == 4
    assert overview["ml_eligible_pct"] == 80.0


def test_compute_market_overview_empty():
    overview = compute_market_overview(pd.DataFrame())
    assert overview["total_listings"] == 0
    assert overview["total_vehicles"] == 0
    assert overview["median_asking_price"] is None
    assert overview["average_asking_price"] is None
    assert overview["distinct_categories"] == 0


def test_compute_market_overview_distinguishes_vehicles_and_listings(sample_market_df):
    # If multiple listings share the same vehicle_id
    df_with_vehicles = sample_market_df.copy()
    df_with_vehicles["vehicle_id"] = [1, 1, 2, 3, 4]  # 5 listings, 4 vehicles
    overview = compute_market_overview(df_with_vehicles)
    assert overview["total_listings"] == 5
    assert overview["total_vehicles"] == 4


def test_get_category_summary(sample_market_df):
    cat_df = get_category_summary(sample_market_df)
    assert not cat_df.empty
    assert "category" in cat_df.columns
    assert "listing_count" in cat_df.columns
    assert "median_asking_price" in cat_df.columns
    assert "mean_asking_price" in cat_df.columns
    # Check cars
    cars_row = cat_df[cat_df["category"] == "Cars"].iloc[0]
    assert cars_row["listing_count"] == 2
    assert cars_row["median_asking_price"] == (9_500_000 + 14_200_000) / 2


def test_get_category_summary_empty():
    cat_df = get_category_summary(pd.DataFrame())
    assert cat_df.empty
    assert "category" in cat_df.columns
    assert "median_asking_price" in cat_df.columns



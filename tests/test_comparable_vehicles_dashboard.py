"""
Unit and integration tests for Phase 10.4 — Comparable Vehicles Search UI.

Validates the 14 requirements specified in Phase 10.4:
1. Page loads and exposes entry point functions.
2. Target vehicle form validation works (domain constraints).
3. Valid input reaches the existing comparable engine.
4. Existing comparable engine is used (no second algorithm).
5. Results table renders expected columns with correct formatting.
6. Empty results render safely without exceptions.
7. Comparable market summary renders correct descriptive statistics.
8. Single comparable works with zero spread.
9. Missing asking prices and optional fields are handled safely.
10. Sorting controls work (similarity, asking price, year).
11. Result controls work (display threshold filtering without modifying similarity).
12. No private seller contact data (phone/email) is displayed.
13. Existing comparable engine tests remain passing.
14. Existing Phase 10.3 market intelligence tests remain passing.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from analytics.comparables.comparable_engine import (
    ComparableVehicle,
    ComparableVehicleEngine,
)
from analytics.comparables.market_summary import (
    ComparableMarketSummary,
    create_comparable_market_summary,
)
from dashboard.pages.comparable_vehicles_page import (
    CURRENT_YEAR,
    VEHICLE_CATEGORIES,
    FUEL_TYPES,
    TRANSMISSIONS,
    SRI_LANKA_DISTRICTS,
    CONDITIONS,
    apply_result_controls,
    build_comparable_table_dataframe,
    compute_comparable_market_summary,
    execute_comparable_search,
    render_comparable_vehicles_page,
    validate_target_vehicle_input,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_valid_query() -> Dict[str, Any]:
    return {
        "category": "Cars",
        "brand": "Toyota",
        "model": "Premio",
        "manufacture_year": 2016,
        "mileage": 85000.0,
        "engine_cc": 1500.0,
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "district": "Colombo",
        "condition": "Registered (Used)",
    }


@pytest.fixture
def sample_comparables() -> List[ComparableVehicle]:
    return [
        ComparableVehicle(
            listing_id="L01",
            category="Cars",
            brand="Toyota",
            model="Premio",
            manufacture_year=2016,
            mileage=80000.0,
            engine_cc=1500.0,
            fuel_type="Petrol",
            transmission="Automatic",
            district="Colombo",
            condition="Registered (Used)",
            asking_price=12_000_000.0,
            similarity_score=0.95,
            similarity_percentage=95.0,
        ),
        ComparableVehicle(
            listing_id="L02",
            category="Cars",
            brand="Toyota",
            model="Allion",
            manufacture_year=2015,
            mileage=90000.0,
            engine_cc=1500.0,
            fuel_type="Petrol",
            transmission="Automatic",
            district="Gampaha",
            condition="Registered (Used)",
            asking_price=11_500_000.0,
            similarity_score=0.82,
            similarity_percentage=82.0,
        ),
        ComparableVehicle(
            listing_id="L03",
            category="Cars",
            brand="Honda",
            model="Grace",
            manufacture_year=2017,
            mileage=70000.0,
            engine_cc=1500.0,
            fuel_type="Hybrid",
            transmission="Automatic",
            district="Colombo",
            condition="Registered (Used)",
            asking_price=13_000_000.0,
            similarity_score=0.74,
            similarity_percentage=74.0,
        ),
    ]


# ---------------------------------------------------------------------------
# 1. Page loads
# ---------------------------------------------------------------------------

def test_page_loads_and_callable():
    """Verifies that the page module loads and exposes required entry points."""
    assert callable(render_comparable_vehicles_page)
    assert callable(validate_target_vehicle_input)
    assert callable(execute_comparable_search)
    assert callable(build_comparable_table_dataframe)
    assert callable(compute_comparable_market_summary)
    assert callable(apply_result_controls)


# ---------------------------------------------------------------------------
# 2. Target vehicle form validation
# ---------------------------------------------------------------------------

def test_target_vehicle_form_valid(sample_valid_query: Dict[str, Any]):
    """Valid target vehicle input passes validation with zero errors."""
    is_valid, errors = validate_target_vehicle_input(sample_valid_query)
    assert is_valid is True
    assert len(errors) == 0


def test_target_vehicle_form_invalid_cases():
    """Ensures sensible validation catches empty, out-of-range, and invalid inputs."""
    # Empty brand and model
    is_valid, errors = validate_target_vehicle_input({"category": "Cars", "brand": "", "model": ""})
    assert is_valid is False
    assert any("Brand" in e for e in errors)
    assert any("Model" in e for e in errors)

    # Invalid category
    is_valid, errors = validate_target_vehicle_input({"category": "Spaceship", "brand": "Tesla", "model": "Model 3"})
    assert is_valid is False
    assert any("Category" in e for e in errors)

    # Out of range manufacture year
    is_valid, errors = validate_target_vehicle_input({
        "category": "Cars", "brand": "Toyota", "model": "Premio", "manufacture_year": 1800
    })
    assert is_valid is False
    assert any("Manufacture Year" in e for e in errors)

    # Negative mileage and engine cc
    is_valid, errors = validate_target_vehicle_input({
        "category": "Cars", "brand": "Toyota", "model": "Premio", "manufacture_year": 2016,
        "mileage": -100, "engine_cc": -50
    })
    assert is_valid is False
    assert any("Mileage" in e for e in errors)
    assert any("Engine CC" in e for e in errors)


# ---------------------------------------------------------------------------
# 3. Valid input reaches comparable engine
# ---------------------------------------------------------------------------

def test_valid_input_reaches_comparable_engine(sample_valid_query: Dict[str, Any]):
    """Verifies target vehicle specification dictionary is passed to find_comparables."""
    mock_engine = MagicMock(spec=ComparableVehicleEngine)
    mock_engine.find_comparables.return_value = []

    execute_comparable_search(
        query=sample_valid_query,
        top_k=5,
        engine=mock_engine,
    )

    mock_engine.find_comparables.assert_called_once()
    called_kwargs = mock_engine.find_comparables.call_args[1]
    assert called_kwargs["query"] == sample_valid_query
    assert called_kwargs["top_k"] == 5
    assert called_kwargs["match_category_strictly"] is True


# ---------------------------------------------------------------------------
# 4. Existing comparable engine is used
# ---------------------------------------------------------------------------

def test_existing_comparable_engine_is_used(sample_valid_query: Dict[str, Any]):
    """Confirms search relies on ComparableVehicleEngine and its weighted similarity scoring."""
    candidate_pool = pd.DataFrame([
        {
            "listing_id": "P01",
            "category": "Cars",
            "brand": "Toyota",
            "model": "Premio",
            "manufacture_year": 2016,
            "mileage": 85000.0,
            "engine_cc": 1500.0,
            "fuel_type": "Petrol",
            "transmission": "Automatic",
            "district": "Colombo",
            "condition": "Registered (Used)",
            "asking_price": 12_500_000.0,
            "ml_eligible": True,
        }
    ])

    results = execute_comparable_search(
        query=sample_valid_query,
        top_k=5,
        candidate_pool=candidate_pool,
    )

    assert len(results) == 1
    assert isinstance(results[0], ComparableVehicle)
    assert results[0].listing_id == "P01"
    # Similarity should be 1.0 since all attributes match identically
    assert pytest.approx(results[0].similarity_score, 0.01) == 1.0


# ---------------------------------------------------------------------------
# 5. Results render
# ---------------------------------------------------------------------------

def test_results_render_table_dataframe(sample_comparables: List[ComparableVehicle]):
    """DataFrame contains required non-private fields with correct formatting."""
    df = build_comparable_table_dataframe(sample_comparables)
    assert len(df) == 3

    expected_columns = [
        "Brand",
        "Model",
        "Category",
        "Manufacture Year",
        "Mileage",
        "Engine CC",
        "Fuel Type",
        "Transmission",
        "District",
        "Condition",
        "Advertised Asking Price (LKR)",
        "Similarity",
    ]
    for col in expected_columns:
        assert col in df.columns

    # Check formatting
    assert df.iloc[0]["Advertised Asking Price (LKR)"] == "12,000,000 LKR"
    assert df.iloc[0]["Mileage"] == "80,000 km"
    assert df.iloc[0]["Engine CC"] == "1,500 cc"
    assert df.iloc[0]["Similarity"] == "95.0%"


# ---------------------------------------------------------------------------
# 6. Empty result renders correctly
# ---------------------------------------------------------------------------

def test_empty_results_handling():
    """Empty comparable results produce empty table and safe null summary."""
    empty_df = build_comparable_table_dataframe([])
    assert isinstance(empty_df, pd.DataFrame)
    assert empty_df.empty

    empty_summary = compute_comparable_market_summary([])
    assert empty_summary.comparable_count == 0
    assert empty_summary.min_asking_price is None
    assert empty_summary.max_asking_price is None
    assert empty_summary.median_asking_price is None
    assert empty_summary.average_asking_price is None
    assert empty_summary.price_spread is None


# ---------------------------------------------------------------------------
# 7. Market summary renders
# ---------------------------------------------------------------------------

def test_market_summary_calculation(sample_comparables: List[ComparableVehicle]):
    """Computes correct asking-price summary statistics reusing market_summary.py."""
    summary = compute_comparable_market_summary(sample_comparables)

    assert summary.comparable_count == 3
    assert summary.min_asking_price == 11_500_000.0
    assert summary.max_asking_price == 13_000_000.0
    assert summary.median_asking_price == 12_000_000.0
    assert summary.average_asking_price == pytest.approx(12_166_666.67, 0.1)
    assert summary.price_spread == 1_500_000.0


# ---------------------------------------------------------------------------
# 8. One comparable works
# ---------------------------------------------------------------------------

def test_one_comparable_handling(sample_comparables: List[ComparableVehicle]):
    """Single comparable yields valid stats with zero price spread."""
    single = sample_comparables[:1]
    df = build_comparable_table_dataframe(single)
    assert len(df) == 1

    summary = compute_comparable_market_summary(single)
    assert summary.comparable_count == 1
    assert summary.min_asking_price == 12_000_000.0
    assert summary.max_asking_price == 12_000_000.0
    assert summary.median_asking_price == 12_000_000.0
    assert summary.average_asking_price == 12_000_000.0
    assert summary.price_spread == 0.0


# ---------------------------------------------------------------------------
# 9. Missing asking prices and optional fields are handled safely
# ---------------------------------------------------------------------------

def test_missing_asking_prices_and_optional_fields():
    """Missing fields are displayed as N/A without raising exceptions."""
    comp_missing = ComparableVehicle(
        listing_id="L_MISSING",
        category="Cars",
        brand="Toyota",
        model="Corolla",
        manufacture_year=None,
        mileage=None,
        engine_cc=None,
        fuel_type=None,
        transmission=None,
        district=None,
        condition=None,
        asking_price=0.0,
        similarity_score=0.5,
        similarity_percentage=50.0,
    )
    df = build_comparable_table_dataframe([comp_missing])
    assert len(df) == 1
    assert df.iloc[0]["Manufacture Year"] == "N/A"
    assert df.iloc[0]["Mileage"] == "N/A"
    assert df.iloc[0]["Engine CC"] == "N/A"
    assert df.iloc[0]["District"] == "N/A"

    # Summary ignores non-positive prices safely
    summary = compute_comparable_market_summary([comp_missing])
    assert summary.comparable_count == 0


# ---------------------------------------------------------------------------
# 10. Sorting works
# ---------------------------------------------------------------------------

def test_sorting_controls(sample_comparables: List[ComparableVehicle]):
    """Tests sorting by similarity, asking price ascending/descending, and year."""
    # Similarity descending (default)
    sorted_sim = apply_result_controls(sample_comparables, sort_by="Similarity (highest first)")
    assert [c.listing_id for c in sorted_sim] == ["L01", "L02", "L03"]

    # Asking Price ascending
    sorted_price_asc = apply_result_controls(sample_comparables, sort_by="Asking Price (lowest first)")
    assert [c.listing_id for c in sorted_price_asc] == ["L02", "L01", "L03"]

    # Asking Price descending
    sorted_price_desc = apply_result_controls(sample_comparables, sort_by="Asking Price (highest first)")
    assert [c.listing_id for c in sorted_price_desc] == ["L03", "L01", "L02"]

    # Year descending
    sorted_year = apply_result_controls(sample_comparables, sort_by="Year (newest first)")
    assert [c.listing_id for c in sorted_year] == ["L03", "L01", "L02"]


# ---------------------------------------------------------------------------
# 11. Result controls work
# ---------------------------------------------------------------------------

def test_result_controls_filtering(sample_comparables: List[ComparableVehicle]):
    """Tests similarity display threshold filtering without altering underlying scores."""
    # Filter with min 80% similarity threshold
    filtered = apply_result_controls(sample_comparables, min_similarity_pct=80.0)
    assert len(filtered) == 2
    assert [c.listing_id for c in filtered] == ["L01", "L02"]

    # Scores must remain unmodified
    assert filtered[0].similarity_percentage == 95.0
    assert filtered[1].similarity_percentage == 82.0


# ---------------------------------------------------------------------------
# 12. No private seller contact data is displayed
# ---------------------------------------------------------------------------

def test_no_private_seller_contact_data(sample_comparables: List[ComparableVehicle]):
    """Strictly verifies that no seller phone, email, or private contact info is displayed."""
    df = build_comparable_table_dataframe(sample_comparables)
    private_keywords = ["phone", "email", "contact", "seller", "telephone", "mobile", "address"]

    for col in df.columns:
        for kw in private_keywords:
            assert kw not in col.lower(), f"Private field '{col}' detected in comparable table!"

    for c in sample_comparables:
        c_dict = c.to_dict()
        for kw in private_keywords:
            assert kw not in c_dict, f"Private field '{kw}' found in comparable vehicle dict!"


# ---------------------------------------------------------------------------
# 13. Semantic wording compliance: "Similarity", not "Confidence" / "Accuracy"
# ---------------------------------------------------------------------------

def test_semantic_wording_compliance(sample_comparables: List[ComparableVehicle]):
    """Verifies that similarity column is labeled 'Similarity', not Confidence/Accuracy."""
    df = build_comparable_table_dataframe(sample_comparables)
    assert "Similarity" in df.columns
    assert "Confidence" not in df.columns
    assert "Accuracy" not in df.columns
    assert "Probability" not in df.columns
    assert "Certainty" not in df.columns

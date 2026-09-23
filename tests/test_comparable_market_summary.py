"""
Tests for Comparable Market Summary Subsystem (Step 9.13).

Validates:
1. Multiple valid comparables: correct count, min, max, median, average, spread.
2. Single comparable handling: correct stats with zero price spread.
3. Zero comparables handling: safe null/empty summary without raising exceptions.
4. Missing asking price values: skipped safely without crashing.
5. Invalid/non-numeric asking price values: handled safely without errors.
6. Deterministic output: consistent repeated evaluations.
7. Existing valuation service integration: comparable_market_summary present in result.
8. Non-probabilistic reporting: no confidence scores, accuracy percentages, or probability fields.
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from analytics.comparables.comparable_engine import ComparableVehicle
from analytics.comparables.market_summary import (
    ComparableMarketSummary,
    create_comparable_market_summary,
)
from ml.valuation.valuation_service import ValuationResult, VehicleValuationService


@pytest.fixture
def mock_comparables_list() -> List[Dict[str, Any]]:
    return [
        {"listing_id": "L1", "brand": "Toyota", "model": "Premio", "asking_price": 8_500_000.0},
        {"listing_id": "L2", "brand": "Toyota", "model": "Premio", "asking_price": 11_200_000.0},
        {"listing_id": "L3", "brand": "Toyota", "model": "Premio", "asking_price": 9_800_000.0},
        {"listing_id": "L4", "brand": "Toyota", "model": "Premio", "asking_price": 10_500_000.0},
        {"listing_id": "L5", "brand": "Toyota", "model": "Premio", "asking_price": 9_100_000.0},
    ]


@pytest.fixture
def mock_comparable_objects() -> List[ComparableVehicle]:
    prices = [8_500_000.0, 11_200_000.0, 9_800_000.0, 10_500_000.0, 9_100_000.0]
    return [
        ComparableVehicle(
            listing_id=f"L{i}",
            category="Cars",
            brand="Toyota",
            model="Premio",
            manufacture_year=2016,
            mileage=85000.0,
            engine_cc=1500.0,
            fuel_type="Petrol",
            transmission="Automatic",
            district="Colombo",
            condition="Registered (Used)",
            asking_price=p,
            similarity_score=0.9,
            similarity_percentage=90.0,
        )
        for i, p in enumerate(prices, start=1)
    ]


def test_multiple_valid_comparables(mock_comparables_list):
    """1. Multiple valid comparables calculate correct count, min, max, median, average, and spread."""
    summary = create_comparable_market_summary(mock_comparables_list)

    assert isinstance(summary, ComparableMarketSummary)
    d = summary.to_dict()

    assert d["comparable_count"] == 5
    assert d["min_asking_price"] == 8_500_000.0
    assert d["max_asking_price"] == 11_200_000.0
    assert d["median_asking_price"] == 9_800_000.0
    assert d["average_asking_price"] == round((8.5 + 11.2 + 9.8 + 10.5 + 9.1) / 5 * 1_000_000, 2)
    assert d["price_spread"] == round(11_200_000.0 - 8_500_000.0, 2)


def test_multiple_valid_comparable_objects(mock_comparable_objects):
    """1b. Works seamlessly with ComparableVehicle dataclass objects."""
    summary = create_comparable_market_summary(mock_comparable_objects)
    d = summary.to_dict()

    assert d["comparable_count"] == 5
    assert d["min_asking_price"] == 8_500_000.0
    assert d["max_asking_price"] == 11_200_000.0
    assert d["median_asking_price"] == 9_800_000.0


def test_single_comparable():
    """2. One comparable returns that price with 0 price spread."""
    single = [{"listing_id": "L1", "asking_price": 9_500_000.0}]
    summary = create_comparable_market_summary(single)
    d = summary.to_dict()

    assert d["comparable_count"] == 1
    assert d["min_asking_price"] == 9_500_000.0
    assert d["max_asking_price"] == 9_500_000.0
    assert d["median_asking_price"] == 9_500_000.0
    assert d["average_asking_price"] == 9_500_000.0
    assert d["price_spread"] == 0.0


def test_zero_comparables():
    """3. Zero comparables return a safe empty summary without exceptions."""
    summary_empty = create_comparable_market_summary([])
    d1 = summary_empty.to_dict()
    assert d1["comparable_count"] == 0
    assert d1["min_asking_price"] is None
    assert d1["max_asking_price"] is None
    assert d1["median_asking_price"] is None
    assert d1["average_asking_price"] is None
    assert d1["price_spread"] is None

    summary_none = create_comparable_market_summary(None)
    d2 = summary_none.to_dict()
    assert d2["comparable_count"] == 0
    assert d2["min_asking_price"] is None


def test_missing_asking_price_values():
    """4. Records with missing asking prices are filtered safely."""
    items = [
        {"listing_id": "L1", "asking_price": 10_000_000.0},
        {"listing_id": "L2"},  # Missing asking_price key
        {"listing_id": "L3", "asking_price": None},
        {"listing_id": "L4", "asking_price": np.nan},
        {"listing_id": "L5", "asking_price": 12_000_000.0},
    ]
    summary = create_comparable_market_summary(items)
    d = summary.to_dict()

    assert d["comparable_count"] == 2
    assert d["min_asking_price"] == 10_000_000.0
    assert d["max_asking_price"] == 12_000_000.0
    assert d["price_spread"] == 2_000_000.0


def test_invalid_non_numeric_asking_price_values():
    """5. Non-numeric or non-positive asking prices are handled safely."""
    items = [
        {"listing_id": "L1", "asking_price": "not_a_number"},
        {"listing_id": "L2", "asking_price": -500_000.0},
        {"listing_id": "L3", "asking_price": 0.0},
        {"listing_id": "L4", "asking_price": 7_500_000.0},
    ]
    summary = create_comparable_market_summary(items)
    d = summary.to_dict()

    assert d["comparable_count"] == 1
    assert d["min_asking_price"] == 7_500_000.0
    assert d["max_asking_price"] == 7_500_000.0


def test_deterministic_output(mock_comparables_list):
    """6. Output is completely deterministic across repeated calls."""
    res1 = create_comparable_market_summary(mock_comparables_list).to_dict()
    res2 = create_comparable_market_summary(mock_comparables_list).to_dict()
    res3 = create_comparable_market_summary(mock_comparables_list).to_dict()

    assert res1 == res2 == res3


def test_no_confidence_or_probability_generated(mock_comparables_list):
    """8. Verifies no confidence score, probability, or accuracy is produced."""
    d = create_comparable_market_summary(mock_comparables_list).to_dict()
    forbidden_terms = ["confidence", "probability", "accuracy", "certainty", "%"]

    for k, v in d.items():
        k_str = str(k).lower()
        v_str = str(v).lower()
        for term in forbidden_terms:
            assert term not in k_str, f"Forbidden term '{term}' in key '{k}'"
            assert term not in v_str, f"Forbidden term '{term}' in value '{v}'"


def test_valuation_service_integration():
    """7. Existing valuation service returns comparable_market_summary in result."""
    model_path = Path("data/analysis/ml/models/model.joblib")
    meta_path = Path("data/analysis/ml/models/model_metadata.json")
    if not model_path.exists():
        pytest.skip(f"Model artifact not found at {model_path}")

    service = VehicleValuationService(model_path=model_path, metadata_path=meta_path)
    vehicle = {
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

    result = service.valuate(vehicle, top_k_comparables=4)
    assert isinstance(result, ValuationResult)
    d = result.to_dict()

    assert "comparable_market_summary" in d
    cms = d["comparable_market_summary"]
    assert "comparable_count" in cms
    assert cms["comparable_count"] == len(d["comparables"])
    if cms["comparable_count"] > 0:
        assert cms["min_asking_price"] is not None
        assert cms["max_asking_price"] is not None
        assert cms["min_asking_price"] <= cms["max_asking_price"]
        assert cms["price_spread"] == round(cms["max_asking_price"] - cms["min_asking_price"], 2)

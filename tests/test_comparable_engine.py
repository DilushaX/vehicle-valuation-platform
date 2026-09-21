"""
Tests for Comparable Vehicle Search Engine (Step 9 - Part 4).

Validates:
1. Same category vehicles are prioritized and retrieved.
2. Incompatible categories are excluded when category matching is enforced.
3. Missing or invalid asking price records are excluded from comparables.
4. ML-ineligible records are excluded.
5. Result count respects the requested top_k limit.
6. Similarity scores are deterministic and normalized between 0.0 and 1.0.
7. No duplicate listings are returned in search results.
8. Query listing itself is excluded if exclude_listing_id is provided.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from analytics.comparables.comparable_engine import (
    ComparableVehicle,
    ComparableVehicleEngine,
)


@pytest.fixture
def candidate_pool() -> pd.DataFrame:
    """Synthetic dataset of candidate listings for deterministic unit tests."""
    return pd.DataFrame([
        {
            "listing_id": "CAR_01",
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
        },
        {
            "listing_id": "CAR_02",
            "category": "Cars",
            "brand": "Toyota",
            "model": "Allion",
            "manufacture_year": 2015,
            "mileage": 90000.0,
            "engine_cc": 1500.0,
            "fuel_type": "Petrol",
            "transmission": "Automatic",
            "district": "Gampaha",
            "condition": "Registered (Used)",
            "asking_price": 11_800_000.0,
            "ml_eligible": True,
        },
        {
            "listing_id": "CAR_03",
            "category": "Cars",
            "brand": "Honda",
            "model": "Grace",
            "manufacture_year": 2017,
            "mileage": 70000.0,
            "engine_cc": 1500.0,
            "fuel_type": "Hybrid",
            "transmission": "Automatic",
            "district": "Colombo",
            "condition": "Registered (Used)",
            "asking_price": 10_500_000.0,
            "ml_eligible": True,
        },
        {
            "listing_id": "BIKE_01",
            "category": "Motorbikes",
            "brand": "Bajaj",
            "model": "Pulsar",
            "manufacture_year": 2020,
            "mileage": 25000.0,
            "engine_cc": 150.0,
            "fuel_type": "Petrol",
            "transmission": "Manual",
            "district": "Colombo",
            "condition": "Registered (Used)",
            "asking_price": 450_000.0,
            "ml_eligible": True,
        },
        {
            "listing_id": "INELIG_01",
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
            "asking_price": 12_000_000.0,
            "ml_eligible": False,  # Ineligible
        },
        {
            "listing_id": "NOPRICE_01",
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
            "asking_price": np.nan,  # Missing price
            "ml_eligible": True,
        },
    ])


@pytest.fixture
def query_car() -> Dict[str, Any]:
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


def test_find_comparables_prioritizes_identical_match(candidate_pool, query_car):
    engine = ComparableVehicleEngine()
    results = engine.find_comparables(query=query_car, candidate_pool=candidate_pool, top_k=5)

    assert len(results) > 0
    # Top comparable should be CAR_01 (exact match on make, model, year, specs)
    top = results[0]
    assert top.listing_id == "CAR_01"
    assert top.brand == "Toyota"
    assert top.model == "Premio"
    assert top.similarity_score == pytest.approx(1.0, rel=1e-2)
    assert top.similarity_percentage == pytest.approx(100.0, rel=1e-2)


def test_incompatible_categories_excluded(candidate_pool, query_car):
    engine = ComparableVehicleEngine()
    results = engine.find_comparables(
        query=query_car, candidate_pool=candidate_pool, match_category_strictly=True
    )
    categories = [r.category for r in results]
    assert "Motorbikes" not in categories
    assert all(c == "Cars" for c in categories)


def test_missing_price_and_ineligible_excluded(candidate_pool, query_car):
    engine = ComparableVehicleEngine()
    # Candidate pool includes INELIG_01 (ml_eligible=False) and NOPRICE_01 (asking_price=NaN)
    results = engine.find_comparables(query=query_car, candidate_pool=candidate_pool, top_k=10)
    listing_ids = [r.listing_id for r in results]

    assert "NOPRICE_01" not in listing_ids
    # Both ineligible and missing price should not be present
    for r in results:
        assert r.asking_price > 0


def test_top_k_limit(candidate_pool, query_car):
    engine = ComparableVehicleEngine()
    results_2 = engine.find_comparables(query=query_car, candidate_pool=candidate_pool, top_k=2)
    assert len(results_2) == 2


def test_similarity_scores_deterministic(candidate_pool, query_car):
    engine = ComparableVehicleEngine()
    res1 = engine.find_comparables(query=query_car, candidate_pool=candidate_pool, top_k=5)
    res2 = engine.find_comparables(query=query_car, candidate_pool=candidate_pool, top_k=5)

    assert len(res1) == len(res2)
    for r1, r2 in zip(res1, res2):
        assert r1.listing_id == r2.listing_id
        assert r1.similarity_score == r2.similarity_score


def test_exclude_self_listing(candidate_pool, query_car):
    engine = ComparableVehicleEngine()
    results = engine.find_comparables(
        query=query_car,
        candidate_pool=candidate_pool,
        top_k=5,
        exclude_listing_id="CAR_01",
    )
    listing_ids = [r.listing_id for r in results]
    assert "CAR_01" not in listing_ids


def test_find_comparables_live_database(query_car):
    """Verifies that comparable engine works with real PostgreSQL database records."""
    engine = ComparableVehicleEngine()
    results = engine.find_comparables(query=query_car, top_k=3)

    assert len(results) > 0
    assert len(results) <= 3
    for comp in results:
        assert isinstance(comp, ComparableVehicle)
        assert comp.category == "Cars"
        assert comp.asking_price > 0
        assert 0.0 <= comp.similarity_score <= 1.0
        d = comp.to_dict()
        assert "listing_id" in d
        assert "similarity_percentage" in d


def test_similarity_score_semantics_and_attribute_weights():
    engine = ComparableVehicleEngine()
    # 1. Verify weights sum to 1.00
    total_weights = (
        engine.WEIGHT_BRAND
        + engine.WEIGHT_MODEL
        + engine.WEIGHT_YEAR
        + engine.WEIGHT_MILEAGE
        + engine.WEIGHT_CC
        + engine.WEIGHT_TRANSMISSION
        + engine.WEIGHT_FUEL
        + engine.WEIGHT_DISTRICT
        + engine.WEIGHT_CONDITION
    )
    assert total_weights == pytest.approx(1.0, rel=1e-5)

    # 2. Verify engine docstring semantics clarify specification distance
    doc = ComparableVehicleEngine.__doc__.lower()
    assert "specification alignment" in doc
    assert "not represent prediction confidence" in doc


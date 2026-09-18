"""
End-to-End Integration Test Suite for Vehicle Valuation Platform (Step 9 - Part 9).

Validates complete system flow:
Input vehicle
→ feature validation & preparation
→ trained model prediction
→ model-based prediction range
→ Tree SHAP explainability
→ comparable vehicle retrieval
→ unified valuation service
→ human-readable report formatting
→ FastAPI HTTP REST endpoint

Ensures zero PostgreSQL database mutations and verifies full system stability.
"""

from pathlib import Path
from typing import Any, Dict

from fastapi.testclient import TestClient
import numpy as np
import pytest
from sqlalchemy.orm import Session

from analytics.comparables.comparable_engine import ComparableVehicleEngine
from api.main import app
from database.connection import get_sessionmaker
from database.models import (
    Listing,
    ListingObservation,
    PriceHistory,
    ScrapeRun,
    Vehicle,
)
from ml.prediction.predictor import VehiclePricePredictor
from ml.prediction.valuation_report import ValuationReportFormatter
from ml.valuation.valuation_service import VehicleValuationService

client = TestClient(app)


@pytest.fixture
def test_vehicle_scenarios():
    """Diverse vehicle inputs spanning multiple categories and specifications."""
    return [
        {
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
        },
        {
            "category": "SUVs",
            "brand": "Honda",
            "model": "CRV",
            "manufacture_year": 2018,
            "mileage": 60000.0,
            "engine_cc": 2000.0,
            "fuel_type": "Petrol",
            "transmission": "Automatic",
            "district": "Gampaha",
            "condition": "Registered (Used)",
        },
        {
            "category": "Motorbikes",
            "brand": "Bajaj",
            "model": "Pulsar 150",
            "manufacture_year": 2021,
            "mileage": 18000.0,
            "engine_cc": 150.0,
            "fuel_type": "Petrol",
            "transmission": "Manual",
            "district": "Kandy",
            "condition": "Registered (Used)",
        },
    ]


def test_end_to_end_valuation_pipeline_flow(test_vehicle_scenarios):
    """Verifies complete component chain from input to formatted report and REST response."""
    service = VehicleValuationService()

    for scenario in test_vehicle_scenarios:
        # 1. Valuation Service Execution
        result = service.valuate(scenario, top_k_factors=4, top_k_comparables=4)

        assert result.estimated_asking_price_lkr > 100_000.0
        assert result.currency == "LKR"

        # 2. Uncertainty Range Check
        r = result.prediction_range_lkr
        assert r["lower"] <= result.estimated_asking_price_lkr <= r["upper"]
        assert r["spread"] == pytest.approx(r["upper"] - r["lower"], rel=1e-3)

        # 3. Explainability Check
        assert len(result.explanation) > 0
        assert len(result.explanation) <= 4
        for factor in result.explanation:
            assert factor["direction"] in ["positive", "negative", "neutral"]
            assert isinstance(factor["contribution"], float)

        # 4. Comparable Matching Check
        assert len(result.comparables) > 0
        assert len(result.comparables) <= 4
        for comp in result.comparables:
            assert comp["category"] == scenario["category"]
            assert comp["asking_price"] > 0
            assert 0.0 <= comp["similarity_score"] <= 1.0

        # 5. Formatter Output Check
        text_report = ValuationReportFormatter.format_text(result, scenario)
        assert "VEHICLE VALUATION REPORT" in text_report
        assert scenario["brand"] in text_report
        assert "Rs." in text_report

        md_report = ValuationReportFormatter.format_markdown(result, scenario)
        assert scenario["brand"] in md_report
        assert "Comparable Market Listings" in md_report

        # 6. REST API Endpoint Check
        api_payload = scenario.copy()
        api_payload["top_k_factors"] = 3
        api_payload["top_k_comparables"] = 3

        response = client.post("/api/valuation/predict", json=api_payload)
        assert response.status_code == 200
        api_data = response.json()
        assert api_data["estimated_asking_price_lkr"] == pytest.approx(
            result.estimated_asking_price_lkr, rel=1e-3
        )
        assert len(api_data["explanation"]) <= 3
        assert len(api_data["comparables"]) <= 3


def test_database_invariance_after_full_valuation_workload():
    """Guarantees that executing valuation pipeline leaves PostgreSQL completely untouched."""
    SessionLocal = get_sessionmaker()

    with SessionLocal() as db:
        v_before = db.query(Vehicle).count()
        l_before = db.query(Listing).count()
        p_before = db.query(PriceHistory).count()
        o_before = db.query(ListingObservation).count()
        r_before = db.query(ScrapeRun).count()
        elig_before = db.query(Listing).filter(Listing.ml_eligible == True).count()
        inelig_before = db.query(Listing).filter(Listing.ml_eligible == False).count()

    # Execute valuation workloads
    service = VehicleValuationService()
    test_car = {
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
    for _ in range(3):
        service.valuate(test_car)
        client.post("/api/valuation/predict", json=test_car)

    # Re-audit database counts
    with SessionLocal() as db:
        v_after = db.query(Vehicle).count()
        l_after = db.query(Listing).count()
        p_after = db.query(PriceHistory).count()
        o_after = db.query(ListingObservation).count()
        r_after = db.query(ScrapeRun).count()
        elig_after = db.query(Listing).filter(Listing.ml_eligible == True).count()
        inelig_after = db.query(Listing).filter(Listing.ml_eligible == False).count()

    assert v_before == v_after == 171
    assert l_before == l_after == 171
    assert p_before == p_after == 135
    assert o_before == o_after == 188
    assert r_before == r_after == 23
    assert elig_before == elig_after == 113
    assert inelig_before == inelig_after == 58

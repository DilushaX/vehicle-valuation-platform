"""
Tests for Unified Vehicle Valuation Workflow Orchestration Layer (Phase 10.5 Part 1).

Validates:
A. Complete valid vehicle input:
   - Workflow executes successfully.
   - Valuation is returned.
   - Range is returned.
   - Explanation is returned.
   - Data quality is returned.
   - Comparables section exists.
   - Market summary exists.
   - Audit information exists.
B. Partial vehicle input:
   - Workflow does not crash unnecessarily.
   - Missing fields are reflected in data quality.
C. Zero comparable vehicles:
   - Valuation still works.
   - Comparable list is empty.
   - Market summary handles zero results safely.
D. Invalid required input:
   - Clear validation failure.
   - No fake valuation is returned.
   - raise_on_error=True raises ValidationError.
E. Database immutability:
   - Verifies the workflow performs strictly read/analysis operations and NO database writes.
F. Schema, serialization, and disclaimers:
   - All 8 sections are present in serialized dictionary.
   - Mandatory operational and methodological disclaimers are present.
   - Backward compatibility with ValuationResult is preserved.
"""

from pathlib import Path
from typing import Any, Dict
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.connection import Base
from database.models import Listing, Vehicle
from database.repository import VehicleRepository
from ml.prediction.predictor import ValidationError
from ml.valuation.valuation_service import (
    STANDARD_VALUATION_LIMITATIONS,
    ValuationResult,
    VehicleValuationService,
)
from ml.valuation.valuation_workflow import (
    AuditSection,
    ComparablesSection,
    DataQualitySection,
    ExplanationSection,
    MarketSummarySection,
    RangeSection,
    ValuationSection,
    ValuationWorkflowInput,
    ValuationWorkflowResult,
    VehicleValuationWorkflow,
    run_valuation_workflow,
)


@pytest.fixture
def valuation_service(db_session) -> VehicleValuationService:
    """Loads existing valuation service with trained model artifact and seeded in-memory listing."""
    model_path = Path("data/analysis/ml/models/model.joblib")
    meta_path = Path("data/analysis/ml/models/model_metadata.json")
    if not model_path.exists():
        pytest.skip(f"Model artifact not found at {model_path}")
    repo = VehicleRepository(db_session)
    repo.sync_listing({
        "listing_id": "TEST_SEED_01",
        "listing_url": "https://example.com/1",
        "category": "Cars",
        "brand": "Toyota",
        "model": "Premio",
        "manufacture_year": 2016,
        "mileage": 80000.0,
        "engine_cc": 1500,
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "district": "Colombo",
        "condition": "Registered (Used)",
        "price": 12_000_000,
        "ml_eligible": True,
    })
    repo.commit()
    return VehicleValuationService(db_session=db_session, model_path=model_path, metadata_path=meta_path)


@pytest.fixture
def workflow(valuation_service: VehicleValuationService) -> VehicleValuationWorkflow:
    """Initializes the valuation workflow using the existing valuation service."""
    return VehicleValuationWorkflow(valuation_service=valuation_service)


@pytest.fixture
def valid_car_spec() -> Dict[str, Any]:
    """A valid, complete vehicle specification."""
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


# ===========================================================================
# Test Suite A: Complete Valid Vehicle Input
# ===========================================================================

def test_complete_valid_vehicle_workflow_execution(workflow: VehicleValuationWorkflow, valid_car_spec: Dict[str, Any]):
    """
    A. Complete valid vehicle input:
       - workflow executes successfully
       - valuation is returned
       - range is returned
       - explanation is returned
       - data quality is returned
       - comparables section exists
       - market summary exists
       - audit information exists
    """
    result = workflow.execute(valid_car_spec, top_k_factors=3, top_k_comparables=3)

    assert isinstance(result, ValuationWorkflowResult)
    assert result.success is True
    assert result.status == "SUCCESS"
    assert result.error is None
    assert result.errors == []

    # 1. Valuation section
    assert result.valuation is not None
    assert isinstance(result.valuation, ValuationSection)
    assert result.valuation.estimated_asking_price_lkr > 1_000_000.0
    assert result.valuation.currency == "LKR"
    assert result.valuation.model["name"] == "RandomForestRegressor"
    assert result.valuation.model["target_variable"] == "asking_price"
    assert "asking price" in result.valuation.disclaimer.lower()

    # 2. Range section
    assert result.range is not None
    assert isinstance(result.range, RangeSection)
    assert result.range.lower <= result.valuation.estimated_asking_price_lkr <= result.range.upper
    assert result.range.lower > 0
    assert result.range.spread > 0
    assert "dispersion" in result.range.method.lower() or "percentile" in result.range.method.lower()
    assert "indicative model-based prediction range" in result.range.disclaimer.lower()

    # 3. Explanation section
    assert result.explanation is not None
    assert isinstance(result.explanation, ExplanationSection)
    assert len(result.explanation) == 3
    for factor in result.explanation:
        assert "feature" in factor
        assert "contribution" in factor
        assert "direction" in factor
        assert "value" in factor
    assert "shap values describe model contribution" in result.explanation.disclaimer.lower()

    # 4. Data Quality section
    assert result.data_quality is not None
    assert isinstance(result.data_quality, DataQualitySection)
    assert result.data_quality.status == "COMPLETE"
    assert result.data_quality.provided_features == 11
    assert result.data_quality.expected_features == 11
    assert result.data_quality.missing_features == []
    assert result.data_quality.comparable_count > 0

    # 5. Comparables section
    assert result.comparables is not None
    assert isinstance(result.comparables, ComparablesSection)
    assert len(result.comparables) > 0
    assert len(result.comparables) <= 3
    assert result.comparables.total_found == len(result.comparables.items)
    for comp in result.comparables:
        assert comp["category"] == "Cars"
        assert comp["asking_price"] > 0
        assert 0.0 <= comp["similarity_score"] <= 1.0
        assert 0.0 <= comp["similarity_percentage"] <= 100.0
    assert "comparable similarity is not a probability" in result.comparables.disclaimer.lower()

    # 6. Market Summary section
    assert result.market_summary is not None
    assert isinstance(result.market_summary, MarketSummarySection)
    assert result.market_summary.comparable_count == len(result.comparables)
    assert result.market_summary.min_asking_price is not None
    assert result.market_summary.max_asking_price is not None
    assert result.market_summary.min_asking_price <= result.market_summary.max_asking_price
    assert result.market_summary.median_asking_price is not None
    assert result.market_summary.average_asking_price is not None
    assert result.market_summary.price_spread is not None
    assert "advertised asking prices" in result.market_summary.disclaimer.lower()

    # 7. Audit section
    assert result.audit is not None
    assert isinstance(result.audit, AuditSection)
    assert result.audit.metadata["model_name"] == "RandomForestRegressor"
    assert result.audit.metadata["target"] == "asking_price"
    assert result.audit.reproducibility["algorithm"] == "SHA-256"
    assert len(result.audit.reproducibility["fingerprint"]) == 64
    assert "reproducibility fingerprint" in result.audit.disclaimer.lower()

    # 8. Limitations
    assert len(result.limitations) == len(STANDARD_VALUATION_LIMITATIONS)
    assert any("not a confirmed transaction" in lim.lower() for lim in result.limitations)


# ===========================================================================
# Test Suite B: Partial Vehicle Input
# ===========================================================================

def test_partial_vehicle_input_missing_mileage(workflow: VehicleValuationWorkflow, valid_car_spec: Dict[str, Any]):
    """
    B. Partial vehicle input:
       - workflow does not crash unnecessarily
       - missing fields are reflected in data quality
    """
    partial_spec = valid_car_spec.copy()
    partial_spec["mileage"] = None

    result = workflow.execute(partial_spec)

    assert result.success is True
    assert result.status == "PARTIAL_DATA"
    assert result.valuation is not None
    assert result.valuation.estimated_asking_price_lkr > 0

    assert result.data_quality is not None
    assert result.data_quality.status == "PARTIAL"
    assert result.data_quality.provided_features == 10
    assert result.data_quality.expected_features == 11
    assert "mileage" in result.data_quality.missing_features


def test_partial_vehicle_input_missing_multiple_optional_fields(
    workflow: VehicleValuationWorkflow,
    valid_car_spec: Dict[str, Any],
):
    """
    Partial input with both mileage and engine_cc omitted:
    does not crash and correctly tracks missing features in data quality.
    """
    partial_spec = valid_car_spec.copy()
    del partial_spec["mileage"]
    partial_spec["engine_cc"] = None

    result = workflow.execute(partial_spec)

    assert result.success is True
    assert result.status == "PARTIAL_DATA"
    assert result.data_quality.status == "PARTIAL"
    assert result.data_quality.provided_features == 9
    assert "mileage" in result.data_quality.missing_features
    assert "engine_cc" in result.data_quality.missing_features


def test_partial_vehicle_input_manufacture_year_derives_age(
    workflow: VehicleValuationWorkflow,
    valid_car_spec: Dict[str, Any],
):
    """
    Providing manufacture_year without vehicle_age works seamlessly and derives age.
    """
    spec = valid_car_spec.copy()
    spec.pop("vehicle_age", None)
    spec["manufacture_year"] = 2018

    result = workflow.execute(spec)

    assert result.success is True
    assert result.valuation is not None
    assert result.valuation.estimated_asking_price_lkr > 0


# ===========================================================================
# Test Suite C: Zero Comparable Vehicles
# ===========================================================================

def test_zero_comparable_vehicles_via_top_k_zero(
    workflow: VehicleValuationWorkflow,
    valid_car_spec: Dict[str, Any],
):
    """
    C. Zero comparable vehicles (top_k_comparables=0):
       - valuation still works
       - range still works
       - explanation still works
       - data quality reflects zero comparables
       - audit information exists
       - comparable list is empty
       - market summary handles zero results safely with None metrics
    """
    result = workflow.execute(valid_car_spec, top_k_comparables=0)

    assert result.success is True
    assert result.valuation is not None
    assert result.valuation.estimated_asking_price_lkr > 0

    assert result.range is not None
    assert result.range.lower > 0

    assert result.explanation is not None
    assert len(result.explanation) > 0

    assert result.data_quality is not None
    assert result.data_quality.comparable_count == 0

    assert result.audit is not None
    assert len(result.audit.reproducibility["fingerprint"]) == 64

    # Comparables section empty
    assert result.comparables is not None
    assert result.comparables.items == []
    assert result.comparables.total_found == 0
    assert len(result.comparables) == 0

    # Market summary handles zero safely
    assert result.market_summary is not None
    assert result.market_summary.comparable_count == 0
    assert result.market_summary.min_asking_price is None
    assert result.market_summary.max_asking_price is None
    assert result.market_summary.median_asking_price is None
    assert result.market_summary.average_asking_price is None
    assert result.market_summary.price_spread is None


def test_zero_comparables_when_candidate_pool_has_no_matches(
    valuation_service: VehicleValuationService,
    valid_car_spec: Dict[str, Any],
):
    """
    Zero comparables when engine returns no candidate listings:
    handles zero results safely without errors.
    """
    # Create workflow whose valuation_service uses an engine with no candidate matches
    import pandas as pd
    from analytics.comparables.comparable_engine import ComparableVehicleEngine

    empty_engine = ComparableVehicleEngine()
    # Mock find_comparables to return empty list
    empty_engine.find_comparables = lambda *args, **kwargs: []

    valuation_service_copy = VehicleValuationService(
        predictor=valuation_service.predictor,
        comparable_engine=empty_engine,
    )
    wf = VehicleValuationWorkflow(valuation_service=valuation_service_copy)

    result = wf.execute(valid_car_spec)

    assert result.success is True
    assert result.valuation.estimated_asking_price_lkr > 0
    assert result.comparables.items == []
    assert result.comparables.total_found == 0
    assert result.market_summary.comparable_count == 0
    assert result.market_summary.min_asking_price is None


# ===========================================================================
# Test Suite D: Invalid Required Input
# ===========================================================================

def test_invalid_input_missing_required_brand(workflow: VehicleValuationWorkflow, valid_car_spec: Dict[str, Any]):
    """
    D. Invalid required input:
       - clear validation failure
       - no fake valuation is returned
    """
    bad_spec = valid_car_spec.copy()
    del bad_spec["brand"]

    result = workflow.execute(bad_spec)

    assert result.success is False
    assert result.status == "VALIDATION_ERROR"
    assert result.valuation is None
    assert result.range is None
    assert result.explanation is None
    assert result.error is not None
    assert "Missing required categorical features" in result.error
    assert len(result.errors) > 0


def test_invalid_input_unrecognized_category(workflow: VehicleValuationWorkflow, valid_car_spec: Dict[str, Any]):
    """Invalid vehicle category yields clear failure and no fake valuation."""
    bad_spec = valid_car_spec.copy()
    bad_spec["category"] = "Hovercrafts"

    result = workflow.execute(bad_spec)

    assert result.success is False
    assert result.status == "VALIDATION_ERROR"
    assert result.valuation is None
    assert "Invalid vehicle category" in result.error


def test_invalid_input_negative_mileage(workflow: VehicleValuationWorkflow, valid_car_spec: Dict[str, Any]):
    """Negative mileage fails validation without returning valuation."""
    bad_spec = valid_car_spec.copy()
    bad_spec["mileage"] = -25000.0

    result = workflow.execute(bad_spec)

    assert result.success is False
    assert result.status == "VALIDATION_ERROR"
    assert result.valuation is None
    assert "mileage cannot be negative" in result.error


def test_invalid_input_future_manufacture_year(workflow: VehicleValuationWorkflow, valid_car_spec: Dict[str, Any]):
    """Future manufacture year leading to negative age fails validation."""
    bad_spec = valid_car_spec.copy()
    bad_spec.pop("vehicle_age", None)
    bad_spec["manufacture_year"] = 2030  # Relative to reference year 2026

    result = workflow.execute(bad_spec)

    assert result.success is False
    assert result.status == "VALIDATION_ERROR"
    assert result.valuation is None
    assert "vehicle_age cannot be negative" in result.error


def test_invalid_input_raise_on_error_mode(workflow: VehicleValuationWorkflow, valid_car_spec: Dict[str, Any]):
    """When raise_on_error=True, raises ValidationError directly."""
    bad_spec = valid_car_spec.copy()
    del bad_spec["brand"]

    with pytest.raises(ValidationError, match="Missing required categorical features"):
        workflow.execute(bad_spec, raise_on_error=True)


# ===========================================================================
# Test Suite E: Database Immutability (No DB Writes)
# ===========================================================================

def test_workflow_performs_no_database_writes(valuation_service: VehicleValuationService, valid_car_spec: Dict[str, Any]):
    """
    E. Verify the workflow does not modify database state.
    Executes multiple workflow calls against an isolated SQLite test session
    and verifies row counts and transaction state remain completely unchanged.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionClass = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = SessionClass()

    try:
        repo = VehicleRepository(session)
        # Seed 2 dummy listings
        repo.sync_listing({
            "listing_id": "TEST_DB_01",
            "listing_url": "https://example.com/1",
            "category": "Cars",
            "brand": "Toyota",
            "model": "Premio",
            "manufacture_year": 2016,
            "mileage": 80000.0,
            "engine_cc": 1500,
            "fuel_type": "Petrol",
            "transmission": "Automatic",
            "district": "Colombo",
            "condition": "Registered (Used)",
            "price": 12_000_000,
        })
        repo.sync_listing({
            "listing_id": "TEST_DB_02",
            "listing_url": "https://example.com/2",
            "category": "Cars",
            "brand": "Toyota",
            "model": "Allion",
            "manufacture_year": 2015,
            "mileage": 90000.0,
            "engine_cc": 1500,
            "fuel_type": "Petrol",
            "transmission": "Automatic",
            "district": "Gampaha",
            "condition": "Registered (Used)",
            "price": 11_500_000,
        })
        repo.commit()

        initial_count = session.query(Listing).count()
        assert initial_count == 2

        # Create workflow using this test session
        wf = VehicleValuationWorkflow(session=session)

        # 1. Execute valid request
        res1 = wf.execute(valid_car_spec)
        assert res1.success is True

        # 2. Execute partial request
        partial_spec = valid_car_spec.copy()
        partial_spec["mileage"] = None
        res2 = wf.execute(partial_spec)
        assert res2.success is True

        # 3. Execute zero-comparables request
        res3 = wf.execute(valid_car_spec, top_k_comparables=0)
        assert res3.success is True

        # 4. Execute invalid request
        bad_spec = valid_car_spec.copy()
        bad_spec["mileage"] = -100.0
        res4 = wf.execute(bad_spec)
        assert res4.success is False

        # Verify database state was NOT modified
        final_count = session.query(Listing).count()
        assert final_count == initial_count
        assert len(session.dirty) == 0
        assert len(session.new) == 0
        assert len(session.deleted) == 0

    finally:
        session.rollback()
        session.close()


# ===========================================================================
# Test Suite F: Schemas, Serialization, and Backward Compatibility
# ===========================================================================

def test_workflow_result_to_dict_structure(workflow: VehicleValuationWorkflow, valid_car_spec: Dict[str, Any]):
    """
    Verifies that result.to_dict() contains all required top-level sections:
    valuation, range, explanation, data_quality, comparables, market_summary, audit, limitations.
    """
    result = workflow.execute(valid_car_spec)
    d = result.to_dict()

    expected_sections = [
        "valuation",
        "range",
        "explanation",
        "data_quality",
        "comparables",
        "market_summary",
        "audit",
        "limitations",
    ]
    for sec in expected_sections:
        assert sec in d, f"Missing section '{sec}' in workflow result dictionary"

    assert isinstance(d["valuation"], dict)
    assert isinstance(d["range"], dict)
    assert isinstance(d["explanation"], dict)
    assert isinstance(d["data_quality"], dict)
    assert isinstance(d["comparables"], dict)
    assert isinstance(d["market_summary"], dict)
    assert isinstance(d["audit"], dict)
    assert isinstance(d["limitations"], list)


def test_conversion_to_legacy_valuation_result(workflow: VehicleValuationWorkflow, valid_car_spec: Dict[str, Any]):
    """
    Verifies result.to_valuation_result() produces a fully compatible legacy ValuationResult.
    """
    result = workflow.execute(valid_car_spec)
    legacy_res = result.to_valuation_result()

    assert isinstance(legacy_res, ValuationResult)
    assert legacy_res.estimated_asking_price_lkr == result.valuation.estimated_asking_price_lkr
    assert legacy_res.currency == "LKR"
    assert len(legacy_res.explanation) == len(result.explanation.factors)
    assert len(legacy_res.comparables) == len(result.comparables.items)
    assert legacy_res.data_quality["status"] == "COMPLETE"
    assert legacy_res.audit["model_name"] == "RandomForestRegressor"
    assert legacy_res.reproducibility["algorithm"] == "SHA-256"


def test_run_valuation_workflow_functional_helper(valid_car_spec: Dict[str, Any], db_session):
    """Verifies module-level run_valuation_workflow helper function."""
    result = run_valuation_workflow(valid_car_spec, session=db_session, top_k_factors=2, top_k_comparables=2)

    assert isinstance(result, ValuationWorkflowResult)
    assert result.success is True
    assert result.valuation.estimated_asking_price_lkr > 0
    assert len(result.explanation) == 2
    assert len(result.comparables) <= 2


def test_pydantic_input_model_support(workflow: VehicleValuationWorkflow, valid_car_spec: Dict[str, Any]):
    """Verifies that ValuationWorkflowInput can be passed directly as input."""
    pydantic_input = ValuationWorkflowInput(**valid_car_spec)
    result = workflow.execute(pydantic_input)

    assert result.success is True
    assert result.valuation.estimated_asking_price_lkr > 0
    assert result.data_quality.status == "COMPLETE"


def test_reproducibility_deterministic(workflow: VehicleValuationWorkflow, valid_car_spec: Dict[str, Any]):
    """Identical inputs produce identical reproducibility fingerprints."""
    res1 = workflow.execute(valid_car_spec)
    res2 = workflow.execute(valid_car_spec)

    assert res1.audit.reproducibility["fingerprint"] == res2.audit.reproducibility["fingerprint"]

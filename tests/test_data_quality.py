"""
Comprehensive test suite for Phase 4 Step 5: Data Quality & Historical Dataset Readiness.
Covers:
1. Missing, invalid, out-of-range, and suspicious price validation.
2. Missing, negative, out-of-range, and suspicious mileage validation.
3. Valid mileage acceptance.
4. Manufacture Year (YOM) validation (missing, invalid, future).
5. Registration Year (YOR) validation (missing, future).
6. Year relationship validation (YOR < YOM -> registration_before_manufacture).
7. Valid YOM and YOR combinations.
8. Engine CC validation (negative, zero, EV tolerance, implausible bounds).
9. Fuel type and transmission normalization.
10. Category validation (missing, invalid, canonicalization).
11. ML eligibility decision (critical vs. non-critical classification).
12. Historical consistency checking (clean DB vs. anomaly detection).
13. Historical lifecycle preservation (first_seen_at immutability, observations, price changes, reactivation).
14. Data quality CLI reporting.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import pytest
from sqlalchemy import select

from data_pipeline.cleaning.cleaners import VehicleCleaner
from data_pipeline.quality.quality_engine import DatasetQualityEngine
from database.models import Listing, ListingObservation, PriceHistory, ScrapeRun, Vehicle
from database.repository import VehicleRepository
from scraper.validators.listing_validator import ListingValidator
import scripts.data_quality_report as report_cli


# ==============================================================================
# 1. PRICE VALIDATION TESTS
# ==============================================================================

def test_missing_price():
    validator = ListingValidator()
    record = {
        "price": None,
        "mileage": 50000,
        "manufacture_year": 2018,
        "brand": "Toyota",
        "model": "Axio",
        "category": "Cars",
    }
    res = validator.validate(record)
    assert res["is_valid"] is False
    assert res["ml_eligible"] is False
    assert "missing_price" in res["validation_issues"]
    assert "missing_price" in res["critical_issues"]
    assert any("price is missing" in r.lower() for r in res["ml_exclusion_reasons"])


def test_invalid_price_zero_or_negative():
    validator = ListingValidator()
    res_zero = validator.validate({"price": 0, "mileage": 50000, "manufacture_year": 2018, "brand": "Toyota", "model": "Axio", "category": "Cars"})
    assert "invalid_price" in res_zero["validation_issues"]
    assert res_zero["ml_eligible"] is False

    res_neg = validator.validate({"price": -50000, "mileage": 50000, "manufacture_year": 2018, "brand": "Toyota", "model": "Axio", "category": "Cars"})
    assert "invalid_price" in res_neg["validation_issues"]
    assert res_neg["ml_eligible"] is False


def test_price_out_of_range_category_aware():
    validator = ListingValidator()
    # Car exceeding 350M
    res_high = validator.validate({"price": 600_000_000, "mileage": 10000, "manufacture_year": 2022, "brand": "Bugatti", "model": "Chiron", "category": "Cars"})
    assert "price_out_of_range" in res_high["validation_issues"]
    assert res_high["ml_eligible"] is False

    # Motorbike with price Rs. 5,000 (below 25,000 minimum)
    res_low = validator.validate({"price": 5000, "mileage": 10000, "manufacture_year": 2020, "brand": "Bajaj", "model": "Pulsar", "category": "Motorbikes"})
    assert "price_out_of_range" in res_low["validation_issues"]
    assert res_low["ml_eligible"] is False


def test_suspicious_price_pattern():
    validator = ListingValidator()
    for dummy_price in [111111, 123456, 999999, 222222]:
        res = validator.validate({"price": dummy_price, "mileage": 50000, "manufacture_year": 2018, "brand": "Toyota", "model": "Axio", "category": "Cars"})
        assert "suspicious_price_pattern" in res["validation_issues"]
        assert res["ml_eligible"] is False


# ==============================================================================
# 2. MILEAGE VALIDATION TESTS
# ==============================================================================

def test_missing_mileage():
    validator = ListingValidator()
    res = validator.validate({"price": 5000000, "mileage": None, "manufacture_year": 2018, "brand": "Toyota", "model": "Axio", "category": "Cars"})
    assert "missing_mileage" in res["validation_issues"]
    assert res["ml_eligible"] is False


def test_invalid_mileage_negative():
    validator = ListingValidator()
    res = validator.validate({"price": 5000000, "mileage": -1500, "manufacture_year": 2018, "brand": "Toyota", "model": "Axio", "category": "Cars"})
    assert "invalid_mileage" in res["validation_issues"]
    assert res["ml_eligible"] is False


def test_suspicious_mileage_pattern():
    validator = ListingValidator()
    for dummy_mileage in [111111, 123456, 999999, 222222]:
        res = validator.validate({"price": 5000000, "mileage": dummy_mileage, "manufacture_year": 2018, "brand": "Toyota", "model": "Axio", "category": "Cars"})
        assert "suspicious_mileage_pattern" in res["validation_issues"]
        assert res["ml_eligible"] is False


def test_valid_mileage():
    validator = ListingValidator()
    res = validator.validate({"price": 5000000, "mileage": 48500, "manufacture_year": 2018, "brand": "Toyota", "model": "Axio", "category": "Cars"})
    assert "missing_mileage" not in res["validation_issues"]
    assert "invalid_mileage" not in res["validation_issues"]
    assert "suspicious_mileage_pattern" not in res["validation_issues"]


# ==============================================================================
# 3. YEAR VALIDATION TESTS (YOM, YOR, RELATIONSHIP)
# ==============================================================================

def test_missing_yom():
    validator = ListingValidator()
    res = validator.validate({"price": 5000000, "mileage": 50000, "manufacture_year": None, "year": None, "brand": "Toyota", "model": "Axio", "category": "Cars"})
    assert "missing_yom" in res["validation_issues"]
    assert res["ml_eligible"] is False


def test_historical_invalid_yom():
    validator = ListingValidator()
    res = validator.validate({"price": 5000000, "mileage": 50000, "manufacture_year": 1930, "brand": "Toyota", "model": "Axio", "category": "Cars"})
    assert "invalid_yom" in res["validation_issues"]
    assert res["ml_eligible"] is False


def test_future_yom():
    validator = ListingValidator()
    future_year = datetime.now().year + 5
    res = validator.validate({"price": 5000000, "mileage": 50000, "manufacture_year": future_year, "brand": "Toyota", "model": "Axio", "category": "Cars"})
    assert "future_yom" in res["validation_issues"]
    assert res["ml_eligible"] is False


def test_missing_yor_is_non_critical():
    """Unregistered or brand new vehicles might lack YOR, but remain ML eligible."""
    validator = ListingValidator()
    res = validator.validate({
        "price": 5000000,
        "mileage": 50000,
        "manufacture_year": 2018,
        "registration_year": None,
        "brand": "Toyota",
        "model": "Axio",
        "category": "Cars",
    })
    # If YOM is valid and all core fields valid, it is ML eligible
    assert res["ml_eligible"] is True
    assert "missing_yom" not in res["validation_issues"]


def test_future_yor():
    validator = ListingValidator()
    future_year = datetime.now().year + 5
    res = validator.validate({
        "price": 5000000,
        "mileage": 50000,
        "manufacture_year": 2018,
        "registration_year": future_year,
        "brand": "Toyota",
        "model": "Axio",
        "category": "Cars",
    })
    assert "future_yor" in res["validation_issues"]
    assert "future_yor" in res["non_critical_issues"]
    # Non-critical issue does not disqualify from valuation training
    assert res["ml_eligible"] is True
    assert res["is_valid"] is False


def test_registration_before_manufacture():
    """Clerical paperwork anomaly (YOR < YOM) is flagged as non-critical."""
    validator = ListingValidator()
    res = validator.validate({
        "price": 5000000,
        "mileage": 50000,
        "manufacture_year": 2018,
        "registration_year": 2016,
        "brand": "Toyota",
        "model": "Axio",
        "category": "Cars",
    })
    assert res["is_valid"] is False
    assert "registration_before_manufacture" in res["validation_issues"]
    assert "registration_before_manufacture" in res["non_critical_issues"]
    assert res["ml_eligible"] is True  # Non-critical!


def test_valid_yom_and_yor():
    validator = ListingValidator()
    res = validator.validate({
        "price": 5000000,
        "mileage": 50000,
        "manufacture_year": 2018,
        "registration_year": 2019,
        "brand": "Toyota",
        "model": "Axio",
        "category": "Cars",
    })
    assert res["is_valid"] is True
    assert res["ml_eligible"] is True
    assert len(res["validation_issues"]) == 0


# ==============================================================================
# 4. ENGINE CC & FUEL/TRANSMISSION NORMALIZATION
# ==============================================================================

def test_engine_cc_validation():
    validator = ListingValidator()

    # Negative CC
    res_neg = validator.validate({"price": 5000000, "mileage": 50000, "manufacture_year": 2018, "brand": "Toyota", "model": "Axio", "category": "Cars", "engine_cc": -100})
    assert "invalid_engine_cc" in res_neg["validation_issues"]
    assert res_neg["ml_eligible"] is True  # Non-critical

    # Zero CC on ICE car
    res_zero = validator.validate({"price": 5000000, "mileage": 50000, "manufacture_year": 2018, "brand": "Toyota", "model": "Axio", "category": "Cars", "engine_cc": 0, "fuel_type": "Petrol"})
    assert "invalid_engine_cc" in res_zero["validation_issues"]

    # Zero CC on Electric Vehicle is valid!
    res_ev = validator.validate({"price": 5000000, "mileage": 50000, "manufacture_year": 2018, "brand": "Nissan", "model": "Leaf", "category": "Cars", "engine_cc": 0, "fuel_type": "Electric"})
    assert "invalid_engine_cc" not in res_ev["validation_issues"]

    # Implausible CC (> 16000)
    res_huge = validator.validate({"price": 5000000, "mileage": 50000, "manufacture_year": 2018, "brand": "Toyota", "model": "Axio", "category": "Cars", "engine_cc": 25000})
    assert "implausible_engine_cc" in res_huge["validation_issues"]


def test_fuel_type_and_transmission_normalization():
    assert VehicleCleaner.normalize_fuel_type("gasoline") == "Petrol"
    assert VehicleCleaner.normalize_fuel_type("super petrol") == "Petrol"
    assert VehicleCleaner.normalize_fuel_type("super diesel") == "Diesel"
    assert VehicleCleaner.normalize_fuel_type("plug-in hybrid") == "Hybrid"
    assert VehicleCleaner.normalize_fuel_type("phev") == "Hybrid"
    assert VehicleCleaner.normalize_fuel_type("ev") == "Electric"
    assert VehicleCleaner.normalize_fuel_type("battery electric") == "Electric"

    assert VehicleCleaner.normalize_transmission("auto") == "Automatic"
    assert VehicleCleaner.normalize_transmission("cvt") == "Automatic"
    assert VehicleCleaner.normalize_transmission("tiptronic") == "Automatic"
    assert VehicleCleaner.normalize_transmission("mt") == "Manual"
    assert VehicleCleaner.normalize_transmission("manual") == "Manual"


# ==============================================================================
# 5. CATEGORY VALIDATION & CANONICALIZATION
# ==============================================================================

def test_category_validation():
    validator = ListingValidator()

    # Missing category
    res_miss = validator.validate({"price": 5000000, "mileage": 50000, "manufacture_year": 2018, "brand": "Toyota", "model": "Axio", "category": None}, require_category=True)
    assert "missing_category" in res_miss["validation_issues"]
    assert res_miss["ml_eligible"] is False

    # Invalid category
    res_inv = validator.validate({"price": 5000000, "mileage": 50000, "manufacture_year": 2018, "brand": "Toyota", "model": "Axio", "category": "Aeroplane"}, require_category=True)
    assert "invalid_category" in res_inv["validation_issues"]
    assert res_inv["ml_eligible"] is False

    # Canonical mapping
    assert VehicleCleaner.canonicalize_category("cars") == "Cars"
    assert VehicleCleaner.canonicalize_category("car") == "Cars"
    assert VehicleCleaner.canonicalize_category("three-wheels") == "Three Wheelers"
    assert VehicleCleaner.canonicalize_category("three wheel") == "Three Wheelers"
    assert VehicleCleaner.canonicalize_category("heavy-duty") == "Heavy-Duty"
    assert VehicleCleaner.canonicalize_category("motorcycles") == "Motorbikes"


# ==============================================================================
# 6. ML ELIGIBILITY (CRITICAL VS NON-CRITICAL)
# ==============================================================================

def test_ml_eligibility_critical_vs_non_critical():
    validator = ListingValidator()

    # Listing with ONLY non-critical issues (missing YOR, registration_before_manufacture, missing engine_cc)
    record_non_crit = {
        "price": 14500000,
        "mileage": 35000,
        "manufacture_year": 2019,
        "registration_year": 2018,  # clerical anomaly
        "brand": "Toyota",
        "model": "Premio",
        "category": "Cars",
        "engine_cc": None,          # optional
    }
    res1 = validator.validate(record_non_crit)
    assert res1["is_valid"] is False  # Has data quality issue
    assert res1["ml_eligible"] is True  # But valid for ML valuation!
    assert len(res1["critical_issues"]) == 0
    assert len(res1["non_critical_issues"]) > 0

    # Listing with a CRITICAL issue (missing price)
    record_crit = dict(record_non_crit)
    record_crit["price"] = None
    res2 = validator.validate(record_crit)
    assert res2["is_valid"] is False
    assert res2["ml_eligible"] is False
    assert "missing_price" in res2["critical_issues"]
    assert len(res2["ml_exclusion_reasons"]) > 0


# ==============================================================================
# 7. HISTORICAL DATASET CONSISTENCY CHECKS
# ==============================================================================

def test_historical_consistency_on_clean_dataset(db_session):
    """Verifies that a well-formed database session reports 0 consistency anomalies."""
    repo = VehicleRepository(db_session)
    run = repo.create_scrape_run(category="Cars")

    data = {
        "listing_id": "test_clean_01",
        "listing_url": "https://riyasewana.com/buy/test-clean-01",
        "category": "Cars",
        "brand": "Toyota",
        "model": "Axio",
        "manufacture_year": 2017,
        "registration_year": 2018,
        "price": 11000000,
        "mileage": 60000,
        "is_valid": True,
        "ml_eligible": True,
    }
    repo.sync_listing(data, scrape_run=run)
    repo.commit()

    engine = DatasetQualityEngine()
    audit = engine.audit_dataset(db_session)

    assert audit["total_listings"] == 1
    assert audit["total_vehicles"] == 1
    assert audit["ml_eligible_count"] == 1
    assert audit["consistency_checks"]["is_consistent"] is True
    assert audit["consistency_checks"]["total_consistency_errors"] == 0


def test_historical_consistency_detects_deliberate_anomalies(db_session):
    """Verifies that the consistency checker detects duplicate listings, timestamp ordering, and orphans."""
    engine = DatasetQualityEngine()

    # Seed an orphan PriceHistory (pointing to non-existent listing_id 999999)
    orphan_price = PriceHistory(listing_id=999999, price=5000000, observed_at=datetime.now(timezone.utc))
    db_session.add(orphan_price)

    # Seed an orphan Vehicle (without any listing)
    orphan_v = Vehicle(category="Cars", brand="Test", model="Orphan", manufacture_year=2015)
    db_session.add(orphan_v)

    # Seed a listing with first_seen_at > last_seen_at
    bad_v = Vehicle(category="Cars", brand="Toyota", model="Corolla")
    db_session.add(bad_v)
    db_session.flush()

    bad_listing = Listing(
        listing_id="bad_time_01",
        vehicle_id=bad_v.id,
        listing_url="https://riyasewana.com/buy/bad-time-01",
        first_seen_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
        last_seen_at=datetime(2026, 9, 1, tzinfo=timezone.utc),  # earlier than first_seen!
        current_status="ACTIVE",
    )
    db_session.add(bad_listing)
    db_session.commit()

    audit = engine.audit_dataset(db_session)
    consistency = audit["consistency_checks"]

    assert consistency["is_consistent"] is False
    assert consistency["total_consistency_errors"] >= 3
    assert consistency["checks"]["orphan_price_histories"]["is_valid"] is False
    assert consistency["checks"]["orphan_vehicles"]["is_valid"] is False
    assert consistency["checks"]["first_seen_after_last_seen"]["is_valid"] is False


# ==============================================================================
# 8. HISTORICAL LIFECYCLE PRESERVATION & REACTIVATION
# ==============================================================================

def to_utc(dt: datetime) -> datetime:
    """Helper ensuring timezone-aware UTC comparison across backends."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def test_historical_lifecycle_preservation_and_reactivation(db_session):
    """
    Verifies:
    1. First observation -> ACTIVE, first_seen_at recorded.
    2. Same price -> observation created, NO new PriceHistory.
    3. Price change -> observation created, new PriceHistory.
    4. Disappearance -> NO_LONGER_OBSERVED (never SOLD).
    5. Reappearance -> ACTIVE, first_seen_at immutable, observations preserved.
    """
    repo = VehicleRepository(db_session)
    t1 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 9, 3, 10, 0, tzinfo=timezone.utc)
    t4 = datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc)

    run1 = repo.create_scrape_run(category="Cars")
    data1 = {
        "listing_id": "hist_lifecycle_test",
        "listing_url": "https://riyasewana.com/buy/hist-test",
        "category": "Cars",
        "brand": "Toyota",
        "model": "Allion",
        "manufacture_year": 2017,
        "price": 12000000,
        "mileage": 55000,
    }

    # Step 1: Initial observation
    l1, is_new1 = repo.sync_listing(data1, scrape_run=run1, observed_at=t1)
    repo.commit()
    assert is_new1 is True
    assert l1.current_status == "ACTIVE"
    assert to_utc(l1.first_seen_at) == t1
    assert len(l1.price_history) == 1
    assert len(l1.observations) == 1

    # Step 2: Second observation, identical price
    run2 = repo.create_scrape_run(category="Cars")
    l2, is_new2 = repo.sync_listing(data1, scrape_run=run2, observed_at=t2)
    repo.commit()
    assert is_new2 is False
    assert to_utc(l2.first_seen_at) == t1  # Immutable!
    assert to_utc(l2.last_seen_at) == t2
    assert len(l2.price_history) == 1  # Unchanged price -> no duplicate PriceHistory!
    assert len(l2.observations) == 2

    # Step 3: Third observation, price reduced to 11.5M
    run3 = repo.create_scrape_run(category="Cars")
    data3 = dict(data1)
    data3["price"] = 11500000
    l3, is_new3 = repo.sync_listing(data3, scrape_run=run3, observed_at=t3)
    repo.commit()
    assert len(l3.price_history) == 2  # New price history recorded!
    assert l3.price_history[-1].price == 11500000
    assert len(l3.observations) == 3

    # Step 4: Disappearance (NO_LONGER_OBSERVED, never SOLD)
    repo.mark_listing_no_longer_observed(l3)
    repo.commit()
    reloaded_disp = repo.find_listing("hist_lifecycle_test")
    assert reloaded_disp.current_status == "NO_LONGER_OBSERVED"

    # Step 5: Reappearance
    run4 = repo.create_scrape_run(category="Cars")
    l4, is_new4 = repo.sync_listing(data3, scrape_run=run4, observed_at=t4)
    repo.commit()
    assert is_new4 is False
    assert l4.current_status == "ACTIVE"
    assert to_utc(l4.first_seen_at) == t1  # Original first_seen preserved!
    assert to_utc(l4.last_seen_at) == t4
    assert len(l4.observations) == 4
    assert len(l4.price_history) == 2  # Same 11.5M price -> no new price history


# ==============================================================================
# 9. CLI DATA QUALITY REPORT TESTS
# ==============================================================================

def test_cli_data_quality_report(capsys):
    """Verifies that data_quality_report CLI can execute without errors."""
    with patch("sys.argv", ["scripts/data_quality_report.py"]):
        exit_code = report_cli.main()
        assert exit_code == 0

    captured = capsys.readouterr()
    assert "VEHICLE VALUATION PLATFORM — DATA QUALITY & READINESS REPORT" in captured.out
    assert "GLOBAL DATASET INVENTORY" in captured.out
    assert "ML VALUATION TRAINING ELIGIBILITY" in captured.out
    assert "HISTORICAL LIFECYCLE CONSISTENCY AUDIT" in captured.out

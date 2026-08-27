"""
Tests for Data Quality Engine & Deduplication
"""
import pytest
from data_pipeline.quality.quality_engine import DataQualityEngine
from data_pipeline.cleaning.cleaners import DataCleaner
from data_pipeline.deduplication.dedup import DeduplicationEngine

def test_clean_standardization():
    raw = {
        "make": "toyota",
        "model": "TOYOTA AQUA",
        "fuel_type": "PETROL HYBRID",
        "transmission": "AUTO",
        "district": "colombo 03",
        "yom": "2018",
        "mileage_km": "65000",
        "price_rs": "8500000"
    }
    cleaned = DataCleaner.clean_record(raw)
    assert cleaned["make"] == "Toyota"
    assert cleaned["model"] == "Aqua"
    assert cleaned["fuel_type"] == "Hybrid"
    assert cleaned["transmission"] == "Automatic"
    assert cleaned["district"] == "Colombo"
    assert cleaned["yom"] == 2018
    assert cleaned["mileage_km"] == 65000
    assert cleaned["price_rs"] == 8500000.0

def test_valid_record_quality():
    engine = DataQualityEngine()
    record = {
        "category": "Cars",
        "make": "Toyota",
        "model": "Aqua",
        "yom": 2018,
        "mileage_km": 65000,
        "price_rs": 8500000.0,
        "fuel_type": "Hybrid",
        "transmission": "Automatic",
        "district": "Colombo"
    }
    status, score, issues, is_outlier = engine.validate_single_record(record)
    assert status == "VALID"
    assert score >= 90.0
    assert len(issues) == 0
    assert is_outlier is False

def test_suspicious_patterns():
    engine = DataQualityEngine()
    # Dummy mileage
    record_bad_mileage = {
        "category": "Cars",
        "make": "Toyota",
        "model": "Aqua",
        "yom": 2018,
        "mileage_km": 123456,
        "price_rs": 8500000.0
    }
    status, score, issues, _ = engine.validate_single_record(record_bad_mileage)
    assert status == "SUSPICIOUS"
    assert any("dummy mileage" in issue.lower() for issue in issues)

    # Dummy price
    record_bad_price = {
        "category": "Cars",
        "make": "Toyota",
        "model": "Aqua",
        "yom": 2018,
        "mileage_km": 65000,
        "price_rs": 123.0
    }
    status_p, score_p, issues_p, _ = engine.validate_single_record(record_bad_price)
    assert status_p == "SUSPICIOUS"
    assert any("price" in issue.lower() for issue in issues_p)

def test_invalid_and_missing():
    engine = DataQualityEngine()
    record_invalid_year = {
        "category": "Cars",
        "make": "Toyota",
        "model": "Aqua",
        "yom": 1950,
        "mileage_km": 65000,
        "price_rs": 8500000.0
    }
    status, _, issues, _ = engine.validate_single_record(record_invalid_year)
    assert status == "INVALID"

    record_missing = {
        "category": "Cars",
        "make": None,
        "model": None,
        "yom": 2018,
        "mileage_km": None,
        "price_rs": 8500000.0
    }
    status_m, _, issues_m, _ = engine.validate_single_record(record_missing)
    assert status_m == "MISSING"

def test_deduplication():
    records = [
        {"make": "Toyota", "model": "Aqua", "yom": 2018, "fuel_type": "Hybrid", "transmission": "Automatic", "mileage_km": 65000, "price_rs": 8500000, "district": "Colombo"},
        {"make": "Toyota", "model": "Aqua", "yom": 2018, "fuel_type": "Hybrid", "transmission": "Automatic", "mileage_km": 65000, "price_rs": 8500000, "district": "Colombo"},
        {"make": "Honda", "model": "Vezel", "yom": 2016, "fuel_type": "Hybrid", "transmission": "Automatic", "mileage_km": 80000, "price_rs": 10200000, "district": "Kandy"}
    ]
    deduped = DeduplicationEngine.detect_duplicates(records)
    assert len(deduped) == 3
    assert deduped[0]["is_duplicate"] is False
    assert deduped[1]["is_duplicate"] is True
    assert deduped[2]["is_duplicate"] is False

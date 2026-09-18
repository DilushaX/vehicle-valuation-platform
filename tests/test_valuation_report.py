"""
Tests for Vehicle Valuation Report Formatter (Step 9 - Part 8).

Validates:
1. Sensible LKR rounding rounds to nearest 10,000 without artificial micro-decimals.
2. Plain-text formatter outputs vehicle specs, price, range, factors, comparables, and legal notice.
3. Markdown formatter outputs structured markdown with tables and badges.
4. Formatter handles partial or empty result objects without raising exceptions.
"""

import pytest

from ml.prediction.valuation_report import (
    ValuationReportFormatter,
    round_sensible_lkr,
)


@pytest.fixture
def mock_valuation_result():
    return {
        "estimated_asking_price_lkr": 15_916_610.43,
        "prediction_range_lkr": {
            "lower": 8_723_171.12,
            "upper": 24_122_604.89,
            "spread": 15_399_433.77,
            "method": "RandomForest tree dispersion",
        },
        "currency": "LKR",
        "model": {
            "name": "RandomForestRegressor",
            "target_variable": "asking_price",
        },
        "explanation": [
            {
                "feature": "transmission",
                "value": "Automatic",
                "contribution": 0.6673,
                "direction": "positive",
                "description": "Transmission (Automatic) contributed positively.",
            },
            {
                "feature": "fuel_type",
                "value": "Petrol",
                "contribution": -0.0136,
                "direction": "negative",
                "description": "Fuel Type (Petrol) contributed negatively.",
            },
        ],
        "comparables": [
            {
                "listing_id": "12293566",
                "category": "Cars",
                "brand": "Toyota",
                "model": "Premio",
                "manufacture_year": 2016,
                "mileage": 85000,
                "district": "Colombo",
                "asking_price": 15_500_000.0,
                "similarity_percentage": 96.0,
            }
        ],
        "limitations": [
            "This estimate models seller advertised asking prices on Riyasewana.",
            "Actual transaction prices may differ.",
        ],
    }


@pytest.fixture
def mock_vehicle_spec():
    return {
        "brand": "Toyota",
        "model": "Premio",
        "manufacture_year": 2016,
        "vehicle_age": 10,
        "mileage": 85000,
        "engine_cc": 1500,
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "district": "Colombo",
        "condition": "Registered (Used)",
    }


def test_round_sensible_lkr():
    assert round_sensible_lkr(15_916_610.43) == 15_920_000
    assert round_sensible_lkr(8_723_171.12) == 8_720_000
    assert round_sensible_lkr(0.0) == 0
    assert round_sensible_lkr(-500.0) == 0


def test_format_text_output(mock_valuation_result, mock_vehicle_spec):
    text = ValuationReportFormatter.format_text(mock_valuation_result, mock_vehicle_spec)

    assert "VEHICLE VALUATION REPORT" in text
    assert "Toyota Premio" in text
    assert "2016" in text
    assert "Estimated Market Asking Price : Rs. 15,920,000" in text
    assert "Indicative Model-Based Range  : Rs. 8,720,000 – Rs. 24,120,000" in text
    assert "Transmission (Automatic)" in text
    assert "12293566" in text
    assert "NOT a confirmed transaction" in text


def test_format_markdown_output(mock_valuation_result, mock_vehicle_spec):
    md = ValuationReportFormatter.format_markdown(mock_valuation_result, mock_vehicle_spec)

    assert "# Vehicle Valuation: Toyota Premio" in md
    assert "Rs. 15,920,000" in md
    assert "Rs. 8,720,000 – Rs. 24,120,000" in md
    assert "| **Transmission** | Automatic | 🟢 Positive" in md
    assert "| `12293566` |" in md
    assert "[!IMPORTANT]" in md
    assert "Asking Price Notice" in md


def test_formatter_handles_empty_or_minimal_data():
    empty_result = {"estimated_asking_price_lkr": 5_000_000.0}
    text = ValuationReportFormatter.format_text(empty_result)
    assert "Estimated Market Asking Price : Rs. 5,000,000" in text
    assert "(No feature attribution available)" in text

    md = ValuationReportFormatter.format_markdown(empty_result)
    assert "Rs. 5,000,000" in md

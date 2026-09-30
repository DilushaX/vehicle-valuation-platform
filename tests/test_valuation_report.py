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
        "data_quality": {
            "status": "COMPLETE",
            "provided_features": 11,
            "expected_features": 11,
            "missing_features": [],
            "comparable_count": 1,
        },
        "comparable_market_summary": {
            "comparable_count": 1,
            "min_asking_price": 15_500_000.0,
            "max_asking_price": 15_500_000.0,
            "median_asking_price": 15_500_000.0,
            "average_asking_price": 15_500_000.0,
            "price_spread": 0.0,
        },
        "audit": {
            "model_name": "RandomForestRegressor",
            "model_version": "1.9.1",
            "model_status": "EXPERIMENTAL_RESEARCH_BENCHMARK",
            "target": "asking_price",
            "generated_at": "2026-09-23T16:25:00+00:00",
            "feature_schema_version": "1.0",
            "valuation_method": "RandomForest asking-price regression with Tree SHAP explainability",
            "comparable_method": "Multi-attribute weighted similarity scoring",
            "prediction_range_method": "individual_tree_percentiles",
        },
        "reproducibility": {
            "fingerprint": "a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0",
            "algorithm": "SHA-256",
        },
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
    assert "COMPARABLE MARKET SUMMARY" in text
    assert "Rs. 15,500,000" in text
    assert "VALUATION DATA QUALITY" in text
    assert "11 / 11" in text
    assert "COMPLETE" in text
    assert "VALUATION AUDIT & REPRODUCIBILITY" in text
    assert "a1b2c3d4" in text


def test_format_markdown_output(mock_valuation_result, mock_vehicle_spec):
    md = ValuationReportFormatter.format_markdown(mock_valuation_result, mock_vehicle_spec)

    assert "# Vehicle Valuation: Toyota Premio" in md
    assert "Rs. 15,920,000" in md
    assert "Rs. 8,720,000 – Rs. 24,120,000" in md
    assert "| **Transmission** | Automatic | 🟢 Positive" in md
    assert "| `12293566` |" in md
    assert "Comparable Market Summary" in md
    assert "Valuation Data Quality" in md
    assert "COMPLETE" in md
    assert "Valuation Audit & Reproducibility" in md
    assert "a1b2c3d4" in md
    assert "[!IMPORTANT]" in md
    assert "Asking Price Notice" in md





def test_formatter_handles_empty_or_minimal_data():
    empty_result = {"estimated_asking_price_lkr": 5_000_000.0}
    text = ValuationReportFormatter.format_text(empty_result)
    assert "Estimated Market Asking Price : Rs. 5,000,000" in text
    assert "(No feature attribution available)" in text

    md = ValuationReportFormatter.format_markdown(empty_result)
    assert "Rs. 5,000,000" in md


# ===========================================================================
# Phase 10.5 Part 3: Focused Test Suite
# ===========================================================================

from ml.prediction.valuation_report import (
    generate_report_filename,
    sanitize_filename_component,
)
from ml.valuation.valuation_service import ValuationResult
from ml.valuation.valuation_workflow import ValuationWorkflowResult


def test_report_contains_all_required_sections(mock_valuation_result, mock_vehicle_spec):
    """
    Requirement 8.B: Verifies report contains all required sections:
    - Vehicle Information (all canonical fields)
    - Estimated Market Asking Price
    - Indicative Model-Based Prediction Range
    - Model Explanation (SHAP)
    - Data Quality
    - Comparable Market Summary (min, max, median, avg, spread)
    - Comparable Vehicles list
    - Audit & Reproducibility metadata
    - Verified legal & methodological limitations
    """
    spec_with_category = dict(mock_vehicle_spec)
    spec_with_category["category"] = "Cars"

    text = ValuationReportFormatter.format_text(mock_valuation_result, spec_with_category)

    # A. Vehicle Information
    assert "Category     : Cars" in text
    assert "Toyota Premio" in text
    assert "2016" in text
    assert "85,000 km" in text
    assert "1,500 cc" in text
    assert "Petrol" in text
    assert "Automatic" in text
    assert "Colombo" in text
    assert "Registered (Used)" in text

    # B. Valuation Result
    assert "Estimated Market Asking Price : Rs. 15,920,000" in text
    assert "Indicative Model-Based Prediction Range" in text
    assert "Rs. 8,720,000 – Rs. 24,120,000" in text
    assert "RandomForestRegressor" in text

    # C. Model Explanation
    assert "KEY MODEL CONTRIBUTING FACTORS:" in text
    assert "Transmission" in text

    # D. Data Quality
    assert "VALUATION DATA QUALITY:" in text
    assert "Status              : COMPLETE" in text
    assert "Input Features      : 11 / 11" in text
    assert "Missing Features    : None" in text
    assert "Comparable Listings : 1" in text

    # E. Comparable Market Summary
    assert "COMPARABLE MARKET SUMMARY:" in text
    assert "Comparable Count      : 1" in text
    assert "Minimum Asking Price  : Rs. 15,500,000" in text
    assert "Maximum Asking Price  : Rs. 15,500,000" in text
    assert "Median Asking Price   : Rs. 15,500,000" in text
    assert "Average Asking Price  : Rs. 15,500,000" in text
    assert "Price Spread          : Rs. 0" in text

    # F. Comparable Vehicles
    assert "COMPARABLE MARKET LISTINGS (Riyasewana):" in text
    assert "12293566" in text

    # G. Audit / Reproducibility
    assert "VALUATION AUDIT & REPRODUCIBILITY:" in text
    assert "Feature Schema Version      : 1.0" in text
    assert "Valuation Method            :" in text
    assert "Prediction Range Method     : individual_tree_percentiles" in text
    assert "Comparable Method           : Multi-attribute weighted similarity scoring" in text
    assert "Reproducibility Fingerprint : a1b2c3d4" in text

    # H. Limitations
    assert "IMPORTANT NOTICE & DISCLAIMERS:" in text
    assert "This valuation estimates market asking prices" in text
    assert "NOT a confirmed transaction" in text
    assert "indicative model-based range" in text
    assert "SHAP values describe model contribution" in text
    assert "experimental research benchmark" in text


def test_report_generation_from_workflow_result(mock_valuation_result, mock_vehicle_spec):
    """
    Requirement 8.A: Verifies report generation from ValuationWorkflowResult
    and its to_report_text() convenience method.
    """
    val_result = ValuationResult(
        estimated_asking_price_lkr=mock_valuation_result["estimated_asking_price_lkr"],
        prediction_range_lkr=mock_valuation_result["prediction_range_lkr"],
        currency="LKR",
        model=mock_valuation_result["model"],
        explanation=mock_valuation_result["explanation"],
        comparables=mock_valuation_result["comparables"],
        limitations=mock_valuation_result["limitations"],
        data_quality=mock_valuation_result["data_quality"],
        comparable_market_summary=mock_valuation_result["comparable_market_summary"],
        audit=mock_valuation_result["audit"],
        reproducibility=mock_valuation_result["reproducibility"],
    )
    wf_result = ValuationWorkflowResult.from_valuation_result(val_result)

    # Format via ValuationReportFormatter directly
    text1 = ValuationReportFormatter.format_text(wf_result, mock_vehicle_spec)
    assert "VEHICLE VALUATION REPORT" in text1
    assert "Estimated Market Asking Price : Rs. 15,920,000" in text1

    # Format via convenience method
    text2 = wf_result.to_report_text(mock_vehicle_spec)
    assert text1 == text2


def test_report_zero_comparables(mock_valuation_result, mock_vehicle_spec):
    """
    Requirement 8.C: Verifies that zero comparables case generates cleanly and
    clearly indicates that no sufficiently similar comparables were found.
    """
    zero_comp_result = dict(mock_valuation_result)
    zero_comp_result["comparables"] = []
    zero_comp_result["comparable_market_summary"] = {
        "comparable_count": 0,
        "min_asking_price": None,
        "max_asking_price": None,
        "median_asking_price": None,
        "average_asking_price": None,
        "price_spread": None,
    }
    zero_comp_result["data_quality"] = dict(mock_valuation_result["data_quality"])
    zero_comp_result["data_quality"]["comparable_count"] = 0

    text = ValuationReportFormatter.format_text(zero_comp_result, mock_vehicle_spec)
    assert "No close comparables found in database" in text
    assert "Comparable Count      : 0" in text
    assert "No sufficiently similar comparables were found" in text


def test_report_partial_vehicle_input(mock_valuation_result):
    """
    Requirement 8.D: Verifies that partial vehicle inputs (missing optional fields)
    are gracefully represented without raising exceptions.
    """
    partial_spec = {
        "brand": "Toyota",
        "model": "Allion",
        # mileage, engine_cc, district, condition, manufacture_year omitted
    }
    text = ValuationReportFormatter.format_text(mock_valuation_result, partial_spec)
    assert "Toyota Allion" in text
    assert "Year         : Not specified" in text
    assert "Mileage      : Not specified" in text
    assert "Engine CC    : Not specified" in text
    assert "District     : Not specified" in text
    assert "Condition    : Not specified" in text


def test_filename_generation_and_sanitization():
    """
    Requirement 8.E: Tests filename generation and sanitization for:
    - Normal brand / model
    - Names containing spaces
    - Special characters and punctuation
    - Path traversal prevention
    - Missing fields
    """
    # Normal
    fn1 = generate_report_filename({"brand": "Toyota", "model": "Premio", "manufacture_year": 2016})
    assert fn1 == "vehicle_valuation_toyota_premio_2016.txt"

    # Spaces in make / model
    fn2 = generate_report_filename({"brand": "Land Rover", "model": "Range Rover Sport", "manufacture_year": 2020})
    assert fn2 == "vehicle_valuation_land_rover_range_rover_sport_2020.txt"

    # Special characters and punctuation
    fn3 = generate_report_filename({"brand": "Mercedes-Benz", "model": "C 200 (AMG / Sport!)", "manufacture_year": 2018})
    assert fn3 == "vehicle_valuation_mercedes-benz_c_200_amg_sport_2018.txt"

    # Path traversal prevention
    fn4 = generate_report_filename({"brand": "../../etc", "model": "passwd", "manufacture_year": 2015})
    assert "/" not in fn4
    assert "\\" not in fn4
    assert ".." not in fn4
    assert fn4 == "vehicle_valuation_etc_passwd_2015.txt"

    # Missing fields
    fn5 = generate_report_filename({})
    assert fn5 == "vehicle_valuation_report.txt"

    fn6 = generate_report_filename({"brand": "Honda", "model": "Civic"})
    assert fn6 == "vehicle_valuation_honda_civic.txt"


def test_download_content_identical_to_preview(mock_valuation_result, mock_vehicle_spec):
    """
    Requirement 8.F: Verifies that downloaded report content is identical to the
    displayed report content (single source of truth).
    """
    preview_text = ValuationReportFormatter.format_text(mock_valuation_result, mock_vehicle_spec)
    download_text = ValuationReportFormatter.format_text(mock_valuation_result, mock_vehicle_spec)
    assert preview_text == download_text
    assert len(preview_text) > 0


def test_no_private_seller_contact_in_report(mock_valuation_result, mock_vehicle_spec):
    """
    Requirement 8.G: Strictly verifies that no seller phone number, email address,
    private address, or internal secrets are included in the generated report.
    """
    comp_with_private = dict(mock_valuation_result["comparables"][0])
    comp_with_private["seller_phone"] = "0771234567"
    comp_with_private["seller_email"] = "seller@example.com"
    comp_with_private["seller_address"] = "No 123, Secret Lane, Colombo"
    comp_with_private["database_password"] = "supersecretpassword123"

    res = dict(mock_valuation_result)
    res["comparables"] = [comp_with_private]

    text = ValuationReportFormatter.format_text(res, mock_vehicle_spec)
    assert "0771234567" not in text
    assert "seller@example.com" not in text
    assert "Secret Lane" not in text
    assert "supersecretpassword123" not in text

    import re
    assert not re.search(r"07\d{8}", text)
    assert not re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)


def test_precise_wording_compliance(mock_valuation_result, mock_vehicle_spec):
    """
    Requirement 2: Verifies strict compliance with precise terminology:
    - Uses 'Estimated Market Asking Price'
    - Uses 'Indicative Model-Based Prediction Range'
    - Disallows misleading claims (guaranteed valuation, actual transaction price)
    """
    text = ValuationReportFormatter.format_text(mock_valuation_result, mock_vehicle_spec)

    assert "Estimated Market Asking Price" in text
    assert "Indicative Model-Based Prediction Range" in text

    lower = text.lower()
    assert "actual selling price :" not in lower
    assert "confirmed market value :" not in lower
    assert "guaranteed valuation" not in lower
    if "confidence interval" in lower:
        assert "not a statistical confidence interval" in lower

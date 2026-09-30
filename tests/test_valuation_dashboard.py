"""
Tests for Vehicle Valuation Dashboard — Final Valuation Report & Export (Phase 10.5 Part 3).

Validates:
1. Final Valuation Report expander is rendered with correct title ("📄 Final Valuation Report").
2. Download button is rendered with expected label, filename, MIME type, and data.
3. Preview text and downloaded text are 100% identical (single source of truth).
4. Zero-comparable results are handled gracefully without errors.
5. Partial vehicle inputs are handled gracefully without errors.
6. No private seller contact data (phone/email/address/secrets) is exposed in preview or download.
7. Report is not generated or displayed on valuation error or unsubmitted state.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch
import pytest

from dashboard.pages.valuation_page import (
    render_final_report_section,
    render_valuation_page,
)
from ml.prediction.valuation_report import (
    ValuationReportFormatter,
    generate_report_filename,
)


@pytest.fixture
def sample_valuation_result() -> Dict[str, Any]:
    return {
        "estimated_asking_price_lkr": 14_500_000.0,
        "prediction_range_lkr": {
            "lower": 12_000_000.0,
            "upper": 17_000_000.0,
            "spread": 5_000_000.0,
            "percentile_lower": 10,
            "percentile_upper": 90,
            "method": "individual_tree_percentiles",
        },
        "currency": "LKR",
        "model": {
            "name": "RandomForestRegressor",
            "target_variable": "asking_price",
            "status": "EXPERIMENTAL_RESEARCH_BENCHMARK",
        },
        "explanation": [
            {
                "feature": "manufacture_year",
                "value": "2016",
                "contribution": 1_200_000.0,
                "direction": "positive",
                "description": "Manufacture year contributed positively.",
            },
            {
                "feature": "mileage",
                "value": "85000",
                "contribution": -450_000.0,
                "direction": "negative",
                "description": "Mileage contributed negatively.",
            },
        ],
        "comparables": [
            {
                "listing_id": "990011",
                "brand": "Toyota",
                "model": "Premio",
                "manufacture_year": 2016,
                "mileage": 82000.0,
                "district": "Colombo",
                "asking_price": 14_800_000.0,
                "similarity_percentage": 95.5,
            }
        ],
        "comparable_market_summary": {
            "comparable_count": 1,
            "min_asking_price": 14_800_000.0,
            "max_asking_price": 14_800_000.0,
            "median_asking_price": 14_800_000.0,
            "average_asking_price": 14_800_000.0,
            "price_spread": 0.0,
        },
        "data_quality": {
            "status": "COMPLETE",
            "provided_features": 11,
            "expected_features": 11,
            "missing_features": [],
            "comparable_count": 1,
        },
        "audit": {
            "model_name": "RandomForestRegressor",
            "model_version": "1.0",
            "model_status": "EXPERIMENTAL_RESEARCH_BENCHMARK",
            "target": "asking_price",
            "generated_at": "2026-09-30T10:00:00+00:00",
            "feature_schema_version": "1.0",
            "valuation_method": "RandomForest asking-price regression with Tree SHAP explainability",
            "prediction_range_method": "individual_tree_percentiles",
            "comparable_method": "Multi-attribute weighted similarity scoring",
        },
        "reproducibility": {
            "fingerprint": "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff",
            "algorithm": "SHA-256",
        },
        "limitations": [
            "Estimates market asking prices observed on Riyasewana.",
            "Not a confirmed transaction or final selling price.",
        ],
    }


@pytest.fixture
def sample_vehicle_payload() -> Dict[str, Any]:
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
        "top_k_factors": 5,
        "top_k_comparables": 5,
        "percentile_lower": 10,
        "percentile_upper": 90,
    }


def test_render_final_report_section_components(sample_valuation_result, sample_vehicle_payload):
    """
    Verifies that render_final_report_section renders the expander with title
    '📄 Final Valuation Report', creates the download button with correct metadata,
    and displays the preview.
    """
    with patch("streamlit.expander") as mock_expander, \
         patch("streamlit.download_button") as mock_download, \
         patch("streamlit.code") as mock_code, \
         patch("streamlit.markdown") as mock_markdown:

        # Configure expander context manager
        expander_cm = MagicMock()
        mock_expander.return_value.__enter__ = MagicMock(return_value=expander_cm)
        mock_expander.return_value.__exit__ = MagicMock(return_value=None)

        render_final_report_section(sample_valuation_result, sample_vehicle_payload)

        # 1. Expander created with correct title
        mock_expander.assert_called_once_with("📄 Final Valuation Report", expanded=False)

        # 2. Download button created
        assert mock_download.called
        download_kwargs = mock_download.call_args[1]
        assert "Download Valuation Report" in download_kwargs["label"]
        assert download_kwargs["file_name"] == "vehicle_valuation_toyota_premio_2016.txt"
        assert download_kwargs["mime"] == "text/plain"
        assert download_kwargs["use_container_width"] is True
        assert download_kwargs["key"] == "download_final_valuation_report"

        # 3. Preview code block displayed
        assert mock_code.called
        code_args = mock_code.call_args[0]
        assert code_args[1] == "text" if len(code_args) > 1 else mock_code.call_args[1].get("language") == "text"


def test_preview_and_download_are_identical_single_source_of_truth(sample_valuation_result, sample_vehicle_payload):
    """
    Requirement 6: Displayed report preview and downloaded file content must be
    100% identical and derived from the exact same report generation function.
    """
    with patch("streamlit.expander") as mock_expander, \
         patch("streamlit.download_button") as mock_download, \
         patch("streamlit.code") as mock_code, \
         patch("streamlit.markdown"):

        mock_expander.return_value.__enter__ = MagicMock()
        mock_expander.return_value.__exit__ = MagicMock()

        render_final_report_section(sample_valuation_result, sample_vehicle_payload)

        download_data = mock_download.call_args[1]["data"]
        preview_data = mock_code.call_args[0][0]

        assert download_data == preview_data
        assert "VEHICLE VALUATION REPORT" in download_data
        assert "Toyota Premio" in download_data
        assert "Estimated Market Asking Price : Rs. 14,500,000" in download_data


def test_dashboard_report_zero_comparables_handling(sample_valuation_result, sample_vehicle_payload):
    """
    Requirement 7.B: When zero comparables are found, report renders safely
    and explicitly states that no comparables were found.
    """
    zero_comp_result = dict(sample_valuation_result)
    zero_comp_result["comparables"] = []
    zero_comp_result["comparable_market_summary"] = {
        "comparable_count": 0,
        "min_asking_price": None,
        "max_asking_price": None,
        "median_asking_price": None,
        "average_asking_price": None,
        "price_spread": None,
    }

    with patch("streamlit.expander") as mock_expander, \
         patch("streamlit.download_button") as mock_download, \
         patch("streamlit.code") as mock_code, \
         patch("streamlit.markdown"):

        mock_expander.return_value.__enter__ = MagicMock()
        mock_expander.return_value.__exit__ = MagicMock()

        # Must not raise exceptions
        render_final_report_section(zero_comp_result, sample_vehicle_payload)

        report_text = mock_download.call_args[1]["data"]
        assert "No close comparables found in database" in report_text
        assert "Comparable Count      : 0" in report_text
        assert "No sufficiently similar comparables were found" in report_text


def test_dashboard_report_partial_input_handling(sample_valuation_result):
    """
    Requirement 7.C: Handles partial inputs where optional vehicle specifications
    (mileage, engine_cc, district, condition) were omitted by user.
    """
    partial_payload = {
        "category": "Cars",
        "brand": "Toyota",
        "model": "Aqua",
        # Optional fields omitted
    }

    with patch("streamlit.expander") as mock_expander, \
         patch("streamlit.download_button") as mock_download, \
         patch("streamlit.code") as mock_code, \
         patch("streamlit.markdown"):

        mock_expander.return_value.__enter__ = MagicMock()
        mock_expander.return_value.__exit__ = MagicMock()

        render_final_report_section(sample_valuation_result, partial_payload)

        report_text = mock_download.call_args[1]["data"]
        assert "Toyota Aqua" in report_text
        assert "Mileage      : Not specified" in report_text
        assert "Engine CC    : Not specified" in report_text
        assert "District     : Not specified" in report_text


def test_dashboard_report_no_private_seller_contact(sample_valuation_result, sample_vehicle_payload):
    """
    Requirement 8.G: Strict verification that no private seller contact details
    (phone, email, secret keys) leak into dashboard preview or download.
    """
    res = dict(sample_valuation_result)
    res["comparables"] = [
        {
            "listing_id": "PRIV01",
            "brand": "Toyota",
            "model": "Premio",
            "manufacture_year": 2016,
            "district": "Colombo",
            "asking_price": 14_000_000.0,
            "similarity_percentage": 92.0,
            "seller_phone": "0779998877",
            "seller_email": "seller_private@test.lk",
            "db_secret": "my_db_secret_key",
        }
    ]

    with patch("streamlit.expander") as mock_expander, \
         patch("streamlit.download_button") as mock_download, \
         patch("streamlit.code") as mock_code, \
         patch("streamlit.markdown"):

        mock_expander.return_value.__enter__ = MagicMock()
        mock_expander.return_value.__exit__ = MagicMock()

        render_final_report_section(res, sample_vehicle_payload)

        report_text = mock_download.call_args[1]["data"]
        assert "0779998877" not in report_text
        assert "seller_private@test.lk" not in report_text
        assert "my_db_secret_key" not in report_text


def test_dashboard_does_not_render_report_when_no_valuation():
    """
    Requirement 7.A: When the user has not submitted the form or valuation failed,
    the final valuation report section must not be displayed.
    """
    with patch("streamlit.form") as mock_form, \
         patch("dashboard.pages.valuation_page.render_final_report_section") as mock_render_report, \
         patch("streamlit.form_submit_button", return_value=False), \
         patch("streamlit.selectbox", return_value="Cars"), \
         patch("streamlit.text_input", return_value=""), \
         patch("streamlit.number_input", return_value=2015), \
         patch("streamlit.slider", return_value=5), \
         patch("streamlit.markdown"), \
         patch("streamlit.divider"), \
         patch("streamlit.columns", side_effect=lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]), \
         patch("streamlit.expander"):

        mock_form.return_value.__enter__ = MagicMock()
        mock_form.return_value.__exit__ = MagicMock(return_value=False)

        # Simulate rendering valuation page without form submission
        render_valuation_page(api_url="http://localhost:8000")

        # Report must NOT have been rendered
        assert not mock_render_report.called

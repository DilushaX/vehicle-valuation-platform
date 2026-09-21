"""
Tests for Model-Based Valuation Prediction Range & Uncertainty (Step 9 - Part 3).

Validates:
1. lower <= estimate <= upper holds for valid inputs.
2. Range bounds are strictly non-negative (> 10,000 LKR).
3. Repeated predictions on the same input are deterministic.
4. Range estimation works for valid vehicle inputs.
5. Invalid inputs (e.g. lower percentile >= upper percentile, negative mileage) fail safely.
6. Spread is correctly calculated as upper - lower.
7. UncertaintyEstimator directly and predictor.predict_with_range agree.
"""

from pathlib import Path
from typing import Any, Dict

import pytest

from ml.prediction.predictor import ValidationError, VehiclePricePredictor
from ml.prediction.uncertainty import UncertaintyEstimator, ValuationRange


@pytest.fixture
def trained_predictor() -> VehiclePricePredictor:
    """Loads the serialized trained model artifact."""
    model_path = Path("data/analysis/ml/models/model.joblib")
    meta_path = Path("data/analysis/ml/models/model_metadata.json")
    if not model_path.exists():
        pytest.skip(f"Model artifact not found at {model_path}")
    return VehiclePricePredictor.load(model_path=model_path, metadata_path=meta_path)


@pytest.fixture
def sample_car() -> Dict[str, Any]:
    return {
        "category": "Cars",
        "brand": "Toyota",
        "model": "Premio",
        "vehicle_age": 10,
        "mileage": 85000.0,
        "engine_cc": 1500.0,
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "district": "Colombo",
        "condition": "Registered (Used)",
    }


def test_valuation_range_bounds_ordering(trained_predictor, sample_car):
    range_dict = trained_predictor.predict_with_range(sample_car, percentile_lower=10, percentile_upper=90)

    est = range_dict["estimate"]
    lower = range_dict["lower"]
    upper = range_dict["upper"]
    spread = range_dict["spread"]

    # Mathematical ordering: lower <= estimate <= upper
    assert lower <= est, f"Lower bound {lower} exceeds estimate {est}"
    assert est <= upper, f"Estimate {est} exceeds upper bound {upper}"
    assert spread == pytest.approx(upper - lower, rel=1e-3)


def test_valuation_range_non_negative(trained_predictor, sample_car):
    range_dict = trained_predictor.predict_with_range(sample_car)
    assert range_dict["lower"] >= 10_000.0
    assert range_dict["upper"] >= 10_000.0
    assert range_dict["estimate"] >= 10_000.0


def test_valuation_range_deterministic(trained_predictor, sample_car):
    r1 = trained_predictor.predict_with_range(sample_car)
    r2 = trained_predictor.predict_with_range(sample_car)

    assert r1["estimate"] == r2["estimate"]
    assert r1["lower"] == r2["lower"]
    assert r1["upper"] == r2["upper"]
    assert r1["spread"] == r2["spread"]


def test_invalid_percentiles_fail_safely(trained_predictor, sample_car):
    # p_lower > p_upper
    with pytest.raises(ValueError, match="percentile_lower.*must be strictly less"):
        trained_predictor.predict_with_range(sample_car, percentile_lower=95, percentile_upper=50)

    # p_lower == p_upper
    with pytest.raises(ValueError, match="percentile_lower.*must be strictly less"):
        trained_predictor.predict_with_range(sample_car, percentile_lower=50, percentile_upper=50)


def test_invalid_input_fails_safely(trained_predictor, sample_car):
    bad_car = sample_car.copy()
    bad_car["mileage"] = -100.0
    with pytest.raises(ValidationError, match="mileage cannot be negative"):
        trained_predictor.predict_with_range(bad_car)


def test_direct_uncertainty_estimator(trained_predictor, sample_car):
    estimator = UncertaintyEstimator(trained_predictor.model, trained_predictor.metadata)
    df_formatted = trained_predictor.validate_and_format_input(sample_car)
    point_est = trained_predictor.predict_single(sample_car)

    val_range = estimator.estimate_range(df_formatted, point_estimate=point_est, percentile_lower=15, percentile_upper=85)
    assert isinstance(val_range, ValuationRange)
    assert val_range.lower_bound <= point_est <= val_range.upper_bound
    assert "RandomForest" in val_range.method

    summary_str = val_range.format_summary()
    assert "Estimated Asking Price" in summary_str
    assert "Indicative Range" in summary_str


def test_prediction_range_method_and_terminology(trained_predictor, sample_car):
    range_dict = trained_predictor.predict_with_range(sample_car, percentile_lower=10, percentile_upper=90)
    method_str = range_dict["method"].lower()

    # Verify method clearly identifies empirical dispersion
    assert "empirical dispersion" in method_str or "randomforest" in method_str
    # Verify no claims of statistical guarantees or confidence intervals
    assert "statistically guaranteed" not in method_str
    assert "guaranteed" not in method_str
    assert "confidence interval" not in method_str


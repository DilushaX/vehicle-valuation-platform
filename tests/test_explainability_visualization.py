"""
Tests for Model Explainability Visualization Subsystem (Step 9.11).

Validates:
1. Valid explanation generates a figure.
2. Output file exists and is a valid image.
3. Output directory is created automatically if non-existent.
4. top_k parameter limits the visualized features appropriately.
5. Positive and negative contributions are handled properly.
6. Empty explanation is handled safely without crashing.
7. Invalid explanation data is handled safely with descriptive errors.
8. Existing valuation service integration functions seamlessly.
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest

from ml.explainability.visualization import (
    FeatureContributionVisualizer,
    create_feature_contribution_plot,
)
from ml.valuation.valuation_service import ValuationResult, VehicleValuationService


@pytest.fixture
def sample_explanation() -> List[Dict[str, Any]]:
    """Controlled sample feature contribution records."""
    return [
        {
            "feature": "transmission",
            "value": "Automatic",
            "contribution": 0.6673,
            "direction": "positive",
            "description": "Transmission (Automatic) contributed positively to the model prediction.",
        },
        {
            "feature": "brand",
            "value": "Toyota",
            "contribution": 0.4541,
            "direction": "positive",
            "description": "Brand (Toyota) contributed positively to the model prediction.",
        },
        {
            "feature": "engine_cc",
            "value": "1,500 cc",
            "contribution": 0.1681,
            "direction": "positive",
            "description": "Engine Cc (1,500 cc) contributed positively to the model prediction.",
        },
        {
            "feature": "vehicle_age",
            "value": "10 years",
            "contribution": 0.1654,
            "direction": "positive",
            "description": "Vehicle Age (10 years) contributed positively to the model prediction.",
        },
        {
            "feature": "fuel_type",
            "value": "Petrol",
            "contribution": -0.0136,
            "direction": "negative",
            "description": "Fuel Type (Petrol) contributed negatively to the model prediction.",
        },
        {
            "feature": "model",
            "value": "Premio",
            "contribution": -0.0015,
            "direction": "negative",
            "description": "Model (Premio) contributed negatively to the model prediction.",
        },
    ]


def test_valid_explanation_generates_figure(tmp_path, sample_explanation):
    """1. Valid explanation generates a figure and returns Path."""
    out_file = tmp_path / "test_contributions.png"
    result_path = create_feature_contribution_plot(
        explanation=sample_explanation,
        output_path=out_file,
    )

    assert isinstance(result_path, Path)
    assert result_path == out_file


def test_output_file_exists_and_is_valid_image(tmp_path, sample_explanation):
    """2. Output file exists, is non-empty, and has PNG header."""
    out_file = tmp_path / "valid_image.png"
    create_feature_contribution_plot(sample_explanation, out_file)

    assert out_file.exists()
    assert out_file.stat().st_size > 1000  # Valid PNG has substantial size

    # Verify PNG magic bytes
    with open(out_file, "rb") as f:
        magic_bytes = f.read(8)
    assert magic_bytes == b"\x89PNG\r\n\x1a\n"


def test_output_directory_created_automatically(tmp_path, sample_explanation):
    """3. Output directory is created automatically if non-existent."""
    nested_dir = tmp_path / "deeply" / "nested" / "figures"
    assert not nested_dir.exists()

    out_file = nested_dir / "valuation_feature_contributions.png"
    create_feature_contribution_plot(sample_explanation, out_file)

    assert nested_dir.exists()
    assert out_file.exists()


def test_top_k_parameter_works(tmp_path, sample_explanation):
    """4. top_k restricts the displayed features to top k by magnitude."""
    out_file = tmp_path / "top_3.png"
    create_feature_contribution_plot(sample_explanation, out_file, top_k=3)
    assert out_file.exists()

    # Visualizer class method with top_k=2
    out_file_2 = tmp_path / "top_2.png"
    FeatureContributionVisualizer.create_plot(sample_explanation, out_file_2, top_k=2)
    assert out_file_2.exists()


def test_positive_and_negative_contributions_handled(tmp_path):
    """5. Both positive and negative contributions are handled cleanly."""
    mixed_explanation = [
        {"feature": "brand", "value": "Mercedes-Benz", "contribution": 1.25, "direction": "positive"},
        {"feature": "mileage", "value": "250,000 km", "contribution": -0.85, "direction": "negative"},
        {"feature": "vehicle_age", "value": "18 years", "contribution": -0.65, "direction": "negative"},
        {"feature": "condition", "value": "Unregistered", "contribution": 0.40, "direction": "positive"},
    ]
    out_file = tmp_path / "mixed_directions.png"
    result = create_feature_contribution_plot(mixed_explanation, out_file)
    assert result.exists()
    assert result.stat().st_size > 0


def test_empty_explanation_handled_safely(tmp_path):
    """6. Empty explanation raises ValueError or generates placeholder figure."""
    out_file = tmp_path / "empty.png"
    # By default, raise_on_empty=True raises clear ValueError
    with pytest.raises(ValueError, match="empty"):
        create_feature_contribution_plot([], out_file, raise_on_empty=True)

    # When raise_on_empty=False, handles safely by creating placeholder
    placeholder_file = tmp_path / "placeholder.png"
    result = create_feature_contribution_plot([], placeholder_file, raise_on_empty=False)
    assert result.exists()
    assert result.stat().st_size > 0


def test_invalid_explanation_data_handled_safely(tmp_path):
    """7. Invalid explanation inputs are handled safely with ValueError."""
    out_file = tmp_path / "invalid.png"

    # None input
    with pytest.raises(ValueError, match="None"):
        create_feature_contribution_plot(None, out_file)

    # Non-list/dict input
    with pytest.raises(ValueError, match="Invalid explanation data type"):
        create_feature_contribution_plot("not a valid list", out_file)

    # Missing required keys
    with pytest.raises(ValueError, match="missing required 'feature' key"):
        create_feature_contribution_plot([{"contribution": 0.5}], out_file)

    with pytest.raises(ValueError, match="missing required 'contribution' key"):
        create_feature_contribution_plot([{"feature": "mileage"}], out_file)

    # Non-numeric contribution
    with pytest.raises(ValueError, match="must be numeric float"):
        create_feature_contribution_plot([{"feature": "mileage", "contribution": "high"}], out_file)


def test_valuation_service_integration(tmp_path, sample_explanation):
    """8. Existing valuation functionality still works and integrates cleanly."""
    model_path = Path("data/analysis/ml/models/model.joblib")
    meta_path = Path("data/analysis/ml/models/model_metadata.json")
    if not model_path.exists():
        pytest.skip(f"Model artifact not found at {model_path}")

    service = VehicleValuationService(model_path=model_path, metadata_path=meta_path)
    car_spec = {
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

    result = service.valuate(car_spec)
    assert isinstance(result, ValuationResult)
    assert len(result.explanation) > 0

    # Test plotting directly from ValuationResult.explanation
    fig1 = tmp_path / "from_result_explanation.png"
    p1 = create_feature_contribution_plot(result.explanation, fig1)
    assert p1.exists()

    # Test plotting passing ValuationResult object directly
    fig2 = tmp_path / "from_result_obj.png"
    p2 = create_feature_contribution_plot(result, fig2)
    assert p2.exists()

    # Test helper method on ValuationResult
    fig3 = tmp_path / "from_helper_method.png"
    p3 = result.plot_explanation(fig3)
    assert p3.exists()

    # Test helper method on VehicleValuationService
    fig4 = tmp_path / "from_service_helper.png"
    p4 = service.plot_explanation(result, fig4)
    assert p4.exists()

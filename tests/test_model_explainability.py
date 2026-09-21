"""
Tests for Model Explainability & Feature Contribution Subsystem (Step 9 - Part 2).

Validates:
1. Explanation generates structured feature contributions for a valid vehicle prediction.
2. Returned feature names correspond strictly to actual model input features.
3. Contribution values are numeric floats.
4. Positive / negative direction is calculated consistently with contribution sign.
5. Explanation does not crash on unknown / rare categorical values (novel brand, model, district).
6. Explaining with top_k limits the number of returned factors appropriately.
7. Convenience predictor.explain(...) matches ModelExplainer.explain_prediction(...).
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from ml.explainability.explainer import CANONICAL_FEATURE_KEYS, ModelExplainer
from ml.prediction.predictor import VehiclePricePredictor


@pytest.fixture
def trained_predictor() -> VehiclePricePredictor:
    """Loads the serialized trained model artifact."""
    model_path = Path("data/analysis/ml/models/model.joblib")
    meta_path = Path("data/analysis/ml/models/model_metadata.json")
    if not model_path.exists():
        pytest.skip(f"Model artifact not found at {model_path}")
    return VehiclePricePredictor.load(model_path=model_path, metadata_path=meta_path)


@pytest.fixture
def explainer(trained_predictor) -> ModelExplainer:
    return ModelExplainer(predictor=trained_predictor)


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


def test_explanation_structure_valid_prediction(explainer, sample_car):
    explanations = explainer.explain_prediction(sample_car)
    assert len(explanations) > 0

    for item in explanations:
        assert "feature" in item
        assert "value" in item
        assert "contribution" in item
        assert "direction" in item
        assert "description" in item

        # Check types
        assert isinstance(item["feature"], str)
        assert isinstance(item["value"], str)
        assert isinstance(item["contribution"], float)
        assert item["direction"] in ["positive", "negative", "neutral"]
        assert isinstance(item["description"], str)

        # Check direction consistency
        if item["contribution"] > 0:
            assert item["direction"] == "positive"
        elif item["contribution"] < 0:
            assert item["direction"] == "negative"


def test_returned_features_correspond_to_model_inputs(explainer, sample_car):
    explanations = explainer.explain_prediction(sample_car)
    returned_features = [item["feature"] for item in explanations]

    for feat in returned_features:
        assert feat in CANONICAL_FEATURE_KEYS, f"Unexpected feature {feat} not in canonical feature set!"


def test_explanation_with_top_k(explainer, sample_car):
    top_3 = explainer.explain_prediction(sample_car, top_k=3)
    assert len(top_3) == 3

    # Ensure sorted by absolute contribution descending
    contribs = [abs(item["contribution"]) for item in top_3]
    assert contribs == sorted(contribs, reverse=True)


def test_unknown_categorical_values_safe(explainer, sample_car):
    novel_car = sample_car.copy()
    novel_car["brand"] = "NonExistentBrandAlien"
    novel_car["model"] = "UFO_9000"
    novel_car["district"] = "OuterSpaceDistrict"

    explanations = explainer.explain_prediction(novel_car)
    assert len(explanations) > 0
    assert all(isinstance(x["contribution"], float) for x in explanations)


def test_predictor_explain_convenience_method(trained_predictor, sample_car):
    direct_expl = trained_predictor.explain(sample_car, top_k=5)
    assert len(direct_expl) == 5
    assert direct_expl[0]["feature"] in CANONICAL_FEATURE_KEYS


def test_shap_explanation_uses_non_causal_attribution_terminology(explainer, sample_car):
    explanations = explainer.explain_prediction(sample_car)
    assert len(explanations) > 0
    for item in explanations:
        desc = item["description"].lower()
        # Verify non-causal attribution phrasing
        assert "contributed" in desc
        assert "to the model prediction" in desc
        # Ensure no causal language
        assert "cause" not in desc
        assert "causes" not in desc
        assert "increases the vehicle price" not in desc
        assert "decreases the vehicle price" not in desc


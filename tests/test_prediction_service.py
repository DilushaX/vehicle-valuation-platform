"""
Tests for Vehicle Valuation Prediction Service (Step 9 - Part 1).

Validates:
1. Valid prediction for standard input vehicle features (DataFrame and single dict).
2. Invalid inputs raise appropriate ValidationError.
3. Missing required features (e.g. category, vehicle_age) raise ValidationError.
4. Unknown categories (unseen brand, model, district) are handled safely without crashing.
5. Missing numerical values (mileage, engine_cc) are imputed consistently via preprocessor.
6. Negative numerical values (vehicle_age, mileage, engine_cc) are rejected.
7. Deriving vehicle_age from manufacture_year works accurately.
8. Model artifact loading raises FileNotFoundError if artifact path does not exist.
9. Predictions are returned in original Sri Lankan Rupees (LKR) scale (> 100,000 LKR).
10. TransformedTargetRegressor log1p inverse transformation yields expected LKR scale.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from ml.prediction.predictor import ValidationError, VehiclePricePredictor


@pytest.fixture
def trained_predictor() -> VehiclePricePredictor:
    """Loads the serialized trained model artifact."""
    model_path = Path("data/analysis/ml/models/model.joblib")
    meta_path = Path("data/analysis/ml/models/model_metadata.json")
    if not model_path.exists():
        pytest.skip(f"Model artifact not found at {model_path}")
    return VehiclePricePredictor.load(model_path=model_path, metadata_path=meta_path)


@pytest.fixture
def valid_car_dict() -> Dict[str, Any]:
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


def test_valid_prediction_dict(trained_predictor, valid_car_dict):
    price = trained_predictor.predict_single(valid_car_dict)
    assert isinstance(price, float)
    # Sri Lankan car prices are typically in the millions LKR
    assert price > 1_000_000.0
    assert price < 50_000_000.0


def test_valid_prediction_dataframe(trained_predictor, valid_car_dict):
    df = pd.DataFrame([valid_car_dict, valid_car_dict])
    preds = trained_predictor.predict(df)
    assert len(preds) == 2
    assert isinstance(preds, np.ndarray)
    assert np.all(preds > 1_000_000.0)
    # Identical inputs produce identical predictions
    assert np.isclose(preds[0], preds[1])


def test_manufacture_year_derivation(trained_predictor, valid_car_dict):
    car = valid_car_dict.copy()
    del car["vehicle_age"]
    car["manufacture_year"] = 2016  # 2026 - 2016 = 10 years

    price = trained_predictor.predict_single(car)
    assert price > 1_000_000.0


def test_missing_required_categorical(trained_predictor, valid_car_dict):
    car = valid_car_dict.copy()
    del car["category"]
    with pytest.raises(ValidationError, match="Missing required categorical features"):
        trained_predictor.predict(car)


def test_invalid_category(trained_predictor, valid_car_dict):
    car = valid_car_dict.copy()
    car["category"] = "Spaceships"
    with pytest.raises(ValidationError, match="Invalid vehicle category"):
        trained_predictor.predict(car)


def test_negative_numerical_values(trained_predictor, valid_car_dict):
    # Negative mileage
    car1 = valid_car_dict.copy()
    car1["mileage"] = -500.0
    with pytest.raises(ValidationError, match="mileage cannot be negative"):
        trained_predictor.predict(car1)

    # Negative vehicle age
    car2 = valid_car_dict.copy()
    car2["vehicle_age"] = -1
    with pytest.raises(ValidationError, match="vehicle_age cannot be negative"):
        trained_predictor.predict(car2)

    # Negative engine cc
    car3 = valid_car_dict.copy()
    car3["engine_cc"] = -1500.0
    with pytest.raises(ValidationError, match="engine_cc cannot be negative"):
        trained_predictor.predict(car3)


def test_unknown_category_handled_gracefully(trained_predictor, valid_car_dict):
    # Pass a completely novel brand and model
    car = valid_car_dict.copy()
    car["brand"] = "NonExistentBrandXYZ"
    car["model"] = "MysteryModel999"
    car["district"] = "UnheardOfDistrict"

    price = trained_predictor.predict_single(car)
    assert isinstance(price, float)
    assert price > 10_000.0  # Safe positive estimate without crashing


def test_missing_numerical_values_imputed(trained_predictor, valid_car_dict):
    # Mileage and engine_cc are None / NaN
    car = valid_car_dict.copy()
    car["mileage"] = None
    car["engine_cc"] = np.nan

    price = trained_predictor.predict_single(car)
    assert isinstance(price, float)
    assert price > 1_000_000.0


def test_model_artifact_not_found():
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        VehiclePricePredictor.load("non_existent_directory/model.joblib")


def test_prediction_returned_in_lkr_not_log(trained_predictor, valid_car_dict):
    preds = trained_predictor.predict(valid_car_dict)
    # Log values for asking price would be in range [11.0, 18.0]
    # LKR values are in range [100,000, 100,000,000]
    assert preds[0] > 100_000.0, f"Prediction {preds[0]} appears to be in log scale!"

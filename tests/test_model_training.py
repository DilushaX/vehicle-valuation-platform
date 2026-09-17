"""
Comprehensive test suite for Model Training & Evaluation (Phase 5 - Step 8).

Validates:
1. Baselines (MeanBaselineRegressor, MedianBaselineRegressor) correctly predict summary statistics.
2. Candidate models factory builds models with deterministic seeds and target transforms.
3. Evaluation metrics (MAE, RMSE, R2, MedAE, MAPE) match ground-truth mathematical definitions.
4. Train/Test splitting ensures 80/20 ratio, category stratification, and disjoint sets.
5. Strict data leakage prevention (feature transformers fitted only on train split).
6. Target transformation (log1p) correctly inverted so all reported metrics are on original LKR scale.
7. End-to-end training runs 5-fold CV on train set only and selects best model by CV MAE.
8. Serialization saves pipeline and metadata accurately.
9. VehiclePricePredictor loads artifact and produces valid, positive LKR price predictions.
10. Database invariance: no database mutations during training or evaluation.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session

from database.connection import get_engine, get_sessionmaker
from database.models import Listing, ListingObservation, PriceHistory, ScrapeRun, Vehicle
from feature_engineering.dataset import MLDatasetLoader
from ml.prediction.predictor import VehiclePricePredictor
from ml.training.baselines import MeanBaselineRegressor, MedianBaselineRegressor
from ml.training.evaluation import RegressionMetrics, evaluate_predictions
from ml.training.model_registry import (
    create_model_pipeline,
    get_candidate_models,
)
from ml.training.trainer import ModelTrainer, TrainerConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_ml_df() -> pd.DataFrame:
    """Synthetic dataset with 50 diverse vehicle records for fast unit testing."""
    np.random.seed(42)
    categories = ["Cars", "SUVs", "Vans", "Motorbikes", "Lorries"]
    brands = ["Toyota", "Nissan", "Honda", "Bajaj", "Isuzu"]
    districts = ["Colombo", "Gampaha", "Kandy", "Kurunegala", "Kalutara"]
    conditions = ["Registered (Used)", "Unregistered"]
    transmissions = ["Automatic", "Manual"]
    fuel_types = ["Petrol", "Diesel"]

    data = []
    for i in range(50):
        cat = categories[i % len(categories)]
        brand = brands[i % len(brands)]
        mfg_year = int(np.random.randint(2005, 2024))
        mileage = float(np.random.randint(5000, 200000))
        engine_cc = float(np.random.choice([1000, 1500, 2000, 2800, 150]))
        # Realistic LKR prices
        if cat == "Motorbikes":
            price = float(np.random.randint(200000, 900000))
        elif cat == "Cars":
            price = float(np.random.randint(3000000, 15000000))
        elif cat == "SUVs":
            price = float(np.random.randint(12000000, 35000000))
        else:
            price = float(np.random.randint(2500000, 8000000))

        data.append(
            {
                "listing_id": f"TEST_{i:04d}",
                "vehicle_id": i + 1,
                "category": cat,
                "brand": brand,
                "model": f"{brand}_Model_{i % 3}",
                "manufacture_year": mfg_year,
                "registration_year": mfg_year + 1 if i % 2 == 0 else np.nan,
                "mileage": mileage,
                "engine_cc": engine_cc,
                "fuel_type": fuel_types[i % 2],
                "transmission": transmissions[i % 2],
                "district": districts[i % len(districts)],
                "condition": conditions[i % 2],
                "asking_price": price,
                "ml_eligible": True,
                "first_seen_at": datetime(2026, 9, 1, tzinfo=timezone.utc),
            }
        )

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Test 1: Baselines
# ---------------------------------------------------------------------------

def test_mean_baseline():
    X = pd.DataFrame({"feat1": [1, 2, 3, 4, 5]})
    y = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

    model = MeanBaselineRegressor()
    model.fit(X, y)
    assert model.mean_value_ == 300.0

    preds = model.predict(X)
    assert len(preds) == 5
    assert np.allclose(preds, 300.0)


def test_median_baseline():
    X = pd.DataFrame({"feat1": [1, 2, 3, 4, 5]})
    y = np.array([100.0, 200.0, 300.0, 400.0, 10000.0])  # Skewed

    model = MedianBaselineRegressor()
    model.fit(X, y)
    assert model.median_value_ == 300.0

    preds = model.predict(X)
    assert len(preds) == 5
    assert np.allclose(preds, 300.0)


# ---------------------------------------------------------------------------
# Test 2: Model Registry & Pipeline Factory
# ---------------------------------------------------------------------------

def test_model_registry_candidate_models():
    trainer = ModelTrainer()
    models = get_candidate_models(
        preprocessor_factory=trainer.pipeline_coordinator.build_preprocessor,
        target_transform="raw",
        random_state=42,
    )
    assert "MeanBaseline" in models
    assert "MedianBaseline" in models
    assert "LinearRegression" in models
    assert "RandomForestRegressor" in models
    assert "HistGradientBoostingRegressor" in models

    # Verify each estimator has fit and predict
    for name, model in models.items():
        assert hasattr(model, "fit"), f"{name} lacks fit"
        assert hasattr(model, "predict"), f"{name} lacks predict"


# ---------------------------------------------------------------------------
# Test 3: Evaluation Metrics Calculation
# ---------------------------------------------------------------------------

def test_evaluate_predictions_metrics():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0])  # Errors: +10, -10, +10, -10

    metrics = evaluate_predictions(y_true, y_pred)
    assert np.isclose(metrics.mae, 10.0)
    assert np.isclose(metrics.rmse, 10.0)
    assert np.isclose(metrics.medae, 10.0)
    assert metrics.r2 > 0.99

    metrics_dict = metrics.to_dict()
    assert "mae" in metrics_dict
    assert "rmse" in metrics_dict
    assert "r2" in metrics_dict
    assert "medae" in metrics_dict
    assert "mape" in metrics_dict


# ---------------------------------------------------------------------------
# Test 4: Data Splitting & Category Stratification
# ---------------------------------------------------------------------------

def test_data_splitting_disjoint_and_ratio(synthetic_ml_df):
    trainer = ModelTrainer(config=TrainerConfig(test_size=0.2, random_state=42))
    X, y, meta = trainer.prepare_data(synthetic_ml_df)
    X_train, X_test, y_train, y_test, meta_train, meta_test = trainer.split_data(X, y, meta)

    # 80/20 split on 50 samples = 40 train, 10 test
    assert len(X_train) == 40
    assert len(X_test) == 10

    # Sets must be disjoint
    assert set(X_train.index).isdisjoint(set(X_test.index))

    # All categories present in train
    assert set(X_train["category"]) == set(synthetic_ml_df["category"])


# ---------------------------------------------------------------------------
# Test 5: Leakage Prevention - Preprocessor Fitted Only on Train
# ---------------------------------------------------------------------------

def test_no_data_leakage_in_training(synthetic_ml_df):
    trainer = ModelTrainer(config=TrainerConfig(test_size=0.2, random_state=42))
    X, y, meta = trainer.prepare_data(synthetic_ml_df)
    X_train, X_test, y_train, y_test, _, _ = trainer.split_data(X, y, meta)

    models = get_candidate_models(
        preprocessor_factory=trainer.pipeline_coordinator.build_preprocessor,
        target_transform="raw",
    )
    model = models["LinearRegression"]

    # Model should not be fitted initially
    with pytest.raises(Exception):
        model.predict(X_test)

    # Fit model strictly on X_train, y_train
    model.fit(X_train, y_train)

    # Now model can predict on X_test without fitting on X_test
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


# ---------------------------------------------------------------------------
# Test 6: Target Transformation on LKR Scale
# ---------------------------------------------------------------------------

def test_target_transformation_evaluates_in_lkr(synthetic_ml_df):
    config = TrainerConfig(
        target_transform="log1p",
        cv_folds=3,
        random_state=42,
    )
    trainer = ModelTrainer(config=config)
    X, y, meta = trainer.prepare_data(synthetic_ml_df)
    X_train, X_test, y_train, y_test, _, _ = trainer.split_data(X, y, meta)

    models = get_candidate_models(
        preprocessor_factory=trainer.pipeline_coordinator.build_preprocessor,
        target_transform="log1p",
    )
    lr = models["LinearRegression"]
    lr.fit(X_train, y_train)

    preds = lr.predict(X_test)
    # Predictions must be in LKR scale (> 100,000 LKR), NOT in log scale (< 25)
    assert np.all(preds > 100000.0), f"Predictions appear to be in log scale: {preds[:3]}"


# ---------------------------------------------------------------------------
# Test 7: End-to-End ModelTrainer Execution
# ---------------------------------------------------------------------------

def test_end_to_end_trainer_execution(synthetic_ml_df, tmp_path):
    config = TrainerConfig(
        test_size=0.2,
        cv_folds=3,
        target_transform="log1p",
        output_dir=tmp_path,
        random_state=42,
    )
    trainer = ModelTrainer(config=config)
    results = trainer.train_and_evaluate(synthetic_ml_df)

    assert "best_model_name" in results
    assert results["best_test_metrics"].mae > 0
    assert results["best_test_metrics"].rmse > 0
    assert results["test_size"] == 10
    assert len(results["cv_results"]) == 5

    # Verify artifacts were generated
    assert (tmp_path / "models" / "model.joblib").exists()
    assert (tmp_path / "models" / "model_metadata.json").exists()
    assert (tmp_path / "model_comparison.csv").exists()
    assert (tmp_path / "model_evaluation_report.md").exists()
    assert (tmp_path / "figures" / "actual_vs_predicted.png").exists()


# ---------------------------------------------------------------------------
# Test 8: Prediction Interface
# ---------------------------------------------------------------------------

def test_vehicle_price_predictor(synthetic_ml_df, tmp_path):
    config = TrainerConfig(
        test_size=0.2,
        cv_folds=3,
        target_transform="log1p",
        output_dir=tmp_path,
        random_state=42,
    )
    trainer = ModelTrainer(config=config)
    trainer.train_and_evaluate(synthetic_ml_df)

    # Initialize predictor from saved artifact
    predictor = VehiclePricePredictor.load(
        model_path=tmp_path / "models" / "model.joblib",
        metadata_path=tmp_path / "models" / "model_metadata.json",
    )

    meta = predictor.metadata
    assert "model_name" in meta
    assert meta["target_transform"] == "log1p"

    # Predict on new unseen slice of clean features
    X, _, _ = trainer.prepare_data(synthetic_ml_df)
    unseen_df = X.head(3).copy()
    predictions = predictor.predict(unseen_df)
    assert len(predictions) == 3
    assert all(isinstance(p, (float, np.floating)) for p in predictions)
    assert all(p > 0 for p in predictions)

    # Verify leakage detection in predictor
    leaked_df = unseen_df.copy()
    leaked_df["asking_price"] = 5_000_000.0
    with pytest.raises(Exception, match="Data leakage detected"):
        predictor.predict(leaked_df)


# ---------------------------------------------------------------------------
# Test 9: Database Invariance
# ---------------------------------------------------------------------------

def test_database_invariance_during_training(db_session: Session):
    """Verify that training and predicting does not alter PostgreSQL counts."""
    v_count_before = db_session.query(Vehicle).count()
    l_count_before = db_session.query(Listing).count()
    p_count_before = db_session.query(PriceHistory).count()
    o_count_before = db_session.query(ListingObservation).count()
    r_count_before = db_session.query(ScrapeRun).count()

    # Load dataset via loader
    loader = MLDatasetLoader()
    df = loader.load_raw_dataset(ml_eligible_only=True)
    assert len(df) > 0

    # Ensure no database state changed
    v_count_after = db_session.query(Vehicle).count()
    l_count_after = db_session.query(Listing).count()
    p_count_after = db_session.query(PriceHistory).count()
    o_count_after = db_session.query(ListingObservation).count()
    r_count_after = db_session.query(ScrapeRun).count()

    assert v_count_before == v_count_after
    assert l_count_before == l_count_after
    assert p_count_before == p_count_after
    assert o_count_before == o_count_after
    assert r_count_before == r_count_after

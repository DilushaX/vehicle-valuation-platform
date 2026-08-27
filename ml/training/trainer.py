"""
ML Model Trainer Module
Trains, cross-validates, evaluates, and registers category-specific
valuation regression models (XGBoost, Random Forest, Ridge).
"""
import os
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
from sqlalchemy.orm import Session
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Gracefully import XGBoost if libomp is available on system
HAS_XGBOOST = False
try:
    from xgboost import XGBRegressor
    # Test initialize
    _test = XGBRegressor()
    HAS_XGBOOST = True
except Exception:
    XGBRegressor = None


from config import settings
from database.connection import SessionLocal
from database.repository import VehicleRepository
from database.models import ModelVersion, utc_now
from ml.preprocessing.pipeline import VehiclePreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, db: Optional[Session] = None):
        self.db = db or SessionLocal()
        self.repo = VehicleRepository(self.db)
        os.makedirs(settings.ML_MODELS_DIR, exist_ok=True)

    def _prepare_data(self, category: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Extracts valid listings from DB and returns (X_df, y_series)."""
        listings = self.repo.get_all_clean_training_data(category=category)
        if not listings:
            return pd.DataFrame(), pd.Series()

        rows = []
        for r in listings:
            rows.append({
                "category": r.category,
                "make": r.make,
                "model": r.model,
                "yom": r.yom,
                "mileage_km": r.mileage_km or 60000,
                "fuel_type": r.fuel_type or "Petrol",
                "transmission": r.transmission or "Automatic",
                "condition": r.condition or "Used",
                "engine_cc": r.engine_cc or 1500,
                "district": r.district or "Colombo",
                "price_rs": r.current_price
            })

        df = pd.DataFrame(rows)
        y = df["price_rs"]
        X = df.drop(columns=["price_rs"])
        return X, y

    def train_category_model(self, category: str = "Cars") -> Dict[str, Any]:
        """
        Trains and compares algorithms for a specific vehicle category.
        Saves the best performing model pipeline.
        """
        logger.info(f"Initiating model training for category '{category}'...")
        X, y = self._prepare_data(category=category)

        if len(X) < settings.MIN_TRAINING_SAMPLES_PER_CATEGORY:
            logger.warning(
                f"Insufficient training data for category '{category}' "
                f"({len(X)} samples found, minimum required is {settings.MIN_TRAINING_SAMPLES_PER_CATEGORY})."
            )
            return {
                "category": category,
                "status": "INSUFFICIENT_DATA",
                "sample_count": len(X),
                "message": "Insufficient market data to train a reliable model"
            }

        prep_pipeline, _ = VehiclePreprocessor.build_pipeline()
        X_transformed = prep_pipeline.fit_transform(X)

        algorithms = {
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=settings.RANDOM_STATE),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=settings.RANDOM_STATE)
        }
        if HAS_XGBOOST and XGBRegressor is not None:
            algorithms["XGBoost"] = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=settings.RANDOM_STATE)

        kfold = KFold(n_splits=min(5, max(2, len(X) // 10)), shuffle=True, random_state=settings.RANDOM_STATE)
        model_scores = {}

        for name, model in algorithms.items():
            maes, rmses, r2s, mapes = [], [], [], []
            for train_idx, val_idx in kfold.split(X_transformed):
                X_tr, X_va = X_transformed[train_idx], X_transformed[val_idx]
                y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_tr, y_tr)
                preds = model.predict(X_va)

                maes.append(mean_absolute_error(y_va, preds))
                rmses.append(root_mean_squared_error(y_va, preds))
                r2s.append(r2_score(y_va, preds))
                mapes.append(np.mean(np.abs((y_va - preds) / y_va)) * 100.0)

            model_scores[name] = {
                "MAE": float(np.mean(maes)),
                "RMSE": float(np.mean(rmses)),
                "R2": float(np.mean(r2s)),
                "MAPE": float(np.mean(mapes))
            }
            logger.info(f"Category '{category}' | Model {name}: R2={np.mean(r2s):.4f}, MAE=Rs.{np.mean(maes):,.0f}, MAPE={np.mean(mapes):.2f}%")

        # Select best model based on highest R2
        best_name = max(model_scores, key=lambda k: model_scores[k]["R2"])
        best_metrics = model_scores[best_name]

        # Fit final model on all data
        best_estimator = algorithms[best_name]
        best_estimator.fit(X_transformed, y)

        # Bundle preprocessor + estimator
        model_bundle = {
            "preprocessor": prep_pipeline,
            "estimator": best_estimator,
            "algorithm": best_name,
            "category": category,
            "metrics": best_metrics,
            "feature_columns": list(X.columns),
            "trained_at": datetime.now(timezone.utc).isoformat()
        }

        model_filename = f"valuation_{category.lower().replace('-', '_')}.joblib"
        artifact_path = os.path.join(settings.ML_MODELS_DIR, model_filename)
        joblib.dump(model_bundle, artifact_path)

        # Deactivate old versions and record new version in DB
        self.db.query(ModelVersion).filter(
            ModelVersion.category == category.capitalize()
        ).update({"is_active": False})

        model_version = ModelVersion(
            model_id=f"mv_{category.lower()}_{uuid.uuid4().hex[:6]}",
            category=category.capitalize(),
            algorithm=best_name,
            metrics_json=json.dumps(best_metrics),
            feature_names_json=json.dumps(list(X.columns)),
            artifact_path=str(artifact_path),
            is_active=True,
            trained_at=utc_now()
        )
        self.db.add(model_version)
        self.db.commit()

        logger.info(f"Saved active model for '{category}' -> {artifact_path}")

        return {
            "category": category,
            "status": "SUCCESS",
            "sample_count": len(X),
            "best_algorithm": best_name,
            "metrics": best_metrics,
            "comparison": model_scores,
            "artifact_path": artifact_path
        }

    def train_all_supported_categories(self) -> Dict[str, Any]:
        """Trains models for all supported vehicle categories in the platform."""
        results = {}
        for cat in settings.SUPPORTED_CATEGORIES:
            res = self.train_category_model(category=cat.capitalize())
            results[cat] = res
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train ML Valuation Models")
    parser.add_argument("--category", type=str, default=None, help="Category to train (e.g. Cars, Vans)")
    parser.add_argument("--train-all", action="store_true", help="Train models for all categories")
    args = parser.parse_args()

    trainer = ModelTrainer()
    if args.train_all:
        res = trainer.train_all_supported_categories()
        print("Training Results Summary:")
        print(json.dumps(res, indent=2))
    elif args.category:
        res = trainer.train_category_model(args.category)
        print(json.dumps(res, indent=2))
    else:
        res = trainer.train_category_model("Cars")
        print(json.dumps(res, indent=2))

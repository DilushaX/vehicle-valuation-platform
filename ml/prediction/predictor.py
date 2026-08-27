"""
Vehicle Valuation Predictor Module
Provides ML inference for Estimated Market Asking Value, Estimated Market Range,
Confidence scoring, and safety fallbacks for sparse vehicle specifications.
"""
import os
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from sqlalchemy.orm import Session
from sqlalchemy import func

from config import settings
from database.connection import SessionLocal
from database.models import PredictionLog, Listing, utc_now

logger = logging.getLogger(__name__)

class VehicleValuationPredictor:
    def __init__(self, db: Optional[Session] = None):
        self.db = db or SessionLocal()
        self._model_cache: Dict[str, Any] = {}

    def _load_model(self, category: str) -> Optional[Dict[str, Any]]:
        """Loads and caches trained model bundle for a category."""
        clean_cat = category.lower().replace("-", "_")
        if clean_cat in self._model_cache:
            return self._model_cache[clean_cat]

        filename = f"valuation_{clean_cat}.joblib"
        filepath = os.path.join(settings.ML_MODELS_DIR, filename)

        if not os.path.exists(filepath):
            logger.warning(f"No trained model artifact found at {filepath}")
            return None

        try:
            bundle = joblib.load(filepath)
            self._model_cache[clean_cat] = bundle
            return bundle
        except Exception as e:
            logger.error(f"Error loading model artifact {filepath}: {e}")
            return None

    def predict_value(
        self,
        category: str,
        make: str,
        model: str,
        yom: int,
        mileage_km: int,
        fuel_type: Optional[str] = "Petrol",
        transmission: Optional[str] = "Automatic",
        condition: Optional[str] = "Used",
        engine_cc: Optional[int] = 1500,
        district: Optional[str] = "Colombo",
        seller_asking_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculates Estimated Market Asking Value, Estimated Market Range, Confidence,
        and optional seller asking price assessment.
        """
        bundle = self._load_model(category)
        if not bundle:
            return {
                "status": "INSUFFICIENT_DATA",
                "category": category,
                "message": f"Insufficient market data to estimate value for category '{category}'."
            }

        preprocessor = bundle["preprocessor"]
        estimator = bundle["estimator"]
        metrics = bundle.get("metrics", {})
        rmse = metrics.get("RMSE", 300_000.0)
        mape = metrics.get("MAPE", 6.0)

        # Check sample density in DB for confidence rating
        similar_count = self.db.query(Listing).filter(
            func.lower(Listing.category) == category.lower(),
            func.lower(Listing.make) == make.lower(),
            func.lower(Listing.model) == model.lower(),
            Listing.data_quality_status == "VALID"
        ).count()

        input_df = pd.DataFrame([{
            "category": category.capitalize(),
            "make": make.title(),
            "model": model.title(),
            "yom": yom,
            "mileage_km": mileage_km,
            "fuel_type": (fuel_type or "Petrol").title(),
            "transmission": (transmission or "Automatic").title(),
            "condition": (condition or "Used").title(),
            "engine_cc": engine_cc or 1500,
            "district": (district or "Colombo").title()
        }])

        try:
            X_trans = preprocessor.transform(input_df)
            raw_prediction = float(estimator.predict(X_trans)[0])
            predicted_value = round(max(100_000.0, raw_prediction), -4)

            # Compute Confidence Range (e.g. ± 1.0 * RMSE or percentage spread)
            spread = max(predicted_value * (mape / 100.0), rmse * 0.8)
            range_low = round(max(100_000.0, predicted_value - spread), -4)
            range_high = round(predicted_value + spread, -4)

            # Confidence Level
            if similar_count >= 10 and metrics.get("R2", 0) >= 0.85:
                confidence = "High"
            elif similar_count >= 3 and metrics.get("R2", 0) >= 0.65:
                confidence = "Medium"
            else:
                confidence = "Low"

            # Asking Price Assessment if provided
            assessment = None
            delta_pct = None
            if seller_asking_price and seller_asking_price > 0:
                delta = seller_asking_price - predicted_value
                delta_pct = round((delta / predicted_value) * 100.0, 1)

                if seller_asking_price < range_low:
                    assessment = "BELOW MARKET RANGE"
                elif seller_asking_price > range_high:
                    assessment = "ABOVE MARKET RANGE"
                else:
                    assessment = "WITHIN MARKET RANGE"

            # Log prediction to DB
            pred_log = PredictionLog(
                category=category.capitalize(),
                make=make.title(),
                model=model.title(),
                yom=yom,
                mileage_km=mileage_km,
                fuel_type=fuel_type,
                transmission=transmission,
                district=district,
                asking_price=seller_asking_price,
                predicted_value=predicted_value,
                range_low=range_low,
                range_high=range_high,
                confidence=confidence,
                assessment=assessment,
                created_at=utc_now()
            )
            self.db.add(pred_log)
            self.db.commit()

            return {
                "status": "SUCCESS",
                "category": category,
                "make": make,
                "model": model,
                "yom": yom,
                "mileage_km": mileage_km,
                "fuel_type": fuel_type,
                "transmission": transmission,
                "district": district,
                "estimated_market_asking_value": predicted_value,
                "estimated_market_range": {
                    "low": range_low,
                    "high": range_high
                },
                "confidence": confidence,
                "similar_listings_count": similar_count,
                "seller_asking_price": seller_asking_price,
                "price_assessment": {
                    "classification": assessment,
                    "difference_percent": delta_pct,
                    "difference_amount": (seller_asking_price - predicted_value) if seller_asking_price else None
                } if assessment else None,
                "algorithm": bundle.get("algorithm", "XGBoost"),
                "model_metrics": metrics
            }

        except Exception as e:
            logger.error(f"Prediction inference error: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "message": f"Valuation calculation failed: {str(e)}"
            }

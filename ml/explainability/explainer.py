"""
Valuation Explainer Module
Uses SHAP (SHapley Additive exPlanations) to explain individual vehicle
price predictions and quantify the financial impact of each vehicle feature.
"""
import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import shap
import joblib

from config import settings
from ml.preprocessing.pipeline import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

logger = logging.getLogger(__name__)

class ValuationExplainer:
    def __init__(self):
        self._explainer_cache: Dict[str, Any] = {}

    def _get_explainer(self, category: str) -> Optional[Dict[str, Any]]:
        clean_cat = category.lower().replace("-", "_")
        if clean_cat in self._explainer_cache:
            return self._explainer_cache[clean_cat]

        filename = f"valuation_{clean_cat}.joblib"
        filepath = os.path.join(settings.ML_MODELS_DIR, filename)
        if not os.path.exists(filepath):
            return None

        bundle = joblib.load(filepath)
        estimator = bundle["estimator"]
        preprocessor = bundle["preprocessor"]

        try:
            # TreeExplainer for XGBoost & Random Forest
            explainer = shap.TreeExplainer(estimator)
            data = {"explainer": explainer, "preprocessor": preprocessor, "bundle": bundle}
            self._explainer_cache[clean_cat] = data
            return data
        except Exception as e:
            logger.warning(f"Could not build TreeExplainer: {e}. Fallback to Explainer.")
            try:
                explainer = shap.Explainer(estimator)
                data = {"explainer": explainer, "preprocessor": preprocessor, "bundle": bundle}
                self._explainer_cache[clean_cat] = data
                return data
            except Exception as e2:
                logger.error(f"Failed to initialize SHAP explainer: {e2}")
                return None

    def explain_prediction(
        self,
        category: str,
        vehicle_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Computes SHAP feature importance breakdown for a single vehicle inference.
        Returns:
          - base_value (Rs.)
          - predicted_value (Rs.)
          - feature_contributions: list of {feature, display_name, impact_rs, impact_direction, impact_strength}
        """
        obj = self._get_explainer(category)
        if not obj:
            return {
                "status": "UNAVAILABLE",
                "message": "Explainability model is not available for this category."
            }

        explainer = obj["explainer"]
        preprocessor = obj["preprocessor"]

        df = pd.DataFrame([{
            "category": category.capitalize(),
            "make": vehicle_dict.get("make", "Unknown").title(),
            "model": vehicle_dict.get("model", "Unknown").title(),
            "yom": int(vehicle_dict.get("yom", 2015)),
            "mileage_km": int(vehicle_dict.get("mileage_km", 60000)),
            "fuel_type": str(vehicle_dict.get("fuel_type", "Petrol")).title(),
            "transmission": str(vehicle_dict.get("transmission", "Automatic")).title(),
            "condition": str(vehicle_dict.get("condition", "Used")).title(),
            "engine_cc": int(vehicle_dict.get("engine_cc", 1500)),
            "district": str(vehicle_dict.get("district", "Colombo")).title()
        }])

        try:
            X_trans = preprocessor.transform(df)
            shap_values = explainer.shap_values(X_trans)

            if isinstance(shap_values, list):
                shap_arr = shap_values[0]
            else:
                shap_arr = shap_values

            # Aggregate SHAP values back to high-level intuitive features
            col_transform = preprocessor.named_steps["column_transform"]
            all_feature_names = col_transform.get_feature_names_out()

            # Group impacts into core user-facing concepts
            feature_impacts = {
                "Vehicle Year / Age": 0.0,
                "Mileage (km)": 0.0,
                "Make & Model": 0.0,
                "Fuel Type": 0.0,
                "Transmission": 0.0,
                "Location / District": 0.0,
                "Condition": 0.0,
                "Engine Capacity": 0.0
            }

            for idx, f_name in enumerate(all_feature_names):
                val = float(shap_arr[0][idx])
                if "yom" in f_name or "vehicle_age" in f_name:
                    feature_impacts["Vehicle Year / Age"] += val
                elif "mileage" in f_name:
                    feature_impacts["Mileage (km)"] += val
                elif "make" in f_name or "model" in f_name:
                    feature_impacts["Make & Model"] += val
                elif "fuel_type" in f_name:
                    feature_impacts["Fuel Type"] += val
                elif "transmission" in f_name:
                    feature_impacts["Transmission"] += val
                elif "district" in f_name:
                    feature_impacts["Location / District"] += val
                elif "condition" in f_name:
                    feature_impacts["Condition"] += val
                elif "engine_cc" in f_name:
                    feature_impacts["Engine Capacity"] += val

            # Format sorted list
            contributions = []
            for name, impact in feature_impacts.items():
                impact_rounded = round(impact, -3)
                abs_imp = abs(impact)
                if abs_imp > 500_000:
                    strength = "High Influence"
                elif abs_imp > 150_000:
                    strength = "Medium Influence"
                else:
                    strength = "Low Influence"

                contributions.append({
                    "feature": name,
                    "impact_rs": impact_rounded,
                    "impact_direction": "POSITIVE (Increases Value)" if impact >= 0 else "NEGATIVE (Decreases Value)",
                    "impact_strength": strength,
                    "abs_impact": abs_imp
                })

            contributions.sort(key=lambda x: x["abs_impact"], reverse=True)

            base_val = getattr(explainer, "expected_value", None)
            if isinstance(base_val, np.ndarray):
                base_val = float(base_val[0])
            elif base_val is not None:
                base_val = float(base_val)
            else:
                base_val = 5_000_000.0

            return {
                "status": "SUCCESS",
                "base_market_value": round(base_val, -4),
                "contributions": contributions,
                "top_positive": [c for c in contributions if c["impact_rs"] > 0][:3],
                "top_negative": [c for c in contributions if c["impact_rs"] < 0][:3]
            }

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "message": f"Could not generate SHAP explanation: {str(e)}"
            }

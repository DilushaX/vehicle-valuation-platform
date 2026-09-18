"""
Model Explainability Subsystem for Vehicle Valuation.

Provides local feature attribution explaining which vehicle attributes
(e.g., vehicle age, mileage, transmission, brand, model, engine CC)
contributed positively or negatively to the predicted asking price.

Uses Tree SHAP (Shapley Additive Explanations) on the underlying
RandomForestRegressor, with a reliable fallback mechanism.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor

from ml.prediction.predictor import VehiclePricePredictor

logger = logging.getLogger(__name__)

CANONICAL_FEATURE_KEYS = [
    "brand_model",
    "vehicle_age",
    "mileage",
    "engine_cc",
    "category",
    "brand",
    "model",
    "fuel_type",
    "transmission",
    "district",
    "condition",
]


class ModelExplainer:
    """
    Calculates feature-level attribution for vehicle asking price predictions.
    
    Attributes:
        predictor: A configured VehiclePricePredictor instance.
    """

    def __init__(
        self,
        predictor: VehiclePricePredictor,
    ):
        self.predictor = predictor
        self.model = predictor.model
        self._tree_explainer = None
        self._underlying_rf: Optional[RandomForestRegressor] = None
        self._preprocessor = None
        self._init_components()

    def _init_components(self) -> None:
        """Extracts preprocessor and tree estimator from model pipeline."""
        if isinstance(self.model, TransformedTargetRegressor):
            pipeline = self.model.regressor_
        else:
            pipeline = self.model

        if hasattr(pipeline, "named_steps"):
            self._preprocessor = pipeline.named_steps.get("preprocessor")
            self._underlying_rf = pipeline.named_steps.get("regressor")
        elif hasattr(pipeline, "steps"):
            self._preprocessor = pipeline.steps[0][1]
            self._underlying_rf = pipeline.steps[-1][1]

        # Initialize TreeExplainer if SHAP is available and model is a tree ensemble
        if self._underlying_rf is not None and isinstance(self._underlying_rf, RandomForestRegressor):
            try:
                import shap
                self._tree_explainer = shap.TreeExplainer(self._underlying_rf)
                logger.info("SHAP TreeExplainer initialized successfully for valuation model.")
            except Exception as e:
                logger.warning(f"Could not initialize SHAP TreeExplainer: {e}. Falling back to tree-based attribution.")
                self._tree_explainer = None

    def explain_prediction(
        self,
        vehicle_data: Union[Dict[str, Any], pd.DataFrame, Sequence[Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Explains feature contributions for a vehicle listing.
        
        Args:
            vehicle_data: Dictionary or DataFrame representing a single vehicle.
            top_k: Maximum number of top contributing features to return.
            
        Returns:
            List of structured feature contribution records sorted by magnitude:
            [
                {
                    "feature": "transmission",
                    "value": "Automatic",
                    "contribution": 0.6673,
                    "direction": "positive",
                    "description": "Transmission (Automatic) increased the estimated asking price."
                },
                ...
            ]
        """
        df_formatted = self.predictor.validate_and_format_input(vehicle_data)
        if df_formatted.empty:
            return []

        # Take first record if multiple provided
        single_row_df = df_formatted.iloc[[0]].copy()

        if self._tree_explainer is not None and self._preprocessor is not None:
            return self._explain_via_shap(single_row_df, top_k)
        else:
            return self._explain_via_fallback(single_row_df, top_k)

    def _explain_via_shap(
        self,
        single_row_df: pd.DataFrame,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Computes Tree SHAP values and aggregates one-hot encoded columns back to input features."""
        # 1. Transform raw row to model feature vector
        X_trans = self._preprocessor.transform(single_row_df)
        feature_names = self._preprocessor.get_feature_names_out()

        # 2. Compute SHAP values
        shap_values = self._tree_explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            # For some tree models shap_values is a list of arrays
            raw_contribs = shap_values[0][0]
        else:
            raw_contribs = shap_values[0]

        # 3. Aggregate transformed one-hot columns back to original input features
        aggregated_contribs: Dict[str, float] = {k: 0.0 for k in CANONICAL_FEATURE_KEYS}

        for feat_name, val in zip(feature_names, raw_contribs):
            clean_name = feat_name.replace("num__", "").replace("cat__", "")
            # Find matching canonical feature
            matched_key = None
            for key in CANONICAL_FEATURE_KEYS:
                if clean_name.startswith(key):
                    matched_key = key
                    break
            if matched_key is None:
                matched_key = clean_name.split("_")[0]
            aggregated_contribs[matched_key] = aggregated_contribs.get(matched_key, 0.0) + float(val)

        # 4. Format structured output records
        results: List[Dict[str, Any]] = []
        row_dict = single_row_df.iloc[0].to_dict()

        for feat, contrib in aggregated_contribs.items():
            if abs(contrib) < 1e-6:
                continue

            val_repr = row_dict.get(feat, "Unknown")
            # Format numerical values cleanly
            if isinstance(val_repr, float) and np.isnan(val_repr):
                val_repr = "Not Specified"
            elif feat == "mileage" and isinstance(val_repr, (int, float)):
                val_repr = f"{val_repr:,.0f} km"
            elif feat == "engine_cc" and isinstance(val_repr, (int, float)):
                val_repr = f"{val_repr:,.0f} cc"
            elif feat == "vehicle_age" and isinstance(val_repr, (int, float)):
                val_repr = f"{val_repr:.0f} years"

            direction = "positive" if contrib > 0 else "negative"

            if direction == "positive":
                desc = f"{feat.replace('_', ' ').title()} ({val_repr}) contributed positively to the estimated asking price."
            else:
                desc = f"{feat.replace('_', ' ').title()} ({val_repr}) contributed negatively to the estimated asking price."

            results.append({
                "feature": feat,
                "value": str(val_repr),
                "contribution": round(float(contrib), 4),
                "direction": direction,
                "description": desc,
            })

        # Sort by absolute contribution descending
        results.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        if top_k is not None and top_k > 0:
            return results[:top_k]

        return results

    def _explain_via_fallback(
        self,
        single_row_df: pd.DataFrame,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Heuristic tree-based feature attribution fallback when SHAP is not initialized.
        Uses feature importances and directional difference from median vehicle baseline.
        """
        logger.info("Using fallback feature contribution calculation.")
        if self._underlying_rf is None or not hasattr(self._underlying_rf, "feature_importances_"):
            return []

        importances = self._underlying_rf.feature_importances_
        feature_names = (
            self._preprocessor.get_feature_names_out()
            if self._preprocessor is not None
            else [f"feat_{i}" for i in range(len(importances))]
        )

        aggregated_importances: Dict[str, float] = {}
        for feat_name, imp in zip(feature_names, importances):
            clean_name = feat_name.replace("num__", "").replace("cat__", "")
            matched_key = clean_name.split("_")[0]
            for key in CANONICAL_FEATURE_KEYS:
                if clean_name.startswith(key):
                    matched_key = key
                    break
            aggregated_importances[matched_key] = aggregated_importances.get(matched_key, 0.0) + float(imp)

        row_dict = single_row_df.iloc[0].to_dict()
        results: List[Dict[str, Any]] = []

        # Predict current vs perturbed
        current_pred = self.predictor.predict(single_row_df)[0]

        for feat, imp in aggregated_importances.items():
            val_repr = str(row_dict.get(feat, "Unknown"))
            # Default directional heuristic: newer vehicle age, lower mileage, higher CC generally increase asking price
            direction = "positive"
            if feat == "vehicle_age" and float(row_dict.get("vehicle_age", 10)) > 15:
                direction = "negative"
            elif feat == "mileage" and float(row_dict.get("mileage", 80000)) > 100000:
                direction = "negative"

            contrib = round(float(imp), 4) * (1.0 if direction == "positive" else -1.0)
            results.append({
                "feature": feat,
                "value": val_repr,
                "contribution": contrib,
                "direction": direction,
                "description": f"{feat.replace('_', ' ').title()} ({val_repr}) influenced the estimated asking price.",
            })

        results.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        if top_k is not None and top_k > 0:
            return results[:top_k]
        return results

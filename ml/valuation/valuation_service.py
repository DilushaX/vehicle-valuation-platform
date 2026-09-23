"""
Unified Vehicle Valuation Service.

Coordinates:
1. Input schema and domain validation.
2. Feature preparation and canonicalization.
3. Machine learning point asking-price prediction.
4. Model-based valuation range (uncertainty) estimation.
5. Tree SHAP feature attribution and explainability.
6. Multi-attribute comparable vehicle retrieval from market listings.
7. Model metadata and explicit legal/methodological limitations.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sqlalchemy.orm import Session

from analytics.comparables.comparable_engine import ComparableVehicleEngine
from ml.explainability.explainer import ModelExplainer
from ml.prediction.predictor import ValidationError, VehiclePricePredictor
from ml.prediction.uncertainty import UncertaintyEstimator

logger = logging.getLogger(__name__)

STANDARD_VALUATION_LIMITATIONS = [
    "This valuation estimates market asking prices observed on Riyasewana. It is not a confirmed transaction or final selling price.",
    "Actual negotiated selling prices may differ from advertised asking prices because the collected dataset does not contain verified final transaction prices.",
    "The underlying valuation model was trained on an experimental benchmark dataset of 113 verified ML-eligible records across 8 vehicle categories; current dataset size limits generalization across sparse categories.",
    "The indicative prediction range reflects ensemble decision tree dispersion, NOT a legally or financially guaranteed appraisal.",
    "Actual vehicle value may differ due to factors not captured by the dataset, including physical condition, accident history, mechanical condition, battery/engine health, and registration documentation.",
]


@dataclass
class ValuationResult:
    """
    Structured outcome of a complete vehicle valuation request.
    """
    estimated_asking_price_lkr: float
    prediction_range_lkr: Dict[str, Any]
    currency: str
    model: Dict[str, Any]
    explanation: List[Dict[str, Any]]
    comparables: List[Dict[str, Any]]
    limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimated_asking_price_lkr": round(float(self.estimated_asking_price_lkr), 2),
            "prediction_range_lkr": self.prediction_range_lkr,
            "currency": self.currency,
            "model": self.model,
            "explanation": self.explanation,
            "comparables": self.comparables,
            "limitations": self.limitations,
        }

    def plot_explanation(
        self,
        output_path: Union[str, Path] = "data/analysis/ml/figures/valuation_feature_contributions.png",
        top_k: Optional[int] = 10,
        show_values: bool = True,
    ) -> Path:
        """
        Generates and saves a model feature contribution plot for this valuation result.
        
        Args:
            output_path: Target filesystem path for the output PNG.
            top_k: Maximum number of contributing features to display.
            show_values: Whether to include feature values in labels.
            
        Returns:
            Path object pointing to the generated image file.
        """
        from ml.explainability.visualization import create_feature_contribution_plot
        return create_feature_contribution_plot(
            explanation=self.explanation,
            output_path=output_path,
            top_k=top_k,
            show_values=show_values,
        )



class VehicleValuationService:
    """
    Unified valuation service for Sri Lankan vehicle market intelligence.
    """

    def __init__(
        self,
        predictor: Optional[VehiclePricePredictor] = None,
        comparable_engine: Optional[ComparableVehicleEngine] = None,
        db_session: Optional[Session] = None,
        model_path: Union[str, Path] = "data/analysis/ml/models/model.joblib",
        metadata_path: Optional[Union[str, Path]] = "data/analysis/ml/models/model_metadata.json",
    ):
        if predictor is not None:
            self.predictor = predictor
        else:
            self.predictor = VehiclePricePredictor.load(
                model_path=model_path, metadata_path=metadata_path
            )

        self.explainer = ModelExplainer(predictor=self.predictor)
        self.uncertainty_estimator = UncertaintyEstimator(
            model=self.predictor.model,
            metadata=self.predictor.metadata,
        )
        self.comparable_engine = comparable_engine or ComparableVehicleEngine(session=db_session)

    def valuate(
        self,
        vehicle_data: Union[Dict[str, Any], pd.DataFrame, pd.Series],
        top_k_factors: int = 5,
        top_k_comparables: int = 5,
        percentile_lower: int = 10,
        percentile_upper: int = 90,
        exclude_listing_id: Optional[str] = None,
    ) -> ValuationResult:
        """
        Executes an end-to-end vehicle valuation workflow.
        
        Args:
            vehicle_data: Input vehicle attributes (dict, Series, or DataFrame).
            top_k_factors: Number of top contributing explainability factors to return.
            top_k_comparables: Number of comparable market listings to retrieve.
            percentile_lower: Lower bound percentile for model-based range (default: 10).
            percentile_upper: Upper bound percentile for model-based range (default: 90).
            exclude_listing_id: Optional listing ID to exclude from comparables.
            
        Returns:
            ValuationResult containing price estimate, range, factors, comparables, and disclaimers.
            
        Raises:
            ValidationError: If required features are missing or values are out of bounds.
        """
        # 1. Validation and feature formatting
        df_formatted = self.predictor.validate_and_format_input(vehicle_data)
        if df_formatted.empty:
            raise ValidationError("Cannot perform valuation on empty vehicle input.")

        # 2. Point Prediction (asking price in LKR)
        point_estimate = self.predictor.predict_single(vehicle_data)

        # 3. Model-Based Prediction Range
        val_range = self.uncertainty_estimator.estimate_range(
            formatted_df=df_formatted,
            point_estimate=point_estimate,
            percentile_lower=percentile_lower,
            percentile_upper=percentile_upper,
        )

        # 4. Model Explainability (Tree SHAP feature attribution)
        explanations = self.explainer.explain_prediction(
            vehicle_data=df_formatted,
            top_k=top_k_factors,
        )

        # 5. Comparable Vehicles Retrieval
        comps = self.comparable_engine.find_comparables(
            query=df_formatted,
            top_k=top_k_comparables,
            match_category_strictly=True,
            exclude_listing_id=exclude_listing_id,
        )
        comparables_dict_list = [c.to_dict() for c in comps]

        # 6. Extract Model Metadata
        meta = self.predictor.metadata or {}
        model_info = {
            "name": meta.get("model_name", "RandomForestRegressor"),
            "target_variable": meta.get("target_variable", "asking_price"),
            "target_transform": meta.get("target_transform", "log1p"),
            "training_records": meta.get("train_samples", 90),
            "test_samples": meta.get("test_samples", 23),
            "status": meta.get("status", "EXPERIMENTAL_RESEARCH_BENCHMARK"),
        }

        # 7. Construct Unified Result
        return ValuationResult(
            estimated_asking_price_lkr=point_estimate,
            prediction_range_lkr=val_range.to_dict(),
            currency="LKR",
            model=model_info,
            explanation=explanations,
            comparables=comparables_dict_list,
            limitations=STANDARD_VALUATION_LIMITATIONS,
        )

    def plot_explanation(
        self,
        valuation_result_or_explanation: Union[ValuationResult, List[Dict[str, Any]], Dict[str, Any]],
        output_path: Union[str, Path] = "data/analysis/ml/figures/valuation_feature_contributions.png",
        top_k: Optional[int] = 10,
        show_values: bool = True,
    ) -> Path:
        """
        Convenience method to render and save a model feature contribution plot.
        
        Args:
            valuation_result_or_explanation: ValuationResult or list/dict containing explanation.
            output_path: Target filesystem path for the output PNG.
            top_k: Maximum number of contributing features to display.
            show_values: Whether to include feature values in labels.
            
        Returns:
            Path to the saved figure.
        """
        from ml.explainability.visualization import create_feature_contribution_plot
        return create_feature_contribution_plot(
            explanation=valuation_result_or_explanation,
            output_path=output_path,
            top_k=top_k,
            show_values=show_values,
        )


"""
Unified Vehicle Valuation Workflow Orchestration Layer (Phase 10.5 Part 1).

Orchestrates the complete vehicle valuation lifecycle:
    User Vehicle Input
        ↓
    Input Validation
        ↓
    ML Valuation
        ↓
    Estimated Asking Price
        ↓
    Indicative Prediction Range
        ↓
    Model Explanation
        ↓
    Data Quality Indicators
        ↓
    Comparable Vehicles
        ↓
    Comparable Market Summary
        ↓
    Audit / Reproducibility
        ↓
    Final Valuation Result

Mandatory Methodological and Operational Disclaimers:
1. Asking Price: The model estimates market ASKING PRICE observed on Riyasewana.
   It does NOT estimate confirmed actual transaction or selling prices.
2. Prediction Range: The range is an indicative model-based prediction range
   reflecting ensemble decision tree dispersion, NOT a statistical confidence interval.
3. Feature Attribution: SHAP values describe model contribution/attribution
   and must not be described as causal effects.
4. Comparable Matching: Comparable similarity measures specification alignment
   relative to the requested vehicle based on weighted feature distances.
   It is NOT a probability, confidence score, accuracy score, or proof that
   two vehicles are the same physical vehicle.
5. Read-Only: This workflow performs strictly read/analysis operations and NO database writes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from feature_engineering.validation import DataLeakageError
from ml.prediction.predictor import ValidationError, VehiclePricePredictor
from ml.valuation.valuation_service import (
    STANDARD_VALUATION_LIMITATIONS,
    ValuationResult,
    VehicleValuationService,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input Schema
# ---------------------------------------------------------------------------

class ValuationWorkflowInput(BaseModel):
    """
    Validated vehicle specification input for the unified valuation workflow.
    Accepts all standard supported vehicle attributes without introducing
    unnecessary new fields.
    """
    category: str = Field(..., description="Canonical vehicle category (e.g. Cars, SUVs, Vans, Motorbikes)")
    brand: str = Field(..., min_length=1, description="Vehicle make / brand (e.g. Toyota, Honda, Nissan)")
    model: str = Field(..., min_length=1, description="Vehicle model (e.g. Premio, Vezel, Fit)")
    manufacture_year: Optional[int] = Field(None, ge=1920, le=2026, description="Year of vehicle manufacture")
    vehicle_age: Optional[float] = Field(None, ge=0, le=100, description="Vehicle age in years (relative to 2026)")
    mileage: Optional[float] = Field(None, ge=0, le=2_000_000, description="Odometer reading in kilometers")
    engine_cc: Optional[float] = Field(None, ge=0, le=25_000, description="Engine capacity in cubic centimeters (cc)")
    fuel_type: str = Field(..., min_length=1, description="Fuel type (e.g. Petrol, Diesel, Hybrid, Electric)")
    transmission: str = Field(..., min_length=1, description="Transmission (e.g. Automatic, Manual)")
    district: str = Field(..., min_length=1, description="Sri Lankan administrative district (e.g. Colombo, Kandy)")
    condition: str = Field(..., min_length=1, description="Condition (e.g. Registered (Used), Unregistered, Brand New)")
    registration_year: Optional[int] = Field(None, ge=1920, le=2026, description="Optional registration year")


# ---------------------------------------------------------------------------
# Section Result Schemas
# ---------------------------------------------------------------------------

class ValuationSection(BaseModel):
    """Point asking-price estimate and model identifier."""
    estimated_asking_price_lkr: float = Field(..., description="Estimated market asking price in LKR")
    currency: str = Field("LKR", description="Currency identifier (Sri Lankan Rupee)")
    model: Dict[str, Any] = Field(..., description="Model metadata summary")
    disclaimer: str = Field(
        "The model estimates market asking prices observed on Riyasewana. "
        "It is not a confirmed transaction or final selling price.",
        description="Operational asking price notice",
    )

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class RangeSection(BaseModel):
    """
    Indicative model-based prediction range reflecting ensemble tree dispersion.
    Not a statistical confidence interval.
    """
    estimate: float = Field(..., description="Point asking price estimate in LKR")
    lower: float = Field(..., description="Lower bound of model-based prediction range in LKR")
    upper: float = Field(..., description="Upper bound of model-based prediction range in LKR")
    spread: float = Field(..., description="Spread between upper and lower bound in LKR")
    percentile_lower: int = Field(10, description="Lower percentile used for tree dispersion")
    percentile_upper: int = Field(90, description="Upper percentile used for tree dispersion")
    method: str = Field("individual_tree_percentiles", description="Uncertainty estimation method")
    disclaimer: str = Field(
        "The prediction range is an indicative model-based prediction range, "
        "NOT a statistical confidence interval.",
        description="Methodological notice for prediction range",
    )

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class ExplanationSection(BaseModel):
    """Model feature attribution factors using Tree SHAP."""
    factors: List[Dict[str, Any]] = Field(default_factory=list, description="Top SHAP feature attribution factors")
    method: str = Field("Tree SHAP feature attribution", description="Explainability methodology")
    disclaimer: str = Field(
        "SHAP values describe model contribution/attribution and must not be described as causal effects.",
        description="Attribution notice",
    )

    def __iter__(self):
        return iter(self.factors)

    def __len__(self) -> int:
        return len(self.factors)

    def __getitem__(self, idx: Union[int, str]) -> Any:
        if isinstance(idx, int):
            return self.factors[idx]
        return getattr(self, idx)


class DataQualitySection(BaseModel):
    """Valuation input data completeness assessment."""
    status: str = Field(..., description="Data completeness status ('COMPLETE' or 'PARTIAL')")
    provided_features: int = Field(..., ge=0, description="Count of valid features provided in the input")
    expected_features: int = Field(11, ge=0, description="Expected feature count (11 canonical features)")
    missing_features: List[str] = Field(default_factory=list, description="List of omitted or null expected features")
    comparable_count: int = Field(0, ge=0, description="Number of comparable market listings retrieved")

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class ComparablesSection(BaseModel):
    """
    Retrieved comparable vehicle listings.
    Similarity score represents specification distance, not confidence or probability.
    """
    items: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved comparable vehicle listings")
    total_found: int = Field(0, ge=0, description="Total count of comparables retrieved")
    disclaimer: str = Field(
        "Comparable similarity is not a probability, confidence score, accuracy score, "
        "or proof that two vehicles are the same physical vehicle.",
        description="Comparable matching notice",
    )

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: Union[int, str]) -> Any:
        if isinstance(idx, int):
            return self.items[idx]
        return getattr(self, idx)


class MarketSummarySection(BaseModel):
    """
    Descriptive summary of asking prices among retrieved comparables.
    Reflects advertised asking prices, not verified transaction prices.
    """
    comparable_count: int = Field(0, ge=0, description="Count of comparables included in summary")
    min_asking_price: Optional[float] = Field(None, description="Minimum asking price in LKR")
    max_asking_price: Optional[float] = Field(None, description="Maximum asking price in LKR")
    median_asking_price: Optional[float] = Field(None, description="Median asking price in LKR")
    average_asking_price: Optional[float] = Field(None, description="Average asking price in LKR")
    price_spread: Optional[float] = Field(None, description="Spread between maximum and minimum asking price in LKR")
    disclaimer: str = Field(
        "Market summary statistics describe advertised asking prices of retrieved comparables, "
        "not verified transaction prices.",
        description="Market summary notice",
    )

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class AuditSection(BaseModel):
    """Valuation audit metadata and deterministic reproducibility fingerprint."""
    metadata: Dict[str, Any] = Field(..., description="Model version, target, timestamp, and methodology metadata")
    reproducibility: Dict[str, Any] = Field(..., description="Deterministic SHA-256 reproducibility fingerprint")
    disclaimer: str = Field(
        "The reproducibility fingerprint identifies equivalent valuation configurations and inputs. "
        "It is not a cryptographic security signature or price warranty.",
        description="Audit disclaimer",
    )

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


# ---------------------------------------------------------------------------
# Unified Workflow Result Schema
# ---------------------------------------------------------------------------

class ValuationWorkflowResult(BaseModel):
    """
    Unified, structured result object produced by VehicleValuationWorkflow.
    
    Exposes 8 distinct sections consumed by the API and Dashboard:
    - valuation
    - range
    - explanation
    - data_quality
    - comparables
    - market_summary
    - audit
    - limitations
    """
    success: bool = Field(True, description="Whether the workflow completed successfully")
    status: str = Field("SUCCESS", description="Execution status: SUCCESS, PARTIAL_DATA, or VALIDATION_ERROR")
    error: Optional[str] = Field(None, description="Primary error message on validation or workflow failure")
    errors: List[str] = Field(default_factory=list, description="All validation or workflow error messages")

    valuation: Optional[ValuationSection] = Field(None, description="Point valuation estimate and model info")
    range: Optional[RangeSection] = Field(None, description="Indicative prediction range")
    explanation: Optional[ExplanationSection] = Field(None, description="Tree SHAP explainability factors")
    data_quality: Optional[DataQualitySection] = Field(None, description="Input data quality assessment")
    comparables: Optional[ComparablesSection] = Field(None, description="Retrieved comparable vehicles")
    market_summary: Optional[MarketSummarySection] = Field(None, description="Comparable market asking price summary")
    audit: Optional[AuditSection] = Field(None, description="Audit trail and deterministic reproducibility fingerprint")
    limitations: List[str] = Field(
        default_factory=lambda: list(STANDARD_VALUATION_LIMITATIONS),
        description="Mandatory legal, financial, and methodological limitations",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result to a dictionary representation."""
        return self.model_dump()

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def to_valuation_result(self) -> ValuationResult:
        """Converts workflow result to the legacy ValuationResult dataclass."""
        if not self.success or self.valuation is None:
            raise ValueError(f"Cannot convert failed workflow result to ValuationResult: {self.error}")
        return ValuationResult(
            estimated_asking_price_lkr=self.valuation.estimated_asking_price_lkr,
            prediction_range_lkr=self.range.model_dump() if self.range else {},
            currency=self.valuation.currency,
            model=self.valuation.model,
            explanation=self.explanation.factors if self.explanation else [],
            comparables=self.comparables.items if self.comparables else [],
            limitations=self.limitations,
            data_quality=self.data_quality.model_dump() if self.data_quality else None,
            comparable_market_summary=self.market_summary.model_dump() if self.market_summary else None,
            audit=self.audit.metadata if self.audit else None,
            reproducibility=self.audit.reproducibility if self.audit else None,
        )

    @classmethod
    def from_valuation_result(
        cls,
        val_result: ValuationResult,
        status: Optional[str] = None,
    ) -> "ValuationWorkflowResult":
        """Constructs ValuationWorkflowResult from an existing ValuationResult."""
        dq_dict = val_result.data_quality or {}
        is_partial = dq_dict.get("status") == "PARTIAL"
        resolved_status = status or ("PARTIAL_DATA" if is_partial else "SUCCESS")

        valuation_sec = ValuationSection(
            estimated_asking_price_lkr=val_result.estimated_asking_price_lkr,
            currency=val_result.currency,
            model=val_result.model,
        )

        rng = val_result.prediction_range_lkr or {}
        range_sec = RangeSection(
            estimate=rng.get("estimate", val_result.estimated_asking_price_lkr),
            lower=rng.get("lower", val_result.estimated_asking_price_lkr * 0.85),
            upper=rng.get("upper", val_result.estimated_asking_price_lkr * 1.15),
            spread=rng.get("spread", 0.0),
            percentile_lower=rng.get("percentile_lower", 10),
            percentile_upper=rng.get("percentile_upper", 90),
            method=rng.get("method", "individual_tree_percentiles"),
        )

        explanation_sec = ExplanationSection(
            factors=val_result.explanation or [],
            method="Tree SHAP feature attribution",
        )

        dq_sec = DataQualitySection(
            status=dq_dict.get("status", "COMPLETE"),
            provided_features=dq_dict.get("provided_features", 11),
            expected_features=dq_dict.get("expected_features", 11),
            missing_features=dq_dict.get("missing_features", []),
            comparable_count=dq_dict.get("comparable_count", len(val_result.comparables)),
        )

        comparables_sec = ComparablesSection(
            items=val_result.comparables or [],
            total_found=len(val_result.comparables or []),
        )

        cms = val_result.comparable_market_summary or {}
        summary_sec = MarketSummarySection(
            comparable_count=cms.get("comparable_count", len(val_result.comparables or [])),
            min_asking_price=cms.get("min_asking_price"),
            max_asking_price=cms.get("max_asking_price"),
            median_asking_price=cms.get("median_asking_price"),
            average_asking_price=cms.get("average_asking_price"),
            price_spread=cms.get("price_spread"),
        )

        audit_sec = AuditSection(
            metadata=val_result.audit or {},
            reproducibility=val_result.reproducibility or {},
        )

        return cls(
            success=True,
            status=resolved_status,
            error=None,
            errors=[],
            valuation=valuation_sec,
            range=range_sec,
            explanation=explanation_sec,
            data_quality=dq_sec,
            comparables=comparables_sec,
            market_summary=summary_sec,
            audit=audit_sec,
            limitations=val_result.limitations or list(STANDARD_VALUATION_LIMITATIONS),
        )


# ---------------------------------------------------------------------------
# Orchestration Layer Implementation
# ---------------------------------------------------------------------------

class VehicleValuationWorkflow:
    """
    Dedicated orchestration layer for the complete vehicle valuation workflow.
    
    Coordinates:
    1. Input validation & domain constraint enforcement.
    2. Missing/partial feature detection and handling.
    3. Machine learning point asking-price estimation.
    4. Model-based prediction range (individual tree percentiles).
    5. Tree SHAP feature attribution and directionality.
    6. Input data quality assessment.
    7. Multi-attribute comparable vehicle retrieval from market listings.
    8. Comparable market descriptive price summary.
    9. Audit trail and deterministic SHA-256 reproducibility fingerprint.
    10. Assembly of the unified 8-section ValuationWorkflowResult.
    
    Reuses existing components without duplicating logic:
    - VehicleValuationService
    - VehiclePricePredictor
    - UncertaintyEstimator
    - ModelExplainer
    - ComparableVehicleEngine
    - create_comparable_market_summary
    - assess_valuation_data_quality
    - create_valuation_audit
    
    Guarantees:
    - Absolutely NO database writes (read/analysis operations only).
    - Preserves all legal, financial, and methodological disclaimers.
    - Gracefully handles partial inputs where supported.
    - Gracefully handles zero comparable listings.
    - Yields clean validation failures without returning fake valuations.
    """

    def __init__(
        self,
        valuation_service: Optional[VehicleValuationService] = None,
        session: Optional[Session] = None,
        model_path: Union[str, Path] = "data/analysis/ml/models/model.joblib",
        metadata_path: Optional[Union[str, Path]] = "data/analysis/ml/models/model_metadata.json",
    ):
        if valuation_service is not None:
            self.valuation_service = valuation_service
        else:
            self.valuation_service = VehicleValuationService(
                db_session=session,
                model_path=model_path,
                metadata_path=metadata_path,
            )

    def validate_input(
        self,
        vehicle_data: Union[Dict[str, Any], pd.DataFrame, pd.Series, BaseModel],
        raise_on_error: bool = False,
    ) -> Tuple[bool, Optional[pd.DataFrame], List[str]]:
        """
        Validates vehicle input using the existing model predictor pipeline.
        
        Args:
            vehicle_data: Input vehicle specifications (dict, DataFrame, Series, or BaseModel).
            raise_on_error: If True, raises ValidationError on invalid input.
            
        Returns:
            Tuple of (is_valid: bool, formatted_df: Optional[pd.DataFrame], errors: List[str]).
        """
        if isinstance(vehicle_data, BaseModel):
            raw_data = vehicle_data.model_dump()
        elif isinstance(vehicle_data, (pd.DataFrame, pd.Series, dict)):
            raw_data = vehicle_data
        else:
            msg = f"Unsupported input type: {type(vehicle_data)}. Must be dict, DataFrame, Series, or BaseModel."
            if raise_on_error:
                raise ValidationError(msg)
            return False, None, [msg]

        try:
            formatted_df = self.valuation_service.predictor.validate_and_format_input(raw_data)
            if formatted_df.empty:
                msg = "Vehicle input cannot be empty."
                if raise_on_error:
                    raise ValidationError(msg)
                return False, None, [msg]
            return True, formatted_df, []
        except (ValidationError, ValueError, DataLeakageError) as exc:
            if raise_on_error:
                raise
            return False, None, [str(exc)]
        except Exception as exc:
            logger.error(f"Unexpected error validating vehicle input: {exc}", exc_info=True)
            if raise_on_error:
                raise ValidationError(f"Validation failure: {str(exc)}") from exc
            return False, None, [f"Validation failure: {str(exc)}"]

    def execute(
        self,
        vehicle_data: Union[Dict[str, Any], pd.DataFrame, pd.Series, BaseModel],
        top_k_factors: int = 5,
        top_k_comparables: int = 5,
        percentile_lower: int = 10,
        percentile_upper: int = 90,
        exclude_listing_id: Optional[str] = None,
        raise_on_error: bool = False,
    ) -> ValuationWorkflowResult:
        """
        Executes the end-to-end vehicle valuation workflow.
        
        Args:
            vehicle_data: Input vehicle specifications.
            top_k_factors: Number of top explainability features to return.
            top_k_comparables: Number of comparable market listings to retrieve.
            percentile_lower: Lower bound percentile for prediction range (default: 10).
            percentile_upper: Upper bound percentile for prediction range (default: 90).
            exclude_listing_id: Optional listing ID to exclude from comparables.
            raise_on_error: If True, raises ValidationError on invalid inputs instead of returning an error result.
            
        Returns:
            ValuationWorkflowResult containing all 8 workflow sections.
        """
        # Step 1 & 2: Input Validation
        is_valid, formatted_df, validation_errors = self.validate_input(
            vehicle_data=vehicle_data,
            raise_on_error=raise_on_error,
        )

        if not is_valid or formatted_df is None or formatted_df.empty:
            primary_error = validation_errors[0] if validation_errors else "Validation failure: Invalid vehicle input."
            return ValuationWorkflowResult(
                success=False,
                status="VALIDATION_ERROR",
                error=primary_error,
                errors=validation_errors,
                valuation=None,
                range=None,
                explanation=None,
                data_quality=None,
                comparables=None,
                market_summary=None,
                audit=None,
                limitations=list(STANDARD_VALUATION_LIMITATIONS),
            )

        # Steps 3 through 9: Execute via existing VehicleValuationService
        try:
            val_result = self.valuation_service.valuate(
                vehicle_data=formatted_df,
                top_k_factors=top_k_factors,
                top_k_comparables=top_k_comparables,
                percentile_lower=percentile_lower,
                percentile_upper=percentile_upper,
                exclude_listing_id=exclude_listing_id,
            )
        except ValidationError as exc:
            if raise_on_error:
                raise
            return ValuationWorkflowResult(
                success=False,
                status="VALIDATION_ERROR",
                error=str(exc),
                errors=[str(exc)],
                valuation=None,
                range=None,
                explanation=None,
                data_quality=None,
                comparables=None,
                market_summary=None,
                audit=None,
                limitations=list(STANDARD_VALUATION_LIMITATIONS),
            )
        except Exception as exc:
            logger.error(f"Valuation workflow execution error: {exc}", exc_info=True)
            if raise_on_error:
                raise
            return ValuationWorkflowResult(
                success=False,
                status="PROCESSING_ERROR",
                error=f"Valuation processing failed: {str(exc)}",
                errors=[str(exc)],
                valuation=None,
                range=None,
                explanation=None,
                data_quality=None,
                comparables=None,
                market_summary=None,
                audit=None,
                limitations=list(STANDARD_VALUATION_LIMITATIONS),
            )

        # Step 10: Assembly of Unified Valuation Result
        dq_dict = val_result.data_quality or {}
        is_partial = dq_dict.get("status") == "PARTIAL"
        status_label = "PARTIAL_DATA" if is_partial else "SUCCESS"

        return ValuationWorkflowResult.from_valuation_result(
            val_result=val_result,
            status=status_label,
        )

    def run(self, *args, **kwargs) -> ValuationWorkflowResult:
        """Alias for execute()."""
        return self.execute(*args, **kwargs)

    def valuate(self, *args, **kwargs) -> ValuationWorkflowResult:
        """Alias for execute()."""
        return self.execute(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> ValuationWorkflowResult:
        """Allows calling workflow instance directly as a callable."""
        return self.execute(*args, **kwargs)


# ---------------------------------------------------------------------------
# Convenience Module Function
# ---------------------------------------------------------------------------

def run_valuation_workflow(
    vehicle_data: Union[Dict[str, Any], pd.DataFrame, pd.Series, BaseModel],
    valuation_service: Optional[VehicleValuationService] = None,
    session: Optional[Session] = None,
    top_k_factors: int = 5,
    top_k_comparables: int = 5,
    percentile_lower: int = 10,
    percentile_upper: int = 90,
    exclude_listing_id: Optional[str] = None,
    raise_on_error: bool = False,
) -> ValuationWorkflowResult:
    """
    Convenience functional interface to execute the unified vehicle valuation workflow.
    """
    workflow = VehicleValuationWorkflow(
        valuation_service=valuation_service,
        session=session,
    )
    return workflow.execute(
        vehicle_data=vehicle_data,
        top_k_factors=top_k_factors,
        top_k_comparables=top_k_comparables,
        percentile_lower=percentile_lower,
        percentile_upper=percentile_upper,
        exclude_listing_id=exclude_listing_id,
        raise_on_error=raise_on_error,
    )

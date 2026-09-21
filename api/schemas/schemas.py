"""
API Pydantic Schemas for Vehicle Valuation Service.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from data_pipeline.cleaning.cleaners import VehicleCleaner
from ml.prediction.predictor import CANONICAL_CATEGORIES


class VehicleValuationRequest(BaseModel):
    """
    Request schema for estimating vehicle asking price.
    """
    category: str = Field(..., description="Canonical vehicle category (e.g. Cars, SUVs, Vans, Motorbikes)")
    brand: str = Field(..., min_length=1, description="Vehicle brand/make (e.g. Toyota, Honda, Nissan)")
    model: str = Field(..., min_length=1, description="Vehicle model (e.g. Premio, Vezel, Fit)")
    vehicle_age: Optional[float] = Field(None, ge=0, le=100, description="Vehicle age in years (relative to 2026)")
    manufacture_year: Optional[int] = Field(None, ge=1920, le=2026, description="Year of vehicle manufacture")
    mileage: Optional[float] = Field(None, ge=0, description="Vehicle odometer reading in kilometers")
    engine_cc: Optional[float] = Field(None, ge=0, description="Engine capacity in cubic centimeters (cc)")
    fuel_type: str = Field(..., min_length=1, description="Fuel type (e.g. Petrol, Diesel, Hybrid, Electric)")
    transmission: str = Field(..., min_length=1, description="Transmission (e.g. Automatic, Manual)")
    district: str = Field(..., min_length=1, description="Sri Lankan administrative district (e.g. Colombo, Kandy)")
    condition: str = Field(..., min_length=1, description="Condition (e.g. Registered (Used), Unregistered, Brand New)")
    registration_year: Optional[int] = Field(None, ge=1920, le=2026, description="Optional year of registration")

    top_k_factors: int = Field(5, ge=1, le=20, description="Number of top explainability features to return")
    top_k_comparables: int = Field(5, ge=0, le=20, description="Number of comparable market listings to return")
    percentile_lower: int = Field(10, ge=1, le=49, description="Lower bound percentile for prediction range")
    percentile_upper: int = Field(90, ge=51, le=99, description="Upper bound percentile for prediction range")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        clean = VehicleCleaner.canonicalize_category(v.strip())
        if not clean or clean not in CANONICAL_CATEGORIES:
            raise ValueError(
                f"Invalid vehicle category '{v}'. Must be one of: {sorted(list(CANONICAL_CATEGORIES))}"
            )
        return clean

    @field_validator("brand", "model", "fuel_type", "transmission", "district", "condition")
    @classmethod
    def validate_non_empty_strings(cls, v: str, info) -> str:
        s = v.strip()
        if not s:
            raise ValueError(f"Field '{info.field_name}' cannot be empty or whitespace.")
        return s

    @model_validator(mode="after")
    def validate_age_or_year(self) -> "VehicleValuationRequest":
        if self.vehicle_age is None and self.manufacture_year is None:
            raise ValueError("Either 'vehicle_age' or 'manufacture_year' must be provided.")
        return self


class PredictionRangeResponse(BaseModel):
    """
    Indicative model-based prediction range representing empirical RandomForest tree dispersion.
    Not a statistically guaranteed confidence interval.
    """
    estimate: float
    lower: float
    upper: float
    spread: float
    percentile_lower: int
    percentile_upper: int
    method: str


class ExplanationFactorResponse(BaseModel):
    feature: str
    value: str
    contribution: float
    direction: str
    description: str


class ComparableVehicleResponse(BaseModel):
    listing_id: str
    category: str
    brand: str
    model: str
    manufacture_year: Optional[int] = None
    mileage: Optional[float] = None
    engine_cc: Optional[float] = None
    fuel_type: Optional[str] = None
    transmission: Optional[str] = None
    district: Optional[str] = None
    condition: Optional[str] = None
    asking_price: float
    similarity_score: float
    similarity_percentage: float


class ModelMetadataResponse(BaseModel):
    name: str
    target_variable: str
    target_transform: str
    training_records: int
    test_samples: int
    status: str


class VehicleValuationResponse(BaseModel):
    estimated_asking_price_lkr: float
    prediction_range_lkr: PredictionRangeResponse
    currency: str = "LKR"
    model: ModelMetadataResponse
    explanation: List[ExplanationFactorResponse]
    comparables: List[ComparableVehicleResponse]
    limitations: List[str]


class ErrorResponse(BaseModel):
    detail: str
    error_type: str

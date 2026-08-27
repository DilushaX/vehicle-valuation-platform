"""
API Schemas Module
Pydantic schemas for request validation and structured API responses.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# --- Valuation Schemas ---

class ValuationRequest(BaseModel):
    category: str = Field(default="Cars", description="Vehicle Category (e.g. Cars, Vans, SUVs, Motorbikes, Three-Wheel, Lorries, Buses)")
    make: str = Field(..., json_schema_extra={"example": "Toyota"}, description="Vehicle Manufacturer")
    model: str = Field(..., json_schema_extra={"example": "Aqua"}, description="Vehicle Model")
    yom: int = Field(..., ge=1970, le=2026, json_schema_extra={"example": 2018}, description="Year of Manufacture")
    mileage_km: int = Field(..., ge=0, json_schema_extra={"example": 65000}, description="Mileage in kilometers")
    fuel_type: str = Field(default="Petrol", json_schema_extra={"example": "Hybrid"}, description="Fuel Type")
    transmission: str = Field(default="Automatic", json_schema_extra={"example": "Automatic"}, description="Transmission Type")
    condition: str = Field(default="Used", json_schema_extra={"example": "Used"}, description="Condition (Brand New, Reconditioned, Used)")
    engine_cc: Optional[int] = Field(default=1500, json_schema_extra={"example": 1500}, description="Engine Capacity in cc")
    district: Optional[str] = Field(default="Colombo", json_schema_extra={"example": "Colombo"}, description="Sri Lankan District")
    seller_asking_price: Optional[float] = Field(default=None, json_schema_extra={"example": 8500000.0}, description="Seller asking price in LKR")
    include_shap_explanation: bool = Field(default=True, description="Whether to compute SHAP factor contributions")
    include_negotiation_insights: bool = Field(default=True, description="Whether to include data-driven negotiation insights")

class PriceRange(BaseModel):
    low: float
    high: float

class ShapContribution(BaseModel):
    feature: str
    impact_rs: float
    impact_direction: str
    impact_strength: str
    abs_impact: float

class AskingPriceAssessment(BaseModel):
    classification: str  # BELOW MARKET RANGE, WITHIN MARKET RANGE, ABOVE MARKET RANGE
    difference_percent: float
    difference_amount: Optional[float]

class ValuationResponse(BaseModel):
    status: str
    category: str
    make: str
    model: str
    yom: int
    mileage_km: int
    estimated_market_asking_value: Optional[float] = None
    estimated_market_range: Optional[PriceRange] = None
    confidence: Optional[str] = None
    similar_listings_count: int = 0
    price_assessment: Optional[AskingPriceAssessment] = None
    shap_explanation: Optional[Dict[str, Any]] = None
    negotiation_insights: Optional[Dict[str, Any]] = None
    disclaimer: str = (
        "Important: The system estimates market asking values based on observed vehicle listings. "
        "It does not claim to predict actual transaction prices, negotiated prices, or guaranteed vehicle values."
    )

# --- Comparable Vehicles Schemas ---

class ComparableSearchRequest(BaseModel):
    category: str = Field(default="Cars", json_schema_extra={"example": "Cars"})
    make: str = Field(..., json_schema_extra={"example": "Toyota"})
    model: str = Field(..., json_schema_extra={"example": "Aqua"})
    yom: int = Field(..., json_schema_extra={"example": 2018})
    mileage_km: int = Field(..., json_schema_extra={"example": 65000})
    fuel_type: Optional[str] = Field(default="Hybrid")
    transmission: Optional[str] = Field(default="Automatic")
    district: Optional[str] = Field(default="Colombo")
    top_k: int = Field(default=6, ge=1, le=20)

class ComparableVehicleItem(BaseModel):
    listing_id: str
    title: str
    url: Optional[str]
    make: str
    model: str
    yom: int
    mileage_km: Optional[int]
    fuel_type: Optional[str]
    transmission: Optional[str]
    district: Optional[str]
    condition: Optional[str]
    price_rs: float
    similarity_score: float
    status: str

class ComparableSearchResponse(BaseModel):
    count: int
    median_price: float
    avg_price: float
    p25_price: float
    p75_price: float
    min_price: float
    max_price: float
    comparables: List[ComparableVehicleItem]

# --- Analytics Schemas ---

class MarketOverviewResponse(BaseModel):
    total_listings: int
    active_listings: int
    valid_listings: int
    median_price: float
    avg_price: float
    min_price: float
    max_price: float
    top_brands: List[Dict[str, Any]]
    top_models: List[Dict[str, Any]]

# --- Data Quality & Scraper Schemas ---

class DataQualitySummaryResponse(BaseModel):
    total_records: int
    valid_records: int
    suspicious_records: int
    invalid_records: int
    missing_records: int
    duplicates_detected: int
    overall_quality_score: float
    last_scrape_run: Optional[Dict[str, Any]] = None

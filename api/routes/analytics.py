"""
Market Analytics API Routes Module
Endpoints for overview KPIs, brand analysis, model rankings, depreciation,
fuel & transmission, district pricing, and historical price movements.
"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from database.connection import get_db
from api.schemas.schemas import MarketOverviewResponse
from analytics.market.market_analytics import MarketAnalyticsEngine
from analytics.trends.trend_engine import MarketTrendEngine

router = APIRouter(prefix="/api/v1/analytics", tags=["Market Analytics"])

@router.get("/overview", response_model=MarketOverviewResponse)
def get_market_overview(category: Optional[str] = None, db: Session = Depends(get_db)):
    engine = MarketAnalyticsEngine(db=db)
    return engine.get_market_overview(category=category)

@router.get("/brands")
def get_brand_analysis(
    category: Optional[str] = None,
    top_n: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    engine = MarketAnalyticsEngine(db=db)
    return engine.get_brand_analysis(category=category, top_n=top_n)

@router.get("/models")
def get_model_analysis(
    make: Optional[str] = None,
    category: Optional[str] = None,
    top_n: int = Query(default=15, ge=1, le=50),
    db: Session = Depends(get_db)
):
    engine = MarketAnalyticsEngine(db=db)
    return engine.get_model_analysis(make=make, category=category, top_n=top_n)

@router.get("/depreciation")
def get_depreciation_analysis(
    make: Optional[str] = None,
    model: Optional[str] = None,
    category: Optional[str] = "Cars",
    db: Session = Depends(get_db)
):
    engine = MarketAnalyticsEngine(db=db)
    return engine.get_depreciation_analysis(make=make, model=model, category=category)

@router.get("/fuel-transmission")
def get_fuel_and_transmission_breakdown(category: Optional[str] = None, db: Session = Depends(get_db)):
    engine = MarketAnalyticsEngine(db=db)
    return engine.get_fuel_and_transmission_breakdown(category=category)

@router.get("/districts")
def get_district_pricing(category: Optional[str] = None, db: Session = Depends(get_db)):
    engine = MarketAnalyticsEngine(db=db)
    return engine.get_district_pricing_heatmap(category=category)

@router.get("/trends/price-movements")
def get_price_movements(days: int = Query(default=30, ge=1, le=365), db: Session = Depends(get_db)):
    engine = MarketTrendEngine(db=db)
    return engine.get_price_movement_summary(days=days)

@router.get("/trends/volume")
def get_volume_trends(db: Session = Depends(get_db)):
    engine = MarketTrendEngine(db=db)
    return engine.get_listing_volume_trends()

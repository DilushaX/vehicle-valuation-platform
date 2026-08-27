"""
Comparable Vehicles API Routes Module
Endpoints for searching similar vehicle listings and market summary metrics.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database.connection import get_db
from api.schemas.schemas import ComparableSearchRequest, ComparableSearchResponse
from analytics.comparables.comparable_engine import ComparableEngine

router = APIRouter(prefix="/api/v1/comparables", tags=["Comparable Vehicles"])

@router.post("/search", response_model=ComparableSearchResponse)
def search_comparable_vehicles(req: ComparableSearchRequest, db: Session = Depends(get_db)):
    """
    Finds top matching active/historical vehicle listings using multi-attribute similarity scoring.
    """
    engine = ComparableEngine(db=db)
    result = engine.find_comparables(
        category=req.category,
        make=req.make,
        model=req.model,
        yom=req.yom,
        mileage_km=req.mileage_km,
        fuel_type=req.fuel_type,
        transmission=req.transmission,
        district=req.district,
        top_k=req.top_k
    )
    return result

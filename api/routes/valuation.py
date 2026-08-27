"""
Valuation API Routes Module
Endpoints for ML Vehicle Price Valuation, SHAP Explainability, and Negotiation Insights.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database.connection import get_db
from api.schemas.schemas import ValuationRequest, ValuationResponse
from ml.prediction.predictor import VehicleValuationPredictor
from ml.explainability.explainer import ValuationExplainer
from ml.negotiation.insights import NegotiationInsightEngine
from analytics.comparables.comparable_engine import ComparableEngine

router = APIRouter(prefix="/api/v1/valuation", tags=["Valuation & ML"])

@router.post("/predict", response_model=ValuationResponse)
def predict_valuation(req: ValuationRequest, db: Session = Depends(get_db)):
    """
    Estimates Market Asking Value, Estimated Range, and Confidence rating using ML.
    Optionally computes SHAP feature importance impact and negotiation insights.
    """
    predictor = VehicleValuationPredictor(db=db)
    result = predictor.predict_value(
        category=req.category,
        make=req.make,
        model=req.model,
        yom=req.yom,
        mileage_km=req.mileage_km,
        fuel_type=req.fuel_type,
        transmission=req.transmission,
        condition=req.condition,
        engine_cc=req.engine_cc,
        district=req.district,
        seller_asking_price=req.seller_asking_price
    )

    if result.get("status") == "INSUFFICIENT_DATA":
        return ValuationResponse(
            status="INSUFFICIENT_DATA",
            category=req.category,
            make=req.make,
            model=req.model,
            yom=req.yom,
            mileage_km=req.mileage_km,
            disclaimer=result.get("message", "Insufficient market data to estimate value.")
        )
    elif result.get("status") == "ERROR":
        raise HTTPException(status_code=500, detail=result.get("message"))

    # Optional SHAP explanation
    shap_res = None
    if req.include_shap_explanation:
        explainer = ValuationExplainer()
        shap_res = explainer.explain_prediction(
            category=req.category,
            vehicle_dict={
                "make": req.make,
                "model": req.model,
                "yom": req.yom,
                "mileage_km": req.mileage_km,
                "fuel_type": req.fuel_type,
                "transmission": req.transmission,
                "condition": req.condition,
                "engine_cc": req.engine_cc,
                "district": req.district
            }
        )

    # Optional Negotiation Insights
    neg_insights = None
    if req.include_negotiation_insights and req.seller_asking_price:
        comp_engine = ComparableEngine(db=db)
        comps = comp_engine.find_comparables(
            category=req.category,
            make=req.make,
            model=req.model,
            yom=req.yom,
            mileage_km=req.mileage_km,
            fuel_type=req.fuel_type,
            transmission=req.transmission,
            district=req.district,
            top_k=5
        )

        neg_insights = NegotiationInsightEngine.generate_negotiation_insights(
            estimated_value=result["estimated_market_asking_value"],
            range_low=result["estimated_market_range"]["low"],
            range_high=result["estimated_market_range"]["high"],
            seller_asking_price=req.seller_asking_price,
            comparable_median_price=comps.get("median_price"),
            comparable_count=comps.get("count", 0)
        )

    return ValuationResponse(
        status="SUCCESS",
        category=req.category,
        make=req.make,
        model=req.model,
        yom=req.yom,
        mileage_km=req.mileage_km,
        estimated_market_asking_value=result.get("estimated_market_asking_value"),
        estimated_market_range=result.get("estimated_market_range"),
        confidence=result.get("confidence"),
        similar_listings_count=result.get("similar_listings_count", 0),
        price_assessment=result.get("price_assessment"),
        shap_explanation=shap_res,
        negotiation_insights=neg_insights
    )

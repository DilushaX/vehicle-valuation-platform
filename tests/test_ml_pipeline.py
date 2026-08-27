"""
Tests for ML Valuation Predictor and SHAP Explainability
"""
import pytest
from ml.prediction.predictor import VehicleValuationPredictor
from ml.explainability.explainer import ValuationExplainer

def test_ml_prediction_inference():
    predictor = VehicleValuationPredictor()
    res = predictor.predict_value(
        category="Cars",
        make="Toyota",
        model="Aqua",
        yom=2018,
        mileage_km=65000,
        fuel_type="Hybrid",
        transmission="Automatic",
        district="Colombo",
        seller_asking_price=8500000.0
    )

    assert res["status"] == "SUCCESS"
    assert res["estimated_market_asking_value"] > 4000000.0
    assert res["estimated_market_range"]["low"] < res["estimated_market_asking_value"]
    assert res["estimated_market_range"]["high"] > res["estimated_market_asking_value"]
    assert res["confidence"] in ["High", "Medium", "Low"]
    assert res["price_assessment"] is not None
    assert res["price_assessment"]["classification"] in ["BELOW MARKET RANGE", "WITHIN MARKET RANGE", "ABOVE MARKET RANGE"]

def test_shap_explanation():
    explainer = ValuationExplainer()
    shap_res = explainer.explain_prediction(
        category="Cars",
        vehicle_dict={
            "make": "Toyota",
            "model": "Aqua",
            "yom": 2018,
            "mileage_km": 65000,
            "fuel_type": "Hybrid",
            "transmission": "Automatic",
            "district": "Colombo"
        }
    )

    assert shap_res["status"] == "SUCCESS"
    assert len(shap_res["contributions"]) > 0
    assert "feature" in shap_res["contributions"][0]
    assert "impact_rs" in shap_res["contributions"][0]

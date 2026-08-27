"""
Integration Tests for FastAPI Backend Endpoints
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "documentation" in response.json()

def test_valuation_predict_api():
    payload = {
        "category": "Cars",
        "make": "Toyota",
        "model": "Aqua",
        "yom": 2018,
        "mileage_km": 65000,
        "fuel_type": "Hybrid",
        "transmission": "Automatic",
        "district": "Colombo",
        "seller_asking_price": 8500000.0,
        "include_shap_explanation": True,
        "include_negotiation_insights": True
    }
    response = client.post("/api/v1/valuation/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["estimated_market_asking_value"] > 0
    assert data["estimated_market_range"]["low"] > 0
    assert data["price_assessment"] is not None

def test_comparables_search_api():
    payload = {
        "category": "Cars",
        "make": "Toyota",
        "model": "Aqua",
        "yom": 2018,
        "mileage_km": 65000,
        "fuel_type": "Hybrid",
        "transmission": "Automatic",
        "district": "Colombo",
        "top_k": 5
    }
    response = client.post("/api/v1/comparables/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "comparables" in data

def test_analytics_overview_api():
    response = client.get("/api/v1/analytics/overview")
    assert response.status_code == 200
    data = response.json()
    assert "total_listings" in data

def test_quality_summary_api():
    response = client.get("/api/v1/quality/summary")
    assert response.status_code == 200
    data = response.json()
    assert "overall_quality_score" in data

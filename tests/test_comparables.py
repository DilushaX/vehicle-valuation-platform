"""
Tests for Comparable Vehicle Similarity Engine
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, Listing
from analytics.comparables.comparable_engine import ComparableEngine

@pytest.fixture
def comp_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Seed sample vehicles
    samples = [
        Listing(
            listing_id="c_1", category="Cars", make="Toyota", model="Aqua",
            yom=2018, mileage_km=65000, fuel_type="Hybrid", transmission="Automatic",
            district="Colombo", current_price=8200000.0, current_status="ACTIVE",
            data_quality_status="VALID"
        ),
        Listing(
            listing_id="c_2", category="Cars", make="Toyota", model="Aqua",
            yom=2017, mileage_km=70000, fuel_type="Hybrid", transmission="Automatic",
            district="Gampaha", current_price=7900000.0, current_status="ACTIVE",
            data_quality_status="VALID"
        ),
        Listing(
            listing_id="c_3", category="Cars", make="Toyota", model="Prius",
            yom=2015, mileage_km=110000, fuel_type="Hybrid", transmission="Automatic",
            district="Kandy", current_price=9500000.0, current_status="ACTIVE",
            data_quality_status="VALID"
        )
    ]
    session.add_all(samples)
    session.commit()
    yield session
    session.close()

def test_comparable_scoring_and_ranking(comp_db):
    engine = ComparableEngine(comp_db)
    result = engine.find_comparables(
        category="Cars",
        make="Toyota",
        model="Aqua",
        yom=2018,
        mileage_km=65000,
        fuel_type="Hybrid",
        transmission="Automatic",
        district="Colombo",
        top_k=5
    )

    assert result["count"] >= 2
    comps = result["comparables"]
    
    # Exact match should have the highest similarity score
    assert comps[0]["listing_id"] == "c_1"
    assert comps[0]["similarity_score"] > 90.0
    assert comps[1]["listing_id"] == "c_2"
    assert comps[0]["similarity_score"] >= comps[1]["similarity_score"]
    assert result["median_price"] > 0

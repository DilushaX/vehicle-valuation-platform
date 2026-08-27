"""
Tests for Database Layer, Delta Tracking, and Price History
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, Listing, PriceHistory, ListingStatusHistory
from database.repository import VehicleRepository

@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_upsert_listing_new_and_price_change(test_db):
    repo = VehicleRepository(test_db)
    
    # 1. Initial New Listing
    data1 = {
        "listing_id": "riyasewana_99901",
        "category": "Cars",
        "make": "Toyota",
        "model": "Aqua",
        "yom": 2018,
        "mileage_km": 65000,
        "fuel_type": "Hybrid",
        "transmission": "Automatic",
        "district": "Colombo",
        "price_rs": 8500000.0,
        "url": "https://riyasewana.com/buy/toyota-aqua-99901"
    }

    listing, action1 = repo.upsert_listing(data1)
    assert action1 == "NEW"
    assert listing.current_price == 8500000.0
    assert listing.current_status == "ACTIVE"

    # Verify initial price history
    prices = test_db.query(PriceHistory).filter(PriceHistory.listing_id == listing.id).all()
    assert len(prices) == 1
    assert prices[0].price == 8500000.0

    # 2. Price Change Observation
    data2 = data1.copy()
    data2["price_rs"] = 8300000.0  # Rs. 200k price drop

    updated_listing, action2 = repo.upsert_listing(data2)
    assert action2 == "PRICE_CHANGED"
    assert updated_listing.current_price == 8300000.0

    # Verify updated price history
    prices2 = test_db.query(PriceHistory).filter(PriceHistory.listing_id == listing.id).all()
    assert len(prices2) == 2
    assert prices2[1].price == 8300000.0
    assert prices2[1].change_amount == -200000.0

def test_mark_unseen_as_inactive(test_db):
    from datetime import datetime, timedelta, timezone
    repo = VehicleRepository(test_db)
    
    data = {
        "listing_id": "riyasewana_99902",
        "category": "Cars",
        "make": "Suzuki",
        "model": "Wagon R",
        "yom": 2017,
        "mileage_km": 50000,
        "fuel_type": "Hybrid",
        "transmission": "Automatic",
        "district": "Gampaha",
        "price_rs": 6000000.0
    }
    listing, _ = repo.upsert_listing(data)

    # Scrape run started after listing was last seen
    run_start = datetime.now(timezone.utc) + timedelta(minutes=5)
    
    # Run crawl where listing 99902 is NOT present
    inactive_count = repo.mark_unseen_as_inactive(
        category="Cars",
        seen_listing_ids=["riyasewana_99999"],
        run_start_time=run_start
    )
    assert inactive_count == 1
    
    refreshed = test_db.query(Listing).filter(Listing.listing_id == "riyasewana_99902").first()
    assert refreshed.current_status == "NO_LONGER_OBSERVED"

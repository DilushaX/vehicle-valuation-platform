import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import select

from database.models import (
    Vehicle,
    Listing,
    PriceHistory,
    ScrapeRun,
    ListingObservation,
)
from database.repository import VehicleRepository


def to_utc(dt: datetime) -> datetime:
    """Helper to ensure consistent UTC comparison across SQLite and Postgres dialects."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def test_init_does_not_import_database_connection():
    """
    Verifies tests/__init__.py does not load database.connection on package import.
    """
    init_path = Path(__file__).parent / "__init__.py"
    content = init_path.read_text()
    assert "database.connection" not in content
    assert "from database.connection import Base" not in content


def test_condition_is_saved_and_retrieved(vehicle_repo: VehicleRepository):
    """
    FIX 1: Verifies vehicle condition (Used, Reconditioned, Brand New)
    is properly saved and retrieved via Vehicle and Listing.
    """
    data = {
        "listing_id": "test_cond_01",
        "listing_url": "https://riyasewana.com/buy/test-cond-01",
        "title": "Toyota Prius Reconditioned 2020",
        "category": "Cars",
        "brand": "Toyota",
        "model": "Prius",
        "manufacture_year": 2020,
        "registration_year": 2021,
        "fuel_type": "Hybrid",
        "transmission": "Automatic",
        "engine_cc": 1800,
        "condition": "Reconditioned",
        "price": 14500000,
        "mileage": 15000,
    }

    listing, is_new = vehicle_repo.sync_listing(data)
    vehicle_repo.commit()

    assert is_new is True
    assert listing.vehicle.condition == "Reconditioned"
    assert listing.condition == "Reconditioned"

    # Query afresh from DB
    retrieved = vehicle_repo.find_listing("test_cond_01")
    assert retrieved is not None
    assert retrieved.condition == "Reconditioned"
    assert retrieved.vehicle.condition == "Reconditioned"


def test_first_listing_creates_one_listing_and_observation(vehicle_repo: VehicleRepository):
    """
    FIX 2: First observation creates exactly one Vehicle, one Listing,
    initial PriceHistory, and one ListingObservation.
    """
    scrape_run = vehicle_repo.create_scrape_run(category="Cars")
    data = {
        "listing_id": "test_sync_01",
        "listing_url": "https://riyasewana.com/buy/test-sync-01",
        "title": "Toyota Axio 2016",
        "category": "Cars",
        "brand": "Toyota",
        "model": "Axio",
        "manufacture_year": 2016,
        "registration_year": 2017,
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "engine_cc": 1500,
        "condition": "Used",
        "price": 12000000,
        "mileage": 118000,
        "is_valid": True,
        "validation_issues": [],
    }

    listing, is_new = vehicle_repo.sync_listing(data, scrape_run=scrape_run)
    vehicle_repo.commit()

    assert is_new is True
    assert listing.current_status == "ACTIVE"
    assert listing.first_seen_at is not None
    assert listing.last_seen_at is not None
    assert len(listing.price_history) == 1
    assert listing.price_history[0].price == 12000000
    assert len(listing.observations) == 1
    assert listing.observations[0].observed_price == 12000000
    assert listing.observations[0].observed_mileage == 118000


def test_same_listing_on_second_scrape_does_not_duplicate_listing(vehicle_repo: VehicleRepository):
    """
    FIX 2: Second observation of the same listing_id does NOT create
    another Listing or another Vehicle record.
    """
    scrape_run_1 = vehicle_repo.create_scrape_run(category="Cars")
    scrape_run_2 = vehicle_repo.create_scrape_run(category="Cars")

    data = {
        "listing_id": "test_sync_dup",
        "listing_url": "https://riyasewana.com/buy/test-sync-dup",
        "title": "Toyota Axio 2016",
        "brand": "Toyota",
        "model": "Axio",
        "price": 12000000,
        "mileage": 118000,
    }

    # Day 1
    listing_1, is_new_1 = vehicle_repo.sync_listing(data, scrape_run=scrape_run_1)
    vehicle_repo.commit()
    assert is_new_1 is True

    # Day 2
    listing_2, is_new_2 = vehicle_repo.sync_listing(data, scrape_run=scrape_run_2)
    vehicle_repo.commit()
    assert is_new_2 is False

    # Verify counts in DB
    all_listings = vehicle_repo.db.scalars(
        select(Listing).where(Listing.listing_id == "test_sync_dup")
    ).all()
    assert len(all_listings) == 1
    assert listing_1.id == listing_2.id

    all_vehicles = vehicle_repo.db.scalars(select(Vehicle)).all()
    assert len(all_vehicles) == 1


def test_same_listing_updates_last_seen_at(vehicle_repo: VehicleRepository):
    """
    FIX 2: Second observation updates last_seen_at while preserving first_seen_at.
    """
    t_day1 = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)
    t_day2 = datetime(2026, 9, 2, 10, 0, 0, tzinfo=timezone.utc)

    data = {
        "listing_id": "test_time_sync",
        "listing_url": "https://riyasewana.com/buy/test-time-sync",
        "title": "Honda Vezel 2015",
        "brand": "Honda",
        "model": "Vezel",
        "price": 10500000,
    }

    listing, _ = vehicle_repo.sync_listing(data, observed_at=t_day1)
    vehicle_repo.commit()
    assert to_utc(listing.first_seen_at) == to_utc(t_day1)
    assert to_utc(listing.last_seen_at) == to_utc(t_day1)

    # Day 2
    listing, _ = vehicle_repo.sync_listing(data, observed_at=t_day2)
    vehicle_repo.commit()
    assert to_utc(listing.first_seen_at) == to_utc(t_day1)
    assert to_utc(listing.last_seen_at) == to_utc(t_day2)


def test_same_price_does_not_create_duplicate_price_history(vehicle_repo: VehicleRepository):
    """
    FIX 3 (Case 1): If the price remains unchanged across consecutive days,
    do NOT add duplicate price_history rows.
    """
    data = {
        "listing_id": "test_price_unchanged",
        "listing_url": "https://riyasewana.com/buy/test-price-unchanged",
        "title": "Toyota Vitz 2018",
        "brand": "Toyota",
        "model": "Vitz",
        "price": 6500000,
        "mileage": 50000,
    }

    run1 = vehicle_repo.create_scrape_run()
    run2 = vehicle_repo.create_scrape_run()
    run3 = vehicle_repo.create_scrape_run()

    # Day 1: 6.5M
    listing, _ = vehicle_repo.sync_listing(data, scrape_run=run1)
    # Day 2: 6.5M (unchanged)
    listing, _ = vehicle_repo.sync_listing(data, scrape_run=run2)
    # Day 3: 6.5M (unchanged)
    listing, _ = vehicle_repo.sync_listing(data, scrape_run=run3)
    vehicle_repo.commit()

    # Should have 3 observations
    assert len(listing.observations) == 3
    # But ONLY 1 price_history row!
    assert len(listing.price_history) == 1
    assert listing.price_history[0].price == 6500000


def test_price_changes_recorded_accurately(vehicle_repo: VehicleRepository):
    """
    FIX 3 (Case 2): Price drops and price increases trigger new price_history records.
    """
    base_data = {
        "listing_id": "test_price_changes",
        "listing_url": "https://riyasewana.com/buy/test-price-changes",
        "title": "Toyota Premio 2017",
        "brand": "Toyota",
        "model": "Premio",
        "mileage": 70000,
    }

    # Day 1: 12,000,000
    d1 = dict(base_data, price=12000000)
    listing, _ = vehicle_repo.sync_listing(d1, scrape_run=vehicle_repo.create_scrape_run())

    # Day 2: 12,000,000 (same)
    d2 = dict(base_data, price=12000000)
    vehicle_repo.sync_listing(d2, scrape_run=vehicle_repo.create_scrape_run())

    # Day 3: 11,800,000 (price drop)
    d3 = dict(base_data, price=11800000)
    vehicle_repo.sync_listing(d3, scrape_run=vehicle_repo.create_scrape_run())

    # Day 4: 11,800,000 (same)
    d4 = dict(base_data, price=11800000)
    vehicle_repo.sync_listing(d4, scrape_run=vehicle_repo.create_scrape_run())

    # Day 5: 11,500,000 (price drop)
    d5 = dict(base_data, price=11500000)
    vehicle_repo.sync_listing(d5, scrape_run=vehicle_repo.create_scrape_run())

    # Day 6: 11,900,000 (price increase)
    d6 = dict(base_data, price=11900000)
    vehicle_repo.sync_listing(d6, scrape_run=vehicle_repo.create_scrape_run())

    vehicle_repo.commit()

    # 6 observations total
    assert len(listing.observations) == 6

    # 4 distinct price points in price_history: 12.0M -> 11.8M -> 11.5M -> 11.9M
    assert len(listing.price_history) == 4
    prices = [ph.price for ph in listing.price_history]
    assert prices == [12000000, 11800000, 11500000, 11900000]


def test_listing_disappearance_marks_no_longer_observed(vehicle_repo: VehicleRepository):
    """
    FIX 3 (Case 3) & FIX 5: When a listing is missing in subsequent scrapes,
    mark as NO_LONGER_OBSERVED without deleting historical data.
    """
    data_a = {
        "listing_id": "listing_staying",
        "listing_url": "https://riyasewana.com/buy/staying",
        "title": "Car A",
        "category": "Cars",
        "brand": "Toyota",
        "price": 8000000,
    }
    data_b = {
        "listing_id": "listing_disappearing",
        "listing_url": "https://riyasewana.com/buy/disappearing",
        "title": "Car B",
        "category": "Cars",
        "brand": "Nissan",
        "price": 6000000,
    }

    run1 = vehicle_repo.create_scrape_run(category="Cars")
    listing_a, _ = vehicle_repo.sync_listing(data_a, scrape_run=run1)
    listing_b, _ = vehicle_repo.sync_listing(data_b, scrape_run=run1)
    vehicle_repo.commit()

    # Day 2: Only Car A was observed on the website
    marked = vehicle_repo.mark_unobserved_listings(
        observed_listing_ids=["listing_staying"],
        source="riyasewana",
        category="Cars",
    )
    vehicle_repo.commit()

    assert len(marked) == 1
    assert marked[0].listing_id == "listing_disappearing"
    assert marked[0].current_status == "NO_LONGER_OBSERVED"

    # Car A remains ACTIVE
    assert listing_a.current_status == "ACTIVE"

    # Car B historical data is completely preserved
    b_found = vehicle_repo.find_listing("listing_disappearing")
    assert b_found is not None
    assert b_found.current_status == "NO_LONGER_OBSERVED"
    assert len(b_found.price_history) == 1
    assert len(b_found.observations) == 1
    assert b_found.first_seen_at is not None
    assert b_found.last_seen_at is not None


def test_no_longer_observed_listing_returns_to_active(vehicle_repo: VehicleRepository):
    """
    FIX 3 (Case 4): If a NO_LONGER_OBSERVED listing reappears:
    - Set current_status = ACTIVE
    - Update last_seen_at
    - Do NOT reset first_seen_at
    - Record new observation
    - Record price_history only if price differs
    """
    t_day1 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    t_day3 = datetime(2026, 9, 3, 10, 0, tzinfo=timezone.utc)

    data = {
        "listing_id": "test_returning_car",
        "listing_url": "https://riyasewana.com/buy/returning",
        "title": "Suzuki Alto 2015",
        "brand": "Suzuki",
        "model": "Alto",
        "price": 3200000,
    }

    # Day 1: Created ACTIVE
    listing, _ = vehicle_repo.sync_listing(data, observed_at=t_day1)
    vehicle_repo.commit()
    assert listing.current_status == "ACTIVE"

    # Day 2: Disappears
    vehicle_repo.mark_listing_no_longer_observed(listing)
    vehicle_repo.commit()
    assert listing.current_status == "NO_LONGER_OBSERVED"

    # Day 3: Returns with a lower price (3.0M)
    d3 = dict(data, price=3000000)
    run3 = vehicle_repo.create_scrape_run()
    listing_ret, is_new = vehicle_repo.sync_listing(
        d3, scrape_run=run3, observed_at=t_day3
    )
    vehicle_repo.commit()

    assert is_new is False
    assert listing_ret.current_status == "ACTIVE"
    assert to_utc(listing_ret.first_seen_at) == to_utc(t_day1)  # NOT reset
    assert to_utc(listing_ret.last_seen_at) == to_utc(t_day3)   # updated to Day 3
    assert len(listing_ret.observations) == 1   # observation from run3
    assert len(listing_ret.price_history) == 2  # initial 3.2M + new 3.0M
    assert listing_ret.price_history[1].price == 3000000


def test_scrape_run_lifecycle_completed(vehicle_repo: VehicleRepository):
    """
    FIX 4: ScrapeRun transitions from RUNNING to COMPLETED when all pages succeed.
    """
    run = vehicle_repo.create_scrape_run(category="Cars", pages_requested=10)
    assert run.status == "RUNNING"
    assert run.completed_at is None

    vehicle_repo.complete_scrape_run(
        scrape_run=run,
        pages_scraped=10,
        failed_pages=0,
        listings_found=150,
        new_listings=20,
        updated_listings=130,
    )
    vehicle_repo.commit()

    assert run.status == "COMPLETED"
    assert run.completed_at is not None
    assert run.pages_scraped == 10
    assert run.failed_pages == 0
    assert run.listings_found == 150


def test_scrape_run_lifecycle_incomplete_when_pages_fail(vehicle_repo: VehicleRepository):
    """
    FIX 4: ScrapeRun transitions to INCOMPLETE if any pages fail or scraped < requested.
    """
    run = vehicle_repo.create_scrape_run(category="Vans", pages_requested=100)
    vehicle_repo.complete_scrape_run(
        scrape_run=run,
        pages_scraped=98,
        failed_pages=2,
        listings_found=800,
        errors="Page 45 timeout; Page 88 502 error",
    )
    vehicle_repo.commit()

    assert run.status == "INCOMPLETE"
    assert run.failed_pages == 2
    assert run.pages_scraped == 98
    assert "Page 45 timeout" in run.errors


def test_scrape_run_lifecycle_failed(vehicle_repo: VehicleRepository):
    """
    FIX 4: ScrapeRun transitions to FAILED on unrecoverable error.
    """
    run = vehicle_repo.create_scrape_run(category="Motorbikes")
    vehicle_repo.fail_scrape_run(run, errors="Fatal network unreachable error")
    vehicle_repo.commit()

    assert run.status == "FAILED"
    assert run.completed_at is not None
    assert "Fatal network" in run.errors


def test_validation_issues_stored_as_valid_json(vehicle_repo: VehicleRepository):
    """
    Verifies that validation_issues are stored as valid JSON strings in the database.
    """
    vehicle_data = {
        "category": "Cars",
        "brand": "Toyota",
        "model": "Corolla Axio",
        "manufacture_year": 2016,
        "registration_year": 2017,
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "engine_cc": 1500,
    }
    vehicle = vehicle_repo.create_vehicle(vehicle_data)

    issues = ["missing_price", "suspicious_mileage"]
    listing_data = {
        "listing_id": "test_json_101",
        "listing_url": "https://riyasewana.com/buy/test-101",
        "title": "Toyota Axio 2016",
        "validation_issues": issues,
        "is_valid": False,
    }
    listing = vehicle_repo.create_listing(vehicle, listing_data)
    vehicle_repo.commit()

    assert isinstance(listing.validation_issues, str)
    parsed = json.loads(listing.validation_issues)
    assert parsed == ["missing_price", "suspicious_mileage"]


def test_validation_issues_read_back_correctly(vehicle_repo: VehicleRepository):
    """
    Verifies that validation_issues can be read back cleanly via helper methods.
    """
    vehicle = vehicle_repo.create_vehicle({"brand": "Honda", "model": "Civic"})
    issues = ["invalid_year", "suspicious_price_pattern"]
    listing = vehicle_repo.create_listing(
        vehicle,
        {
            "listing_id": "test_json_102",
            "listing_url": "https://riyasewana.com/buy/test-102",
            "validation_issues": issues,
            "is_valid": False,
        },
    )
    vehicle_repo.commit()

    assert listing.get_validation_issues() == issues
    assert vehicle_repo.get_validation_issues(listing) == issues

    retrieved = vehicle_repo.find_listing("test_json_102")
    assert retrieved is not None
    assert retrieved.get_validation_issues() == issues
    assert json.loads(retrieved.validation_issues) == issues


def test_scrape_run_and_listing_observation_relationship(vehicle_repo: VehicleRepository):
    """
    Verifies bi-directional relationships between ScrapeRun and ListingObservation.
    """
    vehicle = vehicle_repo.create_vehicle({"brand": "Nissan", "model": "Leaf"})
    listing = vehicle_repo.create_listing(
        vehicle,
        {
            "listing_id": "test_run_201",
            "listing_url": "https://riyasewana.com/buy/test-201",
            "is_valid": True,
        },
    )
    scrape_run = vehicle_repo.create_scrape_run(category="Cars")

    vehicle_repo.add_observation(
        listing=listing,
        scrape_run=scrape_run,
        price=7500000,
        mileage=45000,
    )
    vehicle_repo.commit()

    assert len(scrape_run.observations) == 1
    obs = scrape_run.observations[0]
    assert obs.observed_price == 7500000
    assert obs.observed_mileage == 45000
    assert obs.listing_id == listing.id

    assert obs.scrape_run is not None
    assert obs.scrape_run.id == scrape_run.id
    assert obs.scrape_run.category == "Cars"


def test_existing_relationships_intact(vehicle_repo: VehicleRepository):
    """
    Verifies existing Phase 2 relationships remain functional:
    - Vehicle -> Listing
    - Listing -> PriceHistory
    - Listing -> ListingObservation
    """
    vehicle = vehicle_repo.create_vehicle({
        "category": "Cars",
        "brand": "Suzuki",
        "model": "Wagon R",
        "manufacture_year": 2017,
        "registration_year": 2018,
    })
    listing = vehicle_repo.create_listing(
        vehicle,
        {
            "listing_id": "test_rel_301",
            "listing_url": "https://riyasewana.com/buy/test-301",
            "is_valid": True,
        },
    )
    vehicle_repo.commit()

    assert len(vehicle.listings) == 1
    assert vehicle.listings[0].listing_id == "test_rel_301"
    assert listing.vehicle.brand == "Suzuki"

    vehicle_repo.add_price_history(listing, 5200000)
    vehicle_repo.add_price_history(listing, 5100000)
    vehicle_repo.commit()

    assert len(listing.price_history) == 2
    prices = [ph.price for ph in listing.price_history]
    assert prices == [5200000, 5100000]
    assert listing.price_history[0].listing.listing_id == "test_rel_301"

    scrape_run = vehicle_repo.create_scrape_run(category="Cars")
    vehicle_repo.add_observation(listing, scrape_run, 5100000, 60000)
    vehicle_repo.commit()

    assert len(listing.observations) == 1
    assert listing.observations[0].observed_price == 5100000
    assert listing.observations[0].listing.listing_id == "test_rel_301"

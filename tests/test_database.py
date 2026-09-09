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


def test_ad_date_is_saved_and_retrieved(vehicle_repo: VehicleRepository):
    """
    Verifies that the original ad_date is preserved in PostgreSQL and retrieved accurately.
    """
    data = {
        "listing_id": "test_ad_date_01",
        "listing_url": "https://riyasewana.com/buy/test-ad-date-01",
        "title": "Nissan Leaf 2017",
        "category": "Cars",
        "brand": "Nissan",
        "model": "Leaf",
        "ad_date": "2026 Sep 07, 3:10 pm",
        "manufacture_year": 2017,
        "price": 5500000,
        "mileage": 60000,
    }

    listing, is_new = vehicle_repo.sync_listing(data)
    vehicle_repo.commit()

    assert is_new is True
    assert listing.ad_date == "2026 Sep 07, 3:10 pm"

    # Query afresh from database
    retrieved = vehicle_repo.find_listing("test_ad_date_01")
    assert retrieved is not None
    assert retrieved.ad_date == "2026 Sep 07, 3:10 pm"



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


def test_step4_price_history_deduplication_and_transition(vehicle_repo: VehicleRepository):
    """
    STEP 4 PART 2 REGRESSION TEST:
    Simulates:
    Observation 1: price = 4,200,000 -> 1 observation, 1 price-history record
    Observation 2: price = 4,200,000 -> 2 observations, 1 price-history record (no duplicate)
    Observation 3: price = 4,000,000 -> 3 observations, 2 price-history records
    """
    base_data = {
        "listing_id": "step4_price_test",
        "listing_url": "https://riyasewana.com/buy/step4-price-test",
        "title": "Toyota Hiace Van",
        "category": "Vans",
        "brand": "Toyota",
        "model": "Hiace",
        "mileage": 120000,
    }

    # Observation 1: price = 4,200,000
    run1 = vehicle_repo.create_scrape_run(category="Vans")
    d1 = dict(base_data, price=4200000)
    listing, is_new1 = vehicle_repo.sync_listing(d1, scrape_run=run1)
    vehicle_repo.commit()

    assert is_new1 is True
    assert len(listing.observations) == 1
    assert len(listing.price_history) == 1
    assert listing.price_history[0].price == 4200000

    # Observation 2: price = 4,200,000 (unchanged)
    run2 = vehicle_repo.create_scrape_run(category="Vans")
    d2 = dict(base_data, price=4200000)
    listing, is_new2 = vehicle_repo.sync_listing(d2, scrape_run=run2)
    vehicle_repo.commit()

    assert is_new2 is False
    assert len(listing.observations) == 2
    assert len(listing.price_history) == 1
    assert listing.price_history[0].price == 4200000

    # Observation 3: price = 4,000,000 (price drop)
    run3 = vehicle_repo.create_scrape_run(category="Vans")
    d3 = dict(base_data, price=4000000)
    listing, is_new3 = vehicle_repo.sync_listing(d3, scrape_run=run3)
    vehicle_repo.commit()

    assert is_new3 is False
    assert len(listing.observations) == 3
    assert len(listing.price_history) == 2
    assert listing.price_history[0].price == 4200000
    assert listing.price_history[1].price == 4000000


def test_step4_listing_disappearance_and_reactivation_lifecycle(vehicle_repo: VehicleRepository):
    """
    STEP 4 PART 3 REGRESSION TEST:
    Simulates:
    Run 1: Listing A observed -> status = ACTIVE
    Run 2: Listing A absent from observed set -> status = NO_LONGER_OBSERVED (never SOLD)
    Run 3: Listing A appears again -> status = ACTIVE (first_seen_at preserved, last_seen_at updated)
    """
    t1 = datetime(2026, 9, 1, 9, 0, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 9, 3, 14, 30, 0, tzinfo=timezone.utc)

    data_a = {
        "listing_id": "step4_lifecycle_van",
        "listing_url": "https://riyasewana.com/buy/step4-lifecycle-van",
        "title": "Nissan Caravan Van",
        "category": "Vans",
        "brand": "Nissan",
        "model": "Caravan",
        "price": 3800000,
        "mileage": 140000,
        "ad_date": "2026 Sep 01, 8:00 am",
    }

    # Run 1: Listing observed
    run1 = vehicle_repo.create_scrape_run(category="Vans")
    listing, is_new1 = vehicle_repo.sync_listing(data_a, scrape_run=run1, observed_at=t1)
    vehicle_repo.commit()

    assert is_new1 is True
    assert listing.current_status == "ACTIVE"

    # Run 2: Listing A absent from observed set
    marked = vehicle_repo.mark_unobserved_listings(
        observed_listing_ids=["different_listing_id"],
        source="riyasewana",
        category="Vans",
    )
    vehicle_repo.commit()

    assert len(marked) == 1
    assert marked[0].listing_id == "step4_lifecycle_van"
    assert marked[0].current_status == "NO_LONGER_OBSERVED"
    assert marked[0].current_status != "SOLD"
    assert "SOLD" not in marked[0].current_status

    # Historical data preserved during absence
    reloaded = vehicle_repo.find_listing("step4_lifecycle_van")
    assert reloaded.current_status == "NO_LONGER_OBSERVED"
    assert len(reloaded.observations) == 1
    assert len(reloaded.price_history) == 1

    # Run 3: Listing A appears again
    run3 = vehicle_repo.create_scrape_run(category="Vans")
    listing_ret, is_new3 = vehicle_repo.sync_listing(data_a, scrape_run=run3, observed_at=t3)
    vehicle_repo.commit()

    assert is_new3 is False
    assert listing_ret.current_status == "ACTIVE"
    assert to_utc(listing_ret.first_seen_at) == to_utc(t1)
    assert to_utc(listing_ret.last_seen_at) == to_utc(t3)
    assert len(listing_ret.observations) == 2
    assert len(listing_ret.price_history) == 1


def test_step4_historical_data_integrity_preservation(vehicle_repo: VehicleRepository):
    """
    STEP 4 PART 4 REGRESSION TEST:
    Verifies that across repeated scrapes, the system strictly preserves:
    - first_seen_at, last_seen_at, ad_date, observed_at, observed_price, observed_mileage
    - price history, listing status, source listing ID, listing URL
    - old observations are NOT overwritten
    - old price history is NOT deleted
    """
    t1 = datetime(2026, 9, 5, 8, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 6, 9, 0, 0, tzinfo=timezone.utc)

    data = {
        "listing_id": "step4_integrity_van",
        "listing_url": "https://riyasewana.com/buy/step4-integrity-van",
        "title": "Mazda Bongo Van",
        "category": "Vans",
        "brand": "Mazda",
        "model": "Bongo",
        "price": 2800000,
        "mileage": 160000,
        "ad_date": "2026 Sep 05, 7:30 am",
        "source": "riyasewana",
    }

    # Initial observation
    run1 = vehicle_repo.create_scrape_run(category="Vans")
    listing, _ = vehicle_repo.sync_listing(data, scrape_run=run1, observed_at=t1)
    vehicle_repo.commit()

    # Second observation with price drop
    run2 = vehicle_repo.create_scrape_run(category="Vans")
    data_day2 = dict(data, price=2700000, mileage=160500)
    listing, _ = vehicle_repo.sync_listing(data_day2, scrape_run=run2, observed_at=t2)
    vehicle_repo.commit()

    # Query afresh from DB
    retrieved = vehicle_repo.find_listing("step4_integrity_van")
    assert retrieved is not None
    assert retrieved.listing_id == "step4_integrity_van"
    assert retrieved.source == "riyasewana"
    assert retrieved.listing_url == "https://riyasewana.com/buy/step4-integrity-van"
    assert retrieved.current_status == "ACTIVE"
    assert retrieved.ad_date == "2026 Sep 05, 7:30 am"
    assert to_utc(retrieved.first_seen_at) == to_utc(t1)
    assert to_utc(retrieved.last_seen_at) == to_utc(t2)

    # Observations preserved and not overwritten
    assert len(retrieved.observations) == 2
    obs1, obs2 = retrieved.observations[0], retrieved.observations[1]
    assert obs1.observed_price == 2800000
    assert obs1.observed_mileage == 160000
    assert to_utc(obs1.observed_at) == to_utc(t1)
    assert obs2.observed_price == 2700000
    assert obs2.observed_mileage == 160500
    assert to_utc(obs2.observed_at) == to_utc(t2)

    # Price history preserved and not deleted
    assert len(retrieved.price_history) == 2
    ph1, ph2 = retrieved.price_history[0], retrieved.price_history[1]
    assert ph1.price == 2800000
    assert to_utc(ph1.observed_at) == to_utc(t1)
    assert ph2.price == 2700000
    assert to_utc(ph2.observed_at) == to_utc(t2)


def test_phase4_listing_category_property(vehicle_repo: VehicleRepository):
    """
    PHASE 4 STEP 1: Verifies Listing.category property cleanly delegates
    to vehicle.category for all discovered vehicle types.
    """
    categories = ["Car", "Heavy-Duty", "Lorry", "Motorbike", "Pickup", "SUV", "Three Wheel", "Van"]
    for cat in categories:
        data = {
            "listing_id": f"p4_cat_{cat.lower().replace('-', '_').replace(' ', '_')}",
            "listing_url": f"https://riyasewana.com/buy/{cat.lower()}",
            "title": f"Test {cat}",
            "category": cat,
            "brand": "SampleBrand",
            "model": "SampleModel",
            "price": 1000000,
        }
        listing, is_new = vehicle_repo.sync_listing(data)
        assert is_new is True
        assert listing.category == cat
        assert listing.vehicle.category == cat


def test_phase4_historical_rules_1_to_11(vehicle_repo: VehicleRepository):
    """
    PHASE 4 STEP 1: Exhaustive verification of Rules 1 through 11:
    - Rule 1: first_seen_at immutable after first observation.
    - Rule 2: last_seen_at updates whenever listing is observed.
    - Rule 3: Every observation creates a ListingObservation.
    - Rule 4: Old ListingObservations are NEVER overwritten.
    - Rule 5: Unchanged asking price does NOT create duplicate PriceHistory.
    - Rule 6: Changed asking price creates a new PriceHistory.
    - Rule 7: Missing listing does NOT mean SOLD.
    - Rule 8: Missing listings become NO_LONGER_OBSERVED.
    - Rule 9: Reappearing listing becomes ACTIVE.
    - Rule 10: Historical records are never deleted automatically.
    - Rule 11: Asking price remains explicitly an asking price.
    """
    t1 = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 2, 8, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc)
    t4 = datetime(2026, 9, 4, 8, 0, tzinfo=timezone.utc)

    base = {
        "listing_id": "rule_test_suv",
        "listing_url": "https://riyasewana.com/buy/rule-test-suv",
        "title": "Toyota Fortuner 2020 SUV",
        "category": "SUV",
        "brand": "Toyota",
        "model": "Fortuner",
        "price": 28000000,
        "mileage": 45000,
        "ad_date": "2026 Sep 01, 7:00 am",
    }

    # Step 1: Observation 1
    run1 = vehicle_repo.create_scrape_run(category="SUVs")
    listing, is_new1 = vehicle_repo.sync_listing(base, scrape_run=run1, observed_at=t1)
    vehicle_repo.commit()

    assert is_new1 is True
    # RULE 1 & 2
    assert to_utc(listing.first_seen_at) == to_utc(t1)
    assert to_utc(listing.last_seen_at) == to_utc(t1)
    # RULE 3
    assert len(listing.observations) == 1
    # RULE 11
    assert listing.observations[0].observed_price == 28000000
    assert len(listing.price_history) == 1
    assert listing.price_history[0].price == 28000000

    # Step 2: Observation 2 (Same price, advanced last_seen)
    run2 = vehicle_repo.create_scrape_run(category="SUVs")
    listing, is_new2 = vehicle_repo.sync_listing(base, scrape_run=run2, observed_at=t2)
    vehicle_repo.commit()

    assert is_new2 is False
    # RULE 1: first_seen_at remains unchanged
    assert to_utc(listing.first_seen_at) == to_utc(t1)
    # RULE 2: last_seen_at updated to t2
    assert to_utc(listing.last_seen_at) == to_utc(t2)
    # RULE 3 & 4: new observation appended, old preserved
    assert len(listing.observations) == 2
    assert to_utc(listing.observations[0].observed_at) == to_utc(t1)
    assert to_utc(listing.observations[1].observed_at) == to_utc(t2)
    # RULE 5: Unchanged price -> 0 duplicate PriceHistory records
    assert len(listing.price_history) == 1

    # Step 3: Observation 3 with price change (price drop to 27,500,000)
    run3 = vehicle_repo.create_scrape_run(category="SUVs")
    d3 = dict(base, price=27500000)
    listing, is_new3 = vehicle_repo.sync_listing(d3, scrape_run=run3, observed_at=t3)
    vehicle_repo.commit()

    # RULE 6: Changed price -> new PriceHistory record
    assert len(listing.price_history) == 2
    assert listing.price_history[1].price == 27500000
    assert len(listing.observations) == 3

    # Step 4: Listing absent in next scrape
    marked = vehicle_repo.mark_unobserved_listings(["other_id"], source="riyasewana", category="SUV")
    vehicle_repo.commit()

    # RULE 7 & 8: Becomes NO_LONGER_OBSERVED, NOT SOLD
    assert len(marked) == 1
    assert marked[0].current_status == "NO_LONGER_OBSERVED"
    assert "SOLD" not in marked[0].current_status

    # Step 5: Listing reappears in scrape 4
    run4 = vehicle_repo.create_scrape_run(category="SUVs")
    listing_ret, is_new4 = vehicle_repo.sync_listing(d3, scrape_run=run4, observed_at=t4)
    vehicle_repo.commit()

    # RULE 9: Reappearing becomes ACTIVE
    assert is_new4 is False
    assert listing_ret.current_status == "ACTIVE"
    assert to_utc(listing_ret.first_seen_at) == to_utc(t1)
    assert to_utc(listing_ret.last_seen_at) == to_utc(t4)
    # RULE 10: Historical records never deleted
    assert len(listing_ret.observations) == 4
    assert len(listing_ret.price_history) == 2


def test_phase4_time_series_indexes_and_ordering(vehicle_repo: VehicleRepository):
    """
    PHASE 4 STEP 1: Verifies time-series queries on (listing_id, observed_at)
    return properly ordered observations and price history.
    """
    data = {
        "listing_id": "ts_query_bike",
        "listing_url": "https://riyasewana.com/buy/ts-query-bike",
        "title": "Yamaha FZ 2021 Motorbike",
        "category": "Motorbike",
        "brand": "Yamaha",
        "model": "FZ",
        "price": 750000,
    }
    t_start = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    listing, _ = vehicle_repo.sync_listing(data, scrape_run=vehicle_repo.create_scrape_run(), observed_at=t_start)

    # 3 price adjustments over time
    t_step2 = datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc)
    d2 = dict(data, price=720000)
    vehicle_repo.sync_listing(d2, scrape_run=vehicle_repo.create_scrape_run(), observed_at=t_step2)

    t_step3 = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
    d3 = dict(data, price=690000)
    vehicle_repo.sync_listing(d3, scrape_run=vehicle_repo.create_scrape_run(), observed_at=t_step3)
    vehicle_repo.commit()

    # Query chronological price history via (listing_id, observed_at)
    query_ph = (
        select(PriceHistory)
        .where(PriceHistory.listing_id == listing.id)
        .order_by(PriceHistory.observed_at.asc())
    )
    history = vehicle_repo.db.scalars(query_ph).all()
    assert len(history) == 3
    assert [h.price for h in history] == [750000, 720000, 690000]

    # Query latest price
    latest_p = vehicle_repo.get_latest_price(listing)
    assert latest_p == 690000


def test_phase4_step2_five_run_historical_lifecycle(vehicle_repo: VehicleRepository):
    """
    Validates the exact 5-run historical lifecycle specified in Phase 4 Step 2:
    - Run 1: listing A -> ACTIVE
    - Run 2: listing A still present with same price -> same listing, first_seen_at unchanged,
             last_seen_at updated, new observation, no duplicate price history
    - Run 3: listing A appears with changed price -> new observation, new PriceHistory, previous preserved
    - Run 4: listing A is not observed -> NO_LONGER_OBSERVED, historical records preserved
    - Run 5: listing A appears again -> ACTIVE, same historical identity, first_seen_at preserved,
             new observation, no duplicate listing
    """
    listing_id = "lifecycle_car_001"
    url = f"https://riyasewana.com/buy/toyota-vitz-{listing_id}"
    base_data = {
        "listing_id": listing_id,
        "listing_url": url,
        "title": "Toyota Vitz 2018",
        "category": "Cars",
        "brand": "Toyota",
        "model": "Vitz",
        "manufacture_year": 2018,
        "price": 6500000,
        "mileage": 42000,
        "condition": "Used",
        "district": "Colombo",
        "source": "riyasewana",
    }

    # ----------------------------------------------------
    # RUN 1: First appearance
    # ----------------------------------------------------
    t1 = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)
    sr1 = vehicle_repo.create_scrape_run(category="Cars")
    listing_r1, is_new_r1 = vehicle_repo.sync_listing(base_data, scrape_run=sr1, observed_at=t1)
    vehicle_repo.commit()

    assert is_new_r1 is True
    assert listing_r1.current_status == "ACTIVE"
    assert to_utc(listing_r1.first_seen_at) == t1
    assert to_utc(listing_r1.last_seen_at) == t1
    assert len(listing_r1.price_history) == 1
    assert listing_r1.price_history[0].price == 6500000
    assert len(listing_r1.observations) == 1

    # ----------------------------------------------------
    # RUN 2: Same listing, same price
    # ----------------------------------------------------
    t2 = datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc)
    sr2 = vehicle_repo.create_scrape_run(category="Cars")
    listing_r2, is_new_r2 = vehicle_repo.sync_listing(base_data, scrape_run=sr2, observed_at=t2)
    vehicle_repo.commit()

    assert is_new_r2 is False
    assert listing_r2.id == listing_r1.id
    assert to_utc(listing_r2.first_seen_at) == t1  # Immutable
    assert to_utc(listing_r2.last_seen_at) == t2  # Updated
    assert listing_r2.current_status == "ACTIVE"
    assert len(listing_r2.price_history) == 1     # No duplicate price history
    assert len(listing_r2.observations) == 2      # New observation created

    # ----------------------------------------------------
    # RUN 3: Same listing, changed price (price drop)
    # ----------------------------------------------------
    t3 = datetime(2026, 9, 3, 9, 0, tzinfo=timezone.utc)
    sr3 = vehicle_repo.create_scrape_run(category="Cars")
    data_r3 = dict(base_data, price=6300000)
    listing_r3, is_new_r3 = vehicle_repo.sync_listing(data_r3, scrape_run=sr3, observed_at=t3)
    vehicle_repo.commit()

    assert is_new_r3 is False
    assert listing_r3.id == listing_r1.id
    assert to_utc(listing_r3.first_seen_at) == t1
    assert to_utc(listing_r3.last_seen_at) == t3
    assert len(listing_r3.price_history) == 2     # New PriceHistory appended
    assert [ph.price for ph in sorted(listing_r3.price_history, key=lambda x: x.observed_at)] == [6500000, 6300000]
    assert len(listing_r3.observations) == 3      # 3rd observation logged

    # ----------------------------------------------------
    # RUN 4: Listing is not observed in scrape
    # ----------------------------------------------------
    sr4 = vehicle_repo.create_scrape_run(category="Cars")
    # Active scrape run observes other listings, but NOT lifecycle_car_001
    vehicle_repo.mark_unobserved_listings(observed_listing_ids=["other_car_999"], category="Cars")
    vehicle_repo.commit()

    listing_r4 = vehicle_repo.find_listing(listing_id)
    assert listing_r4.current_status == "NO_LONGER_OBSERVED"
    assert listing_r4.current_status != "SOLD"
    # Historical data fully preserved
    assert len(listing_r4.price_history) == 2
    assert len(listing_r4.observations) == 3
    assert to_utc(listing_r4.first_seen_at) == t1
    assert to_utc(listing_r4.last_seen_at) == t3

    # ----------------------------------------------------
    # RUN 5: Listing reappears on marketplace
    # ----------------------------------------------------
    t5 = datetime(2026, 9, 7, 9, 0, tzinfo=timezone.utc)
    sr5 = vehicle_repo.create_scrape_run(category="Cars")
    # Reappears with same price as run 3 (6300000)
    listing_r5, is_new_r5 = vehicle_repo.sync_listing(data_r3, scrape_run=sr5, observed_at=t5)
    vehicle_repo.commit()

    assert is_new_r5 is False
    assert listing_r5.id == listing_r1.id
    assert listing_r5.current_status == "ACTIVE"
    assert to_utc(listing_r5.first_seen_at) == t1  # Permanently preserved
    assert to_utc(listing_r5.last_seen_at) == t5  # Updated
    assert len(listing_r5.price_history) == 2     # Price was 6300000, no duplicate added
    assert len(listing_r5.observations) == 4      # 4th observation recorded

    # Verify zero duplicate rows exist in database
    all_with_id = vehicle_repo.db.scalars(
        select(Listing).where(Listing.listing_id == listing_id)
    ).all()
    assert len(all_with_id) == 1




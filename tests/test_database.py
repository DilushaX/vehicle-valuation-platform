import json
from pathlib import Path

from database.models import (
    Vehicle,
    Listing,
    PriceHistory,
    ScrapeRun,
    ListingObservation,
)
from database.repository import VehicleRepository


def test_init_does_not_import_database_connection():
    """
    Verifies tests/__init__.py does not load database.connection on package import.
    """
    init_path = Path(__file__).parent / "__init__.py"
    content = init_path.read_text()
    assert "database.connection" not in content
    assert "from database.connection import Base" not in content


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

    # Raw value in database column must be parseable as valid JSON
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

    # Read back via model helper
    assert listing.get_validation_issues() == issues

    # Read back via repository helper
    assert vehicle_repo.get_validation_issues(listing) == issues

    # Query afresh from DB
    retrieved = vehicle_repo.find_listing("test_json_102")
    assert retrieved is not None
    assert retrieved.get_validation_issues() == issues
    assert json.loads(retrieved.validation_issues) == issues


def test_scrape_run_and_listing_observation_relationship(vehicle_repo: VehicleRepository):
    """
    Verifies that ScrapeRun and ListingObservation have bi-directional relationships:
    - ListingObservation -> ScrapeRun (observation.scrape_run)
    - ScrapeRun -> ListingObservation (scrape_run.observations)
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

    # ScrapeRun -> ListingObservation
    assert len(scrape_run.observations) == 1
    obs = scrape_run.observations[0]
    assert obs.observed_price == 7500000
    assert obs.observed_mileage == 45000
    assert obs.listing_id == listing.id

    # ListingObservation -> ScrapeRun
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
    # 1. Vehicle -> Listing
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

    # 2. Listing -> PriceHistory
    vehicle_repo.add_price_history(listing, 5200000)
    vehicle_repo.add_price_history(listing, 5100000)
    vehicle_repo.commit()

    assert len(listing.price_history) == 2
    prices = [ph.price for ph in listing.price_history]
    assert prices == [5200000, 5100000]
    assert listing.price_history[0].listing.listing_id == "test_rel_301"

    # 3. Listing -> ListingObservation
    scrape_run = vehicle_repo.create_scrape_run(category="Cars")
    vehicle_repo.add_observation(listing, scrape_run, 5100000, 60000)
    vehicle_repo.commit()

    assert len(listing.observations) == 1
    assert listing.observations[0].observed_price == 5100000
    assert listing.observations[0].listing.listing_id == "test_rel_301"

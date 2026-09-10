from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
import pytest
from sqlalchemy import select

from config import settings
from data_pipeline.completeness import CategoryCompleteness
from data_pipeline.pipeline_runner import PipelineRunner
from database.models import Listing, ListingObservation, PriceHistory, ScrapeRun, Vehicle
from database.repository import VehicleRepository
from scraper.discovery.category_discovery import DiscoveredCategory, DiscoveredPage
from scraper.discovery.listing_discovery import DiscoveredListing, RiyasewanaListingDiscovery
from scraper.spiders.riyasewana_category_spider import RiyasewanaCategorySpider


def to_utc(dt: datetime) -> datetime:
    """Helper ensuring timezone-aware UTC comparison."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ==============================================================================
# STEP 11 TEST SUITE: INCREMENTAL COLLECTION & LISTING LIFECYCLE
# ==============================================================================

def test_1_new_listing_detection(vehicle_repo: VehicleRepository):
    """1. New listing detection: creates Vehicle, Listing, initial PriceHistory, and Observation."""
    run = vehicle_repo.create_scrape_run(category="Cars")
    data = {
        "listing_id": "car_new_01",
        "listing_url": "https://riyasewana.com/buy/car-new-01",
        "title": "Toyota Premio 2018",
        "category": "Cars",
        "brand": "Toyota",
        "model": "Premio",
        "manufacture_year": 2018,
        "registration_year": 2019,
        "price": 14500000,
        "mileage": 45000,
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "condition": "Used",
        "is_valid": True,
        "validation_issues": [],
    }

    listing, is_new = vehicle_repo.sync_listing(data, scrape_run=run)
    vehicle_repo.commit()

    assert is_new is True
    assert listing.listing_id == "car_new_01"
    assert listing.current_status == "ACTIVE"
    assert listing.vehicle.brand == "Toyota"
    assert listing.vehicle.model == "Premio"
    assert len(listing.price_history) == 1
    assert listing.price_history[0].price == 14500000
    assert len(listing.observations) == 1
    assert listing.observations[0].observed_price == 14500000


def test_2_existing_listing_detection(vehicle_repo: VehicleRepository):
    """2. Existing listing detection: preserves listing identity, updates last_seen_at."""
    t1 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc)

    run1 = vehicle_repo.create_scrape_run(category="Cars")
    data = {
        "listing_id": "car_exist_01",
        "listing_url": "https://riyasewana.com/buy/car-exist-01",
        "title": "Toyota Axio 2016",
        "category": "Cars",
        "brand": "Toyota",
        "model": "Axio",
        "price": 12000000,
        "mileage": 60000,
    }

    listing1, is_new1 = vehicle_repo.sync_listing(data, scrape_run=run1, observed_at=t1)
    vehicle_repo.commit()
    assert is_new1 is True
    initial_id = listing1.id

    # Second observation: existing listing
    run2 = vehicle_repo.create_scrape_run(category="Cars")
    listing2, is_new2 = vehicle_repo.sync_listing(data, scrape_run=run2, observed_at=t2)
    vehicle_repo.commit()

    assert is_new2 is False
    assert listing2.id == initial_id
    assert listing2.listing_id == "car_exist_01"
    assert to_utc(listing2.first_seen_at) == to_utc(t1)
    assert to_utc(listing2.last_seen_at) == to_utc(t2)
    assert len(listing2.observations) == 2


def test_3_repeated_listing_detection(vehicle_repo: VehicleRepository):
    """3. Repeated listing detection across multiple runs maintains exactly one Listing record."""
    data = {
        "listing_id": "car_repeat_01",
        "listing_url": "https://riyasewana.com/buy/car-repeat-01",
        "title": "Honda Civic 2019",
        "category": "Cars",
        "brand": "Honda",
        "model": "Civic",
        "price": 13000000,
    }

    for i in range(1, 4):
        run = vehicle_repo.create_scrape_run(category="Cars")
        listing, is_new = vehicle_repo.sync_listing(data, scrape_run=run)
        if i == 1:
            assert is_new is True
        else:
            assert is_new is False
        vehicle_repo.commit()

    # Query afresh
    all_listings = vehicle_repo.db.scalars(
        select(Listing).where(Listing.listing_id == "car_repeat_01")
    ).all()
    assert len(all_listings) == 1
    assert len(all_listings[0].observations) == 3


def test_4_duplicate_url_handling():
    """4. Duplicate URLs on single page (fragments, queries, repeated links) are deduplicated."""
    html = """
    <html><body>
        <a href="/buy/toyota-vitz-sale-12217088">Vitz Link 1</a>
        <a href="/buy/toyota-vitz-sale-12217088?ref=pagination">Vitz Link with Query</a>
        <a href="/buy/toyota-vitz-sale-12217088#specs">Vitz Link with Fragment</a>
    </body></html>
    """
    discovery = RiyasewanaListingDiscovery()
    urls = discovery.discover_listing_urls(html, "https://riyasewana.com/search/cars")

    assert len(urls) == 1
    assert urls[0] == "https://riyasewana.com/buy/toyota-vitz-sale-12217088"


def test_5_duplicate_listing_id_handling():
    """5. Same listing discovered across multiple pages is extracted with a unique ID."""
    discovery = RiyasewanaListingDiscovery()
    id1 = discovery.extract_listing_id("https://riyasewana.com/buy/toyota-vitz-12217088")
    id2 = discovery.extract_listing_id("https://riyasewana.com/buy/toyota-vitz-sale-colombo-12217088")
    assert id1 == "12217088"
    assert id2 == "12217088"
    assert id1 == id2


def test_6_partial_scope_disappearance_safety(db_session):
    """
    6. Partial-scope disappearance safety:
    Database has 10 active Cars listings.
    Run with max_pages=1, max_listings=2 only observes 2 listings.
    The other 8 listings MUST remain ACTIVE and NOT be marked NO_LONGER_OBSERVED.
    """
    repo = VehicleRepository(db_session)
    run_init = repo.create_scrape_run(category="Cars")

    # Seed 10 active cars
    for i in range(1, 11):
        repo.sync_listing(
            {
                "listing_id": f"car_{i}",
                "listing_url": f"https://riyasewana.com/buy/car-{i}",
                "title": f"Car {i}",
                "category": "Cars",
                "price": 5000000 + (i * 100000),
            },
            scrape_run=run_init,
        )
    repo.commit()

    # Active count before
    active_before = repo.db.scalars(
        select(Listing).join(Vehicle).where(Vehicle.category == "Cars", Listing.current_status == "ACTIVE")
    ).all()
    assert len(active_before) == 10

    # Scoped pipeline run observing only car_1 and car_2
    mock_spider = MagicMock(spec=RiyasewanaCategorySpider)
    mock_spider.scrape_category.return_value = {
        "category_name": "Cars",
        "category_url": "https://riyasewana.com/search/cars",
        "pages_discovered": 5,
        "pages_attempted": 1,
        "pages_scraped": 1,
        "failed_pages": [],
        "listing_urls_discovered": 2,
        "unique_listing_urls": ["https://riyasewana.com/buy/car-1", "https://riyasewana.com/buy/car-2"],
        "listings_attempted": 2,
        "listings_scraped": 2,
        "failed_listings": [],
        "records": [
            {"listing_id": "car_1", "listing_url": "https://riyasewana.com/buy/car-1", "category": "Cars", "price": 5100000},
            {"listing_id": "car_2", "listing_url": "https://riyasewana.com/buy/car-2", "category": "Cars", "price": 5200000},
        ],
        "pagination_exhausted": False,  # Scoped run
        "status": "COMPLETED",
    }

    runner = PipelineRunner(category_spider=mock_spider)
    report = runner.run_pipeline(
        categories=["Cars"],
        max_pages=1,
        max_listings=2,
        dry_run=False,
        export_csv=False,
        db_session=db_session,
    )

    # Disappearance must be skipped
    cat_res = report["category_results"][0]
    assert cat_res["disappearance_definitive"] is False
    assert cat_res["disappeared_listings"] == 0

    # All 10 cars must remain ACTIVE
    active_after = repo.db.scalars(
        select(Listing).join(Vehicle).where(Vehicle.category == "Cars", Listing.current_status == "ACTIVE")
    ).all()
    assert len(active_after) == 10

    no_longer_obs = repo.db.scalars(
        select(Listing).where(Listing.current_status == "NO_LONGER_OBSERVED")
    ).all()
    assert len(no_longer_obs) == 0


def test_7_complete_scope_disappearance_detection(db_session):
    """
    7. Complete-scope disappearance detection:
    Exhaustive run without limits transitions unobserved active listings to NO_LONGER_OBSERVED.
    Never marks them SOLD.
    """
    repo = VehicleRepository(db_session)
    run_init = repo.create_scrape_run(category="Cars")

    # Seed 3 active cars
    for i in range(1, 4):
        repo.sync_listing(
            {
                "listing_id": f"car_{i}",
                "listing_url": f"https://riyasewana.com/buy/car-{i}",
                "title": f"Car {i}",
                "category": "Cars",
                "price": 5000000,
            },
            scrape_run=run_init,
        )
    repo.commit()

    # Complete-scope run (max_pages=None, max_listings=None, pagination_exhausted=True)
    # Only car_1 is observed; car_2 and car_3 have disappeared from the website.
    mock_spider = MagicMock(spec=RiyasewanaCategorySpider)
    mock_spider.scrape_category.return_value = {
        "category_name": "Cars",
        "category_url": "https://riyasewana.com/search/cars",
        "pages_discovered": 1,
        "pages_attempted": 1,
        "pages_scraped": 1,
        "failed_pages": [],
        "listing_urls_discovered": 1,
        "unique_listing_urls": ["https://riyasewana.com/buy/car-1"],
        "listings_attempted": 1,
        "listings_scraped": 1,
        "failed_listings": [],
        "records": [
            {"listing_id": "car_1", "listing_url": "https://riyasewana.com/buy/car-1", "category": "Cars", "price": 5000000},
        ],
        "pagination_exhausted": True,
        "status": "COMPLETED",
    }

    runner = PipelineRunner(category_spider=mock_spider)
    report = runner.run_pipeline(
        categories=["Cars"],
        max_pages=None,
        max_listings=None,
        dry_run=False,
        export_csv=False,
        db_session=db_session,
    )

    cat_res = report["category_results"][0]
    assert cat_res["disappearance_definitive"] is True
    assert cat_res["disappeared_listings"] == 2

    # car_1 is ACTIVE
    car_1 = repo.find_listing("car_1")
    assert car_1.current_status == "ACTIVE"

    # car_2 and car_3 are NO_LONGER_OBSERVED, never SOLD
    car_2 = repo.find_listing("car_2")
    car_3 = repo.find_listing("car_3")
    assert car_2.current_status == "NO_LONGER_OBSERVED"
    assert car_3.current_status == "NO_LONGER_OBSERVED"
    assert car_2.current_status != "SOLD"
    assert car_3.current_status != "SOLD"


def test_8_reappearance_handling(vehicle_repo: VehicleRepository):
    """
    8. Reappearance:
    NO_LONGER_OBSERVED listing reappears -> transitions to ACTIVE.
    Preserves original ID, first_seen_at, historical observations.
    """
    t1 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 9, 3, 10, 0, tzinfo=timezone.utc)

    data = {
        "listing_id": "car_reappear",
        "listing_url": "https://riyasewana.com/buy/car-reappear",
        "title": "Nissan Leaf 2016",
        "category": "Cars",
        "brand": "Nissan",
        "model": "Leaf",
        "price": 4500000,
    }

    # Day 1: Created ACTIVE
    listing, _ = vehicle_repo.sync_listing(data, observed_at=t1)
    vehicle_repo.commit()
    assert listing.current_status == "ACTIVE"

    # Day 2: Disappears
    vehicle_repo.mark_listing_no_longer_observed(listing)
    vehicle_repo.commit()
    assert listing.current_status == "NO_LONGER_OBSERVED"

    # Day 3: Reappears
    run3 = vehicle_repo.create_scrape_run(category="Cars")
    listing_ret, is_new = vehicle_repo.sync_listing(data, scrape_run=run3, observed_at=t3)
    vehicle_repo.commit()

    assert is_new is False
    assert listing_ret.current_status == "ACTIVE"
    assert to_utc(listing_ret.first_seen_at) == to_utc(t1)
    assert to_utc(listing_ret.last_seen_at) == to_utc(t3)
    assert getattr(listing_ret, "_was_reactivated", False) is True


def test_9_unchanged_price(vehicle_repo: VehicleRepository):
    """9. Unchanged price: records observation, does NOT create duplicate PriceHistory."""
    data = {
        "listing_id": "car_same_price",
        "listing_url": "https://riyasewana.com/buy/car-same-price",
        "price": 8000000,
    }

    # Scrape 1
    run1 = vehicle_repo.create_scrape_run()
    listing, _ = vehicle_repo.sync_listing(data, scrape_run=run1)
    vehicle_repo.commit()
    assert len(listing.price_history) == 1

    # Scrape 2 (same price)
    run2 = vehicle_repo.create_scrape_run()
    listing, _ = vehicle_repo.sync_listing(data, scrape_run=run2)
    vehicle_repo.commit()

    assert len(listing.price_history) == 1
    assert len(listing.observations) == 2


def test_10_changed_price(vehicle_repo: VehicleRepository):
    """10. Changed price: records new PriceHistory, preserves old PriceHistory."""
    data1 = {
        "listing_id": "car_change_price",
        "listing_url": "https://riyasewana.com/buy/car-change-price",
        "price": 8000000,
    }
    data2 = dict(data1, price=7500000)

    run1 = vehicle_repo.create_scrape_run()
    listing, _ = vehicle_repo.sync_listing(data1, scrape_run=run1)
    vehicle_repo.commit()
    assert len(listing.price_history) == 1

    run2 = vehicle_repo.create_scrape_run()
    listing, _ = vehicle_repo.sync_listing(data2, scrape_run=run2)
    vehicle_repo.commit()

    assert len(listing.price_history) == 2
    prices = [ph.price for ph in listing.price_history]
    assert prices == [8000000, 7500000]


def test_11_first_seen_at_immutability(vehicle_repo: VehicleRepository):
    """11. first_seen_at immutability: preserved across multiple subsequent scrapes."""
    t1 = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 5, 8, 0, tzinfo=timezone.utc)

    data = {
        "listing_id": "car_immut",
        "listing_url": "https://riyasewana.com/buy/car-immut",
        "price": 6000000,
    }

    listing, _ = vehicle_repo.sync_listing(data, observed_at=t1)
    vehicle_repo.commit()
    assert to_utc(listing.first_seen_at) == to_utc(t1)

    listing, _ = vehicle_repo.sync_listing(data, observed_at=t2)
    vehicle_repo.commit()
    assert to_utc(listing.first_seen_at) == to_utc(t1)


def test_12_last_seen_at_update(vehicle_repo: VehicleRepository):
    """12. last_seen_at update: updates on each observation."""
    t1 = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 5, 8, 0, tzinfo=timezone.utc)

    data = {
        "listing_id": "car_last_seen",
        "listing_url": "https://riyasewana.com/buy/car-last-seen",
        "price": 6000000,
    }

    listing, _ = vehicle_repo.sync_listing(data, observed_at=t1)
    vehicle_repo.commit()
    assert to_utc(listing.last_seen_at) == to_utc(t1)

    listing, _ = vehicle_repo.sync_listing(data, observed_at=t2)
    vehicle_repo.commit()
    assert to_utc(listing.last_seen_at) == to_utc(t2)


def test_13_observation_creation(vehicle_repo: VehicleRepository):
    """13. Observation creation: every successful scrape adds a ListingObservation."""
    data = {
        "listing_id": "car_obs",
        "listing_url": "https://riyasewana.com/buy/car-obs",
        "price": 9000000,
        "mileage": 30000,
    }

    run1 = vehicle_repo.create_scrape_run()
    run2 = vehicle_repo.create_scrape_run()

    listing, _ = vehicle_repo.sync_listing(data, scrape_run=run1)
    listing, _ = vehicle_repo.sync_listing(data, scrape_run=run2)
    vehicle_repo.commit()

    assert len(listing.observations) == 2
    assert listing.observations[0].scrape_run_id == run1.id
    assert listing.observations[1].scrape_run_id == run2.id


def test_14_historical_price_preservation(vehicle_repo: VehicleRepository):
    """14. Historical PriceHistory preservation across sequential price adjustments."""
    data = {
        "listing_id": "car_price_hist",
        "listing_url": "https://riyasewana.com/buy/car-price-hist",
        "price": 10000000,
    }

    listing, _ = vehicle_repo.sync_listing(data)

    # 10M -> 9.5M -> 9.2M
    data["price"] = 9500000
    vehicle_repo.sync_listing(data)

    data["price"] = 9200000
    vehicle_repo.sync_listing(data)
    vehicle_repo.commit()

    assert len(listing.price_history) == 3
    assert [ph.price for ph in listing.price_history] == [10000000, 9500000, 9200000]


def test_15_multi_category_isolation(db_session):
    """
    15. Multi-category isolation:
    A complete-scope scrape for Cars only marks unobserved Cars.
    Active listings in Vans must remain untouched.
    """
    repo = VehicleRepository(db_session)
    run_init = repo.create_scrape_run()

    # Seed 1 car and 1 van
    repo.sync_listing(
        {"listing_id": "car_iso_1", "listing_url": "https://riyasewana.com/buy/car-iso-1", "category": "Cars", "price": 5000000},
        scrape_run=run_init,
    )
    repo.sync_listing(
        {"listing_id": "van_iso_1", "listing_url": "https://riyasewana.com/buy/van-iso-1", "category": "Vans", "price": 7000000},
        scrape_run=run_init,
    )
    repo.commit()

    # Complete-scope Cars run observing NO cars (all disappeared in Cars)
    mock_spider = MagicMock(spec=RiyasewanaCategorySpider)
    mock_spider.scrape_category.return_value = {
        "category_name": "Cars",
        "category_url": "https://riyasewana.com/search/cars",
        "pages_discovered": 1,
        "pages_attempted": 1,
        "pages_scraped": 1,
        "failed_pages": [],
        "listing_urls_discovered": 0,
        "unique_listing_urls": [],
        "listings_attempted": 0,
        "listings_scraped": 0,
        "failed_listings": [],
        "records": [],
        "pagination_exhausted": True,
        "status": "COMPLETED",
    }

    runner = PipelineRunner(category_spider=mock_spider)
    runner.run_pipeline(
        categories=["Cars"],
        max_pages=None,
        max_listings=None,
        dry_run=False,
        export_csv=False,
        db_session=db_session,
    )

    car = repo.find_listing("car_iso_1")
    van = repo.find_listing("van_iso_1")

    assert car.current_status == "NO_LONGER_OBSERVED"
    # Van must remain ACTIVE
    assert van.current_status == "ACTIVE"


def test_16_incomplete_run_does_not_cause_false_disappearance(db_session):
    """
    16. Incomplete run (e.g. failed page or network timeout) skips disappearance detection.
    Active listings are preserved.
    """
    repo = VehicleRepository(db_session)
    run_init = repo.create_scrape_run(category="Cars")
    repo.sync_listing(
        {"listing_id": "car_page_fail", "listing_url": "https://riyasewana.com/buy/car-page-fail", "category": "Cars", "price": 6000000},
        scrape_run=run_init,
    )
    repo.commit()

    # Spider reports failed page -> status INCOMPLETE
    mock_spider = MagicMock(spec=RiyasewanaCategorySpider)
    mock_spider.scrape_category.return_value = {
        "category_name": "Cars",
        "category_url": "https://riyasewana.com/search/cars",
        "pages_discovered": 2,
        "pages_attempted": 2,
        "pages_scraped": 1,
        "failed_pages": [{"url": "page2", "error": "HTTP 500"}],
        "listing_urls_discovered": 0,
        "unique_listing_urls": [],
        "listings_attempted": 0,
        "listings_scraped": 0,
        "failed_listings": [],
        "records": [],
        "pagination_exhausted": False,
        "status": "INCOMPLETE",
    }

    runner = PipelineRunner(category_spider=mock_spider)
    report = runner.run_pipeline(
        categories=["Cars"],
        max_pages=None,
        max_listings=None,
        dry_run=False,
        export_csv=False,
        db_session=db_session,
    )

    cat_res = report["category_results"][0]
    assert cat_res["disappearance_definitive"] is False
    assert cat_res["disappeared_listings"] == 0

    car = repo.find_listing("car_page_fail")
    assert car.current_status == "ACTIVE"


def test_17_dry_run_does_not_mutate_db(db_session):
    """17. Dry-run writes zero rows to PostgreSQL."""
    mock_spider = MagicMock(spec=RiyasewanaCategorySpider)
    mock_spider.scrape_category.return_value = {
        "category_name": "Cars",
        "category_url": "https://riyasewana.com/search/cars",
        "pages_discovered": 1,
        "pages_attempted": 1,
        "pages_scraped": 1,
        "failed_pages": [],
        "listing_urls_discovered": 1,
        "unique_listing_urls": ["https://riyasewana.com/buy/dry-run-01"],
        "listings_attempted": 1,
        "listings_scraped": 1,
        "failed_listings": [],
        "records": [
            {"listing_id": "dry_01", "listing_url": "https://riyasewana.com/buy/dry-run-01", "category": "Cars", "price": 5000000},
        ],
        "pagination_exhausted": True,
        "status": "COMPLETED",
    }

    runner = PipelineRunner(category_spider=mock_spider)
    report = runner.run_pipeline(categories=["Cars"], dry_run=True, export_csv=False, db_session=db_session)

    assert report["dry_run"] is True
    listings = db_session.scalars(select(Listing)).all()
    runs = db_session.scalars(select(ScrapeRun)).all()
    assert len(listings) == 0
    assert len(runs) == 0


def test_18_collection_metrics(db_session):
    """18. Pipeline returns all required Step 8 collection metrics."""
    mock_spider = MagicMock(spec=RiyasewanaCategorySpider)
    mock_spider.scrape_category.return_value = {
        "category_name": "Cars",
        "category_url": "https://riyasewana.com/search/cars",
        "pages_discovered": 2,
        "pages_attempted": 2,
        "pages_scraped": 2,
        "failed_pages": [],
        "listing_urls_discovered": 3,
        "unique_listing_urls": ["url1", "url2", "url3"],
        "listings_attempted": 3,
        "listings_scraped": 3,
        "failed_listings": [],
        "records": [
            {"listing_id": "m1", "listing_url": "url1", "category": "Cars", "price": 1000000},
            {"listing_id": "m2", "listing_url": "url2", "category": "Cars", "price": 2000000},
        ],
        "pagination_exhausted": True,
        "status": "COMPLETED",
    }

    runner = PipelineRunner(category_spider=mock_spider)
    report = runner.run_pipeline(
        categories=["Cars"],
        max_pages=None,
        max_listings=None,
        dry_run=False,
        export_csv=False,
        db_session=db_session,
    )

    cat_res = report["category_results"][0]
    assert "new_listings" in cat_res
    assert "updated_listings" in cat_res
    assert "observations_created" in cat_res
    assert "price_changes" in cat_res
    assert "reactivated_listings" in cat_res
    assert "disappeared_listings" in cat_res
    assert "disappearance_definitive" in cat_res
    assert cat_res["new_listings"] == 2
    assert cat_res["observations_created"] == 2


# ==============================================================================
# STEP 10: COMPLETE FIVE-RUN HISTORICAL LIFECYCLE SIMULATION
# ==============================================================================

def test_step_10_complete_five_run_lifecycle(db_session):
    """
    Executes the exact 5-run sequence from Step 10:
    RUN 1: Listing A is new.
           Expected: ACTIVE, first_seen_at = T1, observation = 1, initial price history = 1
    RUN 2: Listing A appears again with same price.
           Expected: ACTIVE, first_seen_at unchanged, last_seen_at updated, observation = 2, price history unchanged
    RUN 3: Listing A appears again with changed price.
           Expected: ACTIVE, observation = 3, new PriceHistory, old PriceHistory preserved
    RUN 4: A complete category collection does not observe Listing A.
           Expected: NO_LONGER_OBSERVED, Never SOLD.
    RUN 5: Listing A appears again.
           Expected: ACTIVE, same Listing ID, same first_seen_at, historical observations preserved,
                     historical prices preserved, new observation, no duplicate Listing.
    """
    repo = VehicleRepository(db_session)
    t1 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 9, 3, 10, 0, tzinfo=timezone.utc)
    t4 = datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc)
    t5 = datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc)

    # --------------------------------------------------------------------------
    # RUN 1: Listing A is new
    # --------------------------------------------------------------------------
    run1 = repo.create_scrape_run(category="Cars")
    data_run1 = {
        "listing_id": "listing_alpha",
        "listing_url": "https://riyasewana.com/buy/toyota-alpha-12345",
        "title": "Toyota Alpha 2018",
        "category": "Cars",
        "brand": "Toyota",
        "model": "Alpha",
        "price": 10000000,
        "mileage": 50000,
    }
    listing_r1, is_new_r1 = repo.sync_listing(data_run1, scrape_run=run1, observed_at=t1)
    repo.complete_scrape_run(run1, pages_requested=1, pages_scraped=1, listings_found=1, new_listings=1)
    repo.commit()

    assert is_new_r1 is True
    assert listing_r1.current_status == "ACTIVE"
    assert to_utc(listing_r1.first_seen_at) == to_utc(t1)
    assert to_utc(listing_r1.last_seen_at) == to_utc(t1)
    assert len(listing_r1.observations) == 1
    assert len(listing_r1.price_history) == 1
    assert listing_r1.price_history[0].price == 10000000
    original_pk = listing_r1.id

    # --------------------------------------------------------------------------
    # RUN 2: Listing A appears again with same price
    # --------------------------------------------------------------------------
    run2 = repo.create_scrape_run(category="Cars")
    listing_r2, is_new_r2 = repo.sync_listing(data_run1, scrape_run=run2, observed_at=t2)
    repo.complete_scrape_run(run2, pages_requested=1, pages_scraped=1, listings_found=1, updated_listings=1)
    repo.commit()

    assert is_new_r2 is False
    assert listing_r2.id == original_pk
    assert listing_r2.current_status == "ACTIVE"
    assert to_utc(listing_r2.first_seen_at) == to_utc(t1)   # UNCHANGED
    assert to_utc(listing_r2.last_seen_at) == to_utc(t2)    # UPDATED
    assert len(listing_r2.observations) == 2                # observation = 2
    assert len(listing_r2.price_history) == 1               # price history UNCHANGED

    # --------------------------------------------------------------------------
    # RUN 3: Listing A appears again with changed price (10M -> 9.5M)
    # --------------------------------------------------------------------------
    run3 = repo.create_scrape_run(category="Cars")
    data_run3 = dict(data_run1, price=9500000)
    listing_r3, is_new_r3 = repo.sync_listing(data_run3, scrape_run=run3, observed_at=t3)
    repo.complete_scrape_run(run3, pages_requested=1, pages_scraped=1, listings_found=1, updated_listings=1, price_changes=1)
    repo.commit()

    assert is_new_r3 is False
    assert listing_r3.id == original_pk
    assert listing_r3.current_status == "ACTIVE"
    assert len(listing_r3.observations) == 3                # observation = 3
    assert len(listing_r3.price_history) == 2               # new PriceHistory
    assert [ph.price for ph in listing_r3.price_history] == [10000000, 9500000] # old preserved

    # --------------------------------------------------------------------------
    # RUN 4: Complete category collection does not observe Listing A
    # --------------------------------------------------------------------------
    run4 = repo.create_scrape_run(category="Cars")
    # Complete category scrape observes other cars, but not listing_alpha
    other_data = {
        "listing_id": "listing_other_beta",
        "listing_url": "https://riyasewana.com/buy/other-beta",
        "category": "Cars",
        "price": 6000000,
    }
    repo.sync_listing(other_data, scrape_run=run4, observed_at=t4)

    # Exhaustive complete-scope disappearance transition
    marked = repo.mark_unobserved_listings(
        observed_listing_ids=["listing_other_beta"],
        source="riyasewana",
        category="Cars",
    )
    repo.complete_scrape_run(run4, pages_requested=1, pages_scraped=1, listings_found=1, new_listings=1, disappeared_listings=len(marked))
    repo.commit()

    assert len(marked) == 1
    assert marked[0].listing_id == "listing_alpha"
    assert marked[0].current_status == "NO_LONGER_OBSERVED"
    assert marked[0].current_status != "SOLD"

    # --------------------------------------------------------------------------
    # RUN 5: Listing A appears again (with price 9.5M)
    # --------------------------------------------------------------------------
    run5 = repo.create_scrape_run(category="Cars")
    listing_r5, is_new_r5 = repo.sync_listing(data_run3, scrape_run=run5, observed_at=t5)
    repo.complete_scrape_run(run5, pages_requested=1, pages_scraped=1, listings_found=1, updated_listings=1, reactivated_listings=1)
    repo.commit()

    assert is_new_r5 is False
    assert listing_r5.id == original_pk
    assert listing_r5.listing_id == "listing_alpha"
    assert listing_r5.current_status == "ACTIVE"
    assert to_utc(listing_r5.first_seen_at) == to_utc(t1)   # SAME first_seen_at preserved
    assert to_utc(listing_r5.last_seen_at) == to_utc(t5)    # UPDATED to Day 5
    assert len(listing_r5.observations) == 4                # historical observations preserved (runs 1,2,3,5)
    assert len(listing_r5.price_history) == 2               # historical prices preserved (same price as Day 3 -> no duplicate)
    assert [ph.price for ph in listing_r5.price_history] == [10000000, 9500000]

    # Verify exactly one Listing record exists for listing_alpha in the database
    all_alpha = repo.db.scalars(
        select(Listing).where(Listing.listing_id == "listing_alpha")
    ).all()
    assert len(all_alpha) == 1

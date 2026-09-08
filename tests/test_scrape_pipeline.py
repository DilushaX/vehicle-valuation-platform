from unittest.mock import MagicMock
import pytest
from sqlalchemy import select

from data_pipeline.completeness import CategoryCompleteness
from data_pipeline.pipeline_runner import PipelineRunner
from database.models import Listing, ListingObservation, PriceHistory, ScrapeRun
from database.repository import VehicleRepository
from scraper.discovery.category_discovery import (
    DiscoveredCategory,
    DiscoveredPage,
    RiyasewanaCategoryDiscovery,
)
from scraper.discovery.listing_discovery import RiyasewanaListingDiscovery
from scraper.spiders.riyasewana_bulk_spider import RiyasewanaBulkSpider
from scraper.spiders.riyasewana_category_spider import RiyasewanaCategorySpider
from scraper.spiders.riyasewana_spider import RiyasewanaSpider
from scraper.validators.listing_validator import ListingValidator


class FakeClient:
    """Mock HTTP client for testing."""

    def __init__(self, pages=None):
        self.pages = pages or {}

    def get(self, url: str) -> str:
        if url not in self.pages:
            raise RuntimeError(f"Connection failed to: {url}")
        return self.pages[url]

    def close(self):
        pass


def test_bulk_spider_success_and_failure_isolation():
    """Verifies that an error in one listing does not abort other listings."""
    mock_spider = MagicMock(spec=RiyasewanaSpider)

    def side_effect(url):
        if "fail" in url:
            raise ValueError("Malformed listing page")
        return {
            "listing_id": "12217083",
            "listing_url": url,
            "title": "Toyota Prius 2018",
            "price": 8500000,
            "mileage": 60000,
            "year": 2018,
            "brand": "Toyota",
            "model": "Prius",
            "condition": "Used",
            "source": "riyasewana",
        }

    mock_spider.scrape_listing.side_effect = side_effect

    bulk_spider = RiyasewanaBulkSpider(
        spider=mock_spider,
        max_retries=1,
        request_delay=0,
    )

    urls = [
        "https://riyasewana.com/buy/toyota-prius-12217083",
        "https://riyasewana.com/buy/failed-listing-9999999",
    ]

    result = bulk_spider.scrape_listings(urls)

    assert result["total_attempted"] == 2
    assert result["total_successful"] == 1
    assert result["total_failed"] == 1
    assert len(result["successful"]) == 1
    assert len(result["failed"]) == 1
    assert result["failed"][0]["listing_id"] == "9999999"
    assert "Malformed listing page" in result["failed"][0]["error"]


def test_bulk_spider_retry_mechanism():
    """Verifies that transient errors trigger retries and succeed if resolved."""
    mock_spider = MagicMock(spec=RiyasewanaSpider)
    attempts = 0

    def flaky_scrape(url):
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ConnectionError("Temporary network reset")
        return {
            "listing_id": "12217084",
            "listing_url": url,
            "title": "Nissan Leaf 2015",
            "price": 4500000,
            "mileage": 75000,
            "year": 2015,
            "brand": "Nissan",
            "model": "Leaf",
            "condition": "Used",
            "source": "riyasewana",
        }

    mock_spider.scrape_listing.side_effect = flaky_scrape

    bulk_spider = RiyasewanaBulkSpider(
        spider=mock_spider,
        max_retries=3,
        retry_delay=0.01,
        request_delay=0,
    )

    result = bulk_spider.scrape_listings(["https://riyasewana.com/buy/nissan-leaf-12217084"])

    assert result["total_successful"] == 1
    assert result["total_failed"] == 0
    assert attempts == 2


def test_category_spider_page_failure_tracking():
    """Verifies that failed pagination pages are tracked and result in INCOMPLETE status."""
    mock_discovery = MagicMock(spec=RiyasewanaCategoryDiscovery)
    mock_discovery.discover_pages.return_value = [
        DiscoveredPage(url="https://riyasewana.com/search/cars", page_number=1),
        DiscoveredPage(url="https://riyasewana.com/search/cars?page=2", page_number=2),
    ]

    mock_client = MagicMock()

    def get_html(url):
        if "page=2" in url:
            raise RuntimeError("HTTP 500 Server Error")
        return """
        <a href="/buy/toyota-corolla-12217083">Toyota Corolla</a>
        """

    mock_client.get.side_effect = get_html

    mock_bulk_spider = MagicMock(spec=RiyasewanaBulkSpider)
    mock_bulk_spider.scrape_listings.return_value = {
        "successful": [
            {
                "listing_id": "12217083",
                "listing_url": "https://riyasewana.com/buy/toyota-corolla-12217083",
                "title": "Toyota Corolla 2016",
                "category": "Cars",
                "price": 12000000,
                "mileage": 50000,
                "year": 2016,
                "brand": "Toyota",
                "model": "Corolla",
                "condition": "Used",
                "source": "riyasewana",
                "is_valid": True,
                "validation_issues": [],
            }
        ],
        "failed": [],
        "total_attempted": 1,
        "total_successful": 1,
        "total_failed": 0,
    }

    spider = RiyasewanaCategorySpider(
        client=mock_client,
        category_discovery=mock_discovery,
        bulk_spider=mock_bulk_spider,
        max_retries=1,
        request_delay=0,
    )

    result = spider.scrape_category(DiscoveredCategory(name="Cars", url="https://riyasewana.com/search/cars"))

    assert result["pages_discovered"] == 2
    assert result["pages_attempted"] == 2
    assert result["pages_scraped"] == 1
    assert len(result["failed_pages"]) == 1
    assert result["failed_pages"][0]["page_number"] == 2
    assert result["status"] == "INCOMPLETE"


def test_completeness_calculation():
    """Verifies completeness percentages and status calculation."""
    comp = CategoryCompleteness(
        category_name="Cars",
        pages_discovered=10,
        pages_attempted=10,
        pages_scraped=10,
        failed_pages=[],
        listing_urls_discovered=100,
        unique_listing_urls=100,
        listings_attempted=100,
        listings_scraped=100,
        failed_listings=[],
        status="COMPLETED",
    )

    assert comp.page_completeness_pct == 100.0
    assert comp.listing_completeness_pct == 100.0
    assert comp.is_fully_complete is True
    assert "100.0%" in comp.format_summary()

    incomplete = CategoryCompleteness(
        category_name="SUVs",
        pages_discovered=10,
        pages_attempted=10,
        pages_scraped=8,
        failed_pages=[{"error": "timeout"}],
        status="INCOMPLETE",
    )
    assert incomplete.page_completeness_pct == 80.0
    assert incomplete.is_fully_complete is False


def test_pipeline_runner_syncs_to_database_with_scraperun(db_session):
    """Verifies end-to-end integration: ScrapeRun creation, sync_listing, and observation recording."""
    mock_category_spider = MagicMock(spec=RiyasewanaCategorySpider)
    mock_category_spider.scrape_category.return_value = {
        "category_name": "Cars",
        "category_url": "https://riyasewana.com/search/cars",
        "pages_discovered": 1,
        "pages_attempted": 1,
        "pages_scraped": 1,
        "failed_pages": [],
        "listing_urls_discovered": 1,
        "unique_listing_urls": ["https://riyasewana.com/buy/toyota-premio-12217090"],
        "listings_attempted": 1,
        "listings_scraped": 1,
        "failed_listings": [],
        "records": [
            {
                "listing_id": "12217090",
                "listing_url": "https://riyasewana.com/buy/toyota-premio-12217090",
                "title": "Toyota Premio 2017",
                "category": "Cars",
                "brand": "Toyota",
                "model": "Premio",
                "year": 2017,
                "price": 14500000,
                "mileage": 55000,
                "fuel_type": "Petrol",
                "transmission": "Automatic",
                "condition": "Used",
                "location": "Colombo",
                "district": "Colombo",
                "source": "riyasewana",
                "is_valid": True,
                "validation_issues": [],
            }
        ],
        "status": "COMPLETED",
    }

    runner = PipelineRunner(category_spider=mock_category_spider)

    report = runner.run_pipeline(
        categories=["Cars"],
        dry_run=False,
        export_csv=False,
        db_session=db_session,
    )

    assert report["overall_status"] == "COMPLETED"
    assert report["categories_processed"] == 1
    assert report["category_results"][0]["new_listings"] == 1

    # Verify database persistence
    repo = VehicleRepository(db_session)
    listing = repo.find_listing("12217090", source="riyasewana")
    assert listing is not None
    assert listing.title == "Toyota Premio 2017"
    assert listing.current_status == "ACTIVE"

    # Verify ScrapeRun
    scrape_runs = db_session.scalars(select(ScrapeRun)).all()
    assert len(scrape_runs) == 1
    assert scrape_runs[0].status == "COMPLETED"
    assert scrape_runs[0].category == "Cars"
    assert scrape_runs[0].listings_found == 1
    assert scrape_runs[0].new_listings == 1

    # Verify ListingObservation relationship
    observations = db_session.scalars(select(ListingObservation)).all()
    assert len(observations) == 1
    assert observations[0].listing_id == listing.id
    assert observations[0].scrape_run_id == scrape_runs[0].id


def test_pipeline_runner_idempotent_sync_and_price_changes(db_session):
    """Verifies that second scrape run updates last_seen_at and only adds PriceHistory on change."""
    mock_category_spider = MagicMock(spec=RiyasewanaCategorySpider)

    record_run1 = {
        "listing_id": "12217091",
        "listing_url": "https://riyasewana.com/buy/suzuki-alto-12217091",
        "title": "Suzuki Alto 2015",
        "category": "Cars",
        "brand": "Suzuki",
        "model": "Alto",
        "year": 2015,
        "price": 3200000,
        "mileage": 80000,
        "condition": "Used",
        "source": "riyasewana",
        "is_valid": True,
        "validation_issues": [],
    }

    mock_category_spider.scrape_category.return_value = {
        "category_name": "Cars",
        "category_url": "https://riyasewana.com/search/cars",
        "pages_discovered": 1,
        "pages_attempted": 1,
        "pages_scraped": 1,
        "failed_pages": [],
        "listing_urls_discovered": 1,
        "unique_listing_urls": ["https://riyasewana.com/buy/suzuki-alto-12217091"],
        "listings_attempted": 1,
        "listings_scraped": 1,
        "failed_listings": [],
        "records": [record_run1],
        "status": "COMPLETED",
    }

    runner = PipelineRunner(category_spider=mock_category_spider)

    # First run
    runner.run_pipeline(categories=["Cars"], dry_run=False, export_csv=False, db_session=db_session)

    repo = VehicleRepository(db_session)
    listing = repo.find_listing("12217091", source="riyasewana")
    assert len(listing.price_history) == 1
    assert listing.price_history[0].price == 3200000

    # Second run with same price -> NO new PriceHistory
    runner.run_pipeline(categories=["Cars"], dry_run=False, export_csv=False, db_session=db_session)
    db_session.refresh(listing)
    assert len(listing.price_history) == 1

    # Third run with price reduction -> creates new PriceHistory
    record_run3 = dict(record_run1)
    record_run3["price"] = 3000000
    mock_category_spider.scrape_category.return_value["records"] = [record_run3]

    runner.run_pipeline(categories=["Cars"], dry_run=False, export_csv=False, db_session=db_session)
    db_session.refresh(listing)
    assert len(listing.price_history) == 2
    prices = [ph.price for ph in listing.price_history]
    assert 3200000 in prices
    assert 3000000 in prices


def test_pipeline_runner_dry_run_does_not_mutate_db(db_session):
    """Verifies that running in dry-run mode writes 0 rows to PostgreSQL tables."""
    mock_category_spider = MagicMock(spec=RiyasewanaCategorySpider)
    mock_category_spider.scrape_category.return_value = {
        "category_name": "Cars",
        "category_url": "https://riyasewana.com/search/cars",
        "pages_discovered": 1,
        "pages_attempted": 1,
        "pages_scraped": 1,
        "failed_pages": [],
        "listing_urls_discovered": 1,
        "unique_listing_urls": ["https://riyasewana.com/buy/dry-run-listing-100"],
        "listings_attempted": 1,
        "listings_scraped": 1,
        "failed_listings": [],
        "records": [
            {
                "listing_id": "dry-run-100",
                "listing_url": "https://riyasewana.com/buy/dry-run-listing-100",
                "title": "Dry Run Car",
                "category": "Cars",
                "brand": "Toyota",
                "model": "Allion",
                "price": 10000000,
                "is_valid": True,
                "validation_issues": [],
            }
        ],
        "status": "COMPLETED",
    }

    runner = PipelineRunner(category_spider=mock_category_spider)

    report = runner.run_pipeline(
        categories=["Cars"],
        dry_run=True,
        export_csv=False,
        db_session=db_session,
    )

    assert report["dry_run"] is True

    # Ensure 0 listings and 0 scrape_runs were added
    listings = db_session.scalars(select(Listing)).all()
    assert len(listings) == 0

    scrape_runs = db_session.scalars(select(ScrapeRun)).all()
    assert len(scrape_runs) == 0


def test_pipeline_runner_category_isolation(db_session):
    """Verifies that an exception in one category does not prevent subsequent categories from running."""
    mock_category_spider = MagicMock(spec=RiyasewanaCategorySpider)

    def side_effect(category, **kwargs):
        cat_name = category.name if hasattr(category, "name") else str(category)
        if cat_name == "Cars":
            raise RuntimeError("Cars network failure")
        return {
            "category_name": cat_name,
            "category_url": f"https://riyasewana.com/search/{cat_name.lower()}",
            "pages_discovered": 1,
            "pages_attempted": 1,
            "pages_scraped": 1,
            "failed_pages": [],
            "listing_urls_discovered": 1,
            "unique_listing_urls": [f"https://riyasewana.com/buy/{cat_name.lower()}-01"],
            "listings_attempted": 1,
            "listings_scraped": 1,
            "failed_listings": [],
            "records": [],
            "status": "COMPLETED",
        }

    mock_category_spider.scrape_category.side_effect = side_effect

    runner = PipelineRunner(category_spider=mock_category_spider)

    report = runner.run_pipeline(
        categories=["Cars", "Vans"],
        dry_run=False,
        export_csv=False,
        db_session=db_session,
    )

    assert report["categories_processed"] == 2
    assert report["category_results"][0]["category_name"] == "Cars"
    assert report["category_results"][0]["status"] == "FAILED"
    assert report["category_results"][1]["category_name"] == "Vans"
    assert report["category_results"][1]["status"] == "COMPLETED"
    assert report["overall_status"] == "INCOMPLETE"

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
import httpx

from config import settings
from data_pipeline.completeness import CategoryCompleteness
from data_pipeline.pipeline_runner import PipelineRunner
from scraper.discovery.category_discovery import (
    DiscoveredCategory,
    DiscoveredPage,
    RiyasewanaCategoryDiscovery,
)
from scraper.discovery.listing_discovery import RiyasewanaListingDiscovery
from scraper.spiders.riyasewana_bulk_spider import RiyasewanaBulkSpider
from scraper.spiders.riyasewana_category_spider import RiyasewanaCategorySpider
from scraper.utils.retry import execute_with_retry, get_retry_after, is_transient_error


class MockClient:
    def __init__(self, pages_dict):
        self.pages = pages_dict
        self.requested_urls = []

    def get(self, url: str) -> str:
        self.requested_urls.append(url)
        if url in self.pages:
            res = self.pages[url]
            if isinstance(res, Exception):
                raise res
            return res
        raise RuntimeError(f"404 Not Found: {url}")

    def close(self):
        pass


# ----------------------------------------------------
# 1. max_pages enforcement
# ----------------------------------------------------
def test_max_pages_enforcement():
    """Verifies that pagination discovery halts strictly at max_pages."""
    pages = {
        f"https://riyasewana.com/buy/cars?page={i}": f"""
        <html><body>
            <a href="/buy/car-{i}1-100">Car {i}1</a>
            <a rel="next" href="/buy/cars?page={i+1}">Next</a>
        </body></html>
        """
        for i in range(1, 10)
    }
    pages["https://riyasewana.com/buy/cars"] = pages["https://riyasewana.com/buy/cars?page=1"]

    client = MockClient(pages)
    discovery = RiyasewanaCategoryDiscovery(client=client)

    result = discovery.discover_pages(
        "https://riyasewana.com/buy/cars",
        max_pages=3,
    )

    assert len(result) == 3
    assert [p.page_number for p in result] == [1, 2, 3]
    # Traversal should not have requested page 4 or beyond
    assert not any("page=4" in url for url in client.requested_urls)


# ----------------------------------------------------
# 2. max_listings enforcement & early pagination stop
# ----------------------------------------------------
def test_max_listings_enforcement_and_early_stop():
    """
    Verifies that when max_listings is satisfied from early pages,
    subsequent pagination pages are not inspected and at most max_listings detail pages are scraped.
    """
    page1_html = """
    <html><body>
        <a href="/buy/car-1-101">Car 1</a>
        <a href="/buy/car-2-102">Car 2</a>
        <a href="/buy/car-3-103">Car 3</a>
        <a href="/buy/car-4-104">Car 4</a>
        <a href="/buy/car-5-105">Car 5</a>
    </body></html>
    """
    page2_html = """
    <html><body>
        <a href="/buy/car-6-106">Car 6</a>
    </body></html>
    """
    discovered_pages = [
        DiscoveredPage(url="https://riyasewana.com/buy/cars?page=1", page_number=1, html=page1_html),
        DiscoveredPage(url="https://riyasewana.com/buy/cars?page=2", page_number=2, html=page2_html),
    ]

    mock_discovery = MagicMock(spec=RiyasewanaCategoryDiscovery)
    mock_discovery.discover_pages.return_value = discovered_pages

    mock_bulk_spider = MagicMock(spec=RiyasewanaBulkSpider)
    mock_bulk_spider.scrape_listings.return_value = {
        "successful": [{"listing_id": f"10{i}", "price": 1000000} for i in range(1, 3)],
        "failed": [],
        "total_attempted": 2,
        "total_successful": 2,
        "total_failed": 0,
    }

    spider = RiyasewanaCategorySpider(
        category_discovery=mock_discovery,
        bulk_spider=mock_bulk_spider,
        request_delay=0.0,
    )

    result = spider.scrape_category(
        category="Cars",
        max_pages=2,
        max_listings=2,
    )

    # Scraped listing URLs passed to bulk spider must be capped at max_listings
    mock_bulk_spider.scrape_listings.assert_called_once()
    called_urls = mock_bulk_spider.scrape_listings.call_args[1]["urls"]
    assert len(called_urls) >= 2
    assert mock_bulk_spider.scrape_listings.call_args[1]["max_listings"] == 2
    assert result["listings_scraped"] == 2
    assert result["status"] == "COMPLETED"


# ----------------------------------------------------
# 3. pagination loop prevention
# ----------------------------------------------------
def test_pagination_loop_prevention():
    """Verifies that cyclical pagination links break cleanly without an infinite loop."""
    pages = {
        "https://riyasewana.com/buy/cars": """
        <html><body>
            <a rel="next" href="/buy/cars?page=2">Next</a>
        </body></html>
        """,
        "https://riyasewana.com/buy/cars?page=2": """
        <html><body>
            <a rel="next" href="/buy/cars">Cycle Back to Page 1</a>
        </body></html>
        """,
    }
    client = MockClient(pages)
    discovery = RiyasewanaCategoryDiscovery(client=client)

    result = discovery.discover_pages("https://riyasewana.com/buy/cars", max_pages=10)
    assert len(result) == 2
    assert result[0].url == "https://riyasewana.com/buy/cars"
    assert result[1].url == "https://riyasewana.com/buy/cars?page=2"


# ----------------------------------------------------
# 4. duplicate listing URL removal
# ----------------------------------------------------
def test_duplicate_listing_url_removal():
    """Verifies that duplicate listing links appearing on a page are deduplicated."""
    html = """
    <html><body>
        <a href="/buy/toyota-vitz-12345">Toyota Vitz</a>
        <a href="/buy/toyota-vitz-12345">Toyota Vitz (Duplicate)</a>
        <a href="/buy/honda-fit-67890">Honda Fit</a>
    </body></html>
    """
    ld = RiyasewanaListingDiscovery()
    urls = ld.discover_listing_urls(html, "https://riyasewana.com/buy/cars")

    assert len(urls) == 2
    assert urls == [
        "https://riyasewana.com/buy/honda-fit-67890",
        "https://riyasewana.com/buy/toyota-vitz-12345",
    ]


# ----------------------------------------------------
# 5. listing failure isolation
# ----------------------------------------------------
def test_listing_failure_isolation():
    """Verifies that a failure on one listing detail page does not abort remaining listings."""
    mock_spider = MagicMock()
    mock_validator = MagicMock()

    def fake_scrape(url):
        if "failing-car" in url:
            raise RuntimeError("Corrupted HTML on listing page")
        return {
            "listing_id": url.split("-")[-1],
            "listing_url": url,
            "title": "Good Car",
            "price": 5000000,
        }

    mock_spider.scrape_listing.side_effect = fake_scrape
    mock_validator.validate.side_effect = lambda rec: dict(rec, is_valid=True, validation_issues=[])

    bulk_spider = RiyasewanaBulkSpider(
        spider=mock_spider,
        validator=mock_validator,
        max_retries=1,
        request_delay=0.0,
    )

    urls = [
        "https://riyasewana.com/buy/good-car-101",
        "https://riyasewana.com/buy/failing-car-102",
        "https://riyasewana.com/buy/good-car-103",
    ]

    result = bulk_spider.scrape_listings(urls)

    assert result["total_attempted"] == 3
    assert result["total_successful"] == 2
    assert result["total_failed"] == 1
    assert result["failed"][0]["listing_id"] == "102"
    assert "Corrupted HTML" in result["failed"][0]["error"]
    assert [r["listing_id"] for r in result["successful"]] == ["101", "103"]


# ----------------------------------------------------
# 6. page failure accounting & INCOMPLETE status
# ----------------------------------------------------
def test_page_failure_accounting_and_incomplete_status():
    """Verifies that failed pagination pages are tracked and categorize the run as INCOMPLETE."""
    mock_discovery = MagicMock(spec=RiyasewanaCategoryDiscovery)
    mock_discovery.discover_pages.return_value = [
        DiscoveredPage(url="https://riyasewana.com/buy/cars?page=1", page_number=1, html=None),
        DiscoveredPage(url="https://riyasewana.com/buy/cars?page=2", page_number=2, html=None),
    ]

    mock_client = MagicMock()
    def fake_get(url):
        if "page=2" in url:
            raise RuntimeError("500 Internal Server Error on Page 2")
        return '<a href="/buy/toyota-101">Toyota</a>'

    mock_client.get.side_effect = fake_get

    mock_bulk = MagicMock(spec=RiyasewanaBulkSpider)
    mock_bulk.scrape_listings.return_value = {
        "successful": [{"listing_id": "101", "price": 4000000}],
        "failed": [],
        "total_attempted": 1,
        "total_successful": 1,
        "total_failed": 0,
    }

    category_spider = RiyasewanaCategorySpider(
        client=mock_client,
        category_discovery=mock_discovery,
        bulk_spider=mock_bulk,
        max_retries=1,
        request_delay=0.0,
    )

    res = category_spider.scrape_category("Cars")

    assert res["pages_attempted"] == 2
    assert res["pages_scraped"] == 1
    assert len(res["failed_pages"]) == 1
    assert res["failed_pages"][0]["page_number"] == 2
    assert res["status"] == "INCOMPLETE"


# ----------------------------------------------------
# 7. Retry-After header handling
# ----------------------------------------------------
def test_retry_after_header_handling():
    """Verifies that Retry-After header <= max_retry_after is respected and retried."""
    mock_resp = MagicMock()
    mock_resp.headers = {"Retry-After": "2"}
    mock_resp.status_code = 429
    err_429 = httpx.HTTPStatusError("429 Rate Limit", request=MagicMock(), response=mock_resp)

    assert get_retry_after(err_429) == 2.0
    assert is_transient_error(err_429) is True

    attempts = 0
    with patch("time.sleep") as mock_sleep:
        def flaky():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise err_429
            return "SUCCESS"

        result = execute_with_retry(flaky, max_retries=2, max_retry_after=30.0)
        assert result == "SUCCESS"
        assert attempts == 2
        mock_sleep.assert_called_with(2.0)


# ----------------------------------------------------
# 8. Excessive Retry-After safe abort
# ----------------------------------------------------
def test_excessive_retry_after_safe_abort():
    """Verifies that an upstream Retry-After > 30s aborts retries immediately to protect pipeline."""
    mock_resp = MagicMock()
    mock_resp.headers = {"Retry-After": "300"}  # 5 minutes
    mock_resp.status_code = 429
    err_429 = httpx.HTTPStatusError("429 Rate Limit", request=MagicMock(), response=mock_resp)

    attempts = 0
    def blocked():
        nonlocal attempts
        attempts += 1
        raise err_429

    with pytest.raises(httpx.HTTPStatusError):
        execute_with_retry(blocked, max_retries=3, max_retry_after=30.0)

    # Aborts on attempt 1 without sleeping for 300s
    assert attempts == 1


# ----------------------------------------------------
# 9. Completeness reporting with scope
# ----------------------------------------------------
def test_completeness_reporting_scoped():
    """Verifies that CategoryCompleteness cleanly distinguishes scoped completion from whole-site completion."""
    comp = CategoryCompleteness(
        category_name="Cars",
        pages_discovered=5,
        pages_attempted=2,
        pages_scraped=2,
        failed_pages=[],
        listing_urls_discovered=80,
        unique_listing_urls=80,
        listings_attempted=10,
        listings_scraped=10,
        failed_listings=[],
        status="COMPLETED",
        max_pages_requested=2,
        max_listings_requested=10,
    )

    assert comp.is_fully_complete is True
    assert comp.page_completeness_pct == 100.0
    assert comp.listing_completeness_pct == 100.0

    d = comp.to_dict()
    assert d["is_scoped"] is True
    assert d["max_pages_requested"] == 2
    assert d["max_listings_requested"] == 10

    summary = comp.format_summary()
    assert "max_pages=2, max_listings=10" in summary
    assert "Scope Complete:   YES" in summary
    assert "Completeness is evaluated within the requested collection scope" in summary


# ----------------------------------------------------
# 10. Multi-category isolation
# ----------------------------------------------------
def test_multi_category_isolation():
    """Verifies that an unhandled error in one category does not abort other categories."""
    mock_cat_spider = MagicMock()

    def fake_scrape(category, max_pages=None, max_listings=None):
        cat_name = category if isinstance(category, str) else category.name
        if cat_name == "Heavy-Duty":
            raise RuntimeError("Heavy-Duty category network failure")
        return {
            "records": [{"listing_id": f"{cat_name}_01", "price": 3000000}],
            "pages_discovered": 1,
            "pages_attempted": 1,
            "pages_scraped": 1,
            "failed_pages": [],
            "listing_urls_discovered": 1,
            "unique_listing_urls": ["url1"],
            "listings_attempted": 1,
            "listings_scraped": 1,
            "failed_listings": [],
            "status": "COMPLETED",
        }

    mock_cat_spider.scrape_category.side_effect = fake_scrape

    runner = PipelineRunner(category_spider=mock_cat_spider)

    report = runner.run_pipeline(
        categories=["Cars", "Heavy-Duty", "Vans"],
        dry_run=True,
        export_csv=False,
    )

    assert report["categories_processed"] == 3
    results = {r["category_name"]: r["status"] for r in report["category_results"]}
    assert results["Cars"] == "COMPLETED"
    assert results["Heavy-Duty"] == "FAILED"
    assert results["Vans"] == "COMPLETED"
    assert report["overall_status"] == "INCOMPLETE"


# ----------------------------------------------------
# 11. Dry-run safety: no DB mutation
# ----------------------------------------------------
def test_dry_run_no_database_mutation(vehicle_repo):
    """Verifies that dry_run=True performs scraping without mutating PostgreSQL."""
    mock_cat_spider = MagicMock()
    mock_cat_spider.scrape_category.return_value = {
        "records": [
            {
                "listing_id": "dry_run_car_999",
                "listing_url": "https://riyasewana.com/buy/dry-run-car-999",
                "title": "Dry Run Car",
                "category": "Cars",
                "price": 5000000,
            }
        ],
        "pages_discovered": 1,
        "pages_attempted": 1,
        "pages_scraped": 1,
        "failed_pages": [],
        "listing_urls_discovered": 1,
        "unique_listing_urls": ["https://riyasewana.com/buy/dry-run-car-999"],
        "listings_attempted": 1,
        "listings_scraped": 1,
        "failed_listings": [],
        "status": "COMPLETED",
    }

    runner = PipelineRunner(category_spider=mock_cat_spider)

    # Run in dry-run mode with db_session provided
    report = runner.run_pipeline(
        categories=["Cars"],
        dry_run=True,
        export_csv=False,
        db_session=vehicle_repo.db,
    )

    assert report["dry_run"] is True
    # Verify that listing was NOT written to database
    found = vehicle_repo.find_listing("dry_run_car_999")
    assert found is None

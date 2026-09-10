import logging
import time
from typing import Any, Dict, List, Optional

from config import settings
from scraper.client import RiyasewanaClient
from scraper.discovery.category_discovery import (
    DiscoveredCategory,
    DiscoveredPage,
    RiyasewanaCategoryDiscovery,
)
from scraper.discovery.listing_discovery import RiyasewanaListingDiscovery
from scraper.spiders.riyasewana_bulk_spider import RiyasewanaBulkSpider
from scraper.utils.retry import execute_with_retry

logger = logging.getLogger(__name__)


class RiyasewanaCategorySpider:
    """
    Coordinates end-to-end data collection for a single vehicle category:
    Category -> Pagination Pages -> Listing URLs -> Individual Listings.
    Isolates page-level and listing-level failures so transient errors do not abort the scrape.
    """

    def __init__(
        self,
        client: Optional[RiyasewanaClient] = None,
        category_discovery: Optional[RiyasewanaCategoryDiscovery] = None,
        listing_discovery: Optional[RiyasewanaListingDiscovery] = None,
        bulk_spider: Optional[RiyasewanaBulkSpider] = None,
        max_retries: int = settings.MAX_RETRIES,
        retry_delay: float = settings.RETRY_DELAY,
        retry_backoff: float = settings.RETRY_BACKOFF,
        request_delay: float = settings.REQUEST_DELAY,
    ):
        self.client = client or RiyasewanaClient()
        self.category_discovery = (
            category_discovery or RiyasewanaCategoryDiscovery(self.client)
        )
        self.listing_discovery = (
            listing_discovery or RiyasewanaListingDiscovery()
        )
        self.bulk_spider = bulk_spider or RiyasewanaBulkSpider(
            max_retries=max_retries,
            retry_delay=retry_delay,
            retry_backoff=retry_backoff,
            request_delay=request_delay,
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.request_delay = request_delay

    def scrape_category(
        self,
        category: DiscoveredCategory | str,
        max_pages: Optional[int] = None,
        max_listings: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes coordinated scraping for a category across all accessible pagination pages.
        """
        if isinstance(category, str):
            category_name = category
            category_url = (
                category
                if category.startswith("http")
                else f"https://riyasewana.com/search/{category.lower()}"
            )
        else:
            category_name = category.name
            category_url = category.url

        logger.info(f"Starting category scrape: {category_name} ({category_url})")

        # 1. Discover all pagination pages for the category (limited by max_pages)
        discovered_pages = self.category_discovery.discover_pages(
            category_url,
            max_pages=max_pages,
            request_delay=self.request_delay,
        )
        target_pages = discovered_pages

        if not target_pages:
            logger.warning(
                f"No pagination pages could be discovered for {category_name} ({category_url})"
            )
            return {
                "category_name": category_name,
                "category_url": category_url,
                "pages_discovered": 0,
                "pages_attempted": 1,
                "pages_scraped": 0,
                "failed_pages": [
                    {
                        "url": category_url,
                        "page_number": 1,
                        "error": "Failed to discover pagination pages (network or rate limit error)",
                        "retries_attempted": 0,
                    }
                ],
                "listing_urls_discovered": 0,
                "unique_listing_urls": [],
                "listings_attempted": 0,
                "listings_scraped": 0,
                "failed_listings": [],
                "records": [],
                "status": "FAILED",
            }

        pages_attempted = 0
        pages_scraped = 0
        failed_pages: List[Dict[str, Any]] = []

        all_listing_urls_discovered: List[str] = []
        unique_listing_urls: set[str] = set()
        stopped_early_for_max_listings = False

        # 2. Scrape each pagination page to extract listing URLs
        for index, page in enumerate(target_pages):
            pages_attempted += 1
            retries = 0

            def _fetch_page_html() -> str:
                nonlocal retries
                return self.client.get(page.url)

            def _on_retry(exc: Exception, attempt: int, delay: float):
                nonlocal retries
                retries = attempt
                logger.warning(
                    f"Retry {attempt}/{self.max_retries} for page {page.url}: {exc}"
                )

            try:
                if page.html:
                    html = page.html
                else:
                    html = execute_with_retry(
                        _fetch_page_html,
                        max_retries=self.max_retries,
                        initial_delay=self.retry_delay,
                        backoff_factor=self.retry_backoff,
                        on_retry=_on_retry,
                    )

                page_listing_urls = self.listing_discovery.discover_listing_urls(
                    html=html,
                    page_url=page.url,
                )

                all_listing_urls_discovered.extend(page_listing_urls)
                unique_listing_urls.update(page_listing_urls)
                pages_scraped += 1

                # Stop inspecting further pagination pages if max_listings is satisfied
                if max_listings is not None and len(unique_listing_urls) >= max_listings:
                    stopped_early_for_max_listings = True
                    logger.info(
                        f"Found {len(unique_listing_urls)} listing URLs, satisfying requested max_listings={max_listings}. "
                        "Stopping pagination inspection early."
                    )
                    break

            except Exception as exc:
                logger.error(
                    f"Failed to scrape page {page.url} after {retries + 1} attempts: {exc}"
                )
                failed_pages.append(
                    {
                        "url": page.url,
                        "page_number": page.page_number,
                        "error": str(exc),
                        "retries_attempted": retries,
                    }
                )

            # Politeness delay between page requests if live request was made
            if not page.html and self.request_delay > 0 and index < len(target_pages) - 1:
                time.sleep(self.request_delay)

        sorted_unique_urls = sorted(unique_listing_urls)

        # Check if pagination naturally reached the end of the category
        natural_pagination_exhausted = True
        if max_pages is not None and len(discovered_pages) >= max_pages:
            last_page = discovered_pages[-1]
            if last_page.html:
                has_next = bool(self.category_discovery._find_next_page(last_page.html, last_page.url))
                if has_next:
                    natural_pagination_exhausted = False
            else:
                natural_pagination_exhausted = False

        pagination_exhausted = (
            natural_pagination_exhausted
            and not stopped_early_for_max_listings
            and pages_scraped == len(target_pages)
            and len(failed_pages) == 0
        )

        # 3. Bulk scrape individual vehicle listings
        bulk_results = self.bulk_spider.scrape_listings(
            urls=sorted_unique_urls,
            max_listings=max_listings,
        )

        # Determine status
        if pages_scraped == 0 and pages_attempted > 0:
            status = "FAILED"
        elif (
            len(failed_pages) > 0
            or (pages_attempted > 0 and pages_scraped < pages_attempted)
            or len(bulk_results.get("failed", [])) > 0
        ):
            status = "INCOMPLETE"
        else:
            status = "COMPLETED"

        return {
            "category_name": category_name,
            "category_url": category_url,
            "pages_discovered": len(discovered_pages),
            "pages_attempted": pages_attempted,
            "pages_scraped": pages_scraped,
            "failed_pages": failed_pages,
            "listing_urls_discovered": len(all_listing_urls_discovered),
            "unique_listing_urls": sorted_unique_urls,
            "listings_attempted": bulk_results["total_attempted"],
            "listings_scraped": bulk_results["total_successful"],
            "failed_listings": bulk_results["failed"],
            "records": bulk_results["successful"],
            "pagination_exhausted": pagination_exhausted,
            "status": status,
        }

    def close(self):
        """Closes spider and client sessions."""
        self.client.close()
        self.bulk_spider.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

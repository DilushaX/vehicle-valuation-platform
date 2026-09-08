import logging
import time
from typing import Any, Dict, List, Optional

from config import settings
from scraper.discovery.listing_discovery import RiyasewanaListingDiscovery
from scraper.spiders.riyasewana_spider import RiyasewanaSpider
from scraper.utils.retry import execute_with_retry
from scraper.validators.listing_validator import ListingValidator

logger = logging.getLogger(__name__)


class RiyasewanaBulkSpider:
    """
    Bulk scraping spider for multiple individual Riyasewana vehicle listing URLs.
    Reuses existing RiyasewanaSpider, RiyasewanaParser, VehicleExtractor, and ListingValidator.
    Isolates per-listing failures, applies limited retries with backoff, and tracks failed listings.
    """

    def __init__(
        self,
        spider: Optional[RiyasewanaSpider] = None,
        validator: Optional[ListingValidator] = None,
        max_retries: int = settings.MAX_RETRIES,
        retry_delay: float = settings.RETRY_DELAY,
        retry_backoff: float = settings.RETRY_BACKOFF,
        request_delay: float = settings.REQUEST_DELAY,
    ):
        self.spider = spider or RiyasewanaSpider()
        self.validator = validator or ListingValidator()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.request_delay = request_delay

    def scrape_listings(
        self,
        urls: List[str],
        max_listings: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Scrapes a batch of listing URLs with error isolation and validation.

        Returns a dictionary containing:
        - successful: list of validated, normalized vehicle records
        - failed: list of failed listing descriptors with error details
        - total_attempted: count of URLs processed
        - total_successful: count of successfully scraped records
        - total_failed: count of failed listings
        """
        target_urls = urls[:max_listings] if max_listings is not None else urls

        successful: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []

        for index, url in enumerate(target_urls):
            listing_id = RiyasewanaListingDiscovery.extract_listing_id(url)
            retries_attempted = 0

            def _fetch_and_extract() -> Dict[str, Any]:
                nonlocal retries_attempted
                return self.spider.scrape_listing(url)

            def _on_retry(exc: Exception, attempt: int, delay: float):
                nonlocal retries_attempted
                retries_attempted = attempt
                logger.warning(
                    f"Retry {attempt}/{self.max_retries} for listing {url}: {exc}"
                )

            try:
                raw_record = execute_with_retry(
                    _fetch_and_extract,
                    max_retries=self.max_retries,
                    initial_delay=self.retry_delay,
                    backoff_factor=self.retry_backoff,
                    on_retry=_on_retry,
                )

                validated_record = self.validator.validate(raw_record)
                successful.append(validated_record)

            except Exception as exc:
                logger.error(
                    f"Failed to scrape listing {url} after {retries_attempted + 1} attempts: {exc}"
                )
                failed.append(
                    {
                        "url": url,
                        "listing_id": listing_id,
                        "error": str(exc),
                        "retries_attempted": retries_attempted,
                    }
                )

            # Politeness delay between requests
            if self.request_delay > 0 and index < len(target_urls) - 1:
                time.sleep(self.request_delay)

        return {
            "successful": successful,
            "failed": failed,
            "total_attempted": len(target_urls),
            "total_successful": len(successful),
            "total_failed": len(failed),
        }

    def close(self):
        """Closes the underlying spider client session."""
        self.spider.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

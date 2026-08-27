"""
Riyasewana Spider Module
Coordinates multi-category incremental delta scraping, polite crawling,
and structured record assembly.
"""
import logging
from typing import List, Dict, Any, Set, Tuple
from config import settings
from scraper.client import PoliteHttpClient
from scraper.parsers.riyasewana_parser import RiyasewanaParser

logger = logging.getLogger(__name__)

CATEGORY_PATH_MAP = {
    "cars": "cars",
    "vans": "vans",
    "suvs": "suvs",
    "motorbikes": "motorbikes",
    "three-wheel": "three-wheel",
    "lorries": "lorries",
    "buses": "buses"
}

class RiyasewanaSpider:
    def __init__(
        self,
        http_client: PoliteHttpClient = None,
        max_pages: int = settings.SCRAPER_MAX_PAGES_PER_RUN
    ):
        self.client = http_client or PoliteHttpClient()
        self.max_pages = max_pages

    def crawl_category(
        self,
        category: str = "cars",
        known_listing_ids: Set[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Crawls a specific category using incremental delta scraping.
        Stops paginating when consecutive known listing IDs are encountered.
        Returns: (parsed_records, all_observed_listing_ids)
        """
        known_ids = known_listing_ids or set()
        cat_slug = CATEGORY_PATH_MAP.get(category.lower(), "cars")
        
        parsed_records: List[Dict[str, Any]] = []
        observed_ids: List[str] = []

        consecutive_known_count = 0
        max_consecutive_known = 15  # Stop early if 15 existing ads are seen in a row

        logger.info(f"Starting crawl for category '{category}' (Max pages: {self.max_pages})...")

        for page in range(1, self.max_pages + 1):
            page_url = f"{settings.SCRAPER_BASE_URL}/search/{cat_slug}?page={page}"
            logger.info(f"Fetching search page: {page_url}")
            
            html = self.client.get(page_url)
            if not html:
                logger.warning(f"Empty response for {page_url}. Ending pagination.")
                break

            card_items = RiyasewanaParser.parse_search_page(html, category=category.capitalize())
            if not card_items:
                logger.info(f"No more listings found on page {page}. Done.")
                break

            stop_pagination = False
            for card in card_items:
                lid = card["listing_id"]
                observed_ids.append(lid)

                # Check delta condition
                if lid in known_ids:
                    consecutive_known_count += 1
                    if consecutive_known_count >= max_consecutive_known:
                        logger.info(f"Reached {consecutive_known_count} known listings. Delta threshold reached.")
                        stop_pagination = True
                        break
                else:
                    consecutive_known_count = 0

                # Fetch detail page
                if card.get("url"):
                    detail_html = self.client.get(card["url"])
                    if detail_html:
                        record = RiyasewanaParser.parse_detail_page(detail_html, fallback_data=card)
                        parsed_records.append(record)
                    else:
                        # Use card fallback
                        parsed_records.append(card)

            if stop_pagination:
                break

        logger.info(f"Finished crawling '{category}'. Extracted {len(parsed_records)} records.")
        return parsed_records, observed_ids

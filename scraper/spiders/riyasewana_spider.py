import logging
from typing import Dict, Any, Optional
from scraper.client import RiyasewanaClient
from scraper.parsers.riyasewana_parser import RiyasewanaParser
from scraper.extractors.vehicle_extractor import VehicleExtractor

logger = logging.getLogger(__name__)


class RiyasewanaSpider:
    """
    Spider for fetching and processing individual vehicle listings from Riyasewana.
    """

    def __init__(self, client: Optional[RiyasewanaClient] = None):
        self.client = client or RiyasewanaClient()
        self.parser = RiyasewanaParser()
        self.extractor = VehicleExtractor()

    def scrape_listing(self, url: str) -> Dict[str, Any]:
        """
        Scrapes a single listing URL end-to-end:
        1. Fetch HTML using HTTP Client
        2. Extract raw fields using HTML Parser
        3. Normalize attributes using Vehicle Extractor
        """
        logger.info(f"Fetching listing from URL: {url}")
        html = self.client.get(url)

        logger.info("Parsing raw HTML fields...")
        raw_data = self.parser.parse_listing(html, url)

        logger.info("Normalizing extracted attributes...")
        normalized = self.extractor.normalize(raw_data)

        return normalized

    def close(self):
        """Close client connection."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

"""
Polite HTTP Client
Implements rate-limiting, exponential backoff, browser emulation, and resilient error recovery.
"""
import time
import random
import logging
from typing import Optional, Dict, Any
import httpx
from config import settings

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0"
]

class PoliteHttpClient:
    def __init__(
        self,
        min_delay: float = settings.SCRAPER_MIN_DELAY_SECONDS,
        max_delay: float = settings.SCRAPER_MAX_DELAY_SECONDS,
        timeout: int = settings.SCRAPER_TIMEOUT_SECONDS,
        max_retries: int = 3
    ):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.last_request_time = 0.0

    def _get_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,si;q=0.8",
            "Referer": settings.SCRAPER_BASE_URL,
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

    def _wait_polite_interval(self) -> None:
        """Enforces polite rate limiting between sequential requests."""
        elapsed = time.time() - self.last_request_time
        target_delay = random.uniform(self.min_delay, self.max_delay)
        if elapsed < target_delay:
            time.sleep(target_delay - elapsed)
        self.last_request_time = time.time()

    def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Performs a polite synchronous GET request with exponential backoff."""
        for attempt in range(1, self.max_retries + 1):
            self._wait_polite_interval()
            try:
                with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                    response = client.get(url, params=params, headers=self._get_headers())
                    if response.status_code == 200:
                        return response.text
                    elif response.status_code == 429:
                        wait_time = (2 ** attempt) + random.uniform(1.0, 3.0)
                        logger.warning(f"Rate limited (429) on {url}. Backing off for {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    elif response.status_code in [500, 502, 503, 504]:
                        logger.warning(f"Server error ({response.status_code}) on {url}. Retry {attempt}/{self.max_retries}...")
                        time.sleep(2 ** attempt)
                    else:
                        logger.error(f"Failed request to {url}: Status {response.status_code}")
                        return None
            except Exception as e:
                logger.error(f"Request error on {url} (attempt {attempt}): {e}")
                time.sleep(2 ** attempt)
        return None

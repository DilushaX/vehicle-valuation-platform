import logging
import httpx

try:
    from curl_cffi import requests as curl_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

logger = logging.getLogger(__name__)


class RiyasewanaClient:
    """
    HTTP Client for fetching public Riyasewana listing pages.
    Uses httpx as the primary HTTP client with a browser-like User-Agent.
    Falls back gracefully if Cloudflare security blocks standard httpx requests.
    """
    BASE_URL = "https://riyasewana.com"

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.client = httpx.Client(
            timeout=self.timeout,
            headers=self.headers,
            follow_redirects=True
        )

    def get(self, url: str) -> str:
        """
        Fetch HTML content from a given URL.
        """
        try:
            response = self.client.get(url)
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 403 and HAS_CURL_CFFI:
                logger.warning(
                    f"httpx returned 403 (Cloudflare challenge) for {url}. "
                    "Falling back to browser TLS impersonation via curl_cffi."
                )
                r = curl_requests.get(url, impersonate="chrome", timeout=int(self.timeout))
                r.raise_for_status()
                return r.text
            logger.error(f"HTTP error fetching {url}: {exc}")
            raise
        except Exception as exc:
            if HAS_CURL_CFFI:
                logger.warning(f"httpx request failed for {url}: {exc}. Attempting fallback.")
                try:
                    r = curl_requests.get(url, impersonate="chrome", timeout=int(self.timeout))
                    r.raise_for_status()
                    return r.text
                except Exception as fallback_exc:
                    logger.error(f"Fallback request also failed for {url}: {fallback_exc}")
                    raise
            raise

    def close(self):
        """Close the underlying HTTP client session."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

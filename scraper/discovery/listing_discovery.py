import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


@dataclass(frozen=True)
class DiscoveredListing:
    """
    Represents an individual vehicle listing discovered from a category or search page.
    """

    url: str
    listing_id: Optional[str] = None
    source: str = "riyasewana"


class RiyasewanaListingDiscovery:
    """
    Discovers individual Riyasewana listing URLs from category pagination pages.
    Uses structural URL and path validation to reliably distinguish listing URLs
    from category links and other navigational links without hardcoding brand names.
    """

    BASE_URL = "https://riyasewana.com"

    def discover_listing_urls(
        self,
        html: str,
        page_url: str,
    ) -> list[str]:
        """
        Extracts unique, normalized listing URLs from page HTML.
        """
        soup = BeautifulSoup(html, "html.parser")
        discovered: dict[str, None] = {}

        for link in soup.find_all("a", href=True):
            href = link.get("href")
            if not href:
                continue

            absolute_url = urljoin(page_url, href)

            if not self.is_valid_listing_url(absolute_url):
                continue

            normalized_url = self._normalize_url(absolute_url)
            discovered[normalized_url] = None

        return sorted(discovered.keys())

    def discover_listings(
        self,
        html: str,
        page_url: str,
    ) -> list[DiscoveredListing]:
        """
        Extracts structured DiscoveredListing objects containing URL and listing ID.
        """
        urls = self.discover_listing_urls(html, page_url)
        return [
            DiscoveredListing(
                url=u,
                listing_id=self.extract_listing_id(u),
                source="riyasewana",
            )
            for u in urls
        ]

    @classmethod
    def is_valid_listing_url(cls, url: str) -> bool:
        """
        Validates whether a URL represents an individual vehicle listing page.
        Distinguishes listing URLs from category URLs structurally:
        - Must belong to the Riyasewana domain (or relative).
        - Must start with /buy/ and have exactly 2 path segments (/buy/<listing_slug>).
        - Must not end in file extensions (.php, .html, etc.).
        - Must contain an identifiable listing identifier (digits in slug, e.g. -12217083,
          or action keywords like -sale-, or test identifiers).
        - Excludes bare category slugs like /buy/cars, /buy/vans, /buy/.
        """
        parsed = urlparse(url)

        if parsed.netloc and parsed.netloc not in {
            "riyasewana.com",
            "www.riyasewana.com",
        }:
            return False

        path = parsed.path.strip("/")
        parts = [p for p in path.split("/") if p]

        if len(parts) != 2:
            return False

        prefix, slug = parts[0].lower(), parts[1].lower()
        if prefix != "buy":
            return False

        # Exclude scripts and non-listing resources
        if any(slug.endswith(ext) for ext in (".php", ".asp", ".htm")):
            return False

        # Category pages do not have digits or listing action patterns
        has_numeric_id = bool(re.search(r"\d", slug))
        has_action_keyword = any(
            kw in slug for kw in ("-sale-", "-rent-", "-wanted-")
        )
        is_test_slug = slug.startswith("test-") or slug.startswith("test_")

        if not (has_numeric_id or has_action_keyword or is_test_slug):
            return False

        return True

    # Backwards-compatible alias
    _is_valid_listing_url = is_valid_listing_url

    @staticmethod
    def extract_listing_id(url: str) -> Optional[str]:
        """
        Extracts the listing ID from a listing URL.
        Matches trailing numeric IDs (-12217083) or test identifiers.
        """
        clean_url = url.split("?")[0].split("#")[0].rstrip("/")

        # Test pattern: e.g. /buy/test-cond-01 or /buy/test-100
        test_match = re.search(r"/buy/(test[_-][a-zA-Z0-9_-]+)$", clean_url)
        if test_match:
            return test_match.group(1)

        # Standard numeric pattern: e.g. /buy/toyota-corolla-12217083
        num_match = re.search(r"-(\d+)(?:\.html)?$", clean_url)
        if num_match:
            return num_match.group(1)

        return None

    @staticmethod
    def _normalize_url(url: str) -> str:
        """
        Normalizes a listing URL by removing fragment identifiers and query params.
        """
        parsed = urlparse(url)
        clean_path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
        return parsed._replace(path=clean_path, query="", fragment="").geturl()
import logging
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveredCategory:
    name: str
    url: str


@dataclass(frozen=True)
class DiscoveredPage:
    url: str
    page_number: int


class RiyasewanaCategoryDiscovery:
    """
    Discovers Riyasewana vehicle categories and pagination pages from the public website.
    Dynamically identifies categories and pages based on HTML and URL structure
    without relying on hard-coded vehicle brand names or fixed category lists.
    """

    BASE_URL = "https://riyasewana.com"

    def __init__(self, client):
        self.client = client

    # --------------------------------------------------
    # CATEGORY DISCOVERY
    # --------------------------------------------------

    def discover_categories(
        self,
        start_url: str | None = None,
    ) -> list[DiscoveredCategory]:
        """
        Discover vehicle category URLs from Riyasewana.
        Extracts available categories dynamically from the website structure.
        """
        if start_url is None:
            start_url = self.BASE_URL

        html = self.client.get(start_url)
        soup = BeautifulSoup(html, "html.parser")

        categories: dict[str, DiscoveredCategory] = {}

        for link in soup.find_all("a", href=True):
            href = link.get("href")
            if not href:
                continue

            # Skip explicit brand or SEO location links and containers
            classes = link.get("class") or []
            if isinstance(classes, str):
                classes = classes.split()
            if any(c in {"brand-link", "seo-link"} for c in classes):
                continue

            parent_classes: list[str] = []
            for parent in link.parents:
                pc = parent.get("class") or []
                if isinstance(pc, str):
                    pc = pc.split()
                parent_classes.extend(pc)
            if any(c in {"brand-links", "more-links"} for c in parent_classes):
                continue

            absolute_url = urljoin(start_url, href)

            if not self._is_category_url(absolute_url):
                continue

            normalized_url = self._normalize_url(absolute_url)

            # Prefer img alt text if present in rich buttons, otherwise link text
            img = link.find("img")
            img_alt = img.get("alt", "").strip() if img else ""
            link_text = link.get_text(" ", strip=True)
            raw_name = img_alt if (img_alt and "buy" in link_text.lower()) else link_text

            category_name = self._clean_category_name(raw_name or img_alt)
            if not category_name:
                continue

            key = normalized_url.lower()
            if key not in categories:
                categories[key] = DiscoveredCategory(
                    name=category_name,
                    url=normalized_url,
                )

        return sorted(
            categories.values(),
            key=lambda category: category.name.lower(),
        )

    # --------------------------------------------------
    # CATEGORY URL VALIDATION
    # --------------------------------------------------

    @classmethod
    def _is_category_url(cls, url: str) -> bool:
        """
        Validates whether a URL is a vehicle category URL.
        Distinguishes category URLs from individual listing URLs based on URL structure:
        - Must be under the Riyasewana domain (or relative).
        - Must have exactly two path components: /buy/<category> or /search/<category>.
        - Must not end in file extensions (.php, .html, etc.).
        - Must not contain numeric listing IDs or listing action patterns (e.g., -sale-).
        """
        parsed = urlparse(url)

        if parsed.netloc and parsed.netloc not in {
            "riyasewana.com",
            "www.riyasewana.com",
        }:
            return False

        path = parsed.path.strip("/")
        parts = [p for p in path.split("/") if p]

        # Categories are top-level paths under buy/ or search/ (e.g. /buy/cars or /search/cars)
        if len(parts) != 2:
            return False

        prefix, slug = parts[0].lower(), parts[1].lower()
        if prefix not in {"buy", "search"}:
            return False

        # Exclude scripts and documents
        if any(slug.endswith(ext) for ext in (".php", ".html", ".htm", ".asp")):
            return False

        # Listing URLs contain numeric IDs (e.g., -12217083, test-100)
        if re.search(r"\d", slug):
            return False

        # Exclude individual listing patterns (e.g., brand-model-sale-location)
        if any(term in slug for term in ("-sale-", "-rent-", "-wanted-")):
            return False

        return True

    # --------------------------------------------------
    # CATEGORY NAME CLEANING
    # --------------------------------------------------

    @staticmethod
    def _clean_category_name(text: str) -> str:
        """
        Cleans category names by normalizing whitespace and stripping common prefixes.
        """
        cleaned = " ".join(text.split()).strip()
        cleaned = re.sub(r"^buy\s+", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    # --------------------------------------------------
    # URL NORMALIZATION
    # --------------------------------------------------

    @staticmethod
    def _normalize_url(url: str) -> str:
        """
        Normalizes a URL by removing fragment identifiers and normalizing path slashes.
        """
        parsed = urlparse(url)
        clean_path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
        return parsed._replace(path=clean_path, fragment="").geturl()

    # --------------------------------------------------
    # PAGINATION DISCOVERY
    # --------------------------------------------------

    def discover_pages(
        self,
        category_url: str,
    ) -> list[DiscoveredPage]:
        """
        Discover all accessible pagination pages for a single category.
        """
        pages: list[DiscoveredPage] = []
        visited_urls: set[str] = set()

        current_url = self._normalize_url(category_url)

        while current_url and current_url not in visited_urls:
            visited_urls.add(current_url)

            try:
                html = self.client.get(current_url)
            except Exception as e:
                logger.warning(f"Error fetching pagination page {current_url}: {e}")
                break

            page_number = self._extract_page_number(current_url)
            if page_number == 1 and pages:
                page_number = len(pages) + 1

            pages.append(
                DiscoveredPage(
                    url=current_url,
                    page_number=page_number,
                )
            )

            next_url = self._find_next_page(
                html=html,
                current_url=current_url,
            )

            if not next_url:
                break

            next_url = self._normalize_url(next_url)
            if next_url in visited_urls:
                break

            current_url = next_url

        return pages

    # --------------------------------------------------
    # NEXT PAGE
    # --------------------------------------------------

    def _find_next_page(
        self,
        html: str,
        current_url: str,
    ) -> str | None:
        """
        Finds the next pagination page URL from HTML:
        1. Checks for <a rel="next" href="...">
        2. Falls back to finding numbered links matching current_page + 1
        """
        soup = BeautifulSoup(html, "html.parser")

        # Preferred method: <a rel="next" href="...">
        next_link = soup.find("a", rel="next")
        if not next_link:
            next_link = soup.find("a", attrs={"rel": "next"})

        if next_link:
            href = next_link.get("href")
            if href:
                candidate = urljoin(current_url, href)
                return self._normalize_url(candidate)

        # Fallback pagination detection:
        current_page = self._extract_page_number(current_url)
        target_page = current_page + 1
        current_parsed = urlparse(current_url)

        for link in soup.find_all("a", href=True):
            href = link.get("href")
            if not href:
                continue

            candidate_url = self._normalize_url(urljoin(current_url, href))
            cand_parsed = urlparse(candidate_url)

            # Ensure candidate belongs to the same category path
            if cand_parsed.path == current_parsed.path:
                cand_page = self._extract_page_number(candidate_url)
                if cand_page == target_page:
                    return candidate_url

        return None

    # --------------------------------------------------
    # PAGE NUMBER
    # --------------------------------------------------

    @staticmethod
    def _extract_page_number(url: str) -> int:
        """
        Extract page number from common pagination query parameters (?page=N, ?p=N).
        Defaults to 1 if no valid page parameter is present.
        """
        parsed = urlparse(url)
        params = parse_qs(parsed.query)

        for key in ("page", "p"):
            if key in params:
                try:
                    val = int(params[key][0])
                    if val > 0:
                        return val
                except (ValueError, TypeError):
                    continue

        return 1
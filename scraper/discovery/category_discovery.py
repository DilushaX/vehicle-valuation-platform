from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, parse_qs

from bs4 import BeautifulSoup


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
    Discovers Riyasewana vehicle categories and
    pagination pages from the public website.
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

        The website structure is treated as the source
        of truth instead of hard-coding category URLs.
        """

        if start_url is None:
            start_url = self.BASE_URL

        html = self.client.get(start_url)

        soup = BeautifulSoup(html, "html.parser")

        categories: dict[str, DiscoveredCategory] = {}

        for link in soup.find_all("a", href=True):
            href = link.get("href")
            text = link.get_text(" ", strip=True)

            if not href or not text:
                continue

            absolute_url = urljoin(
                start_url,
                href,
            )

            if not self._is_category_url(
                absolute_url
            ):
                continue

            normalized_url = self._normalize_url(
                absolute_url
            )

            category_name = self._clean_category_name(
                text
            )

            if not category_name:
                continue

            key = normalized_url.lower()

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
    def _is_category_url(
        cls,
        url: str,
    ) -> bool:
        parsed = urlparse(url)

        if parsed.netloc not in {
            "riyasewana.com",
            "www.riyasewana.com",
        }:
            return False

        path = parsed.path.rstrip("/")

        # Category pages normally belong to /buy/
        if not path.startswith("/buy/"):
            return False

        # Avoid individual vehicle listings.
        if path.startswith("/buy/") and "/buy/" in path:
            remaining_path = path[len("/buy/"):]

            if remaining_path.startswith(
                (
                    "toyota-",
                    "honda-",
                    "suzuki-",
                    "nissan-",
                    "mitsubishi-",
                )
            ):
                return False

        return True

    # --------------------------------------------------
    # CATEGORY NAME CLEANING
    # --------------------------------------------------

    @staticmethod
    def _clean_category_name(
        text: str,
    ) -> str:
        return " ".join(
            text.split()
        ).strip()

    # --------------------------------------------------
    # URL NORMALIZATION
    # --------------------------------------------------

    @staticmethod
    def _normalize_url(
        url: str,
    ) -> str:
        parsed = urlparse(url)

        return parsed._replace(
            fragment="",
        ).geturl()

    # --------------------------------------------------
    # PAGINATION DISCOVERY
    # --------------------------------------------------

    def discover_pages(
        self,
        category_url: str,
    ) -> list[DiscoveredPage]:
        """
        Discover all accessible pagination pages
        for a single category.
        """

        pages: list[DiscoveredPage] = []

        visited_urls: set[str] = set()

        current_url = category_url
        page_number = 1

        while (
            current_url
            and current_url not in visited_urls
        ):
            visited_urls.add(current_url)

            try:
                html = self.client.get(
                    current_url
                )
            except Exception:
                break

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

            if next_url in visited_urls:
                break

            current_url = next_url
            page_number += 1

        return pages

    # --------------------------------------------------
    # NEXT PAGE
    # --------------------------------------------------

    def _find_next_page(
        self,
        html: str,
        current_url: str,
    ) -> str | None:

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        # Preferred method:
        # <a rel="next" href="...">
        next_link = soup.find(
            "a",
            attrs={"rel": "next"},
        )

        if next_link:
            href = next_link.get("href")

            if href:
                return urljoin(
                    current_url,
                    href,
                )

        # Fallback pagination detection
        current_page = (
            self._extract_page_number(
                current_url
            )
        )

        candidates: list[str] = []

        for link in soup.find_all(
            "a",
            href=True,
        ):
            href = link.get("href")

            if not href:
                continue

            absolute_url = urljoin(
                current_url,
                href,
            )

            candidate_page = (
                self._extract_page_number(
                    absolute_url
                )
            )

            if (
                candidate_page
                == current_page + 1
            ):
                candidates.append(
                    absolute_url
                )

        if not candidates:
            return None

        return candidates[0]

    # --------------------------------------------------
    # PAGE NUMBER
    # --------------------------------------------------

    @staticmethod
    def _extract_page_number(
        url: str,
    ) -> int:

        parsed = urlparse(url)

        params = parse_qs(
            parsed.query
        )

        for key in (
            "page",
            "p",
        ):
            if key not in params:
                continue

            try:
                return int(
                    params[key][0]
                )
            except (
                ValueError,
                TypeError,
            ):
                continue

        return 1
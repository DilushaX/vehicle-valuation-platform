from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


class RiyasewanaListingDiscovery:
    """
    Discovers individual Riyasewana listing URLs
    from a category page.
    """

    BASE_URL = "https://riyasewana.com"

    def discover_listing_urls(
        self,
        html: str,
        page_url: str,
    ) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")

        listing_urls: set[str] = set()

        for link in soup.find_all("a", href=True):
            href = link["href"]

            absolute_url = urljoin(
                page_url,
                href,
            )

            if not self._is_valid_listing_url(
                absolute_url
            ):
                continue

            listing_urls.add(
                self._normalize_url(absolute_url)
            )

        return sorted(listing_urls)

    @staticmethod
    def _is_valid_listing_url(url: str) -> bool:
        parsed = urlparse(url)

        if parsed.netloc not in {
            "riyasewana.com",
            "www.riyasewana.com",
        }:
            return False

        return parsed.path.startswith("/buy/")

    @staticmethod
    def _normalize_url(url: str) -> str:
        """
        Remove unnecessary URL fragments.
        """

        parsed = urlparse(url)

        return parsed._replace(
            fragment=""
        ).geturl()
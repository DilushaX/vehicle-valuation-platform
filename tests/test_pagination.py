import pytest
from scraper.discovery.category_discovery import (
    DiscoveredPage,
    RiyasewanaCategoryDiscovery,
)


class FakeClient:
    """Mock HTTP client returning pre-configured HTML for testing."""

    def __init__(self, pages: dict[str, str] | None = None):
        self.pages = pages or {}

    def get(self, url: str) -> str:
        if url not in self.pages:
            raise RuntimeError(f"Page not found in fake client: {url}")
        return self.pages[url]


def test_page_1_detection():
    """Verifies that URLs without page parameters default to page 1."""
    assert (
        RiyasewanaCategoryDiscovery._extract_page_number(
            "https://riyasewana.com/buy/cars"
        )
        == 1
    )
    assert (
        RiyasewanaCategoryDiscovery._extract_page_number(
            "https://riyasewana.com/search/cars"
        )
        == 1
    )


def test_page_param_page_2_detection():
    """Verifies that ?page=2 query parameter is extracted as page 2."""
    assert (
        RiyasewanaCategoryDiscovery._extract_page_number(
            "https://riyasewana.com/buy/cars?page=2"
        )
        == 2
    )
    assert (
        RiyasewanaCategoryDiscovery._extract_page_number(
            "https://riyasewana.com/search/cars?page=15"
        )
        == 15
    )


def test_page_param_p_2_detection():
    """Verifies that ?p=2 (and other values) query parameter is extracted correctly."""
    assert (
        RiyasewanaCategoryDiscovery._extract_page_number(
            "https://riyasewana.com/buy/cars?p=2"
        )
        == 2
    )
    assert (
        RiyasewanaCategoryDiscovery._extract_page_number(
            "https://riyasewana.com/buy/cars?p=5"
        )
        == 5
    )


def test_invalid_page_parameter_handling():
    """Verifies non-integer or invalid page parameter values safely default to 1."""
    assert (
        RiyasewanaCategoryDiscovery._extract_page_number(
            "https://riyasewana.com/buy/cars?page=abc"
        )
        == 1
    )
    assert (
        RiyasewanaCategoryDiscovery._extract_page_number(
            "https://riyasewana.com/buy/cars?page=-5"
        )
        == 1
    )
    assert (
        RiyasewanaCategoryDiscovery._extract_page_number(
            "https://riyasewana.com/buy/cars?page="
        )
        == 1
    )
    assert (
        RiyasewanaCategoryDiscovery._extract_page_number(
            "https://riyasewana.com/buy/cars?p=invalid"
        )
        == 1
    )


def test_next_page_discovery_using_rel_next():
    """Verifies preferred next-page detection using <a rel="next" href="...">."""
    discovery = RiyasewanaCategoryDiscovery(client=None)

    html = """
    <html>
        <body>
            <a rel="next" href="/buy/cars?page=2">Next &gt;</a>
        </body>
    </html>
    """
    next_url = discovery._find_next_page(
        html=html,
        current_url="https://riyasewana.com/buy/cars",
    )

    assert next_url == "https://riyasewana.com/buy/cars?page=2"


def test_fallback_pagination_discovery():
    """Verifies fallback next-page detection when rel="next" is absent but numeric links exist."""
    discovery = RiyasewanaCategoryDiscovery(client=None)

    html = """
    <html>
        <body>
            <div class="pagination">
                <span class="current">1</span>
                <a href="/buy/cars?page=2">2</a>
                <a href="/buy/cars?page=3">3</a>
            </div>
        </body>
    </html>
    """
    next_url = discovery._find_next_page(
        html=html,
        current_url="https://riyasewana.com/buy/cars",
    )

    assert next_url == "https://riyasewana.com/buy/cars?page=2"


def test_pagination_stops_when_no_next_page_exists():
    """Verifies discovery stops when no next page link or candidates are present."""
    pages = {
        "https://riyasewana.com/buy/cars": """
        <html>
            <body>
                <p>Only one page of results.</p>
            </body>
        </html>
        """
    }
    client = FakeClient(pages)
    discovery = RiyasewanaCategoryDiscovery(client)

    result = discovery.discover_pages("https://riyasewana.com/buy/cars")

    assert len(result) == 1
    assert result[0].page_number == 1
    assert result[0].url == "https://riyasewana.com/buy/cars"


def test_pagination_loop_prevention():
    """Verifies that circular pagination links do not cause an infinite loop."""
    pages = {
        "https://riyasewana.com/buy/cars": """
        <a rel="next" href="/buy/cars?page=2">Next</a>
        """,
        "https://riyasewana.com/buy/cars?page=2": """
        <a rel="next" href="/buy/cars">Previous (Loop)</a>
        """,
    }
    client = FakeClient(pages)
    discovery = RiyasewanaCategoryDiscovery(client)

    result = discovery.discover_pages("https://riyasewana.com/buy/cars")

    assert len(result) == 2
    assert result[0].url == "https://riyasewana.com/buy/cars"
    assert result[1].url == "https://riyasewana.com/buy/cars?page=2"


def test_duplicate_pages_are_not_returned():
    """Verifies duplicate URLs or URLs differing only by fragment are not duplicated."""
    pages = {
        "https://riyasewana.com/buy/cars": """
        <a rel="next" href="/buy/cars?page=2#top">Next</a>
        """,
        "https://riyasewana.com/buy/cars?page=2": """
        <a rel="next" href="/buy/cars?page=2#bottom">Self Loop</a>
        """,
    }
    client = FakeClient(pages)
    discovery = RiyasewanaCategoryDiscovery(client)

    result = discovery.discover_pages("https://riyasewana.com/buy/cars")

    assert len(result) == 2
    assert result[0].page_number == 1
    assert result[1].page_number == 2


def test_discovers_multiple_pages_end_to_end():
    """Verifies full traversal across 3 pagination pages."""
    pages = {
        "https://riyasewana.com/buy/cars": """
        <a rel="next" href="/buy/cars?page=2">Next</a>
        """,
        "https://riyasewana.com/buy/cars?page=2": """
        <a rel="next" href="/buy/cars?page=3">Next</a>
        """,
        "https://riyasewana.com/buy/cars?page=3": """
        <html><body>Final page without next link</body></html>
        """,
    }
    client = FakeClient(pages)
    discovery = RiyasewanaCategoryDiscovery(client)

    result = discovery.discover_pages("https://riyasewana.com/buy/cars")

    assert len(result) == 3
    assert [p.page_number for p in result] == [1, 2, 3]
    assert result[0].url == "https://riyasewana.com/buy/cars"
    assert result[1].url == "https://riyasewana.com/buy/cars?page=2"
    assert result[2].url == "https://riyasewana.com/buy/cars?page=3"
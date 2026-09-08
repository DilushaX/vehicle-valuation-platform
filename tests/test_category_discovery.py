import pytest
from scraper.discovery.category_discovery import (
    DiscoveredCategory,
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


def test_category_links_can_be_discovered():
    """Verifies valid category links are discovered and mapped to DiscoveredCategory objects."""
    html = """
    <html>
        <body>
            <a href="/buy/cars">Cars</a>
            <a href="/buy/motorbikes">Motorbikes</a>
            <a href="/buy/vans">Vans</a>
        </body>
    </html>
    """
    client = FakeClient({"https://riyasewana.com": html})
    discovery = RiyasewanaCategoryDiscovery(client)

    categories = discovery.discover_categories()

    assert len(categories) == 3
    urls = {c.url for c in categories}
    assert "https://riyasewana.com/buy/cars" in urls
    assert "https://riyasewana.com/buy/motorbikes" in urls
    assert "https://riyasewana.com/buy/vans" in urls


def test_external_links_are_ignored():
    """Verifies links pointing to external domains are excluded from discovery."""
    html = """
    <html>
        <body>
            <a href="https://google.com">Google</a>
            <a href="https://external-domain.com/buy/cars">External Category</a>
            <a href="/buy/cars">Cars</a>
        </body>
    </html>
    """
    client = FakeClient({"https://riyasewana.com": html})
    discovery = RiyasewanaCategoryDiscovery(client)

    categories = discovery.discover_categories()

    assert len(categories) == 1
    assert categories[0].name == "Cars"
    assert categories[0].url == "https://riyasewana.com/buy/cars"


def test_duplicate_category_urls_are_removed():
    """Verifies that duplicate category links or links with fragments are deduplicated."""
    html = """
    <html>
        <body>
            <a href="/buy/cars">Cars</a>
            <a href="/buy/cars#top">Cars Top</a>
            <a href="https://riyasewana.com/buy/cars/">Cars with Trailing Slash</a>
        </body>
    </html>
    """
    client = FakeClient({"https://riyasewana.com": html})
    discovery = RiyasewanaCategoryDiscovery(client)

    categories = discovery.discover_categories()

    assert len(categories) == 1
    assert categories[0].url == "https://riyasewana.com/buy/cars"


def test_category_urls_are_normalized():
    """Verifies category URLs are resolved to absolute URLs and stripped of fragments."""
    html = """
    <html>
        <body>
            <a href="/buy/suvs#heading">SUVs</a>
        </body>
    </html>
    """
    client = FakeClient({"https://riyasewana.com": html})
    discovery = RiyasewanaCategoryDiscovery(client)

    categories = discovery.discover_categories()

    assert len(categories) == 1
    assert categories[0].url == "https://riyasewana.com/buy/suvs"


def test_category_names_are_extracted_correctly():
    """Verifies whitespace normalization, prefix removal, and rich button label extraction."""
    html = """
    <html>
        <body>
            <a href="/buy/cars">   Cars   </a>
            <a href="/buy/three-wheels">Buy\nThree Wheels</a>
            <a class="gbtn2" href="/search/lorries">
                <img alt="Lorries" src="/images/lorry.png"/>
                <span>Buy<br/>Lorries</span>
            </a>
        </body>
    </html>
    """
    client = FakeClient({"https://riyasewana.com": html})
    discovery = RiyasewanaCategoryDiscovery(client)

    categories = discovery.discover_categories()

    assert len(categories) == 3
    names = [c.name for c in categories]
    assert "Cars" in names
    assert "Three Wheels" in names
    assert "Lorries" in names


def test_invalid_and_non_category_links_are_ignored():
    """
    Verifies that non-category URLs (listings with IDs, contact pages,
    search root, scripts, etc.) are excluded without hard-coding vehicle brands.
    """
    html = """
    <html>
        <body>
            <a href="/contact">Contact Us</a>
            <a href="/about">About</a>
            <a href="/search">Search All</a>
            <a href="/buy/">Buy Root</a>
            <a href="/spare-parts-accessories.php">Spare Parts</a>
            <a href="/buy/toyota-corolla-sale-homagama-12217083">Toyota Corolla Sale</a>
            <a href="/buy/honda-civic-sale-kandy-12217084">Honda Civic Listing</a>
            <a href="/buy/test-listing-100">Test Listing</a>
            <a href="/buy/cars">Cars</a>
        </body>
    </html>
    """
    client = FakeClient({"https://riyasewana.com": html})
    discovery = RiyasewanaCategoryDiscovery(client)

    categories = discovery.discover_categories()

    assert len(categories) == 1
    assert categories[0].name == "Cars"
    assert categories[0].url == "https://riyasewana.com/buy/cars"


def test_discovered_categories_sorted_by_name():
    """Verifies that discovered categories are returned sorted alphabetically by category name."""
    html = """
    <html>
        <body>
            <a href="/buy/vans">Vans</a>
            <a href="/buy/cars">Cars</a>
            <a href="/buy/motorbikes">Motorbikes</a>
        </body>
    </html>
    """
    client = FakeClient({"https://riyasewana.com": html})
    discovery = RiyasewanaCategoryDiscovery(client)

    categories = discovery.discover_categories()

    assert len(categories) == 3
    assert [c.name for c in categories] == ["Cars", "Motorbikes", "Vans"]
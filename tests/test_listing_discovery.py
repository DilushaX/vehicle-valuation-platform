import pytest
from scraper.discovery.listing_discovery import (
    DiscoveredListing,
    RiyasewanaListingDiscovery,
)


def test_valid_listing_url_extraction():
    """Verifies that valid individual listing URLs are extracted from page HTML."""
    html = """
    <html>
        <body>
            <a href="/buy/toyota-corolla-axio-sale-homagama-12217083">Toyota Axio</a>
            <a href="/buy/honda-vezel-z-sale-nugegoda-12217084">Honda Vezel</a>
            <a href="/buy/suzuki-wagon-r-sale-gampaha-12217085">Suzuki Wagon R</a>
        </body>
    </html>
    """
    discovery = RiyasewanaListingDiscovery()
    urls = discovery.discover_listing_urls(
        html, "https://riyasewana.com/search/cars"
    )

    assert len(urls) == 3
    assert (
        "https://riyasewana.com/buy/toyota-corolla-axio-sale-homagama-12217083"
        in urls
    )
    assert (
        "https://riyasewana.com/buy/honda-vezel-z-sale-nugegoda-12217084"
        in urls
    )
    assert (
        "https://riyasewana.com/buy/suzuki-wagon-r-sale-gampaha-12217085"
        in urls
    )


def test_invalid_url_rejection():
    """Verifies non-listing URLs like contact, about, root buy, and scripts are rejected."""
    discovery = RiyasewanaListingDiscovery()

    assert not discovery.is_valid_listing_url("https://riyasewana.com/contact")
    assert not discovery.is_valid_listing_url("https://riyasewana.com/about")
    assert not discovery.is_valid_listing_url("https://riyasewana.com/buy/")
    assert not discovery.is_valid_listing_url("https://riyasewana.com/buy")
    assert not discovery.is_valid_listing_url(
        "https://riyasewana.com/spare-parts-accessories.php"
    )
    assert not discovery.is_valid_listing_url("https://riyasewana.com/search")


def test_external_url_rejection():
    """Verifies URLs on external domains are excluded even if they have /buy/ in the path."""
    html = """
    <html>
        <body>
            <a href="https://external-classifieds.com/buy/toyota-corolla-12217083">External</a>
            <a href="https://google.com/search?q=riyasewana">Google</a>
            <a href="/buy/toyota-corolla-12217083">Internal</a>
        </body>
    </html>
    """
    discovery = RiyasewanaListingDiscovery()
    urls = discovery.discover_listing_urls(
        html, "https://riyasewana.com/search/cars"
    )

    assert len(urls) == 1
    assert urls[0] == "https://riyasewana.com/buy/toyota-corolla-12217083"


def test_duplicate_listing_url_removal():
    """Verifies that multiple occurrences or variations with fragments/queries are deduplicated."""
    html = """
    <html>
        <body>
            <a href="/buy/toyota-corolla-12217083">Link 1</a>
            <a href="/buy/toyota-corolla-12217083#specs">Link 1 with fragment</a>
            <a href="https://riyasewana.com/buy/toyota-corolla-12217083?ref=search">Link 1 with query</a>
        </body>
    </html>
    """
    discovery = RiyasewanaListingDiscovery()
    urls = discovery.discover_listing_urls(
        html, "https://riyasewana.com/search/cars"
    )

    assert len(urls) == 1
    assert urls[0] == "https://riyasewana.com/buy/toyota-corolla-12217083"


def test_url_normalization():
    """Verifies that listing URLs are converted to clean absolute URLs."""
    discovery = RiyasewanaListingDiscovery()
    normalized = discovery._normalize_url(
        "https://riyasewana.com/buy/honda-civic-12217084/?ref=pagination#overview"
    )
    assert (
        normalized
        == "https://riyasewana.com/buy/honda-civic-12217084"
    )


def test_listing_id_extraction():
    """Verifies numeric listing IDs and test listing IDs are extracted accurately."""
    assert (
        RiyasewanaListingDiscovery.extract_listing_id(
            "https://riyasewana.com/buy/toyota-corolla-axio-sale-homagama-12217083"
        )
        == "12217083"
    )
    assert (
        RiyasewanaListingDiscovery.extract_listing_id(
            "https://riyasewana.com/buy/bajaj-qute-sale-marawila-12281922"
        )
        == "12281922"
    )
    assert (
        RiyasewanaListingDiscovery.extract_listing_id(
            "https://riyasewana.com/buy/test-cond-01"
        )
        == "test-cond-01"
    )
    assert (
        RiyasewanaListingDiscovery.extract_listing_id(
            "https://riyasewana.com/buy/cars"
        )
        is None
    )


def test_category_links_are_not_treated_as_listings():
    """
    Verifies that category links (/buy/cars, /buy/motorbikes, etc.)
    are structurally distinguished and rejected as individual listings.
    """
    discovery = RiyasewanaListingDiscovery()

    assert not discovery.is_valid_listing_url("https://riyasewana.com/buy/cars")
    assert not discovery.is_valid_listing_url(
        "https://riyasewana.com/buy/motorbikes"
    )
    assert not discovery.is_valid_listing_url("https://riyasewana.com/buy/vans")
    assert not discovery.is_valid_listing_url("https://riyasewana.com/buy/suvs")
    assert not discovery.is_valid_listing_url(
        "https://riyasewana.com/buy/three-wheels"
    )


def test_unrelated_buy_links_are_handled_correctly():
    """Verifies that sub-paths or non-listing buy URLs are safely rejected."""
    discovery = RiyasewanaListingDiscovery()

    assert not discovery.is_valid_listing_url(
        "https://riyasewana.com/buy/cars/toyota"
    )
    assert not discovery.is_valid_listing_url(
        "https://riyasewana.com/buy/spare-parts.php"
    )
    assert not discovery.is_valid_listing_url("https://riyasewana.com/buy")


def test_discover_listings_structured_objects():
    """Verifies discover_listings returns DiscoveredListing objects with source identity."""
    html = """
    <html>
        <body>
            <a href="/buy/nissan-leaf-12217086">Nissan Leaf</a>
        </body>
    </html>
    """
    discovery = RiyasewanaListingDiscovery()
    listings = discovery.discover_listings(
        html, "https://riyasewana.com/search/cars"
    )

    assert len(listings) == 1
    assert isinstance(listings[0], DiscoveredListing)
    assert listings[0].listing_id == "12217086"
    assert (
        listings[0].url
        == "https://riyasewana.com/buy/nissan-leaf-12217086"
    )
    assert listings[0].source == "riyasewana"

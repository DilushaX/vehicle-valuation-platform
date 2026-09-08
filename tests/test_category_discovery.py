from scraper.discovery.category_discovery import (
    RiyasewanaCategoryDiscovery,
)


class FakeClient:
    def __init__(self, pages):
        self.pages = pages

    def get(self, url):
        if url not in self.pages:
            raise RuntimeError(
                f"Page not found: {url}"
            )

        return self.pages[url]


def test_discovers_multiple_pages():

    pages = {
        "https://riyasewana.com/buy/cars":
            """
            <a rel="next"
               href="/buy/cars?page=2">
               Next
            </a>
            """,

        "https://riyasewana.com/buy/cars?page=2":
            """
            <a rel="next"
               href="/buy/cars?page=3">
               Next
            </a>
            """,

        "https://riyasewana.com/buy/cars?page=3":
            """
            <html>
                Last page
            </html>
            """,
    }

    client = FakeClient(pages)

    discovery = RiyasewanaCategoryDiscovery(
        client
    )

    result = discovery.discover_pages(
        "https://riyasewana.com/buy/cars"
    )

    assert len(result) == 3

    assert result[0].page_number == 1
    assert result[1].page_number == 2
    assert result[2].page_number == 3

    assert result[0].url == (
        "https://riyasewana.com/buy/cars"
    )

    assert result[1].url == (
        "https://riyasewana.com/buy/cars?page=2"
    )

    assert result[2].url == (
        "https://riyasewana.com/buy/cars?page=3"
    )


def test_stops_when_next_page_does_not_exist():

    pages = {
        "https://riyasewana.com/buy/cars":
            """
            <html>
                No next page
            </html>
            """
    }

    client = FakeClient(pages)

    discovery = RiyasewanaCategoryDiscovery(
        client
    )

    result = discovery.discover_pages(
        "https://riyasewana.com/buy/cars"
    )

    assert len(result) == 1


def test_prevents_pagination_loop():

    pages = {
        "https://riyasewana.com/buy/cars":
            """
            <a rel="next"
               href="/buy/cars?page=2">
               Next
            </a>
            """,

        "https://riyasewana.com/buy/cars?page=2":
            """
            <a rel="next"
               href="/buy/cars">
               Previous
            </a>
            """,
    }

    client = FakeClient(pages)

    discovery = RiyasewanaCategoryDiscovery(
        client
    )

    result = discovery.discover_pages(
        "https://riyasewana.com/buy/cars"
    )

    assert len(result) == 2

    from scraper.discovery.category_discovery import (
    RiyasewanaCategoryDiscovery,
)


class FakeClient:
    def __init__(self, pages):
        self.pages = pages

    def get(self, url):
        if url not in self.pages:
            raise RuntimeError(
                f"Page not found: {url}"
            )

        return self.pages[url]


def test_discovers_categories():

    html = """
    <html>
        <body>
            <a href="/buy/cars">
                Cars
            </a>

            <a href="/buy/motorbikes">
                Motorbikes
            </a>

            <a href="/buy/vans">
                Vans
            </a>

            <a href="https://example.com/test">
                External
            </a>
        </body>
    </html>
    """

    client = FakeClient(
        {
            "https://riyasewana.com": html
        }
    )

    discovery = RiyasewanaCategoryDiscovery(
        client
    )

    categories = discovery.discover_categories()

    assert len(categories) == 3

    urls = {
        category.url
        for category in categories
    }

    assert (
        "https://riyasewana.com/buy/cars"
        in urls
    )

    assert (
        "https://riyasewana.com/buy/motorbikes"
        in urls
    )

    assert (
        "https://riyasewana.com/buy/vans"
        in urls
    )


def test_duplicate_category_urls_are_removed():

    html = """
    <html>
        <body>
            <a href="/buy/cars">
                Cars
            </a>

            <a href="/buy/cars#top">
                Cars
            </a>
        </body>
    </html>
    """

    client = FakeClient(
        {
            "https://riyasewana.com": html
        }
    )

    discovery = RiyasewanaCategoryDiscovery(
        client
    )

    categories = discovery.discover_categories()

    assert len(categories) == 1


def test_external_links_are_ignored():

    html = """
    <html>
        <body>
            <a href="https://google.com">
                Google
            </a>

            <a href="/contact">
                Contact
            </a>
        </body>
    </html>
    """

    client = FakeClient(
        {
            "https://riyasewana.com": html
        }
    )

    discovery = RiyasewanaCategoryDiscovery(
        client
    )

    categories = discovery.discover_categories()

    assert categories == []
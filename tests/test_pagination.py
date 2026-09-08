from scraper.discovery.category_discovery import (
    RiyasewanaCategoryDiscovery,
)


def test_extract_page_number():

    assert (
        RiyasewanaCategoryDiscovery._extract_page_number(
            "https://riyasewana.com/buy/cars"
        )
        == 1
    )

    assert (
        RiyasewanaCategoryDiscovery._extract_page_number(
            "https://riyasewana.com/buy/cars?page=2"
        )
        == 2
    )

    assert (
        RiyasewanaCategoryDiscovery._extract_page_number(
            "https://riyasewana.com/buy/cars?p=5"
        )
        == 5
    )


def test_extract_invalid_page_number():

    assert (
        RiyasewanaCategoryDiscovery._extract_page_number(
            "https://riyasewana.com/buy/cars?page=abc"
        )
        == 1
    )


def test_next_page_detection():

    discovery = RiyasewanaCategoryDiscovery(
        client=None
    )

    html = """
    <html>
        <body>
            <a href="/buy/cars?page=2">
                Page 2
            </a>
        </body>
    </html>
    """

    result = discovery._find_next_page(
        html=html,
        current_url="https://riyasewana.com/buy/cars",
    )

    assert result == (
        "https://riyasewana.com/buy/cars?page=2"
    )
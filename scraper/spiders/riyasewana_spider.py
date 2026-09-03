from scraper.client import RiyasewanaClient
from scraper.parsers.riyasewana_parser import RiyasewanaParser


class RiyasewanaSpider:

    def __init__(self):
        self.client = RiyasewanaClient()
        self.parser = RiyasewanaParser()

    def scrape_listing(self, url: str) -> dict:
        html = self.client.get(url)

        return self.parser.parse_listing(
            html=html,
            url=url
        )

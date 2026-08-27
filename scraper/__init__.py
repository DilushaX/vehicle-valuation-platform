"""Scraper module initialization."""
from scraper.client import PoliteHttpClient
from scraper.parsers.riyasewana_parser import RiyasewanaParser
from scraper.spiders.riyasewana_spider import RiyasewanaSpider
from scraper.mock_data_generator import generate_realistic_dataset

__all__ = [
    "PoliteHttpClient",
    "RiyasewanaParser",
    "RiyasewanaSpider",
    "generate_realistic_dataset"
]

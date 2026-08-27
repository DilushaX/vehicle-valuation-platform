"""
Tests for Riyasewana Scraper & Parsers
"""
import pytest
from scraper.parsers.riyasewana_parser import RiyasewanaParser

SAMPLE_SEARCH_HTML = """
<div id="content">
    <ul class="listing">
        <li class="item">
            <h2 class="more"><a href="https://riyasewana.com/buy/toyota-aqua-sale-colombo-8491023">Toyota Aqua 2018</a></h2>
            <div class="boxintxt">Rs. 8,500,000</div>
            <div class="boxintxt">65,000 km, Colombo</div>
        </li>
        <li class="item">
            <h2 class="more"><a href="/buy/honda-vezel-sale-kandy-8491024">Honda Vezel 2016</a></h2>
            <div class="boxintxt">Rs. 10,200,000</div>
            <div class="boxintxt">85,000 km, Kandy</div>
        </li>
    </ul>
</div>
"""

SAMPLE_DETAIL_HTML = """
<html>
    <head><title>Toyota Aqua 2018 For Sale</title></head>
    <body>
        <h1>Toyota Aqua 2018</h1>
        <table>
            <tr><td>Make</td><td>Toyota</td></tr>
            <tr><td>Model</td><td>Aqua</td></tr>
            <tr><td>Year of Manufacture</td><td>2018</td></tr>
            <tr><td>Condition</td><td>Used</td></tr>
            <tr><td>Mileage</td><td>65,000 km</td></tr>
            <tr><td>Fuel Type</td><td>Hybrid</td></tr>
            <tr><td>Gear</td><td>Automatic</td></tr>
            <tr><td>Engine (cc)</td><td>1500 cc</td></tr>
            <tr><td>Location</td><td>Colombo</td></tr>
            <tr><td>Price (Rs)</td><td>Rs. 8,500,000</td></tr>
        </table>
    </body>
</html>
"""

def test_extract_listing_id_from_url():
    url1 = "https://riyasewana.com/buy/toyota-aqua-sale-colombo-8491023"
    assert RiyasewanaParser.extract_listing_id_from_url(url1) == "8491023"

    url2 = "/buy/honda-vezel-sale-kandy-8491024.html"
    assert RiyasewanaParser.extract_listing_id_from_url(url2) == "8491024"

def test_clean_price():
    assert RiyasewanaParser.clean_price("Rs. 8,500,000") == 8500000.0
    assert RiyasewanaParser.clean_price("8500000") == 8500000.0
    assert RiyasewanaParser.clean_price("Invalid") is None

def test_clean_numeric():
    assert RiyasewanaParser.clean_numeric("65,000 km") == 65000
    assert RiyasewanaParser.clean_numeric("1500 cc") == 1500
    assert RiyasewanaParser.clean_numeric("None") is None

def test_parse_search_page():
    items = RiyasewanaParser.parse_search_page(SAMPLE_SEARCH_HTML, category="Cars")
    assert len(items) == 2
    assert items[0]["listing_id"] == "riyasewana_8491023"
    assert "Toyota Aqua 2018" in items[0]["title"]
    assert items[1]["listing_id"] == "riyasewana_8491024"

def test_parse_detail_page():
    parsed = RiyasewanaParser.parse_detail_page(SAMPLE_DETAIL_HTML)
    assert parsed["make"] == "Toyota"
    assert parsed["model"] == "Aqua"
    assert parsed["yom"] == 2018
    assert parsed["mileage_km"] == 65000
    assert parsed["fuel_type"] == "Hybrid"
    assert parsed["transmission"] == "Automatic"
    assert parsed["engine_cc"] == 1500
    assert parsed["price_rs"] == 8500000.0
    assert parsed["district"] == "Colombo"

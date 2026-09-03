import re
from bs4 import BeautifulSoup


class RiyasewanaParser:

    def parse_listing(self, html: str, url: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")

        return {
            "listing_id": self._extract_listing_id(url),
            "listing_url": url,
            "title": self._extract_title(soup),
            "price": self._extract_price(soup),
            "year": self._extract_field(soup, "Year"),
            "mileage": self._extract_mileage(soup),
            "brand": self._extract_field(soup, "Make"),
            "model": self._extract_field(soup, "Model"),
            "gear": self._extract_field(soup, "Gear"),
            "fuel_type": self._extract_field(soup, "Fuel Type"),
            "engine_cc": self._extract_field(soup, "Engine"),
            "condition": self._extract_field(soup, "Condition"),
            "location": self._extract_field(soup, "Location"),
            "ad_date": self._extract_field(soup, "Ad Date"),
            "description": self._extract_description(soup),
            "source": "riyasewana",
        }

    def _extract_listing_id(self, url: str):
        match = re.search(r"-(\d+)$", url.rstrip("/"))

        return match.group(1) if match else None

    def _extract_title(self, soup: BeautifulSoup):
        heading = soup.find("h1")

        if heading:
            return heading.get_text(" ", strip=True)

        return None

    def _extract_price(self, soup: BeautifulSoup):
        text = soup.get_text(" ", strip=True)

        match = re.search(
            r"Rs\.?\s*([\d,]+)",
            text,
            re.IGNORECASE
        )

        if not match:
            return None

        return int(match.group(1).replace(",", ""))

    def _extract_mileage(self, soup: BeautifulSoup):
        text = soup.get_text(" ", strip=True)

        match = re.search(
            r"([\d,]+)\s*km",
            text,
            re.IGNORECASE
        )

        if not match:
            return None

        return int(match.group(1).replace(",", ""))

    def _extract_field(self, soup: BeautifulSoup, field_name: str):
        element = soup.find(
            string=lambda value:
            value and value.strip().lower() == field_name.lower()
        )

        if not element:
            return None

        parent = element.parent

        next_element = parent.find_next()

        if next_element:
            return next_element.get_text(
                " ",
                strip=True
            )

        return None

    def _extract_description(self, soup: BeautifulSoup):
        text = soup.get_text(" ", strip=True)

        return text if text else None

import re
import logging
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class RiyasewanaParser:
    """
    HTML parser for Riyasewana vehicle listing pages.
    Extracts raw structured data fields from both modern div-based layouts
    and legacy table-based HTML layouts.
    """

    KNOWN_CATEGORIES = {
        "cars": "Car",
        "suvs": "SUV",
        "vans": "Van",
        "motorcycles": "Motorbike",
        "three-wheels": "Three Wheel",
        "lorries": "Lorry",
        "buses": "Bus",
        "pickups": "Pickup",
        "heavy-duties": "Heavy-Duty",
    }

    def parse_listing(self, html: str, url: str) -> Dict[str, Any]:
        """
        Parses raw HTML from a Riyasewana listing page and returns a dictionary
        of raw extracted fields.
        """
        soup = BeautifulSoup(html, "html.parser")

        fields = {
            "listing_id": self._extract_listing_id(url),
            "listing_url": url,
            "title": self._extract_title(soup),
            "category": self._extract_category(soup, url),
            "price": self._extract_price(soup),
            "year": self._extract_detail(soup, "Year"),
            "manufacture_year": self._extract_detail(soup, ["Manufacture Year", "Year of Manufacture", "YOM", "Y.O.M", "Mfd Year"]),
            "registration_year": self._extract_detail(soup, ["Registration Year", "Year of Registration", "Registered Year", "First Registration", "1st Registration", "YOR", "Y.O.R", "Reg Year"]),
            "mileage": self._extract_detail(soup, "Mileage"),
            "brand": self._extract_detail(soup, ["Make", "Brand"]),
            "model": self._extract_detail(soup, "Model"),
            "gear": self._extract_detail(soup, ["Gear", "Transmission"]),
            "fuel_type": self._extract_detail(soup, "Fuel Type"),
            "engine_cc": self._extract_detail(soup, ["Engine (cc)", "Engine", "Engine Capacity"]),
            "condition": self._extract_detail(soup, "Condition"),
            "location": self._extract_detail(soup, "Location"),
            "district": self._extract_district(soup),
            "ad_date": self._extract_detail(soup, ["Ad Date", "Date"]),
            "description": self._extract_description(soup),
            "source": "riyasewana",
        }

        return fields

    def _extract_listing_id(self, url: str) -> Optional[str]:
        match = re.search(r"-(\d+)(?:\.html)?$", url.rstrip("/"))
        return match.group(1) if match else None

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(" ", strip=True)
        return None

    def _extract_category(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        # Try breadcrumb links first
        for a in soup.find_all("a", href=True):
            href = a["href"].lower()
            for cat_key, cat_name in self.KNOWN_CATEGORIES.items():
                if f"/search/{cat_key}" in href:
                    return cat_name

        # Try to infer from title or URL
        url_lower = url.lower()
        for cat_key, cat_name in self.KNOWN_CATEGORIES.items():
            if cat_key in url_lower:
                return cat_name

        return "Vehicle"

    def _extract_price(self, soup: BeautifulSoup) -> Optional[str]:
        # Modern element
        price_el = soup.find(class_="price-amount")
        if price_el:
            return price_el.get_text(" ", strip=True)

        # Legacy text search for Rs. pattern
        text = soup.get_text(" ", strip=True)
        match = re.search(r"Rs\.?\s*([\d,]+)", text, re.IGNORECASE)
        if match:
            return f"Rs. {match.group(1)}"
        return None

    def _extract_detail(self, soup: BeautifulSoup, field_names: Any) -> Optional[str]:
        if isinstance(field_names, str):
            field_names = [field_names]

        field_names_lower = [f.lower() for f in field_names]

        # 1. Search in modern div.detail-row
        for row in soup.find_all(class_="detail-row"):
            text = row.get_text(" ", strip=True)
            parts = [p.strip() for p in text.split("|") if p.strip()]
            if len(parts) >= 2:
                label = parts[0].lower()
                if any(lbl in label for lbl in field_names_lower):
                    return " ".join(parts[1:])

        # 2. Search in table rows (legacy layout)
        for tr in soup.find_all("tr"):
            cols = tr.find_all(["td", "th"])
            if len(cols) >= 2:
                label = cols[0].get_text(strip=True).lower().replace(":", "")
                if any(lbl == label for lbl in field_names_lower):
                    return cols[1].get_text(" ", strip=True)

        # 3. Fallback: text search near label
        for fn in field_names:
            element = soup.find(string=lambda value: value and fn.lower() in value.strip().lower())
            if element:
                parent = element.parent
                next_el = parent.find_next()
                if next_el:
                    val = next_el.get_text(" ", strip=True)
                    if val and val.lower() != fn.lower():
                        return val

        return None

    def _extract_district(self, soup: BeautifulSoup) -> Optional[str]:
        # Look for district in location breadcrumbs or details
        location = self._extract_detail(soup, "Location")
        if location:
            # Common Sri Lankan districts mapping helper will be handled in Extractor,
            # but raw district text can be returned here.
            return location
        return None

    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        # Modern container
        more_body = soup.find(class_="more-card-body")
        if more_body:
            return more_body.get_text("\n", strip=True)

        # Legacy container or div id="content"
        content_div = soup.find("div", id="content")
        if content_div:
            # Look for paragraph or description block
            paras = content_div.find_all("p")
            if paras:
                return "\n".join(p.get_text(" ", strip=True) for p in paras)

        return None

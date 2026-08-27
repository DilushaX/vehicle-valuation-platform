"""
Riyasewana HTML Parser
Extracts structured vehicle attributes from search listing cards and detail pages.
"""
import re
import logging
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class RiyasewanaParser:
    @staticmethod
    def extract_listing_id_from_url(url: str) -> Optional[str]:
        """Extracts numerical listing ID from Riyasewana URL (e.g., ...-colombo-8439201 -> 8439201)."""
        match = re.search(r"-(\d+)(?:\.html)?$", url.strip())
        if match:
            return match.group(1)
        match = re.search(r"/buy/.*?(\d+)", url.strip())
        if match:
            return match.group(1)
        return None

    @staticmethod
    def clean_numeric(val: Optional[str]) -> Optional[int]:
        """Strips commas, units, non-digits and returns clean integer."""
        if not val:
            return None
        cleaned = re.sub(r"[^\d]", "", str(val))
        return int(cleaned) if cleaned else None

    @staticmethod
    def clean_price(val: Optional[str]) -> Optional[float]:
        """Extracts currency value from strings like 'Rs. 8,500,000' or '8500000'."""
        if not val:
            return None
        # Remove currency indicators like 'Rs.', 'Rs', 'LKR'
        cleaned_str = re.sub(r"(?i)(rs\.?|lkr)", "", str(val))
        cleaned = re.sub(r"[^\d.]", "", cleaned_str.replace(",", "")).strip(".")
        try:
            return float(cleaned) if cleaned else None
        except ValueError:
            return None

    @staticmethod
    def parse_search_page(html_content: str, category: str = "Cars") -> List[Dict[str, Any]]:
        """Parses a search results page and returns cards with basic info and detail URLs."""
        results = []
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Target content container
        content_div = soup.find("div", id="content") or soup.find("div", class_="content") or soup
        items = content_div.find_all("li", class_=lambda c: c and ("item" in c or "box" in c or "list" in c))
        
        if not items:
            # Fallback to finding all h2 a links
            items = content_div.find_all("li")

        for item in items:
            link_tag = item.find("h2")
            if link_tag:
                link_tag = link_tag.find("a")
            else:
                link_tag = item.find("a", href=re.compile(r"/buy/"))

            if not link_tag or not link_tag.get("href"):
                continue

            url = link_tag.get("href")
            if not url.startswith("http"):
                url = f"https://riyasewana.com{url}"

            listing_id = RiyasewanaParser.extract_listing_id_from_url(url)
            if not listing_id:
                continue

            title = link_tag.get_text(strip=True)
            
            # Extract price text
            price_text = None
            price_tag = item.find(string=re.compile(r"Rs\.?\s*[\d,]+", re.I))
            if price_tag:
                price_text = price_tag.strip()

            # Extract location/district
            district = None
            location_tag = item.find("div", class_="boxintxt")
            if location_tag:
                txt = location_tag.get_text(strip=True)
                parts = txt.split(",")
                if parts:
                    district = parts[-1].strip()

            results.append({
                "listing_id": f"riyasewana_{listing_id}",
                "source": "riyasewana",
                "category": category,
                "title": title,
                "url": url,
                "raw_price": price_text,
                "district": district
            })

        return results

    @staticmethod
    def parse_detail_page(html_content: str, fallback_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parses full detail specification table.
        Extracts: Make, Model, YOM, Condition, Mileage, Fuel, Gear/Transmission, Engine CC, District, Price.
        """
        data = fallback_data.copy() if fallback_data else {}
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract title
        h1 = soup.find("h1")
        if h1:
            data["title"] = h1.get_text(strip=True)

        # Parse key-value specifications from tables or divs
        specs = {}
        for row in soup.find_all("tr"):
            cols = row.find_all(["th", "td"])
            if len(cols) >= 2:
                key = cols[0].get_text(strip=True).lower()
                val = cols[1].get_text(strip=True)
                specs[key] = val

        for div in soup.find_all("div", class_=lambda c: c and "detail" in c.lower()):
            text = div.get_text(separator=":", strip=True)
            parts = text.split(":")
            if len(parts) >= 2:
                specs[parts[0].strip().lower()] = parts[1].strip()

        # Map Make & Model
        make = specs.get("make") or specs.get("brand")
        model = specs.get("model")
        
        # If not structured in table, infer from title (e.g. "Toyota Aqua 2018")
        if not make or not model:
            title = data.get("title", "")
            title_parts = title.split()
            if len(title_parts) >= 2:
                make = make or title_parts[0]
                model = model or title_parts[1]

        data["make"] = make.title() if make else "Unknown"
        data["model"] = model.title() if model else "Unknown"

        # Year of Manufacture (YOM)
        yom_raw = specs.get("year") or specs.get("yom") or specs.get("year of manufacture") or specs.get("manufactured year")
        if not yom_raw and data.get("title"):
            year_match = re.search(r"\b(19\d\d|20[0-2]\d)\b", data["title"])
            if year_match:
                yom_raw = year_match.group(1)
        data["yom"] = RiyasewanaParser.clean_numeric(yom_raw) or 2015

        # Condition
        data["condition"] = specs.get("condition", "Used").title()

        # Mileage
        mileage_raw = specs.get("mileage") or specs.get("mileage (km)") or specs.get("km")
        data["mileage_km"] = RiyasewanaParser.clean_numeric(mileage_raw)

        # Fuel Type
        fuel_raw = specs.get("fuel type") or specs.get("fuel") or "Petrol"
        data["fuel_type"] = fuel_raw.title()

        # Transmission / Gear
        gear_raw = specs.get("gear") or specs.get("transmission") or "Automatic"
        data["transmission"] = gear_raw.title()

        # Engine Capacity (cc)
        engine_raw = specs.get("engine (cc)") or specs.get("engine capacity") or specs.get("engine")
        data["engine_cc"] = RiyasewanaParser.clean_numeric(engine_raw)

        # Location / District
        location_raw = specs.get("location") or specs.get("city") or specs.get("district") or data.get("district")
        data["district"] = location_raw.title() if location_raw else "Colombo"

        # Price
        price_raw = specs.get("price") or specs.get("price (rs)") or specs.get("negotiable price") or data.get("raw_price")
        data["price_rs"] = RiyasewanaParser.clean_price(price_raw) or 0.0

        return data

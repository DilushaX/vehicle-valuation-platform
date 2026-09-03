import re
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class VehicleExtractor:
    """
    Normalizes raw extracted text values into strongly-typed, standardized vehicle attributes.
    Preserves original raw data in the 'raw_data' key for auditing.
    """

    DISTRICT_MAPPING = {
        "homagama": "Colombo",
        "nugegoda": "Colombo",
        "maharagama": "Colombo",
        "kottawa": "Colombo",
        "dehiwala": "Colombo",
        "malabe": "Colombo",
        "colombo": "Colombo",
        "battaramulla": "Colombo",
        "piliyandala": "Colombo",
        "rajagiriya": "Colombo",
        "kaduwela": "Colombo",
        "moratuwa": "Colombo",
        "gampaha": "Gampaha",
        "negombo": "Gampaha",
        "wattala": "Gampaha",
        "kadawatha": "Gampaha",
        "ja-ela": "Gampaha",
        "kelaniya": "Gampaha",
        "kandy": "Kandy",
        "peradeniya": "Kandy",
        "katugastota": "Kandy",
        "galle": "Galle",
        "matara": "Matara",
        "kurunegala": "Kurunegala",
        "kalutara": "Kalutara",
        "jaffna": "Jaffna",
        "ratnapura": "Ratnapura",
        "badulla": "Badulla",
        "anuradhapura": "Anuradhapura",
        "trincomalee": "Trincomalee",
        "batticaloa": "Batticaloa",
        "puttalam": "Puttalam",
        "kegalle": "Kegalle",
        "matale": "Matale",
        "nuwara eliya": "Nuwara Eliya",
        "hambantota": "Hambantota",
    }

    YOM_PATTERN = re.compile(
        r"(?:year\s+of\s+manufactur(?:e|ing)|manufactur(?:e|ing|ed)(?:\s+year)?|y\.?o\.?m\.?|mfd(?:\.?\s+year)?|mfg(?:\.?\s+year)?)\s*(?:[:=\-]|in)?\s*(\b(?:19\d{2}|20\d{2})\b)",
        re.IGNORECASE
    )

    YOR_PATTERN = re.compile(
        r"(?:year\s+of\s+registration|first\s+registration|1st\s+registration|regist(?:ered|ration)(?:\s+year)?|y\.?o\.?r\.?|reg(?:\.|\s+year)?)\s*(?:[:=\-]|in)?\s*(\b(?:19\d{2}|20\d{2})\b)",
        re.IGNORECASE
    )

    @classmethod
    def extract_years(cls, raw_data: Dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
        """
        Extracts both Manufacture Year (YOM) and Registration Year (YOR)
        from structured fields or from description text variations.
        Returns: (manufacture_year, registration_year)
        """
        desc = raw_data.get("description") or ""
        yom_from_desc = None
        yor_from_desc = None

        yom_match = cls.YOM_PATTERN.search(desc)
        if yom_match:
            try:
                yom_from_desc = int(yom_match.group(1))
            except ValueError:
                pass

        yor_match = cls.YOR_PATTERN.search(desc)
        if yor_match:
            try:
                yor_from_desc = int(yor_match.group(1))
            except ValueError:
                pass

        structured_yom = cls.parse_year(raw_data.get("manufacture_year"))
        structured_yor = cls.parse_year(raw_data.get("registration_year"))
        structured_year = cls.parse_year(raw_data.get("year"))

        manufacture_year = yom_from_desc or structured_yom or structured_year
        registration_year = yor_from_desc or structured_yor

        return manufacture_year, registration_year

    def normalize(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes a raw data dictionary into a standardized vehicle record.
        """
        location_raw = raw_data.get("location")
        district_raw = raw_data.get("district")

        manufacture_year, registration_year = self.extract_years(raw_data)

        record = {
            "listing_id": str(raw_data.get("listing_id")).strip() if raw_data.get("listing_id") else None,
            "listing_url": raw_data.get("listing_url"),
            "title": self._clean_string(raw_data.get("title")),
            "category": self._clean_string(raw_data.get("category")),
            "brand": self._clean_string(raw_data.get("brand")),
            "model": self._clean_string(raw_data.get("model")),
            "year": manufacture_year or registration_year,
            "manufacture_year": manufacture_year,
            "registration_year": registration_year,
            "mileage": self.parse_mileage(raw_data.get("mileage")),
            "price": self.parse_price(raw_data.get("price")),
            "fuel_type": self._clean_string(raw_data.get("fuel_type")),
            "transmission": self._clean_string(raw_data.get("gear")),
            "engine_cc": self.parse_engine_cc(raw_data.get("engine_cc")),
            "condition": self._clean_string(raw_data.get("condition")),
            "location": self._clean_string(location_raw),
            "district": self.determine_district(district_raw or location_raw),
            "ad_date": self._clean_string(raw_data.get("ad_date")),
            "description": raw_data.get("description"),
            "source": raw_data.get("source", "riyasewana"),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "raw_data": raw_data,
        }

        return record

    @staticmethod
    def _clean_string(val: Optional[str]) -> Optional[str]:
        if not val:
            return None
        cleaned = " ".join(str(val).split())
        return cleaned if cleaned else None

    @classmethod
    def parse_price(cls, val: Any) -> Optional[int]:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return int(val)

        val_str = str(val).lower()
        if "million" in val_str or "mn" in val_str:
            m = re.search(r"([\d.]+)\s*(?:million|mn)", val_str)
            if m:
                return int(float(m.group(1)) * 1_000_000)
        if "lakh" in val_str or "lacs" in val_str:
            m = re.search(r"([\d.]+)\s*(?:lakh|lacs)", val_str)
            if m:
                return int(float(m.group(1)) * 100_000)

        match = re.search(r"([\d,]+)", val_str)
        if match:
            digits = match.group(1).replace(",", "")
            if digits:
                try:
                    return int(digits)
                except ValueError:
                    return None
        return None

    @classmethod
    def parse_mileage(cls, val: Any) -> Optional[int]:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return int(val)

        val_str = str(val).lower()
        match = re.search(r"([\d,]+)", val_str)
        if match:
            digits = match.group(1).replace(",", "")
            if digits:
                try:
                    return int(digits)
                except ValueError:
                    return None
        return None

    @classmethod
    def parse_year(cls, val: Any) -> Optional[int]:
        if val is None:
            return None
        if isinstance(val, int):
            return val

        val_str = str(val)
        match = re.search(r"\b(19\d{2}|20\d{2})\b", val_str)
        if match:
            return int(match.group(1))
        return None

    @classmethod
    def parse_engine_cc(cls, val: Any) -> Optional[int]:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return int(val)

        val_str = str(val).lower()
        match = re.search(r"([\d,]+)", val_str)
        if match:
            digits = match.group(1).replace(",", "")
            if digits:
                try:
                    return int(digits)
                except ValueError:
                    return None
        return None

    @classmethod
    def determine_district(cls, location_text: Optional[str]) -> Optional[str]:
        if not location_text:
            return None

        loc_lower = location_text.lower().strip()

        for town, district in cls.DISTRICT_MAPPING.items():
            if town in loc_lower:
                return district

        return location_text.title().strip()

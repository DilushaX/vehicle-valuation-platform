"""
Data Cleaner Module
Cleans, normalizes, and standardizes raw vehicle listing attributes.
"""
import re
from typing import Dict, Any, Optional

MAKE_CANONICAL_MAP = {
    "maruti": "Suzuki",
    "maruti suzuki": "Suzuki",
    "suzuki": "Suzuki",
    "toyota": "Toyota",
    "honda": "Honda",
    "nissan": "Nissan",
    "mitsubishi": "Mitsubishi",
    "hyundai": "Hyundai",
    "kia": "Kia",
    "mazda": "Mazda",
    "bajaj": "Bajaj",
    "tvs": "TVS",
    "yamaha": "Yamaha",
    "hero": "Hero",
    "tata": "Tata",
    "isuzu": "Isuzu",
    "ashok leyland": "Ashok Leyland",
    "leyland": "Ashok Leyland",
    "bmw": "BMW",
    "mercedes": "Mercedes-Benz",
    "mercedes-benz": "Mercedes-Benz",
    "benz": "Mercedes-Benz",
    "audi": "Audi",
    "land rover": "Land Rover",
    "range rover": "Land Rover",
    "daihatsu": "Daihatsu",
    "subaru": "Subaru",
    "ford": "Ford",
    "peugeot": "Peugeot",
    "micro": "Micro",
    "dfsk": "DFSK",
    "mahindra": "Mahindra"
}

FUEL_CANONICAL_MAP = {
    "petrol": "Petrol",
    "gasoline": "Petrol",
    "diesel": "Diesel",
    "hybrid": "Hybrid",
    "petrol hybrid": "Hybrid",
    "diesel hybrid": "Hybrid",
    "electric": "Electric",
    "ev": "Electric",
    "cng": "CNG",
    "gas": "LPG/CNG"
}

TRANSMISSION_CANONICAL_MAP = {
    "auto": "Automatic",
    "automatic": "Automatic",
    "manual": "Manual",
    "tiptronic": "Tiptronic",
    "cvt": "Automatic",
    "other": "Manual"
}

DISTRICT_CANONICAL_LIST = [
    "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "Nuwara Eliya",
    "Galle", "Matara", "Hambantota", "Jaffna", "Kilinochchi", "Mannar",
    "Vavuniya", "Mullaitivu", "Batticaloa", "Ampara", "Trincomalee",
    "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa", "Badulla",
    "Monaragala", "Ratnapura", "Kegalle"
]

class DataCleaner:
    @staticmethod
    def standardize_make(raw_make: Optional[str]) -> str:
        if not raw_make:
            return "Unknown"
        cleaned = raw_make.strip().lower()
        return MAKE_CANONICAL_MAP.get(cleaned, raw_make.strip().title())

    @staticmethod
    def standardize_model(raw_model: Optional[str], make: str = "") -> str:
        if not raw_model:
            return "Unknown"
        model_str = raw_model.strip()
        # Remove repeated make name from model string (e.g. "Toyota Aqua" -> "Aqua")
        if make and model_str.lower().startswith(make.lower()):
            model_str = model_str[len(make):].strip()
        return model_str.title() if model_str else "Unknown"

    @staticmethod
    def standardize_fuel(raw_fuel: Optional[str]) -> str:
        if not raw_fuel:
            return "Petrol"
        cleaned = raw_fuel.strip().lower()
        return FUEL_CANONICAL_MAP.get(cleaned, raw_fuel.strip().title())

    @staticmethod
    def standardize_transmission(raw_trans: Optional[str]) -> str:
        if not raw_trans:
            return "Automatic"
        cleaned = raw_trans.strip().lower()
        return TRANSMISSION_CANONICAL_MAP.get(cleaned, raw_trans.strip().title())

    @staticmethod
    def standardize_district(raw_district: Optional[str]) -> str:
        if not raw_district:
            return "Colombo"
        cleaned = raw_district.strip().lower()
        for d in DISTRICT_CANONICAL_LIST:
            if d.lower() in cleaned or cleaned in d.lower():
                return d
        return raw_district.strip().title()

    @staticmethod
    def clean_record(raw_record: Dict[str, Any]) -> Dict[str, Any]:
        """Cleans and standardizes all fields in a raw listing dictionary."""
        cleaned = raw_record.copy()
        
        cleaned["make"] = DataCleaner.standardize_make(raw_record.get("make"))
        cleaned["model"] = DataCleaner.standardize_model(raw_record.get("model"), cleaned["make"])
        cleaned["fuel_type"] = DataCleaner.standardize_fuel(raw_record.get("fuel_type"))
        cleaned["transmission"] = DataCleaner.standardize_transmission(raw_record.get("transmission"))
        cleaned["district"] = DataCleaner.standardize_district(raw_record.get("district"))
        
        # Condition
        cond = raw_record.get("condition", "Used")
        cleaned["condition"] = cond.title() if cond else "Used"

        # Year
        try:
            cleaned["yom"] = int(raw_record.get("yom", 2015))
        except (ValueError, TypeError):
            cleaned["yom"] = 2015

        # Mileage
        if raw_record.get("mileage_km") is not None:
            try:
                cleaned["mileage_km"] = int(raw_record["mileage_km"])
            except (ValueError, TypeError):
                cleaned["mileage_km"] = None
        else:
            cleaned["mileage_km"] = None

        # Price
        try:
            cleaned["price_rs"] = float(raw_record.get("price_rs", 0.0))
        except (ValueError, TypeError):
            cleaned["price_rs"] = 0.0

        return cleaned

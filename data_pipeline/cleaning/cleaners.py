"""
Vehicle attribute cleaner and normalizer.
Normalizes common Sri Lankan vehicle market data variations into standard canonical forms
while preserving original raw values for auditing.
"""

import re
from typing import Any, Dict, Optional


class VehicleCleaner:
    """
    Standardizes categorical and numeric attributes of vehicle records.
    """

    # Canonical 8 vehicle categories
    CANONICAL_CATEGORIES = {
        "cars": "Cars",
        "car": "Cars",
        "heavy-duties": "Heavy-Duty",
        "heavy-duty": "Heavy-Duty",
        "heavyduty": "Heavy-Duty",
        "lorries": "Lorries",
        "lorry": "Lorries",
        "motorbikes": "Motorbikes",
        "motorbike": "Motorbikes",
        "motorcycles": "Motorbikes",
        "motorcycle": "Motorbikes",
        "pickups": "Pickups",
        "pickup": "Pickups",
        "suvs": "SUVs",
        "suv": "SUVs",
        "three wheelers": "Three Wheelers",
        "three wheeler": "Three Wheelers",
        "three-wheels": "Three Wheelers",
        "three wheel": "Three Wheelers",
        "three-wheel": "Three Wheelers",
        "threewheel": "Three Wheelers",
        "vans": "Vans",
        "van": "Vans",
    }

    # Fuel type mappings
    FUEL_TYPE_MAPPINGS = {
        "petrol": "Petrol",
        "gasoline": "Petrol",
        "unleaded": "Petrol",
        "super petrol": "Petrol",
        "diesel": "Diesel",
        "super diesel": "Diesel",
        "hybrid": "Hybrid",
        "petrol hybrid": "Hybrid",
        "diesel hybrid": "Hybrid",
        "plug-in hybrid": "Hybrid",
        "phev": "Hybrid",
        "electric": "Electric",
        "ev": "Electric",
        "battery electric": "Electric",
        "bev": "Electric",
        "gas": "Gas",
        "lpg": "LPG",
        "cng": "CNG",
        "auto gas": "LPG",
    }

    # Transmission mappings
    TRANSMISSION_MAPPINGS = {
        "auto": "Automatic",
        "automatic": "Automatic",
        "at": "Automatic",
        "a/t": "Automatic",
        "cvt": "Automatic",
        "tiptronic": "Automatic",
        "dual clutch": "Automatic",
        "dct": "Automatic",
        "manual": "Manual",
        "mt": "Manual",
        "m/t": "Manual",
    }

    @classmethod
    def clean_string(cls, val: Any, max_len: Optional[int] = None) -> Optional[str]:
        """Strips excessive whitespace and controls maximum length."""
        if val is None:
            return None
        cleaned = " ".join(str(val).split())
        if not cleaned:
            return None
        if max_len and len(cleaned) > max_len:
            cleaned = cleaned[:max_len].strip()
        return cleaned if cleaned else None

    @classmethod
    def normalize_fuel_type(cls, val: Any) -> Optional[str]:
        """
        Normalizes fuel type variations into canonical forms:
        'Petrol', 'Diesel', 'Hybrid', 'Electric', 'LPG', 'CNG', or 'Gas'.
        """
        cleaned = cls.clean_string(val)
        if not cleaned:
            return None
        lower = cleaned.lower()
        if lower in cls.FUEL_TYPE_MAPPINGS:
            return cls.FUEL_TYPE_MAPPINGS[lower]

        # Substring heuristics for compound phrases
        if "plug-in" in lower or "phev" in lower or "hybrid" in lower:
            return "Hybrid"
        if "electric" in lower or lower == "ev":
            return "Electric"
        if "diesel" in lower:
            return "Diesel"
        if "petrol" in lower or "gasoline" in lower:
            return "Petrol"
        if "lpg" in lower or "cng" in lower or "gas" in lower:
            return "Gas"

        return cleaned.title()

    @classmethod
    def normalize_transmission(cls, val: Any) -> Optional[str]:
        """
        Normalizes transmission variations into 'Automatic' or 'Manual'.
        """
        cleaned = cls.clean_string(val)
        if not cleaned:
            return None
        lower = cleaned.lower()
        if lower in cls.TRANSMISSION_MAPPINGS:
            return cls.TRANSMISSION_MAPPINGS[lower]

        if "auto" in lower or "cvt" in lower or "tiptronic" in lower:
            return "Automatic"
        if "manual" in lower:
            return "Manual"

        return cleaned.title()

    @classmethod
    def canonicalize_category(cls, val: Any) -> Optional[str]:
        """
        Maps any recognized category string or slug to its canonical plural form:
        'Cars', 'Heavy-Duty', 'Lorries', 'Motorbikes', 'Pickups', 'SUVs', 'Three Wheelers', 'Vans'.
        """
        cleaned = cls.clean_string(val)
        if not cleaned:
            return None
        lower = cleaned.lower().strip()
        return cls.CANONICAL_CATEGORIES.get(lower)

    @classmethod
    def is_electric(cls, fuel_type: Optional[str]) -> bool:
        """Returns True if the fuel type indicates an all-electric vehicle."""
        if not fuel_type:
            return False
        return cls.normalize_fuel_type(fuel_type) == "Electric"

"""
Data Quality Engine Module
Performs multi-stage validation, pattern anomaly detection, category range checks,
grouped IQR statistical outlier detection, and quality scoring.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
from config import settings

SUSPICIOUS_MILEAGE_PATTERNS = {1, 12, 123, 1234, 12345, 123456, 111111, 222222, 999999, 1000000}
SUSPICIOUS_PRICE_PATTERNS = {1, 12, 123, 1234, 12345, 123456, 111111, 222222, 999999}

CATEGORY_PRICE_RANGES = {
    "Cars": (400_000, 200_000_000),
    "SUVs": (1_500_000, 250_000_000),
    "Vans": (1_000_000, 50_000_000),
    "Motorbikes": (100_000, 10_000_000),
    "Three-Wheel": (300_000, 4_000_000),
    "Lorries": (500_000, 40_000_000),
    "Buses": (1_000_000, 60_000_000)
}

class DataQualityEngine:
    def __init__(self):
        # Cache for statistical reference distributions
        self.stats_cache: Dict[str, Dict[str, float]] = {}

    def validate_single_record(
        self,
        record: Dict[str, Any],
        price_iqr_bounds: Tuple[float, float] = None
    ) -> Tuple[str, float, List[str], bool]:
        """
        Evaluates data quality of a single vehicle listing.
        Returns: (quality_status, score, issues_list, is_outlier)
        where status is 'VALID', 'SUSPICIOUS', 'INVALID', or 'MISSING'.
        """
        issues = []
        is_invalid = False
        is_missing = False
        is_suspicious = False
        is_outlier = False
        score = 100.0

        category = record.get("category", "Cars").capitalize()
        make = record.get("make")
        model = record.get("model")
        yom = record.get("yom")
        price = record.get("price_rs", 0.0)
        mileage = record.get("mileage_km")

        # 1. Missing Critical Fields
        if not make or make.lower() in ["unknown", "other"]:
            issues.append("Missing or unknown vehicle Make")
            is_missing = True
            score -= 20.0
        if not model or model.lower() in ["unknown", "other"]:
            issues.append("Missing or unknown vehicle Model")
            is_missing = True
            score -= 20.0
        if mileage is None:
            issues.append("Missing mileage")
            is_missing = True
            score -= 15.0

        # 2. Fatal / Invalid Range Checks
        if yom is None or yom < settings.MIN_VALID_YEAR or yom > settings.MAX_VALID_YEAR:
            issues.append(f"Invalid Year of Manufacture: {yom}")
            is_invalid = True
            score -= 40.0

        if price is None or price <= 0:
            issues.append(f"Invalid non-positive asking price: {price}")
            is_invalid = True
            score -= 50.0
        elif price < settings.MIN_VALID_PRICE_LKR:
            issues.append(f"Unrealistically low asking price: Rs. {price:,.0f}")
            is_suspicious = True
            score -= 30.0

        if mileage is not None:
            if mileage < 0:
                issues.append(f"Negative mileage: {mileage} km")
                is_invalid = True
                score -= 40.0
            elif mileage > settings.MAX_VALID_MILEAGE:
                issues.append(f"Extreme mileage: {mileage:,} km")
                is_suspicious = True
                score -= 15.0

        # 3. Suspicious Pattern Detection
        if mileage is not None and mileage in SUSPICIOUS_MILEAGE_PATTERNS:
            issues.append(f"Suspicious dummy mileage pattern detected: {mileage}")
            is_suspicious = True
            score -= 25.0

        if price is not None and int(price) in SUSPICIOUS_PRICE_PATTERNS:
            issues.append(f"Suspicious dummy price pattern detected: {price}")
            is_suspicious = True
            score -= 30.0

        # 4. Category-Specific Price Range Checks
        expected_range = CATEGORY_PRICE_RANGES.get(category)
        if expected_range and price > 0:
            min_exp, max_exp = expected_range
            if price < min_exp * 0.4:
                issues.append(f"Price Rs. {price:,.0f} unusually low for category {category}")
                is_suspicious = True
                score -= 15.0
            elif price > max_exp * 1.5:
                issues.append(f"Price Rs. {price:,.0f} unusually high for category {category}")
                is_suspicious = True
                score -= 15.0

        # 5. Statistical Outlier Check
        if price_iqr_bounds and price > 0:
            lower_bound, upper_bound = price_iqr_bounds
            if price < lower_bound or price > upper_bound:
                issues.append(f"Statistical price outlier (Price Rs. {price:,.0f} outside IQR bounds [{lower_bound:,.0f} - {upper_bound:,.0f}])")
                is_outlier = True
                is_suspicious = True
                score -= 10.0

        score = max(0.0, min(100.0, score))

        if is_invalid:
            status = "INVALID"
        elif is_missing:
            status = "MISSING"
        elif is_suspicious or is_outlier:
            status = "SUSPICIOUS"
        else:
            status = "VALID"

        return status, score, issues, is_outlier

    def compute_group_iqr_bounds(self, records: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        """
        Computes IQR bounds for price grouped by (category, make, model).
        Returns a lookup map: 'category|make|model' -> (lower_bound, upper_bound)
        """
        groups: Dict[str, List[float]] = {}
        for r in records:
            price = r.get("price_rs", 0.0)
            if price and price > 0:
                key = f"{r.get('category', 'Cars')}|{r.get('make', '')}|{r.get('model', '')}".lower()
                groups.setdefault(key, []).append(price)

        bounds: Dict[str, Tuple[float, float]] = {}
        for key, prices in groups.items():
            if len(prices) >= 5:
                q25, q75 = np.percentile(prices, [25, 75])
                iqr = q75 - q25
                lower = max(100_000.0, q25 - (2.5 * iqr))
                upper = q75 + (2.5 * iqr)
                bounds[key] = (lower, upper)

        self.stats_cache = bounds
        return bounds

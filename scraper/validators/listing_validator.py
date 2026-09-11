from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple

from data_pipeline.cleaning.cleaners import VehicleCleaner


class ListingValidator:
    """
    Data quality validator for vehicle listing records.
    Attaches validation flags, critical/non-critical classifications, and ML eligibility
    decisions to audit data quality without deleting or mutating raw records.
    """

    MIN_YEAR = 1950
    DEFAULT_MIN_PRICE = 50_000
    DEFAULT_MAX_PRICE = 500_000_000

    MIN_MILEAGE = 0
    MAX_MILEAGE = 2_000_000

    MIN_ENGINE_CC = 25
    MAX_ENGINE_CC = 16_000

    # Category-specific price ranges (Sri Lankan Rupees)
    CATEGORY_PRICE_RANGES: Dict[str, Tuple[int, int]] = {
        "Cars": (100_000, 350_000_000),
        "Heavy-Duty": (200_000, 500_000_000),
        "Lorries": (150_000, 250_000_000),
        "Motorbikes": (25_000, 25_000_000),
        "Pickups": (200_000, 300_000_000),
        "SUVs": (200_000, 350_000_000),
        "Three Wheelers": (50_000, 15_000_000),
        "Vans": (100_000, 250_000_000),
    }

    # Issues that disqualify a record from ML valuation modeling
    CRITICAL_ISSUE_CODES: Set[str] = {
        "missing_price",
        "invalid_price",
        "price_out_of_range",
        "suspicious_price",
        "suspicious_price_pattern",
        "missing_yom",
        "invalid_yom",
        "future_yom",
        "missing_year",
        "invalid_year",
        "missing_mileage",
        "invalid_mileage",
        "mileage_out_of_range",
        "suspicious_mileage",
        "suspicious_mileage_pattern",
        "missing_category",
        "invalid_category",
        "missing_brand",
        "missing_model",
    }

    # Issues that represent data quality flaws but do not disqualify from valuation modeling
    NON_CRITICAL_ISSUE_CODES: Set[str] = {
        "missing_yor",
        "invalid_yor",
        "future_yor",
        "invalid_registration_year",
        "registration_before_manufacture",
        "registration_precedes_manufacture",
        "invalid_engine_cc",
        "implausible_engine_cc",
        "missing_fuel_type",
        "missing_transmission",
    }

    def validate(self, vehicle: Dict[str, Any], require_category: bool = False) -> Dict[str, Any]:
        """
        Validates normalized vehicle fields, appends structured issue codes,
        computes ML eligibility, and preserves data auditability.
        """
        issues: List[str] = []

        price = vehicle.get("price")
        mileage = vehicle.get("mileage")
        manufacture_year = vehicle.get("manufacture_year")
        registration_year = vehicle.get("registration_year")
        year = manufacture_year if manufacture_year is not None else vehicle.get("year")
        brand = vehicle.get("brand")
        model = vehicle.get("model")
        category = vehicle.get("category")
        engine_cc = vehicle.get("engine_cc")
        fuel_type = vehicle.get("fuel_type")

        current_year = datetime.now().year

        # -------------------------
        # Category validation
        # -------------------------
        canonical_category = VehicleCleaner.canonicalize_category(category) if category else None
        if "category" in vehicle or require_category:
            if not category:
                issues.append("missing_category")
            elif not canonical_category:
                issues.append("invalid_category")

        # -------------------------
        # Price validation
        # -------------------------
        min_price, max_price = (
            self.CATEGORY_PRICE_RANGES.get(canonical_category, (self.DEFAULT_MIN_PRICE, self.DEFAULT_MAX_PRICE))
            if canonical_category
            else (self.DEFAULT_MIN_PRICE, self.DEFAULT_MAX_PRICE)
        )

        if price is None:
            issues.append("missing_price")
        else:
            try:
                price_num = int(price)
                if price_num <= 0:
                    issues.append("invalid_price")
                elif self._is_suspicious_number(price_num):
                    issues.append("suspicious_price_pattern")
                elif price_num < min_price or price_num > max_price:
                    issues.append("price_out_of_range")
                    issues.append("suspicious_price")
            except (ValueError, TypeError):
                issues.append("invalid_price")

        # -------------------------
        # Mileage validation
        # -------------------------
        if mileage is None:
            issues.append("missing_mileage")
        else:
            try:
                mileage_num = int(mileage)
                if mileage_num < self.MIN_MILEAGE:
                    issues.append("invalid_mileage")
                elif self._is_suspicious_number(mileage_num):
                    issues.append("suspicious_mileage_pattern")
                elif mileage_num > self.MAX_MILEAGE:
                    issues.append("mileage_out_of_range")
                    issues.append("suspicious_mileage")
            except (ValueError, TypeError):
                issues.append("invalid_mileage")

        # -------------------------
        # Manufacture Year (YOM) validation
        # -------------------------
        year_val: Optional[int] = None
        if year is None and manufacture_year is None:
            issues.append("missing_yom")
            if registration_year is None:
                issues.append("missing_year")
        else:
            raw_yom = manufacture_year if manufacture_year is not None else year
            try:
                year_val = int(raw_yom)
                if year_val < self.MIN_YEAR:
                    issues.append("invalid_yom")
                    issues.append("invalid_year")
                elif year_val > current_year:
                    issues.append("future_yom")
                    issues.append("invalid_year")
            except (ValueError, TypeError):
                issues.append("invalid_yom")
                issues.append("invalid_year")

        # -------------------------
        # Registration Year (YOR) validation
        # -------------------------
        reg_val: Optional[int] = None
        if registration_year is not None:
            try:
                reg_val = int(registration_year)
                if reg_val < self.MIN_YEAR:
                    issues.append("invalid_yor")
                    issues.append("invalid_registration_year")
                elif reg_val > current_year:
                    issues.append("future_yor")
                    issues.append("invalid_registration_year")
            except (ValueError, TypeError):
                issues.append("invalid_yor")
                issues.append("invalid_registration_year")

        # -------------------------
        # Year Relationship: YOR >= YOM
        # -------------------------
        if year_val is not None and reg_val is not None:
            if reg_val < year_val:
                issues.append("registration_before_manufacture")
                issues.append("registration_precedes_manufacture")

        # -------------------------
        # Engine CC validation
        # -------------------------
        if engine_cc is not None:
            is_ev = VehicleCleaner.is_electric(fuel_type)
            try:
                cc_val = int(engine_cc)
                if cc_val < 0:
                    issues.append("invalid_engine_cc")
                elif cc_val == 0 and not is_ev:
                    issues.append("invalid_engine_cc")
                elif cc_val > 0:
                    if cc_val < self.MIN_ENGINE_CC and not is_ev:
                        issues.append("implausible_engine_cc")
                    elif cc_val > self.MAX_ENGINE_CC:
                        issues.append("implausible_engine_cc")
            except (ValueError, TypeError):
                issues.append("invalid_engine_cc")

        # -------------------------
        # Brand & Model validation
        # -------------------------
        if not brand:
            issues.append("missing_brand")
        if not model:
            issues.append("missing_model")

        # Deduplicate issues while preserving order
        unique_issues = list(dict.fromkeys(issues))
        critical_issues = [iss for iss in unique_issues if iss in self.CRITICAL_ISSUE_CODES]
        non_critical_issues = [iss for iss in unique_issues if iss not in self.CRITICAL_ISSUE_CODES]

        # ML eligibility: zero critical issues
        ml_eligible = len(critical_issues) == 0

        # Generate human-readable exclusion reasons if ineligible
        ml_exclusion_reasons: List[str] = []
        if not ml_eligible:
            for c_iss in critical_issues:
                ml_exclusion_reasons.append(self._format_exclusion_reason(c_iss, vehicle))

        # -------------------------
        # Attach quality metadata
        # -------------------------
        vehicle["validation_issues"] = unique_issues
        vehicle["critical_issues"] = critical_issues
        vehicle["non_critical_issues"] = non_critical_issues
        vehicle["is_valid"] = len(unique_issues) == 0
        vehicle["ml_eligible"] = ml_eligible
        vehicle["ml_exclusion_reasons"] = ml_exclusion_reasons
        vehicle["quality_score"] = None  # Kept NULL until defensible multi-factor model is developed

        return vehicle

    @classmethod
    def _format_exclusion_reason(cls, issue_code: str, vehicle: Dict[str, Any]) -> str:
        """Formats human-readable explanation for an ML exclusion reason."""
        reasons = {
            "missing_price": "Asking price is missing.",
            "invalid_price": f"Price is invalid or zero ({vehicle.get('price')}).",
            "price_out_of_range": f"Price is outside plausible category bounds ({vehicle.get('price')}).",
            "suspicious_price": f"Price is outside plausible bounds ({vehicle.get('price')}).",
            "suspicious_price_pattern": f"Price contains a dummy sequence ({vehicle.get('price')}).",
            "missing_yom": "Year of manufacture is missing.",
            "invalid_yom": f"Year of manufacture is before historical threshold ({vehicle.get('manufacture_year') or vehicle.get('year')}).",
            "future_yom": f"Year of manufacture is in the future ({vehicle.get('manufacture_year') or vehicle.get('year')}).",
            "missing_year": "Manufacture and registration year are both missing.",
            "invalid_year": f"Vehicle year is invalid or outside plausible range ({vehicle.get('year')}).",
            "missing_mileage": "Mileage is missing.",
            "invalid_mileage": f"Mileage is negative ({vehicle.get('mileage')}).",
            "mileage_out_of_range": f"Mileage exceeds plausible maximum ({vehicle.get('mileage')}).",
            "suspicious_mileage": f"Mileage is outside plausible bounds ({vehicle.get('mileage')}).",
            "suspicious_mileage_pattern": f"Mileage contains a repeating or dummy pattern ({vehicle.get('mileage')}).",
            "missing_category": "Vehicle category is missing.",
            "invalid_category": f"Vehicle category '{vehicle.get('category')}' is not supported.",
            "missing_brand": "Vehicle brand is missing.",
            "missing_model": "Vehicle model is missing.",
        }
        return reasons.get(issue_code, f"Critical issue: {issue_code}.")

    @classmethod
    def _is_suspicious_number(cls, value: Any) -> bool:
        """
        Detects obvious dummy digit sequences such as 111111, 123456, 999999.
        """
        if value is None:
            return False

        try:
            val_int = int(value)
        except (ValueError, TypeError):
            return False

        val_str = str(val_int).strip()

        if len(val_str) < 3:
            return True if val_str in {"0", "123"} else False

        # All identical digits (e.g., 111111, 999999, 222222)
        if len(set(val_str)) == 1:
            return True

        # Known repeating patterns
        known_dummies = {
            "123", "1234", "12345", "123456", "1234567", "12345678",
            "654321", "987654", "999999", "111111", "222222", "333333",
            "444444", "555555", "666666", "777777", "888888", "000000",
            "12300", "123000", "123450"
        }
        if val_str in known_dummies:
            return True

        # Strict ascending / descending sequences
        ascending = "0123456789"
        descending = "9876543210"
        if val_str in ascending or val_str in descending:
            return True

        return False
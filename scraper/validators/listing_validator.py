from datetime import datetime
from typing import Dict, Any, List


class ListingValidator:
    """
    Data quality validator for vehicle listing records.
    Attaches validation flags and quality scores to audit data quality
    without deleting or permanently mutating raw records.
    """

    MIN_YEAR = 1950
    MIN_PRICE = 50_000
    MAX_PRICE = 500_000_000

    MIN_MILEAGE = 0
    MAX_MILEAGE = 2_000_000

    def validate(self, vehicle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates normalized vehicle fields and appends validation_issues and is_valid status.
        """
        issues: List[str] = []

        price = vehicle.get("price")
        mileage = vehicle.get("mileage")
        manufacture_year = vehicle.get("manufacture_year")
        registration_year = vehicle.get("registration_year")
        year = manufacture_year if manufacture_year is not None else vehicle.get("year")
        brand = vehicle.get("brand")
        model = vehicle.get("model")

        # -------------------------
        # Price validation
        # -------------------------
        if price is None:
            issues.append("missing_price")
        elif self._is_suspicious_number(price):
            issues.append("suspicious_price_pattern")
        elif price < self.MIN_PRICE or price > self.MAX_PRICE:
            issues.append("suspicious_price")

        # -------------------------
        # Mileage validation
        # -------------------------
        if mileage is None:
            issues.append("missing_mileage")
        elif self._is_suspicious_number(mileage):
            issues.append("suspicious_mileage_pattern")
        elif mileage < self.MIN_MILEAGE:
            issues.append("invalid_mileage")
        elif mileage > self.MAX_MILEAGE:
            issues.append("suspicious_mileage")

        # -------------------------
        # Year validation
        # -------------------------
        if year is None and registration_year is None:
            issues.append("missing_year")
        else:
            current_year = datetime.now().year
            if year is not None:
                try:
                    year_int = int(year)
                    if year_int < self.MIN_YEAR or year_int > current_year + 1:
                        issues.append("invalid_year")
                except (ValueError, TypeError):
                    issues.append("invalid_year")

            if registration_year is not None:
                try:
                    reg_int = int(registration_year)
                    if reg_int < self.MIN_YEAR or reg_int > current_year + 1:
                        issues.append("invalid_registration_year")
                    elif year is not None and reg_int < int(year):
                        issues.append("registration_precedes_manufacture")
                except (ValueError, TypeError):
                    issues.append("invalid_registration_year")

        # -------------------------
        # Brand & Model validation
        # -------------------------
        if not brand:
            issues.append("missing_brand")

        if not model:
            issues.append("missing_model")

        # -------------------------
        # Attach quality flags
        # -------------------------
        vehicle["validation_issues"] = issues
        vehicle["is_valid"] = len(issues) == 0

        return vehicle

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

        # All identical digits (e.g., 111111, 999999)
        if len(set(val_str)) == 1:
            return True

        # Known repeating patterns
        known_dummies = {
            "123", "1234", "12345", "123456", "1234567",
            "654321", "987654", "999999", "111111", "000000",
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
from datetime import datetime


class ListingValidator:

    MIN_YEAR = 1950
    MIN_PRICE = 50_000
    MAX_PRICE = 500_000_000

    MIN_MILEAGE = 0
    MAX_MILEAGE = 2_000_000

    def validate(self, vehicle: dict) -> dict:

        issues = []

        price = vehicle.get("price")
        mileage = vehicle.get("mileage")
        year = vehicle.get("year")

        # -------------------------
        # Price validation
        # -------------------------

        if price is None:
            issues.append("missing_price")

        elif self._is_suspicious_number(price):
            issues.append("suspicious_price_pattern")

        elif price < self.MIN_PRICE:
            issues.append("suspicious_price")

        elif price > self.MAX_PRICE:
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

        if year is None:
            issues.append("missing_year")

        else:
            try:
                year = int(year)

                current_year = datetime.now().year

                if year < self.MIN_YEAR or year > current_year + 1:
                    issues.append("invalid_year")

            except (ValueError, TypeError):
                issues.append("invalid_year")

        # -------------------------
        # Brand / Model
        # -------------------------

        if not vehicle.get("brand"):
            issues.append("missing_brand")

        if not vehicle.get("model"):
            issues.append("missing_model")

        # -------------------------
        # Final result
        # -------------------------

        vehicle["validation_issues"] = issues

        vehicle["is_valid"] = len(issues) == 0

        return vehicle

    @staticmethod
    def _is_suspicious_number(value) -> bool:

        try:
            value = int(value)
        except (ValueError, TypeError):
            return False

        value_str = str(value)

        # Examples:
        # 111111
        # 999999
        # 123456
        # 654321

        if len(set(value_str)) == 1:
            return True

        if value_str in {
            "123",
            "1234",
            "12345",
            "123456",
            "1234567",
            "111111",
            "999999",
            "000000",
        }:
            return True

        ascending = "0123456789"
        descending = "9876543210"

        if value_str in ascending or value_str in descending:
            return True

        return False
import pytest
from scraper.parsers.riyasewana_parser import RiyasewanaParser
from scraper.extractors.vehicle_extractor import VehicleExtractor
from scraper.validators.listing_validator import ListingValidator


class TestRiyasewanaParser:
    def test_extract_listing_id(self):
        parser = RiyasewanaParser()
        url = "https://riyasewana.com/buy/toyota-corolla-axio-sale-homagama-12217083"
        listing_id = parser._extract_listing_id(url)
        assert listing_id == "12217083"

    def test_extract_listing_id_invalid(self):
        parser = RiyasewanaParser()
        assert parser._extract_listing_id("https://riyasewana.com/search") is None

    def test_parse_listing_missing_fields(self):
        parser = RiyasewanaParser()
        empty_html = "<html><body></body></html>"
        raw_data = parser.parse_listing(empty_html, "https://riyasewana.com/buy/test-100")
        assert raw_data["listing_id"] == "100"
        assert raw_data["title"] is None
        assert raw_data["price"] is None
        assert raw_data["brand"] is None


class TestVehicleExtractor:
    def test_price_normalization(self):
        extractor = VehicleExtractor()
        assert extractor.parse_price("Rs. 12,300,000") == 12300000
        assert extractor.parse_price("Rs 8,500,000") == 8500000
        assert extractor.parse_price("12.3 million") == 12300000
        assert extractor.parse_price("45 lakh") == 4500000
        assert extractor.parse_price(None) is None

    def test_mileage_normalization(self):
        extractor = VehicleExtractor()
        assert extractor.parse_mileage("118,000 km") == 118000
        assert extractor.parse_mileage("75000km") == 75000
        assert extractor.parse_mileage(None) is None

    def test_year_normalization(self):
        extractor = VehicleExtractor()
        assert extractor.parse_year("2016") == 2016
        assert extractor.parse_year("Year 2018") == 2018
        assert extractor.parse_year("abc") is None
        assert extractor.parse_year(None) is None

    def test_district_determination(self):
        extractor = VehicleExtractor()
        assert extractor.determine_district("Homagama") == "Colombo"
        assert extractor.determine_district("Nugegoda") == "Colombo"
        assert extractor.determine_district("Negombo") == "Gampaha"
        assert extractor.determine_district("Peradeniya") == "Kandy"
        assert extractor.determine_district("Galle") == "Galle"

    def test_extract_years_variations(self):
        extractor = VehicleExtractor()

        cases = [
            ("YOM: 2016\nYOR: 2017", 2016, 2017),
            ("Year of Manufacture: 2015\nYear of Registration: 2016", 2015, 2016),
            ("Manufacture Year - 2014\nFirst Registration - 2015", 2014, 2015),
            ("Y.O.M: 2013, Y.O.R: 2014", 2013, 2014),
            ("Y.O.M. 2012, Y.O.R. 2013", 2012, 2013),
            ("Mfd: 2011, Reg: 2012", 2011, 2012),
            ("Registered in 2020, manufactured in 2019", 2019, 2020),
            ("YOM 2016 YOR 2017", 2016, 2017),
        ]

        for desc, expected_yom, expected_yor in cases:
            yom, yor = extractor.extract_years({"description": desc})
            assert yom == expected_yom, f"Failed YOM for '{desc}': got {yom}, expected {expected_yom}"
            assert yor == expected_yor, f"Failed YOR for '{desc}': got {yor}, expected {expected_yor}"

    def test_extract_years_only_one_value(self):
        extractor = VehicleExtractor()

        # Only YOM present
        yom1, yor1 = extractor.extract_years({"description": "Manufacture Year: 2016"})
        assert yom1 == 2016
        assert yor1 is None

        # Only generic year present in structured fields
        yom2, yor2 = extractor.extract_years({"year": "2018", "description": "Clean car"})
        assert yom2 == 2018
        assert yor2 is None

        # Only YOR present
        yom3, yor3 = extractor.extract_years({"description": "First Registration: 2019"})
        assert yom3 is None
        assert yor3 == 2019

    def test_normalize_includes_separate_year_fields(self):
        extractor = VehicleExtractor()
        raw_data = {
            "title": "Toyota Axio 2016",
            "year": "2016",
            "description": "Year of Manufacture - 2016\nFirst Registration - 2017",
            "price": "Rs. 12,300,000",
            "mileage": "118,000 km",
            "brand": "Toyota",
            "model": "Axio"
        }
        record = extractor.normalize(raw_data)
        assert record["manufacture_year"] == 2016
        assert record["registration_year"] == 2017
        assert record["year"] == 2016



class TestListingValidator:
    def test_valid_listing(self):
        validator = ListingValidator()
        vehicle = {
            "price": 12300000,
            "mileage": 118000,
            "year": 2016,
            "brand": "Toyota",
            "model": "Corolla Axio"
        }
        res = validator.validate(vehicle)
        assert res["is_valid"] is True
        assert res["validation_issues"] == []

    def test_missing_price(self):
        validator = ListingValidator()
        vehicle = {
            "price": None,
            "mileage": 100000,
            "year": 2018,
            "brand": "Toyota",
            "model": "Vitz"
        }
        res = validator.validate(vehicle)
        assert res["is_valid"] is False
        assert "missing_price" in res["validation_issues"]

    def test_missing_mileage(self):
        validator = ListingValidator()
        vehicle = {
            "price": 5000000,
            "mileage": None,
            "year": 2018,
            "brand": "Honda",
            "model": "Fit"
        }
        res = validator.validate(vehicle)
        assert res["is_valid"] is False
        assert "missing_mileage" in res["validation_issues"]

    def test_suspicious_number_detection(self):
        validator = ListingValidator()
        
        # Price 123456
        v1 = {"price": 123456, "mileage": 50000, "year": 2015, "brand": "Toyota", "model": "Axio"}
        res1 = validator.validate(v1)
        assert "suspicious_price_pattern" in res1["validation_issues"]

        # Mileage 111111
        v2 = {"price": 5000000, "mileage": 111111, "year": 2015, "brand": "Toyota", "model": "Axio"}
        res2 = validator.validate(v2)
        assert "suspicious_mileage_pattern" in res2["validation_issues"]

    def test_invalid_year(self):
        validator = ListingValidator()
        v = {"price": 5000000, "mileage": 50000, "year": 1890, "brand": "Toyota", "model": "Axio"}
        res = validator.validate(v)
        assert "invalid_year" in res["validation_issues"]

    def test_valid_with_both_years(self):
        validator = ListingValidator()
        v = {
            "price": 5000000,
            "mileage": 50000,
            "manufacture_year": 2016,
            "registration_year": 2017,
            "brand": "Toyota",
            "model": "Axio"
        }
        res = validator.validate(v)
        assert res["is_valid"] is True
        assert res["validation_issues"] == []

    def test_registration_precedes_manufacture(self):
        validator = ListingValidator()
        v = {
            "price": 5000000,
            "mileage": 50000,
            "manufacture_year": 2018,
            "registration_year": 2016,
            "brand": "Toyota",
            "model": "Axio"
        }
        res = validator.validate(v)
        assert res["is_valid"] is False
        assert "registration_precedes_manufacture" in res["validation_issues"]


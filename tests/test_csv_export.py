import csv
import json
from pathlib import Path

import pytest

from data_pipeline.export.csv_exporter import CSV_COLUMNS, CSVExporter


def test_csv_export_creates_file(tmp_path: Path):
    """Verifies CSV file is created in the target directory with expected name format."""
    exporter = CSVExporter(output_dir=tmp_path)
    records = [
        {
            "listing_id": "12217083",
            "listing_url": "https://riyasewana.com/buy/toyota-corolla-12217083",
            "title": "Toyota Corolla 2016",
            "category": "Cars",
            "brand": "Toyota",
            "model": "Corolla",
            "year": 2016,
            "manufacture_year": 2016,
            "registration_year": 2016,
            "mileage": 118000,
            "price": 12300000,
            "fuel_type": "Hybrid",
            "transmission": "Automatic",
            "engine_cc": 1500,
            "condition": "Used",
            "location": "Homagama",
            "district": "Colombo",
            "ad_date": "2026-08-27",
            "source": "riyasewana",
            "scraped_at": "2026-09-08T12:00:00Z",
            "validation_issues": [],
            "is_valid": True,
        }
    ]

    output_path = exporter.export_category_records(
        category="Cars",
        records=records,
        date_stamp="2026_09_08",
    )

    assert output_path.exists()
    assert output_path.name == "riyasewana_cars_2026_09_08.csv"


def test_csv_headers_and_data_accuracy(tmp_path: Path):
    """Verifies that CSV columns match schema and field values are correctly written."""
    exporter = CSVExporter(output_dir=tmp_path)
    records = [
        {
            "listing_id": "12217084",
            "listing_url": "https://riyasewana.com/buy/honda-vezel-12217084",
            "title": "Honda Vezel 2015",
            "category": "SUVs",
            "brand": "Honda",
            "model": "Vezel",
            "year": 2015,
            "manufacture_year": 2015,
            "registration_year": 2016,
            "mileage": 95000,
            "price": 9800000,
            "fuel_type": "Hybrid",
            "transmission": "Automatic",
            "engine_cc": 1500,
            "condition": "Used",
            "location": "Nugegoda",
            "district": "Colombo",
            "ad_date": "2026-08-27",
            "source": "riyasewana",
            "scraped_at": "2026-09-08T12:00:00Z",
            "validation_issues": ["suspicious_price_pattern"],
            "is_valid": False,
            "raw_data": {"internal": "should_be_stripped"},
        }
    ]

    output_path = exporter.export_category_records(
        category="SUVs",
        records=records,
        date_stamp="2026_09_08",
    )

    with open(output_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    assert headers == CSV_COLUMNS
    assert len(rows) == 1
    assert rows[0]["listing_id"] == "12217084"
    assert rows[0]["brand"] == "Honda"
    assert rows[0]["price"] == "9800000"
    assert rows[0]["is_valid"] == "False"
    assert json.loads(rows[0]["validation_issues"]) == ["suspicious_price_pattern"]
    assert "raw_data" not in rows[0]


def test_csv_exporter_handles_empty_records(tmp_path: Path):
    """Verifies that an empty records list produces a valid CSV file with headers only."""
    exporter = CSVExporter(output_dir=tmp_path)
    output_path = exporter.export_category_records(
        category="Vans",
        records=[],
        date_stamp="2026_09_08",
    )

    assert output_path.exists()
    with open(output_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        lines = list(reader)

    assert len(lines) == 1
    assert lines[0] == CSV_COLUMNS

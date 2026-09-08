import csv
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "listing_id",
    "listing_url",
    "title",
    "category",
    "brand",
    "model",
    "year",
    "manufacture_year",
    "registration_year",
    "mileage",
    "price",
    "fuel_type",
    "transmission",
    "engine_cc",
    "condition",
    "location",
    "district",
    "ad_date",
    "source",
    "scraped_at",
    "validation_issues",
    "is_valid",
]


class CSVExporter:
    """
    Exports normalized vehicle records to raw CSV snapshots in data/raw/.
    PostgreSQL remains the source of truth; CSV serves as an auditable export/snapshot.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.RAW_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_category_records(
        self,
        category: str,
        records: List[Dict[str, Any]],
        date_stamp: Optional[str] = None,
    ) -> Path:
        """
        Exports a list of normalized vehicle records for a category to CSV.
        Filename format: riyasewana_<category_slug>_YYYY_MM_DD.csv
        """
        clean_category = re.sub(r"[^a-zA-Z0-9_-]+", "_", category.lower()).strip("_")
        if not date_stamp:
            date_stamp = datetime.now(timezone.utc).strftime("%Y_%m_%d")

        file_name = f"riyasewana_{clean_category}_{date_stamp}.csv"
        file_path = self.output_dir / file_name

        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()

            for record in records:
                row = dict(record)

                # Format validation issues as JSON
                issues = row.get("validation_issues", [])
                if isinstance(issues, (list, dict)):
                    row["validation_issues"] = json.dumps(issues)
                elif issues is None:
                    row["validation_issues"] = json.dumps([])

                # Exclude internal raw_data from CSV
                row.pop("raw_data", None)

                writer.writerow(row)

        logger.info(f"Exported {len(records)} records to {file_path}")
        return file_path

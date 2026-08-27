"""
Pipeline Runner Module
Orchestrates the end-to-end data lifecycle:
Scrape -> Clean -> Quality Validation -> DB Delta Upsert -> Status History -> Telemetry.
"""
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from config import settings
from database.connection import init_db, SessionLocal
from database.repository import VehicleRepository
from database.models import Listing
from data_pipeline.cleaning.cleaners import DataCleaner
from data_pipeline.quality.quality_engine import DataQualityEngine
from data_pipeline.deduplication.dedup import DeduplicationEngine
from scraper.spiders.riyasewana_spider import RiyasewanaSpider
from scraper.mock_data_generator import generate_realistic_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(self, db: Optional[Session] = None):
        self.db = db or SessionLocal()
        self.repo = VehicleRepository(self.db)
        self.quality_engine = DataQualityEngine()

    def run_pipeline(
        self,
        category: Optional[str] = "cars",
        use_mock_data: bool = False,
        mock_count: int = 1500
    ) -> Dict[str, Any]:
        """
        Executes a complete pipeline run for a given category or all categories.
        """
        init_db()
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        logger.info(f"Starting pipeline execution: {run_id} (Category: {category}, Mock: {use_mock_data})")
        
        self.repo.start_scrape_run(run_id=run_id, category=category)
        run_start_time = datetime.now(timezone.utc)

        raw_records: List[Dict[str, Any]] = []
        observed_ids: List[str] = []

        try:
            if use_mock_data:
                logger.info(f"Generating {mock_count} realistic Sri Lankan vehicle records...")
                raw_records = generate_realistic_dataset(count=mock_count, inject_anomalies=True)
                observed_ids = [r["listing_id"] for r in raw_records]
            else:
                spider = RiyasewanaSpider()
                # Get known listing IDs to enable delta scraping
                known_listings = self.db.query(Listing.listing_id).all()
                known_set = {row[0] for row in known_listings}

                categories_to_crawl = [category] if category else settings.SUPPORTED_CATEGORIES
                for cat in categories_to_crawl:
                    records, ids = spider.crawl_category(category=cat, known_listing_ids=known_set)
                    raw_records.extend(records)
                    observed_ids.extend(ids)

            logger.info(f"Collected {len(raw_records)} raw records. Proceeding to cleaning...")

            # 1. Clean & Standardize
            cleaned_records = [DataCleaner.clean_record(r) for r in raw_records]

            # 2. Deduplication
            deduped_records = DeduplicationEngine.detect_duplicates(cleaned_records)

            # 3. Quality Validation & Outlier Detection
            iqr_bounds = self.quality_engine.compute_group_iqr_bounds(deduped_records)

            new_count = 0
            updated_count = 0
            price_change_count = 0
            invalid_count = 0
            duplicate_count = 0
            valid_count = 0
            suspicious_count = 0
            missing_count = 0

            # 4. Upsert to Database with Delta & Quality Tracking
            for rec in deduped_records:
                if rec.get("is_duplicate"):
                    duplicate_count += 1

                bounds_key = f"{rec.get('category', 'Cars')}|{rec.get('make', '')}|{rec.get('model', '')}".lower()
                group_bounds = iqr_bounds.get(bounds_key)

                q_status, q_score, q_issues, is_outlier = self.quality_engine.validate_single_record(
                    rec,
                    price_iqr_bounds=group_bounds
                )

                if q_status == "VALID":
                    valid_count += 1
                elif q_status == "SUSPICIOUS":
                    suspicious_count += 1
                elif q_status == "INVALID":
                    invalid_count += 1
                elif q_status == "MISSING":
                    missing_count += 1

                # Upsert into DB
                _, action = self.repo.upsert_listing(
                    listing_data=rec,
                    quality_status=q_status,
                    quality_score=q_score,
                    quality_issues=q_issues,
                    is_outlier=is_outlier
                )

                if action == "NEW":
                    new_count += 1
                elif action == "PRICE_CHANGED":
                    price_change_count += 1
                elif action == "UPDATED":
                    updated_count += 1

            # 5. Mark Unseen Listings as NO_LONGER_OBSERVED (only in live mode or multi-day runs)
            inactive_count = 0
            if not use_mock_data and observed_ids:
                inactive_count = self.repo.mark_unseen_as_inactive(
                    category=category,
                    seen_listing_ids=observed_ids,
                    run_start_time=run_start_time
                )

            # 6. Complete Scrape Run Telemetry
            self.repo.finish_scrape_run(
                run_id=run_id,
                total=len(raw_records),
                new_c=new_count,
                updated_c=updated_count,
                price_c=price_change_count,
                inactive_c=inactive_count,
                duplicate_c=duplicate_count,
                invalid_c=invalid_count,
                status="COMPLETED"
            )

            summary = {
                "run_id": run_id,
                "status": "COMPLETED",
                "total_scraped": len(raw_records),
                "new_listings": new_count,
                "updated_listings": updated_count,
                "price_changes": price_change_count,
                "no_longer_observed": inactive_count,
                "duplicates": duplicate_count,
                "valid_records": valid_count,
                "suspicious_records": suspicious_count,
                "invalid_records": invalid_count,
                "missing_records": missing_count,
                "data_quality_score": round((valid_count / len(raw_records) * 100), 2) if raw_records else 100.0
            }
            logger.info(f"Pipeline {run_id} completed successfully: {summary}")
            return summary

        except Exception as e:
            logger.error(f"Pipeline run {run_id} failed: {e}", exc_info=True)
            self.repo.finish_scrape_run(
                run_id=run_id,
                total=len(raw_records),
                new_c=0,
                updated_c=0,
                price_c=0,
                inactive_c=0,
                duplicate_c=0,
                invalid_c=0,
                status="FAILED",
                error_msg=str(e)
            )
            return {"run_id": run_id, "status": "FAILED", "error": str(e)}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Vehicle Data Pipeline")
    parser.add_argument("--seed-samples", action="store_true", help="Seed database with realistic mock data")
    parser.add_argument("--count", type=int, default=1500, help="Number of mock records to seed")
    parser.add_argument("--category", type=str, default=None, help="Specific category to scrape (default: all)")
    args = parser.parse_args()

    runner = PipelineRunner()
    res = runner.run_pipeline(
        category=args.category,
        use_mock_data=args.seed_samples,
        mock_count=args.count
    )
    print("Pipeline Execution Result:", res)

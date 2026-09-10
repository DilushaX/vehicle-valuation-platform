"""
Comprehensive test suite for Phase 4 Step 4: Automated Scheduling & Long-Term Operational Pipeline.
Covers:
1. Scheduler configuration loading & defaults
2. Scheduled command execution logic (sequential 8 categories)
3. Overlap prevention via CollectionLock (concurrent run detection)
4. Lock file cleanup on successful completion
5. Lock file cleanup on failed execution / exception
6. Failure isolation across categories
7. Disappearance safety under scheduling (complete vs partial vs failed)
8. Historical data preservation (first_seen_at, last_seen_at, observations, price history)
9. Rate limiting respected during scheduled runs
10. Dry-run mode without database mutations
11. Operational logging and aggregate metrics reporting
12. Independence and integrity of existing manual scraper
13. macOS launchd plist generation, time parsing, and status inspection
14. Crontab schedule format validation
"""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select

from config import settings
from data_pipeline.completeness import CategoryCompleteness
from data_pipeline.pipeline_runner import PipelineRunner
from data_pipeline.scheduler.launchd import (
    LAUNCHD_LABEL,
    generate_launchd_plist,
    get_launchd_status,
    parse_schedule_time,
)
from data_pipeline.scheduler.lock import CollectionLock
from database.models import Listing, ListingObservation, PriceHistory, ScrapeRun, Vehicle
from database.repository import VehicleRepository
from scraper.discovery.category_discovery import DiscoveredCategory, DiscoveredPage
from scraper.discovery.listing_discovery import DiscoveredListing
from scraper.spiders.riyasewana_category_spider import RiyasewanaCategorySpider
from scripts.scheduled_collection import (
    DEFAULT_SCHEDULED_CATEGORIES,
    print_completion_report,
    run_scheduled_collection,
)


# ==============================================================================
# 1. SCHEDULER CONFIGURATION LOADING & DEFAULTS
# ==============================================================================

def test_1_scheduler_configuration_defaults():
    """Verify scheduler configuration defaults and supported categories."""
    assert settings.COLLECTION_SCHEDULE == "daily"
    assert settings.COLLECTION_TIME == "02:00"
    assert isinstance(settings.SUPPORTED_CATEGORIES, list)
    assert len(settings.SUPPORTED_CATEGORIES) == 8
    expected_categories = [
        "Cars",
        "Heavy-Duty",
        "Lorries",
        "Motorbikes",
        "Pickups",
        "SUVs",
        "Three Wheelers",
        "Vans",
    ]
    assert settings.SUPPORTED_CATEGORIES == expected_categories
    assert settings.LOCK_FILE_PATH == Path("data/.collection.lock")
    assert DEFAULT_SCHEDULED_CATEGORIES == expected_categories


# ==============================================================================
# 2. SCHEDULED COMMAND EXECUTION LOGIC (ALL 8 CATEGORIES IN SEQUENCE)
# ==============================================================================

def test_2_scheduled_execution_calls_all_8_categories_in_sequence():
    """Verify run_scheduled_collection executes target categories sequentially with complete scope defaults."""
    with patch("scripts.scheduled_collection.PipelineRunner") as mock_runner_cls, \
         patch("scripts.scheduled_collection.RiyasewanaClient") as mock_client_cls, \
         patch("scripts.scheduled_collection.RiyasewanaCategorySpider") as mock_spider_cls:

        mock_runner_instance = MagicMock()
        mock_runner_cls.return_value = mock_runner_instance
        mock_runner_instance.run_pipeline.return_value = {
            "overall_status": "COMPLETED",
            "categories_processed": 8,
            "category_results": [
                {"category_name": cat, "status": "COMPLETED", "new_listings": 1, "observations_created": 1}
                for cat in DEFAULT_SCHEDULED_CATEGORIES
            ],
            "completeness_reports": [],
        }

        report = run_scheduled_collection(dry_run=True)

        assert report["overall_status"] == "COMPLETED"
        assert report["categories_processed"] == 8
        assert "elapsed_seconds" in report
        assert "aggregate_metrics" in report
        assert report["aggregate_metrics"]["total_new_listings"] == 8

        # Ensure run_pipeline was called with all 8 categories and complete scope (None, None)
        mock_runner_instance.run_pipeline.assert_called_once_with(
            categories=DEFAULT_SCHEDULED_CATEGORIES,
            max_pages=None,
            max_listings=None,
            dry_run=True,
            export_csv=True,
            request_delay=settings.REQUEST_DELAY,
        )


# ==============================================================================
# 3. OVERLAP PREVENTION (MUTUAL EXCLUSION LOCK)
# ==============================================================================

def test_3_overlap_prevention_second_process_exits(tmp_path):
    """Verify a second process attempting to acquire an active lock fails cleanly."""
    lock_file = tmp_path / "test.lock"

    lock1 = CollectionLock(lock_file_path=lock_file)
    lock2 = CollectionLock(lock_file_path=lock_file)

    # Process 1 acquires lock
    acquired1 = lock1.acquire()
    assert acquired1 is True
    assert lock1.is_locked is True
    assert lock_file.exists()

    # Process 2 attempts non-blocking acquisition and fails
    acquired2 = lock2.acquire()
    assert acquired2 is False
    assert lock2.is_locked is False

    # Process 1 releases
    lock1.release()
    assert lock1.is_locked is False

    # Now process 2 can acquire
    acquired2_retry = lock2.acquire()
    assert acquired2_retry is True
    lock2.release()


# ==============================================================================
# 4. LOCK FILE CLEANUP ON SUCCESSFUL COMPLETION
# ==============================================================================

def test_4_lock_cleanup_on_successful_completion(tmp_path):
    """Verify context manager acquires and cleanly releases the lock file on normal exit."""
    lock_file = tmp_path / "test_success.lock"

    with CollectionLock(lock_file_path=lock_file) as lock:
        assert lock.is_locked is True
        assert lock_file.exists()
        content = lock_file.read_text()
        assert f"pid={os.getpid()}" in content

    # Exited context manager
    assert lock.is_locked is False
    assert not lock_file.exists()


# ==============================================================================
# 5. LOCK FILE CLEANUP ON FAILED EXECUTION / EXCEPTION
# ==============================================================================

def test_5_lock_cleanup_on_exception(tmp_path):
    """Verify lock is released and unlinked even if an unhandled exception occurs."""
    lock_file = tmp_path / "test_exception.lock"

    with pytest.raises(RuntimeError, match="Simulated crash"):
        with CollectionLock(lock_file_path=lock_file) as lock:
            assert lock.is_locked is True
            assert lock_file.exists()
            raise RuntimeError("Simulated crash")

    assert lock.is_locked is False
    assert not lock_file.exists()

    # Verify a subsequent run can immediately acquire the lock
    subsequent_lock = CollectionLock(lock_file_path=lock_file)
    assert subsequent_lock.acquire() is True
    subsequent_lock.release()


# ==============================================================================
# 6. FAILURE ISOLATION ACROSS CATEGORIES
# ==============================================================================

def test_6_category_failure_isolation(db_session):
    """Verify an error in category 1 does not abort categories 2-8."""
    mock_spider = MagicMock(spec=RiyasewanaCategorySpider)

    # Category 1 fails with exception, Category 2 succeeds
    def mock_scrape(category, max_pages=None, max_listings=None):
        cat_name = getattr(category, "name", str(category))
        if cat_name == "Cars":
            raise ConnectionError("Riyasewana temporary network drop")
        return {
            "status": "COMPLETED",
            "pages_discovered": 1,
            "pages_attempted": 1,
            "pages_scraped": 1,
            "failed_pages": [],
            "listing_urls_discovered": 1,
            "unique_listing_urls": {"https://riyasewana.com/buy/van-01"},
            "listings_attempted": 1,
            "listings_scraped": 1,
            "failed_listings": [],
            "records": [
                {
                    "listing_id": "van_01",
                    "listing_url": "https://riyasewana.com/buy/van-01",
                    "title": "Toyota Hiace 2015",
                    "category": "Vans",
                    "brand": "Toyota",
                    "model": "Hiace",
                    "price": 12000000,
                    "is_valid": True,
                    "validation_issues": [],
                }
            ],
            "pagination_exhausted": True,
        }

    mock_spider.scrape_category.side_effect = mock_scrape

    runner = PipelineRunner(category_spider=mock_spider)
    report = runner.run_pipeline(
        categories=["Cars", "Vans"],
        dry_run=False,
        export_csv=False,
        db_session=db_session,
    )

    assert report["overall_status"] == "INCOMPLETE"
    assert report["categories_processed"] == 2

    res_cars = next(r for r in report["category_results"] if r["category_name"] == "Cars")
    assert res_cars["status"] == "FAILED"
    assert "Riyasewana temporary network drop" in res_cars["error"]

    res_vans = next(r for r in report["category_results"] if r["category_name"] == "Vans")
    assert res_vans["status"] == "COMPLETED"
    assert res_vans["new_listings"] == 1
    assert res_vans["observations_created"] == 1


# ==============================================================================
# 7. DISAPPEARANCE SAFETY UNDER SCHEDULING (COMPLETE VS PARTIAL VS FAILED)
# ==============================================================================

def test_7_disappearance_safety_complete_vs_partial_vs_failed(db_session):
    """
    Verify disappearance detection:
    - Only marks NO_LONGER_OBSERVED on a genuinely complete category run.
    - Never marks NO_LONGER_OBSERVED on a partial run (max_pages or max_listings set).
    - Never marks NO_LONGER_OBSERVED on a failed run.
    """
    repo = VehicleRepository(db_session)
    # Create baseline listing
    run1 = repo.create_scrape_run(category="Cars")
    listing_data = {
        "listing_id": "car_disappear_test",
        "listing_url": "https://riyasewana.com/buy/car-disappear-test",
        "title": "Nissan Leaf 2018",
        "category": "Cars",
        "brand": "Nissan",
        "model": "Leaf",
        "price": 6500000,
        "is_valid": True,
        "validation_issues": [],
    }
    listing, _ = repo.sync_listing(listing_data, scrape_run=run1)
    repo.commit()
    assert listing.current_status == "ACTIVE"

    mock_spider = MagicMock(spec=RiyasewanaCategorySpider)

    # 1. Partial run: max_pages=1 -> unobserved listing MUST REMAIN ACTIVE
    mock_spider.scrape_category.return_value = {
        "status": "COMPLETED",
        "pages_discovered": 5,
        "pages_attempted": 1,
        "pages_scraped": 1,
        "failed_pages": [],
        "listing_urls_discovered": 20,
        "unique_listing_urls": set(),
        "listings_attempted": 0,
        "listings_scraped": 0,
        "failed_listings": [],
        "records": [],
        "pagination_exhausted": False,
    }
    runner = PipelineRunner(category_spider=mock_spider)
    report_partial = runner.run_pipeline(
        categories=["Cars"],
        max_pages=1,
        dry_run=False,
        export_csv=False,
        db_session=db_session,
    )

    res_partial = report_partial["category_results"][0]
    assert res_partial["disappearance_definitive"] is False
    assert res_partial["disappeared_listings"] == 0

    reloaded = repo.find_listing("car_disappear_test")
    assert reloaded.current_status == "ACTIVE"  # Safe! Not marked NO_LONGER_OBSERVED

    # 2. Failed run: pagination failed -> unobserved listing MUST REMAIN ACTIVE
    mock_spider.scrape_category.return_value = {
        "status": "COMPLETED",
        "pages_discovered": 5,
        "pages_attempted": 5,
        "pages_scraped": 4,
        "failed_pages": ["https://riyasewana.com/search/cars?page=5"],
        "listing_urls_discovered": 80,
        "unique_listing_urls": set(),
        "listings_attempted": 0,
        "listings_scraped": 0,
        "failed_listings": [],
        "records": [],
        "pagination_exhausted": False,
    }
    report_failed_page = runner.run_pipeline(
        categories=["Cars"],
        dry_run=False,
        export_csv=False,
        db_session=db_session,
    )

    res_failed_page = report_failed_page["category_results"][0]
    assert res_failed_page["disappearance_definitive"] is False
    reloaded = repo.find_listing("car_disappear_test")
    assert reloaded.current_status == "ACTIVE"  # Safe!

    # 3. Genuinely complete run: pagination exhausted, zero failures -> definitively mark NO_LONGER_OBSERVED
    mock_spider.scrape_category.return_value = {
        "status": "COMPLETED",
        "pages_discovered": 1,
        "pages_attempted": 1,
        "pages_scraped": 1,
        "failed_pages": [],
        "listing_urls_discovered": 1,
        "unique_listing_urls": {"https://riyasewana.com/buy/other-car"},
        "listings_attempted": 1,
        "listings_scraped": 1,
        "failed_listings": [],
        "records": [
            {
                "listing_id": "other_car",
                "listing_url": "https://riyasewana.com/buy/other-car",
                "title": "Suzuki Alto 2015",
                "category": "Cars",
                "price": 3000000,
                "is_valid": True,
                "validation_issues": [],
            }
        ],
        "pagination_exhausted": True,
    }
    report_complete = runner.run_pipeline(
        categories=["Cars"],
        dry_run=False,
        export_csv=False,
        db_session=db_session,
    )

    res_complete = report_complete["category_results"][0]
    assert res_complete["disappearance_definitive"] is True
    assert res_complete["disappeared_listings"] == 1

    reloaded = repo.find_listing("car_disappear_test")
    assert reloaded.current_status == "NO_LONGER_OBSERVED"  # Safely marked disappeared


# ==============================================================================
# 8. HISTORICAL DATA PRESERVATION ACROSS MULTIPLE RUNS
# ==============================================================================

def test_8_historical_data_preservation_across_runs(vehicle_repo: VehicleRepository):
    """
    Verify historical preservation:
    - first_seen_at remains unchanged across scheduled runs.
    - last_seen_at updates.
    - Point-in-time observations are created on each run.
    - Price changes create new PriceHistory.
    - Identical prices do not duplicate PriceHistory.
    """
    # Run 1: Initial observation
    run1 = vehicle_repo.create_scrape_run(category="Cars")
    data_run1 = {
        "listing_id": "hist_car_01",
        "listing_url": "https://riyasewana.com/buy/hist-car-01",
        "title": "Honda Civic 2020",
        "category": "Cars",
        "price": 18000000,
        "is_valid": True,
        "validation_issues": [],
    }
    listing1, is_new1 = vehicle_repo.sync_listing(data_run1, scrape_run=run1)
    vehicle_repo.commit()

    assert is_new1 is True
    original_first_seen = listing1.first_seen_at
    original_last_seen = listing1.last_seen_at
    assert len(listing1.price_history) == 1
    assert len(listing1.observations) == 1

    # Run 2: Unchanged price
    run2 = vehicle_repo.create_scrape_run(category="Cars")
    listing2, is_new2 = vehicle_repo.sync_listing(data_run1, scrape_run=run2)
    vehicle_repo.commit()

    assert is_new2 is False
    assert listing2.first_seen_at == original_first_seen  # Immutable!
    assert listing2.last_seen_at >= original_last_seen
    assert len(listing2.price_history) == 1  # Unchanged price -> no duplicate PriceHistory
    assert len(listing2.observations) == 2   # New point-in-time observation recorded

    # Run 3: Changed price (18M -> 17.5M)
    run3 = vehicle_repo.create_scrape_run(category="Cars")
    data_run3 = dict(data_run1)
    data_run3["price"] = 17500000
    listing3, is_new3 = vehicle_repo.sync_listing(data_run3, scrape_run=run3)
    vehicle_repo.commit()

    assert is_new3 is False
    assert listing3.first_seen_at == original_first_seen  # Immutable!
    assert len(listing3.price_history) == 2  # New price -> price history created
    assert listing3.price_history[-1].price == 17500000
    assert len(listing3.observations) == 3   # Third observation created


# ==============================================================================
# 9. RATE LIMITING RESPECTED DURING SCHEDULED RUNS
# ==============================================================================

def test_9_rate_limiting_passed_to_spider_and_pipeline():
    """Verify politeness delay and retry parameters are forwarded to scraper and runner."""
    with patch("scripts.scheduled_collection.RiyasewanaCategorySpider") as mock_spider_cls, \
         patch("scripts.scheduled_collection.PipelineRunner") as mock_runner_cls, \
         patch("scripts.scheduled_collection.RiyasewanaClient") as mock_client_cls:

        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner
        mock_runner.run_pipeline.return_value = {
            "overall_status": "COMPLETED",
            "categories_processed": 1,
            "category_results": [],
            "completeness_reports": [],
        }

        run_scheduled_collection(
            categories=["Cars"],
            request_delay=2.5,
            request_timeout=35.0,
            max_retries=5,
            retry_backoff=3.0,
            dry_run=True,
        )

        mock_client_cls.assert_called_once_with(timeout=35.0)
        mock_spider_cls.assert_called_once_with(
            client=mock_client_cls.return_value,
            max_retries=5,
            retry_backoff=3.0,
            request_delay=2.5,
        )
        mock_runner.run_pipeline.assert_called_once_with(
            categories=["Cars"],
            max_pages=None,
            max_listings=None,
            dry_run=True,
            export_csv=True,
            request_delay=2.5,
        )


# ==============================================================================
# 10. DRY RUN MODE WORKS WITHOUT DATABASE WRITES
# ==============================================================================

def test_10_dry_run_mode_does_not_mutate_database(vehicle_repo: VehicleRepository):
    """Verify dry_run=True skips database synchronizations and scrape run creation."""
    mock_spider = MagicMock(spec=RiyasewanaCategorySpider)
    mock_spider.scrape_category.return_value = {
        "status": "COMPLETED",
        "pages_discovered": 1,
        "pages_attempted": 1,
        "pages_scraped": 1,
        "failed_pages": [],
        "listing_urls_discovered": 1,
        "unique_listing_urls": {"https://riyasewana.com/buy/car-dry"},
        "listings_attempted": 1,
        "listings_scraped": 1,
        "failed_listings": [],
        "records": [
            {
                "listing_id": "car_dry_01",
                "listing_url": "https://riyasewana.com/buy/car-dry",
                "title": "Dry Run Car",
                "category": "Cars",
                "price": 5000000,
                "is_valid": True,
                "validation_issues": [],
            }
        ],
        "pagination_exhausted": True,
    }

    runner = PipelineRunner(category_spider=mock_spider)
    with patch("data_pipeline.pipeline_runner.VehicleRepository", return_value=vehicle_repo):
        report = runner.run_pipeline(categories=["Cars"], dry_run=True, export_csv=False)

    assert report["overall_status"] == "COMPLETED"
    # Ensure zero records exist in repo
    all_listings = list(vehicle_repo.db.scalars(select(Listing)).all())
    assert len(all_listings) == 0


# ==============================================================================
# 11. OPERATIONAL LOGGING & AGGREGATE METRICS REPORTING
# ==============================================================================

def test_11_operational_logging_and_report_printing(capsys):
    """Verify print_completion_report formats aggregate metrics, failure reasons, and duration."""
    mock_report = {
        "overall_status": "COMPLETED",
        "categories_processed": 2,
        "elapsed_seconds": 45.2,
        "started_at": "2026-09-10 02:00:00 UTC",
        "category_results": [
            {
                "category_name": "Cars",
                "status": "COMPLETED",
                "new_listings": 10,
                "updated_listings": 25,
                "observations_created": 35,
                "price_changes": 2,
                "reactivated_listings": 1,
                "disappeared_listings": 0,
                "disappearance_definitive": True,
                "completeness": {
                    "pages_discovered": 5,
                    "pages_attempted": 5,
                    "pages_scraped": 5,
                    "failed_pages_count": 0,
                    "page_completeness_pct": 100.0,
                    "unique_listing_urls": 35,
                    "listing_urls_discovered": 35,
                    "listings_scraped": 35,
                    "listings_attempted": 35,
                    "failed_listings_count": 0,
                    "listing_completeness_pct": 100.0,
                },
            },
            {
                "category_name": "Vans",
                "status": "FAILED",
                "error": "HTTP 503 Service Unavailable",
            },
        ],
        "aggregate_metrics": {
            "total_new_listings": 10,
            "total_updated_listings": 25,
            "total_observations": 35,
            "total_price_changes": 2,
            "total_reactivated": 1,
            "total_disappeared": 0,
            "failed_categories_count": 1,
            "failed_categories": ["Vans"],
        },
    }

    print_completion_report(mock_report, dry_run=False)
    captured = capsys.readouterr()

    assert "SCHEDULED COLLECTION OPERATIONAL REPORT" in captured.out
    assert "Overall Status       : COMPLETED" in captured.out
    assert "Total Duration       : 45.2s" in captured.out
    assert "Category: Cars [COMPLETED]" in captured.out
    assert "Category: Vans [FAILED]" in captured.out
    assert "Failure Reason       : HTTP 503 Service Unavailable" in captured.out
    assert "AGGREGATE COLLECTION METRICS" in captured.out
    assert "Total New Listings Inserted   : 10" in captured.out
    assert "Total Observations Created    : 35" in captured.out
    assert "Failed Categories (1)       : ['Vans']" in captured.out


# ==============================================================================
# 12. EXISTING MANUAL SCRAPER WORKS INDEPENDENTLY
# ==============================================================================

def test_12_manual_scraper_cli_intact():
    """Verify manual scraper script `scripts/scrape_riyasewana.py` CLI is functional and distinct."""
    result = subprocess.run(
        [sys.executable, "scripts/scrape_riyasewana.py", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0
    assert "Riyasewana Vehicle Market Data Collection Pipeline" in result.stdout
    assert "--category" in result.stdout
    assert "--max-pages" in result.stdout


# ==============================================================================
# 13. MACOS LAUNCHD INTEGRATION & PROPERTY LIST VALIDATION
# ==============================================================================

def test_13_launchd_plist_generation_and_time_parsing():
    """Verify time parsing, XML plist structure, calendar interval, and status inspection."""
    # Test valid time parsing
    assert parse_schedule_time("02:00") == (2, 0)
    assert parse_schedule_time("14:30") == (14, 30)
    assert parse_schedule_time("00:00") == (0, 0)
    assert parse_schedule_time("23:59") == (23, 59)

    # Test invalid time formats
    with pytest.raises(ValueError, match="HH:MM"):
        parse_schedule_time("invalid")
    with pytest.raises(ValueError, match="between 0 and 23"):
        parse_schedule_time("25:00")
    with pytest.raises(ValueError, match="between 0 and 59"):
        parse_schedule_time("02:60")

    # Generate Plist XML for 03:15
    plist_xml = generate_launchd_plist(collection_time="03:15")
    assert "com.vehicle_valuation.collection" in plist_xml
    assert "scripts/scheduled_collection.py" in plist_xml

    # Parse XML and verify structure
    root = ET.fromstring(plist_xml)
    assert root.tag == "plist"
    dict_elem = root.find("dict")
    assert dict_elem is not None

    keys = [child.text for child in dict_elem.findall("key")]
    assert "Label" in keys
    assert "ProgramArguments" in keys
    assert "StartCalendarInterval" in keys
    assert "WorkingDirectory" in keys
    assert "StandardOutPath" in keys
    assert "StandardErrorPath" in keys

    # Verify status function returns dictionary
    status = get_launchd_status()
    assert isinstance(status, dict)
    assert status["label"] == LAUNCHD_LABEL
    assert "is_installed" in status
    assert "is_loaded" in status


# ==============================================================================
# 14. CRONTAB SCHEDULE FORMAT VALIDATION
# ==============================================================================

def test_14_crontab_schedule_format_validation():
    """Verify crontab format string matches standard POSIX 5-field cron specifications."""
    hour, minute = parse_schedule_time("02:00")
    cron_expression = f"{minute} {hour} * * *"
    assert cron_expression == "0 2 * * *"

    hour_custom, minute_custom = parse_schedule_time("04:30")
    cron_expression_custom = f"{minute_custom} {hour_custom} * * *"
    assert cron_expression_custom == "30 4 * * *"

    # Validate complete crontab command string
    proj_dir = Path("/mock/path")
    cron_entry = (
        f"{minute} {hour} * * * cd {proj_dir} && "
        f"PYTHONPATH=. {proj_dir / '.venv' / 'bin' / 'python'} "
        f"scripts/scheduled_collection.py >> {proj_dir / 'logs' / 'cron.log'} 2>&1"
    )
    assert "0 2 * * * cd /mock/path" in cron_entry
    assert "scripts/scheduled_collection.py" in cron_entry

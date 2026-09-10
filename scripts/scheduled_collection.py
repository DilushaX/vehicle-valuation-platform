#!/usr/bin/env python3
"""
Scheduled collection entry point for automated vehicle market data collection.
Executes complete-scope data collection across all 8 supported vehicle categories sequentially.
Invokes existing PipelineRunner without duplicating scraper or lifecycle logic.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from data_pipeline.export.csv_exporter import CSVExporter
from data_pipeline.pipeline_runner import PipelineRunner
from data_pipeline.scheduler.launchd import (
    generate_launchd_plist,
    get_launchd_status,
    install_launchd_agent,
    uninstall_launchd_agent,
)
from data_pipeline.scheduler.lock import CollectionLock
from scraper.client import RiyasewanaClient
from scraper.spiders.riyasewana_category_spider import RiyasewanaCategorySpider

# Default 8 canonical vehicle categories to collect in sequence
DEFAULT_SCHEDULED_CATEGORIES: List[str] = [
    "Cars",
    "Heavy-Duty",
    "Lorries",
    "Motorbikes",
    "Pickups",
    "SUVs",
    "Three Wheelers",
    "Vans",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scheduled_collection")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated Scheduled Vehicle Market Collection (Phase 4 Step 4)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Execute collection without mutating PostgreSQL tables or creating observations.",
    )
    parser.add_argument(
        "--category",
        nargs="+",
        default=None,
        help="Optional category override (default: all 8 canonical categories in sequence).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional pagination limit for testing/debugging (default: None for complete scope).",
    )
    parser.add_argument(
        "--max-listings",
        type=int,
        default=None,
        help="Optional listing limit for testing/debugging (default: None for complete scope).",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=settings.REQUEST_DELAY,
        help=f"Politeness delay between outbound requests in seconds (default: {settings.REQUEST_DELAY}s).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=settings.REQUEST_TIMEOUT,
        help=f"HTTP request timeout in seconds (default: {settings.REQUEST_TIMEOUT}s).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=settings.MAX_RETRIES,
        help=f"Maximum retries for transient errors (default: {settings.MAX_RETRIES}).",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=settings.RETRY_BACKOFF,
        help=f"Exponential retry backoff multiplier (default: {settings.RETRY_BACKOFF}).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        default=False,
        help="Disable CSV snapshot export to data/raw/.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for CSV snapshots (default: data/raw).",
    )
    parser.add_argument(
        "--lock-file",
        type=str,
        default=None,
        help="Custom lock file path (default: data/.collection.lock).",
    )
    parser.add_argument(
        "--no-lock",
        action="store_true",
        default=False,
        help="Bypass lock check (caution: may cause overlapping runs).",
    )
    parser.add_argument(
        "--time",
        type=str,
        default=None,
        help=f"Daily collection time in 24-hour HH:MM format (default: {settings.COLLECTION_TIME}).",
    )
    parser.add_argument(
        "--generate-plist",
        action="store_true",
        default=False,
        help="Generate and print macOS LaunchAgent property list XML to stdout and exit.",
    )
    parser.add_argument(
        "--install-launchd",
        action="store_true",
        default=False,
        help="Install and load macOS LaunchAgent into ~/Library/LaunchAgents.",
    )
    parser.add_argument(
        "--uninstall-launchd",
        action="store_true",
        default=False,
        help="Unload and delete macOS LaunchAgent from ~/Library/LaunchAgents.",
    )
    parser.add_argument(
        "--status-launchd",
        action="store_true",
        default=False,
        help="Check and display macOS LaunchAgent status and exit.",
    )
    return parser.parse_args()


def run_scheduled_collection(
    categories: Optional[List[str]] = None,
    max_pages: Optional[int] = None,
    max_listings: Optional[int] = None,
    dry_run: bool = False,
    request_delay: float = settings.REQUEST_DELAY,
    request_timeout: float = settings.REQUEST_TIMEOUT,
    max_retries: int = settings.MAX_RETRIES,
    retry_backoff: float = settings.RETRY_BACKOFF,
    export_csv: bool = True,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Executes a complete scheduled collection run across target categories.
    Guarantees:
    - Sequential execution of categories (never concurrent)
    - Complete category scope when max_pages and max_listings are None
    - Respects existing rate limits and politeness delays
    - Category failure isolation: failure in one category does not abort the rest
    - Returns structured completion report
    """
    target_categories = categories or DEFAULT_SCHEDULED_CATEGORIES
    start_time = time.time()
    started_at_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(start_time))

    logger.info("=" * 70)
    logger.info("SCHEDULED DATA COLLECTION INITIATED")
    logger.info(f"Start Time      : {started_at_str}")
    logger.info(f"Target Categories ({len(target_categories)}): {target_categories}")
    logger.info(f"Scope           : max_pages={max_pages or 'All (Complete Scope)'}, "
                f"max_listings={max_listings or 'All (Complete Scope)'}")
    logger.info(f"Rate Limiting   : delay={request_delay}s, timeout={request_timeout}s, "
                f"retries={max_retries} (backoff={retry_backoff}x)")
    logger.info(f"Dry Run Mode    : {'ENABLED (No DB mutations)' if dry_run else 'DISABLED (Live PostgreSQL Sync)'}")
    logger.info("=" * 70)

    # Initialize database schema if running live
    if not dry_run:
        try:
            from database.connection import Base, get_engine
            Base.metadata.create_all(bind=get_engine())
        except Exception as schema_err:
            logger.warning(f"Database schema check warning: {schema_err}")

    csv_exporter = CSVExporter(output_dir=Path(output_dir)) if output_dir else None

    client = RiyasewanaClient(timeout=request_timeout)
    category_spider = RiyasewanaCategorySpider(
        client=client,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        request_delay=request_delay,
    )
    runner = PipelineRunner(
        category_spider=category_spider,
        csv_exporter=csv_exporter,
    )

    try:
        report = runner.run_pipeline(
            categories=target_categories,
            max_pages=max_pages,
            max_listings=max_listings,
            dry_run=dry_run,
            export_csv=export_csv,
            request_delay=request_delay,
        )

        elapsed = round(time.time() - start_time, 2)
        report["elapsed_seconds"] = elapsed
        report["started_at"] = started_at_str

        # Compute aggregate operational metrics
        total_new = sum(r.get("new_listings", 0) for r in report.get("category_results", []))
        total_updated = sum(r.get("updated_listings", 0) for r in report.get("category_results", []))
        total_obs = sum(r.get("observations_created", 0) for r in report.get("category_results", []))
        total_prices = sum(r.get("price_changes", 0) for r in report.get("category_results", []))
        total_reactivated = sum(r.get("reactivated_listings", 0) for r in report.get("category_results", []))
        total_disappeared = sum(r.get("disappeared_listings", 0) for r in report.get("category_results", []))
        failed_categories = [
            r["category_name"] for r in report.get("category_results", []) if r.get("status") == "FAILED"
        ]

        report["aggregate_metrics"] = {
            "total_new_listings": total_new,
            "total_updated_listings": total_updated,
            "total_observations": total_obs,
            "total_price_changes": total_prices,
            "total_reactivated": total_reactivated,
            "total_disappeared": total_disappeared,
            "failed_categories_count": len(failed_categories),
            "failed_categories": failed_categories,
        }

        logger.info("=" * 70)
        logger.info("SCHEDULED DATA COLLECTION SUMMARY")
        logger.info(f"Overall Status       : {report.get('overall_status')}")
        logger.info(f"Duration             : {elapsed}s")
        logger.info(f"Categories Processed : {report.get('categories_processed')}/{len(target_categories)}")
        if failed_categories:
            logger.warning(f"Failed Categories    : {failed_categories}")
        if not dry_run:
            logger.info(f"New Listings Created : {total_new}")
            logger.info(f"Existing Observed    : {total_updated}")
            logger.info(f"Total Observations   : {total_obs}")
            logger.info(f"Price Changes        : {total_prices}")
            logger.info(f"Reactivated Listings : {total_reactivated}")
            logger.info(f"Disappeared Listings : {total_disappeared}")
        logger.info("=" * 70)

        return report

    finally:
        client.close()


def print_completion_report(report: dict, dry_run: bool):
    """Prints a structured operational completion report to stdout."""
    print("\n" + "=" * 70)
    print(" SCHEDULED COLLECTION OPERATIONAL REPORT")
    print("=" * 70)
    print(f"Overall Status       : {report['overall_status']}")
    print(f"Categories Processed : {report['categories_processed']}")
    print(f"Total Duration       : {report.get('elapsed_seconds', 0)}s")
    print("-" * 70)

    for res in report.get("category_results", []):
        cat_name = res["category_name"]
        status = res["status"]
        comp = res.get("completeness", {})

        print(f"\nCategory: {cat_name} [{status}]")
        if comp:
            print(f"  Pages (Discovered/Attempted/Scraped/Failed): "
                  f"{comp.get('pages_discovered')}/{comp.get('pages_attempted')}/"
                  f"{comp.get('pages_scraped')}/{comp.get('failed_pages_count')}")
            print(f"  Page Completeness    : {comp.get('page_completeness_pct')}%")
            print(f"  Listing URLs (Found) : {comp.get('unique_listing_urls')} unique "
                  f"({comp.get('listing_urls_discovered')} discovered)")
            print(f"  Listings Scraped     : {comp.get('listings_scraped')}/{comp.get('listings_attempted')} "
                  f"(Failed: {comp.get('failed_listings_count')})")
            print(f"  Listing Completeness : {comp.get('listing_completeness_pct')}%")

        if not dry_run:
            print(f"  New Listings         : {res.get('new_listings', 0)}")
            print(f"  Existing Listings    : {res.get('updated_listings', 0)}")
            print(f"  Observations Created : {res.get('observations_created', 0)}")
            print(f"  Price Changes        : {res.get('price_changes', 0)}")
            print(f"  Reactivated Listings : {res.get('reactivated_listings', 0)}")
            if res.get("disappearance_definitive"):
                print(f"  Disappeared Listings : {res.get('disappeared_listings', 0)} (NO_LONGER_OBSERVED)")
            else:
                print(f"  Disappearance Check  : SKIPPED (Partial/scoped collection; active listings preserved)")

        if res.get("csv_path"):
            print(f"  CSV Snapshot         : {res['csv_path']}")
        if res.get("error"):
            print(f"  Failure Reason       : {res['error']}")

    agg = report.get("aggregate_metrics", {})
    if agg and not dry_run:
        print("\n" + "-" * 70)
        print(" AGGREGATE COLLECTION METRICS")
        print("-" * 70)
        print(f"  Total New Listings Inserted   : {agg.get('total_new_listings', 0)}")
        print(f"  Total Existing Observed       : {agg.get('total_updated_listings', 0)}")
        print(f"  Total Observations Created    : {agg.get('total_observations', 0)}")
        print(f"  Total Price Changes Detected  : {agg.get('total_price_changes', 0)}")
        print(f"  Total Reactivated Listings    : {agg.get('total_reactivated', 0)}")
        print(f"  Total Disappeared Listings    : {agg.get('total_disappeared', 0)}")
        if agg.get("failed_categories_count", 0) > 0:
            print(f"  Failed Categories ({agg['failed_categories_count']})       : {agg.get('failed_categories')}")

    print("\n" + "=" * 70)


def main():
    args = parse_args()

    # Handle LaunchAgent inspection and management commands early
    if args.generate_plist:
        plist_xml = generate_launchd_plist(collection_time=args.time)
        print(plist_xml)
        return 0

    if args.status_launchd:
        status = get_launchd_status()
        print("=" * 60)
        print(" macOS LaunchAgent Status (com.vehicle_valuation.collection)")
        print("=" * 60)
        print(f"  Plist Path      : {status['plist_path']}")
        print(f"  Installed       : {'YES' if status['is_installed'] else 'NO'}")
        print(f"  Loaded (Active) : {'YES' if status['is_loaded'] else 'NO'}")
        print(f"  Configured Time : {status['configured_time']}")
        print(f"  Schedule        : {status['schedule']}")
        print("=" * 60)
        return 0

    if args.install_launchd:
        try:
            plist_p = install_launchd_agent(collection_time=args.time)
            print(f"[OK] Successfully installed and loaded LaunchAgent at: {plist_p}")
            print(f"[OK] Scheduled collection configured for daily at: {args.time or settings.COLLECTION_TIME}")
            return 0
        except Exception as e:
            print(f"[ERROR] Failed to install LaunchAgent: {e}")
            return 1

    if args.uninstall_launchd:
        removed = uninstall_launchd_agent()
        if removed:
            print("[OK] Successfully unloaded and removed macOS LaunchAgent.")
        else:
            print("[NOTICE] LaunchAgent was not installed.")
        return 0

    lock = None
    if not args.no_lock:
        lock_path = Path(args.lock_file) if args.lock_file else None
        lock = CollectionLock(lock_file_path=lock_path)
        if not lock.acquire():
            msg = (
                "Another collection run is currently active. "
                "Exiting safely to prevent overlapping collection runs."
            )
            logger.warning(msg)
            print(f"\n[LOCK NOTICE] {msg}")
            return 0

    try:
        report = run_scheduled_collection(
            categories=args.category,
            max_pages=args.max_pages,
            max_listings=args.max_listings,
            dry_run=args.dry_run,
            request_delay=args.request_delay,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            export_csv=not args.no_csv,
            output_dir=args.output_dir,
        )

        print_completion_report(report, args.dry_run)
        return 0 if report.get("overall_status") == "COMPLETED" else 1

    except Exception as exc:
        logger.error(f"Fatal error during scheduled collection: {exc}", exc_info=True)
        print(f"\n[FATAL ERROR] Scheduled collection failed: {exc}")
        return 1

    finally:
        if lock and lock.is_locked:
            lock.release()


if __name__ == "__main__":
    sys.exit(main())

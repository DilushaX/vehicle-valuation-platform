import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from data_pipeline.export.csv_exporter import CSVExporter
from data_pipeline.pipeline_runner import PipelineRunner
from scraper.client import RiyasewanaClient
from scraper.spiders.riyasewana_category_spider import RiyasewanaCategorySpider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scrape_riyasewana")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Riyasewana Vehicle Market Data Collection Pipeline (Phase 4)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Execute discovery and scraping without mutating the PostgreSQL database.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pagination pages to scrape per category (e.g. 2).",
    )
    parser.add_argument(
        "--max-listings",
        type=int,
        default=None,
        help="Maximum number of listings to scrape per category (e.g. 20).",
    )
    parser.add_argument(
        "--category",
        nargs="+",
        default=None,
        help="Specific categories to scrape (e.g. --category Cars SUVs or --category Cars,SUVs).",
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
    return parser.parse_args()


def main():
    args = parse_args()

    mode_label = "DRY RUN (No Database Mutations)" if args.dry_run else "LIVE PERSISTENCE (PostgreSQL Sync)"
    print("=" * 70)
    print(f" RIYASEWANA DATA COLLECTION PIPELINE — {mode_label}")
    print("=" * 70)
    print(f"Categories     : {args.category or 'Auto-discover public categories'}")
    print(f"Max Pages      : {args.max_pages or 'All accessible'}")
    print(f"Max Listings   : {args.max_listings or 'All accessible'}")
    print(f"Request Delay  : {args.request_delay}s")
    print(f"Request Timeout: {args.request_timeout}s")
    print(f"Max Retries    : {args.max_retries} (Backoff: {args.retry_backoff}x)")
    print(f"CSV Export     : {'Disabled' if args.no_csv else 'Enabled'}")
    print("=" * 70)

    csv_exporter = None
    if args.output_dir:
        csv_exporter = CSVExporter(output_dir=Path(args.output_dir))

    if not args.dry_run:
        try:
            from database.connection import Base, get_engine
            Base.metadata.create_all(bind=get_engine())
        except Exception as schema_err:
            logger.warning(f"Database schema initialization warning: {schema_err}")

    client = RiyasewanaClient(timeout=args.request_timeout)
    category_spider = RiyasewanaCategorySpider(
        client=client,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        request_delay=args.request_delay,
    )
    runner = PipelineRunner(
        category_spider=category_spider,
        csv_exporter=csv_exporter,
    )

    try:
        report = runner.run_pipeline(
            categories=args.category,
            max_pages=args.max_pages,
            max_listings=args.max_listings,
            dry_run=args.dry_run,
            export_csv=not args.no_csv,
            request_delay=args.request_delay,
        )

        print("\n" + "=" * 70)
        print(" PIPELINE COMPLETION REPORT")
        print("=" * 70)
        print(f"Overall Status       : {report['overall_status']}")
        print(f"Categories Processed : {report['categories_processed']}")
        print("-" * 70)

        for res in report["category_results"]:
            cat_name = res["category_name"]
            status = res["status"]
            comp = res.get("completeness", {})

            print(f"\nCategory: {cat_name} [{status}]")
            print(f"  Pages (Discovered/Attempted/Scraped/Failed): "
                  f"{comp.get('pages_discovered')}/{comp.get('pages_attempted')}/"
                  f"{comp.get('pages_scraped')}/{comp.get('failed_pages_count')}")
            print(f"  Page Completeness: {comp.get('page_completeness_pct')}%")
            print(f"  Listings Found   : {comp.get('unique_listing_urls')} unique")
            print(f"  Listings Scraped : {comp.get('listings_scraped')}/{comp.get('listings_attempted')} "
                  f"(Failed: {comp.get('failed_listings_count')})")
            print(f"  Listing Completeness: {comp.get('listing_completeness_pct')}%")

            if not args.dry_run:
                print(f"  Database Sync    : {res.get('new_listings', 0)} new, {res.get('updated_listings', 0)} updated")
            if res.get("csv_path"):
                print(f"  CSV Snapshot     : {res['csv_path']}")
            if res.get("error"):
                print(f"  Error            : {res['error']}")

        print("\n" + "=" * 70)
        return 0

    except Exception as exc:
        logger.error(f"Fatal pipeline error: {exc}", exc_info=True)
        print(f"\n[FATAL ERROR] Pipeline terminated unexpectedly: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

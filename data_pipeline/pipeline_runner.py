import logging
from typing import Any, Callable, Dict, List, Optional
from sqlalchemy.orm import Session

from config import settings
from data_pipeline.completeness import CategoryCompleteness
from data_pipeline.export.csv_exporter import CSVExporter
from database.connection import get_sessionmaker
from database.repository import VehicleRepository
from scraper.discovery.category_discovery import (
    DiscoveredCategory,
    RiyasewanaCategoryDiscovery,
)
from scraper.spiders.riyasewana_category_spider import RiyasewanaCategorySpider

logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    Coordinates the complete Phase 3 data collection pipeline:
    Category Discovery -> Pagination -> Listing Discovery -> Scraping ->
    Validation -> Database Synchronization -> ScrapeRun Completion ->
    CSV Export -> Completeness Verification.
    """

    def __init__(
        self,
        category_discovery: Optional[RiyasewanaCategoryDiscovery] = None,
        category_spider: Optional[RiyasewanaCategorySpider] = None,
        csv_exporter: Optional[CSVExporter] = None,
        session_factory: Optional[Callable[[], Session]] = None,
    ):
        self.category_discovery = (
            category_discovery or RiyasewanaCategoryDiscovery(client=None)
        )
        self.category_spider = category_spider or RiyasewanaCategorySpider()
        self.csv_exporter = csv_exporter or CSVExporter()
        self._session_factory = session_factory

    @property
    def session_factory(self) -> Callable[[], Session]:
        if self._session_factory is None:
            self._session_factory = get_sessionmaker()
        return self._session_factory

    def run_pipeline(
        self,
        categories: Optional[List[str]] = None,
        max_pages: Optional[int] = None,
        max_listings: Optional[int] = None,
        dry_run: bool = False,
        export_csv: bool = True,
        db_session: Optional[Session] = None,
        request_delay: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes the data collection pipeline across one or multiple categories.
        Ensures category isolation: failure in one category does not abort remaining categories.
        """
        delay = request_delay if request_delay is not None else settings.REQUEST_DELAY
        if request_delay is not None:
            self.category_spider.request_delay = delay
            self.category_spider.bulk_spider.request_delay = delay

        logger.info(
            f"Starting Phase 4 Pipeline (dry_run={dry_run}, max_pages={max_pages}, "
            f"max_listings={max_listings}, request_delay={delay}s)"
        )

        # 1. Determine categories to process
        target_categories = self._resolve_categories(categories)
        logger.info(f"Target categories: {[c.name for c in target_categories]}")

        category_results: List[Dict[str, Any]] = []
        completeness_reports: List[CategoryCompleteness] = []

        for cat_idx, cat in enumerate(target_categories):
            cat_name = cat.name
            logger.info(f"--- Processing Category: {cat_name} ---")

            session = db_session
            should_close_session = False
            repo: Optional[VehicleRepository] = None
            scrape_run = None

            if not dry_run:
                try:
                    if session is None:
                        session = self.session_factory()
                        should_close_session = True
                    repo = VehicleRepository(session)
                    scrape_run = repo.create_scrape_run(
                        category=cat_name,
                        source=settings.SOURCE_NAME,
                        pages_requested=0,
                    )
                except Exception as db_init_err:
                    logger.error(
                        f"Database initialization failed for {cat_name}: {db_init_err}"
                    )

            try:
                # 2. Scrape category
                cat_scrape_res = self.category_spider.scrape_category(
                    category=cat,
                    max_pages=max_pages,
                    max_listings=max_listings,
                )

                records = cat_scrape_res["records"]
                new_listings_count = 0
                updated_listings_count = 0
                reactivated_listings_count = 0
                price_changes_count = 0
                observations_count = 0
                disappeared_listings_count = 0
                disappearance_definitive = False
                observed_listing_ids: set[str] = set()

                # 3. Database synchronization (skipped if dry_run)
                if not dry_run and repo is not None and scrape_run is not None:
                    for record in records:
                        try:
                            listing, is_new = repo.sync_listing(
                                data=record,
                                scrape_run=scrape_run,
                            )
                            observed_listing_ids.add(listing.listing_id)
                            observations_count += 1
                            if is_new:
                                new_listings_count += 1
                            else:
                                updated_listings_count += 1
                                if getattr(listing, "_was_reactivated", False):
                                    reactivated_listings_count += 1
                            if getattr(listing, "_price_changed", False):
                                price_changes_count += 1
                        except Exception as sync_err:
                            logger.error(
                                f"Failed to sync listing {record.get('listing_id')}: {sync_err}"
                            )
                            repo.rollback()

                    # 3b. Scope-aware disappearance safety
                    # Enforce strict safety: Partial/scoped runs must NEVER mark unobserved listings as NO_LONGER_OBSERVED.
                    # Only a genuinely complete category-scope run can perform disappearance detection.
                    # Never infer SOLD.
                    is_complete_scope = (
                        cat_scrape_res["status"] == "COMPLETED"
                        and len(cat_scrape_res["failed_pages"]) == 0
                        and len(cat_scrape_res["failed_listings"]) == 0
                        and cat_scrape_res.get("pagination_exhausted", False)
                        and max_pages is None
                        and max_listings is None
                    )

                    if is_complete_scope:
                        marked = repo.mark_unobserved_listings(
                            observed_listing_ids=observed_listing_ids,
                            source=settings.SOURCE_NAME,
                            category=cat_name,
                        )
                        disappeared_listings_count = len(marked)
                        disappearance_definitive = True
                        logger.info(
                            f"Complete category scope confirmed for {cat_name}: "
                            f"{disappeared_listings_count} unobserved active listings transitioned to NO_LONGER_OBSERVED."
                        )
                    else:
                        logger.info(
                            f"Disappearance detection skipped for {cat_name}: collection scope was partial/scoped "
                            f"(max_pages={max_pages}, max_listings={max_listings}, "
                            f"pagination_exhausted={cat_scrape_res.get('pagination_exhausted')}). "
                            f"Existing ACTIVE listings remain unchanged."
                        )

                    # Update ScrapeRun completion
                    error_msg = None
                    if cat_scrape_res["failed_pages"] or cat_scrape_res["failed_listings"]:
                        error_msg = (
                            f"Failed pages: {len(cat_scrape_res['failed_pages'])}, "
                            f"Failed listings: {len(cat_scrape_res['failed_listings'])}"
                        )

                    repo.complete_scrape_run(
                        scrape_run=scrape_run,
                        pages_requested=cat_scrape_res["pages_attempted"],
                        pages_scraped=cat_scrape_res["pages_scraped"],
                        failed_pages=len(cat_scrape_res["failed_pages"]),
                        listings_found=len(cat_scrape_res["unique_listing_urls"]),
                        new_listings=new_listings_count,
                        updated_listings=updated_listings_count,
                        errors=error_msg,
                        failed_listings=len(cat_scrape_res["failed_listings"]),
                        price_changes=price_changes_count,
                        disappeared_listings=disappeared_listings_count,
                        reactivated_listings=reactivated_listings_count,
                        observations_created=observations_count,
                    )
                    repo.commit()

                # 4. CSV export (optional snapshot, skipped on dry_run)
                csv_path = None
                if export_csv and not dry_run and records:
                    csv_path = self.csv_exporter.export_category_records(
                        category=cat_name,
                        records=records,
                    )

                # 5. Completeness report
                is_complete_scope = (
                    cat_scrape_res["status"] == "COMPLETED"
                    and len(cat_scrape_res["failed_pages"]) == 0
                    and len(cat_scrape_res["failed_listings"]) == 0
                    and cat_scrape_res.get("pagination_exhausted", False)
                    and max_pages is None
                    and max_listings is None
                )
                completeness = CategoryCompleteness(
                    category_name=cat_name,
                    pages_discovered=cat_scrape_res["pages_discovered"],
                    pages_attempted=cat_scrape_res["pages_attempted"],
                    pages_scraped=cat_scrape_res["pages_scraped"],
                    failed_pages=cat_scrape_res["failed_pages"],
                    listing_urls_discovered=cat_scrape_res["listing_urls_discovered"],
                    unique_listing_urls=len(cat_scrape_res["unique_listing_urls"]),
                    listings_attempted=cat_scrape_res["listings_attempted"],
                    listings_scraped=cat_scrape_res["listings_scraped"],
                    failed_listings=cat_scrape_res["failed_listings"],
                    status=cat_scrape_res["status"],
                    max_pages_requested=max_pages,
                    max_listings_requested=max_listings,
                    is_complete_scope=is_complete_scope,
                )
                completeness_reports.append(completeness)

                cat_summary = {
                    "category_name": cat_name,
                    "dry_run": dry_run,
                    "status": cat_scrape_res["status"],
                    "completeness": completeness.to_dict(),
                    "csv_path": str(csv_path) if csv_path else None,
                    "new_listings": new_listings_count,
                    "updated_listings": updated_listings_count,
                    "observations_created": observations_count,
                    "price_changes": price_changes_count,
                    "reactivated_listings": reactivated_listings_count,
                    "disappeared_listings": disappeared_listings_count,
                    "disappearance_definitive": disappearance_definitive,
                }
                category_results.append(cat_summary)

            except Exception as cat_err:
                logger.error(f"Category {cat_name} encountered an error: {cat_err}")
                if not dry_run and repo is not None and scrape_run is not None:
                    try:
                        repo.rollback()
                        repo.fail_scrape_run(scrape_run, str(cat_err))
                        repo.commit()
                    except Exception as fail_err:
                        logger.error(f"Could not record scrape run failure: {fail_err}")

                completeness_reports.append(
                    CategoryCompleteness(
                        category_name=cat_name,
                        status="FAILED",
                        max_pages_requested=max_pages,
                        max_listings_requested=max_listings,
                    )
                )
                category_results.append(
                    {
                        "category_name": cat_name,
                        "dry_run": dry_run,
                        "status": "FAILED",
                        "error": str(cat_err),
                    }
                )

            finally:
                if should_close_session and session is not None:
                    session.close()

            # Polite delay between categories
            if delay > 0 and cat_idx < len(target_categories) - 1:
                import time
                time.sleep(delay)

        # Overall pipeline status
        all_completed = (
            all(cr["status"] == "COMPLETED" for cr in category_results)
            if category_results
            else False
        )
        overall_status = "COMPLETED" if all_completed else "INCOMPLETE"

        return {
            "overall_status": overall_status,
            "dry_run": dry_run,
            "scope": {
                "max_pages": max_pages,
                "max_listings": max_listings,
            },
            "categories_processed": len(category_results),
            "category_results": category_results,
            "completeness_reports": [cr.to_dict() for cr in completeness_reports],
        }

    def _resolve_categories(
        self, categories: Optional[List[str]]
    ) -> List[DiscoveredCategory]:
        """
        Resolves target categories: discovers public categories if not specified,
        or converts user-provided names/slugs into DiscoveredCategory instances.
        """
        if not categories:
            try:
                discovered = self.category_discovery.discover_categories()
                if discovered:
                    return discovered
            except Exception as e:
                logger.warning(
                    f"Dynamic category discovery failed ({e}). Falling back to defaults."
                )

            # Fallback default categories
            default_names = ["Cars", "SUVs", "Vans", "Motorbikes"]
            return [
                DiscoveredCategory(
                    name=name,
                    url=f"https://riyasewana.com/search/{name.lower()}",
                )
                for name in default_names
            ]

        resolved: List[DiscoveredCategory] = []
        raw_list: List[str] = []
        for cat in categories:
            for part in cat.split(","):
                part_clean = part.strip()
                if part_clean:
                    raw_list.append(part_clean)

        for cat_clean in raw_list:
            if cat_clean.startswith("http"):
                name = cat_clean.rstrip("/").split("/")[-1]
                name = name if any(c.isupper() for c in name) else name.capitalize()
                url = cat_clean
            else:
                name = cat_clean if any(c.isupper() for c in cat_clean) else cat_clean.capitalize()
                url = f"https://riyasewana.com/search/{cat_clean.lower()}"
            resolved.append(DiscoveredCategory(name=name, url=url))

        return resolved

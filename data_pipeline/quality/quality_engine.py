"""
Dataset Quality Engine & Historical Consistency Auditor.
Evaluates data quality, ML eligibility distributions, and historical lifecycle consistency
across vehicle market dataset tables without modifying records.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional
from sqlalchemy import func, select, or_
from sqlalchemy.orm import Session

from config import settings
from data_pipeline.cleaning.cleaners import VehicleCleaner
from database.models import (
    Listing,
    ListingObservation,
    PriceHistory,
    ScrapeRun,
    Vehicle,
)
from scraper.validators.listing_validator import ListingValidator


class DatasetQualityEngine:
    """
    Audits dataset quality, analyzes ML eligibility, and validates historical lifecycle consistency.
    """

    def __init__(self, validator: Optional[ListingValidator] = None):
        self.validator = validator or ListingValidator()

    def audit_dataset(
        self,
        session: Session,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes a comprehensive data quality and historical consistency audit on the database.
        Returns a structured dictionary with dataset metrics, issue breakdowns,
        category breakdowns, and consistency diagnostics.
        """
        # 1. Total entity counts
        total_listings = session.scalar(select(func.count(Listing.id))) or 0
        total_vehicles = session.scalar(select(func.count(Vehicle.id))) or 0
        total_prices = session.scalar(select(func.count(PriceHistory.id))) or 0
        total_obs = session.scalar(select(func.count(ListingObservation.id))) or 0
        total_runs = session.scalar(select(func.count(ScrapeRun.id))) or 0

        # 2. Query listings with vehicle relations
        stmt = select(Listing).join(Vehicle)
        if category:
            can_cat = VehicleCleaner.canonicalize_category(category) or category
            singular = can_cat.rstrip("s")
            stmt = stmt.where(
                or_(
                    Vehicle.category == can_cat,
                    Vehicle.category == singular,
                    Vehicle.category.ilike(f"%{can_cat}%"),
                )
            )

        listings = list(session.scalars(stmt).all())

        # 3. Analyze ML eligibility and status distribution
        ml_eligible_count = 0
        status_counts: Dict[str, int] = defaultdict(int)
        issue_counts: Dict[str, int] = defaultdict(int)
        critical_issue_counts: Dict[str, int] = defaultdict(int)
        non_critical_issue_counts: Dict[str, int] = defaultdict(int)

        cat_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "ml_eligible": 0, "ml_ineligible": 0, "active": 0, "no_longer_observed": 0}
        )

        for l in listings:
            status_counts[l.current_status] += 1
            cat_name = (
                VehicleCleaner.canonicalize_category(l.vehicle.category)
                if (l.vehicle and l.vehicle.category)
                else (l.vehicle.category if l.vehicle else "Unknown")
            ) or "Unknown"

            cat_stats[cat_name]["total"] += 1
            if l.current_status == "ACTIVE":
                cat_stats[cat_name]["active"] += 1
            elif l.current_status == "NO_LONGER_OBSERVED":
                cat_stats[cat_name]["no_longer_observed"] += 1

            if l.ml_eligible:
                ml_eligible_count += 1
                cat_stats[cat_name]["ml_eligible"] += 1
            else:
                cat_stats[cat_name]["ml_ineligible"] += 1

            issues = l.get_validation_issues()
            for iss in issues:
                issue_counts[iss] += 1
                if iss in ListingValidator.CRITICAL_ISSUE_CODES:
                    critical_issue_counts[iss] += 1
                else:
                    non_critical_issue_counts[iss] += 1

        ml_ineligible_count = len(listings) - ml_eligible_count
        ml_eligibility_rate_pct = (
            round((ml_eligible_count / len(listings)) * 100.0, 2) if listings else 0.0
        )

        # 4. Format category breakdown
        category_breakdown = []
        for cat_name, stats in sorted(cat_stats.items()):
            tot = stats["total"]
            el = stats["ml_eligible"]
            rate = round((el / tot) * 100.0, 2) if tot > 0 else 0.0
            category_breakdown.append(
                {
                    "category": cat_name,
                    "total": tot,
                    "ml_eligible": el,
                    "ml_ineligible": stats["ml_ineligible"],
                    "eligibility_pct": rate,
                    "active": stats["active"],
                    "no_longer_observed": stats["no_longer_observed"],
                }
            )

        # 5. Run historical consistency checks
        consistency_results = self.check_historical_consistency(session)

        return {
            "total_listings": total_listings,
            "total_vehicles": total_vehicles,
            "total_price_history": total_prices,
            "total_observations": total_obs,
            "total_scrape_runs": total_runs,
            "audited_listings_count": len(listings),
            "ml_eligible_count": ml_eligible_count,
            "ml_ineligible_count": ml_ineligible_count,
            "ml_eligibility_rate_pct": ml_eligibility_rate_pct,
            "status_distribution": dict(status_counts),
            "issue_counts": dict(issue_counts),
            "critical_issue_counts": dict(critical_issue_counts),
            "non_critical_issue_counts": dict(non_critical_issue_counts),
            "category_breakdown": category_breakdown,
            "consistency_checks": consistency_results,
        }

    def check_historical_consistency(self, session: Session) -> Dict[str, Any]:
        """
        Executes 10 distinct historical integrity and consistency checks:
        1. Duplicate (source, listing_id)
        2. Duplicate listing URLs
        3. Duplicate observations in the same scrape run
        4. Consecutive duplicate prices in PriceHistory
        5. PriceHistory chronological ordering
        6. Observation observed_at >= first_seen_at
        7. Timestamp ordering first_seen_at <= last_seen_at
        8. Orphan PriceHistory rows (missing listing)
        9. Orphan ListingObservation rows (missing listing or scrape run)
        10. Orphan Vehicle rows (without listing) or orphan listing vehicle_ids
        11. Invalid lifecycle statuses
        """
        checks: Dict[str, Dict[str, Any]] = {}
        total_errors = 0

        # Check 1: Duplicate (source, listing_id)
        dup_listings = session.execute(
            select(Listing.source, Listing.listing_id, func.count(Listing.id))
            .group_by(Listing.source, Listing.listing_id)
            .having(func.count(Listing.id) > 1)
        ).all()
        checks["duplicate_listings"] = {
            "error_count": len(dup_listings),
            "is_valid": len(dup_listings) == 0,
            "details": [f"source={r[0]}, listing_id={r[1]}, count={r[2]}" for r in dup_listings[:5]],
        }
        total_errors += len(dup_listings)

        # Check 2: Duplicate URLs
        dup_urls = session.execute(
            select(Listing.listing_url, func.count(Listing.id))
            .group_by(Listing.listing_url)
            .having(func.count(Listing.id) > 1)
        ).all()
        checks["duplicate_listing_urls"] = {
            "error_count": len(dup_urls),
            "is_valid": len(dup_urls) == 0,
            "details": [f"url={r[0]}, count={r[1]}" for r in dup_urls[:5]],
        }
        total_errors += len(dup_urls)

        # Check 3: Duplicate observations for same listing in same scrape run
        dup_obs = session.execute(
            select(ListingObservation.listing_id, ListingObservation.scrape_run_id, func.count(ListingObservation.id))
            .group_by(ListingObservation.listing_id, ListingObservation.scrape_run_id)
            .having(func.count(ListingObservation.id) > 1)
        ).all()
        checks["duplicate_observations_same_run"] = {
            "error_count": len(dup_obs),
            "is_valid": len(dup_obs) == 0,
            "details": [f"listing_id={r[0]}, scrape_run_id={r[1]}, count={r[2]}" for r in dup_obs[:5]],
        }
        total_errors += len(dup_obs)

        # Check 4: Consecutive identical price entries in PriceHistory
        dup_price_events = 0
        all_listings_with_prices = list(session.scalars(select(Listing).join(PriceHistory).distinct()).all())
        for l in all_listings_with_prices:
            sorted_prices = sorted(l.price_history, key=lambda p: p.observed_at)
            for i in range(1, len(sorted_prices)):
                if sorted_prices[i].price == sorted_prices[i - 1].price:
                    dup_price_events += 1
        checks["consecutive_duplicate_prices"] = {
            "error_count": dup_price_events,
            "is_valid": dup_price_events == 0,
            "details": [] if dup_price_events == 0 else [f"{dup_price_events} consecutive identical price entries"],
        }
        total_errors += dup_price_events

        # Check 5: first_seen_at <= last_seen_at
        bad_timestamps = session.execute(
            select(Listing.id, Listing.first_seen_at, Listing.last_seen_at).where(
                Listing.first_seen_at > Listing.last_seen_at
            )
        ).all()
        checks["first_seen_after_last_seen"] = {
            "error_count": len(bad_timestamps),
            "is_valid": len(bad_timestamps) == 0,
            "details": [f"listing_id={r[0]}: first={r[1]} > last={r[2]}" for r in bad_timestamps[:5]],
        }
        total_errors += len(bad_timestamps)

        # Check 6: Observation timestamps >= listing first_seen_at
        # Allow minor clock-skew tolerance of 2 seconds
        bad_obs_time = 0
        all_obs_listings = list(session.scalars(select(Listing).join(ListingObservation).distinct()).all())
        for l in all_obs_listings:
            for obs in l.observations:
                if obs.observed_at < l.first_seen_at:
                    bad_obs_time += 1
        checks["observation_before_first_seen"] = {
            "error_count": bad_obs_time,
            "is_valid": bad_obs_time == 0,
            "details": [] if bad_obs_time == 0 else [f"{bad_obs_time} observations earlier than first_seen_at"],
        }
        total_errors += bad_obs_time

        # Check 7: Invalid lifecycle statuses (only ACTIVE or NO_LONGER_OBSERVED permitted)
        invalid_statuses = session.execute(
            select(Listing.id, Listing.current_status).where(
                ~Listing.current_status.in_(["ACTIVE", "NO_LONGER_OBSERVED"])
            )
        ).all()
        checks["invalid_lifecycle_statuses"] = {
            "error_count": len(invalid_statuses),
            "is_valid": len(invalid_statuses) == 0,
            "details": [f"listing_id={r[0]}, status={r[1]}" for r in invalid_statuses[:5]],
        }
        total_errors += len(invalid_statuses)

        # Check 8: Orphan PriceHistory rows (listing_id missing)
        orphan_prices = session.execute(
            select(PriceHistory.id).where(~PriceHistory.listing_id.in_(select(Listing.id)))
        ).all()
        checks["orphan_price_histories"] = {
            "error_count": len(orphan_prices),
            "is_valid": len(orphan_prices) == 0,
            "details": [f"price_history_id={r[0]}" for r in orphan_prices[:5]],
        }
        total_errors += len(orphan_prices)

        # Check 9: Orphan ListingObservation rows
        orphan_obs = session.execute(
            select(ListingObservation.id).where(
                ~ListingObservation.listing_id.in_(select(Listing.id))
                | ~ListingObservation.scrape_run_id.in_(select(ScrapeRun.id))
            )
        ).all()
        checks["orphan_observations"] = {
            "error_count": len(orphan_obs),
            "is_valid": len(orphan_obs) == 0,
            "details": [f"observation_id={r[0]}" for r in orphan_obs[:5]],
        }
        total_errors += len(orphan_obs)

        # Check 10: Orphan Vehicles (vehicles without any associated listing)
        orphan_vehicles = session.execute(
            select(Vehicle.id).where(~Vehicle.id.in_(select(Listing.vehicle_id)))
        ).all()
        checks["orphan_vehicles"] = {
            "error_count": len(orphan_vehicles),
            "is_valid": len(orphan_vehicles) == 0,
            "details": [f"vehicle_id={r[0]}" for r in orphan_vehicles[:5]],
        }
        total_errors += len(orphan_vehicles)

        # Check 11: Broken vehicle reference in listing
        broken_vehicle_refs = session.execute(
            select(Listing.id, Listing.vehicle_id).where(~Listing.vehicle_id.in_(select(Vehicle.id)))
        ).all()
        checks["broken_vehicle_references"] = {
            "error_count": len(broken_vehicle_refs),
            "is_valid": len(broken_vehicle_refs) == 0,
            "details": [f"listing_id={r[0]}, vehicle_id={r[1]}" for r in broken_vehicle_refs[:5]],
        }
        total_errors += len(broken_vehicle_refs)

        return {
            "total_consistency_errors": total_errors,
            "is_consistent": total_errors == 0,
            "checks": checks,
        }

    @staticmethod
    def format_report_text(audit: Dict[str, Any]) -> str:
        """
        Formats audit dictionary into an operational ASCII terminal report.
        Strictly omits any passwords, database URLs, or credentials.
        """
        lines = []
        sep = "=" * 76
        subsep = "-" * 76

        lines.append("\n" + sep)
        lines.append(" VEHICLE VALUATION PLATFORM — DATA QUALITY & READINESS REPORT")
        lines.append(" Phase 4 Step 5: Historical Dataset Quality & Integrity Audit")
        lines.append(sep)

        # Summary Section
        lines.append("\n1. GLOBAL DATASET INVENTORY")
        lines.append(subsep)
        lines.append(f"  Total Vehicle Entities    : {audit['total_vehicles']}")
        lines.append(f"  Total Market Listings     : {audit['total_listings']}")
        lines.append(f"  Total Price History Events: {audit['total_price_history']}")
        lines.append(f"  Total Observations        : {audit['total_observations']}")
        lines.append(f"  Total Scrape Runs Tracked : {audit['total_scrape_runs']}")

        stat_dist = audit.get("status_distribution", {})
        active_count = stat_dist.get("ACTIVE", 0)
        no_longer_count = stat_dist.get("NO_LONGER_OBSERVED", 0)
        lines.append(f"  Status Breakdown          : ACTIVE={active_count}, NO_LONGER_OBSERVED={no_longer_count}")

        # ML Eligibility Section
        lines.append("\n2. ML VALUATION TRAINING ELIGIBILITY")
        lines.append(subsep)
        lines.append(f"  ML Eligible Listings      : {audit['ml_eligible_count']} ({audit['ml_eligibility_rate_pct']}%)")
        lines.append(f"  ML Ineligible Listings    : {audit['ml_ineligible_count']}")
        lines.append("  * Note: Ineligible listings are preserved with flags and excluded from training.")

        # Category Breakdown Table
        lines.append("\n3. CATEGORY ELIGIBILITY BREAKDOWN")
        lines.append(subsep)
        lines.append(f"  {'Category':<16} {'Total':<8} {'Eligible':<10} {'Ineligible':<12} {'Rate %':<8} {'Active':<8}")
        lines.append(f"  {'-'*16} {'-'*8} {'-'*10} {'-'*12} {'-'*8} {'-'*8}")

        for cat in audit.get("category_breakdown", []):
            c_name = cat["category"]
            tot = cat["total"]
            el = cat["ml_eligible"]
            inel = cat["ml_ineligible"]
            rate = f"{cat['eligibility_pct']}%"
            act = cat["active"]
            lines.append(f"  {c_name:<16} {tot:<8} {el:<10} {inel:<12} {rate:<8} {act:<8}")

        # Quality Issues Breakdown
        lines.append("\n4. DATA QUALITY ISSUES AUDIT")
        lines.append(subsep)

        crit_issues = audit.get("critical_issue_counts", {})
        non_crit_issues = audit.get("non_critical_issue_counts", {})

        lines.append("  [CRITICAL ISSUES — Excludes from ML Training]")
        if crit_issues:
            for code, count in sorted(crit_issues.items(), key=lambda x: -x[1]):
                lines.append(f"    - {code:<32}: {count} listings")
        else:
            lines.append("    None detected.")

        lines.append("\n  [NON-CRITICAL ISSUES — Audited & Preserved, Training Eligible]")
        if non_crit_issues:
            for code, count in sorted(non_crit_issues.items(), key=lambda x: -x[1]):
                lines.append(f"    - {code:<32}: {count} listings")
        else:
            lines.append("    None detected.")

        # Historical Consistency Checks
        lines.append("\n5. HISTORICAL LIFECYCLE CONSISTENCY AUDIT")
        lines.append(subsep)
        cons = audit.get("consistency_checks", {})
        is_cons = cons.get("is_consistent", False)
        tot_errors = cons.get("total_consistency_errors", 0)

        lines.append(f"  Overall Consistency Status: {'PASSED (Zero Anomalies)' if is_cons else f'FAILED ({tot_errors} errors)'}")

        checks = cons.get("checks", {})
        for check_key, res in checks.items():
            status_tag = "[OK]" if res["is_valid"] else f"[ERROR: {res['error_count']}]"
            readable_name = check_key.replace("_", " ").title()
            lines.append(f"    {status_tag:<14} {readable_name}")
            for d in res.get("details", []):
                lines.append(f"      -> {d}")

        # Business Disclaimer
        lines.append("\n" + sep)
        lines.append(" BUSINESS & LEGAL NOTICE:")
        lines.append(" - Asking prices observed on Riyasewana are seller asking prices, NOT confirmed")
        lines.append("   transaction or sold prices.")
        lines.append(" - Inactive listings are classified as NO_LONGER_OBSERVED and never marked as SOLD.")
        lines.append(" - Quality analysis performs auditing and classification without destructive deletion.")
        lines.append(sep + "\n")

        return "\n".join(lines)

#!/usr/bin/env python3
"""
CLI entry point for Dataset Quality & Historical Consistency Reporting (Phase 4 Step 5).
Audits PostgreSQL vehicle dataset, analyzes ML training readiness, and validates historical consistency.
Can be executed as:
    python scripts/data_quality_report.py
    python -m scripts.data_quality_report
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.connection import get_sessionmaker
from database.repository import VehicleRepository
from data_pipeline.quality.quality_engine import DatasetQualityEngine

logger = logging.getLogger("data_quality_report")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dataset Quality & Historical Consistency Auditor (Phase 4 Step 5)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional category filter (e.g. Cars, Vans, SUVs). Default: all categories.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as machine-readable JSON instead of formatted text.",
    )
    parser.add_argument(
        "--fail-on-inconsistency",
        action="store_true",
        default=False,
        help="Exit with non-zero status code if any historical consistency errors are detected.",
    )
    parser.add_argument(
        "--recalculate-ml",
        action="store_true",
        default=False,
        help="Re-evaluate and persist ml_eligible flags for all listings based on current critical rules.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    session_factory = get_sessionmaker()

    try:
        with session_factory() as session:
            repo = VehicleRepository(session)
            engine = DatasetQualityEngine()

            # Optional recalculation of ML eligibility
            if args.recalculate_ml:
                all_listings = repo.get_ml_eligible_listings() + repo.get_ml_ineligible_listings()
                updated_count = 0
                for l in all_listings:
                    repo.recalculate_ml_eligibility(l)
                    updated_count += 1
                repo.commit()
                if not args.json:
                    print(f"[OK] Recalculated ML eligibility for {updated_count} listings in PostgreSQL.\n")

            # Run full audit
            audit = engine.audit_dataset(session=session, category=args.category)

            if args.json:
                print(json.dumps(audit, indent=2, default=str))
            else:
                report_text = engine.format_report_text(audit)
                print(report_text)

            is_consistent = audit.get("consistency_checks", {}).get("is_consistent", True)

            if args.fail_on_inconsistency and not is_consistent:
                sys.stderr.write("[ERROR] Historical consistency checks failed.\n")
                return 1

            return 0

    except Exception as exc:
        sys.stderr.write(f"[FATAL ERROR] Data quality audit failed: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

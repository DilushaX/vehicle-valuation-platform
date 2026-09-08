"""
scripts/preflight_check.py

Production Preflight Validation Script for PostgreSQL Data Collection.

Performs non-destructive environment, configuration, dependency, and database checks
prior to running production data collection.

Does NOT scrape Riyasewana.
Does NOT insert test records into the production database.
"""

import argparse
import importlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from database.connection import Base
import database.models  # noqa: F401 - Register models with Base.metadata

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("preflight_check")

REQUIRED_PACKAGES = [
    ("psycopg2", "psycopg2-binary or psycopg2"),
    ("sqlalchemy", "SQLAlchemy"),
    ("httpx", "httpx"),
    ("bs4", "beautifulsoup4"),
    ("pydantic_settings", "pydantic-settings"),
    ("pydantic", "pydantic"),
    ("dotenv", "python-dotenv"),
]

REQUIRED_TABLES = [
    "vehicles",
    "listings",
    "price_history",
    "scrape_runs",
    "listing_observations",
]


def mask_db_url(url: str) -> str:
    """Masks password credentials in a database connection URL for safe logging/display."""
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)


def check_dependencies() -> Tuple[bool, List[str]]:
    """Verifies that all required production Python packages are installed and importable."""
    messages = []
    all_ok = True
    for module_name, display_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module_name)
            messages.append(f"[OK] Dependency '{display_name}' ({module_name}) is available.")
        except ImportError:
            all_ok = False
            messages.append(f"[FAIL] Missing required dependency: '{display_name}' (import {module_name}).")
    return all_ok, messages


def check_scraper_config(cfg: Optional[Any] = None) -> Tuple[bool, List[str]]:
    """
    Validates scraper configuration values:
    - REQUEST_DELAY must be >= 1.0 (recommended >= 1.5) to avoid HTTP 429 rate limits.
    - MAX_RETRIES must be between 1 and 10.
    - RETRY_BACKOFF must be >= 1.0.
    - REQUEST_TIMEOUT must be >= 5.0.
    """
    s = cfg or settings
    messages = []
    all_ok = True

    # 1. REQUEST_DELAY
    if s.REQUEST_DELAY <= 0:
        all_ok = False
        messages.append(f"[FAIL] REQUEST_DELAY ({s.REQUEST_DELAY}s) must be greater than 0.")
    elif s.REQUEST_DELAY < 1.0:
        all_ok = False
        messages.append(
            f"[FAIL] REQUEST_DELAY ({s.REQUEST_DELAY}s) is dangerously low! "
            f"Must be >= 1.0s (recommended >= 1.5s) to avoid upstream rate limits."
        )
    elif s.REQUEST_DELAY < 1.5:
        messages.append(
            f"[WARN] REQUEST_DELAY is {s.REQUEST_DELAY}s. "
            f"Production recommended is >= 1.5s for safe long-running scrapes."
        )
    else:
        messages.append(f"[OK] REQUEST_DELAY is {s.REQUEST_DELAY}s (safe conservative rate limit).")

    # 2. MAX_RETRIES
    if not (1 <= s.MAX_RETRIES <= 10):
        all_ok = False
        messages.append(f"[FAIL] MAX_RETRIES ({s.MAX_RETRIES}) must be between 1 and 10.")
    else:
        messages.append(f"[OK] MAX_RETRIES is {s.MAX_RETRIES}.")

    # 3. RETRY_BACKOFF
    if s.RETRY_BACKOFF < 1.0:
        all_ok = False
        messages.append(f"[FAIL] RETRY_BACKOFF ({s.RETRY_BACKOFF}) must be >= 1.0.")
    else:
        messages.append(f"[OK] RETRY_BACKOFF is {s.RETRY_BACKOFF}.")

    # 4. REQUEST_TIMEOUT
    if s.REQUEST_TIMEOUT < 5.0:
        all_ok = False
        messages.append(f"[FAIL] REQUEST_TIMEOUT ({s.REQUEST_TIMEOUT}s) is too low. Must be >= 5.0s.")
    else:
        messages.append(f"[OK] REQUEST_TIMEOUT is {s.REQUEST_TIMEOUT}s.")

    return all_ok, messages


def check_output_directory(raw_dir: Optional[Path] = None) -> Tuple[bool, List[str]]:
    """Verifies that the CSV output directory exists or can be created and is writable."""
    target_dir = raw_dir or settings.RAW_DATA_DIR
    messages = []
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        # Test write permission without leaving persistent artifacts
        test_file = target_dir / ".preflight_write_test"
        test_file.write_text("preflight_ok", encoding="utf-8")
        test_file.unlink()
        messages.append(f"[OK] Output directory '{target_dir}' exists and is writable.")
        return True, messages
    except Exception as exc:
        messages.append(f"[FAIL] Output directory '{target_dir}' write check error: {exc}")
        return False, messages


def check_database(
    db_url: Optional[str] = None,
    init_tables: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validates the database connection and schema:
    - Checks that DATABASE_URL is set and non-empty.
    - Connects to database and runs `SELECT 1;`.
    - Inspects existing tables.
    - If tables are missing and init_tables is True, initializes them safely using Base.metadata.create_all.
    - Never inserts test data.
    - Never drops existing tables.
    """
    from sqlalchemy import create_engine, inspect, text

    if db_url is not None:
        url = db_url
    else:
        url = os.getenv("DATABASE_URL") or settings.DATABASE_URL
    messages = []

    if not url or not url.strip():
        messages.append("[FAIL] DATABASE_URL is not set or is empty.")
        return False, messages

    masked = mask_db_url(url)
    messages.append(f"Target Database: {masked}")

    # Check dialect
    if not url.startswith("postgresql"):
        messages.append(
            f"[WARN] DATABASE_URL does not use PostgreSQL dialect (current: {url.split('://')[0] if '://' in url else url})."
        )
    else:
        messages.append("[OK] DATABASE_URL specifies PostgreSQL dialect.")

    # Attempt connection
    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
        messages.append("[OK] Database connectivity check succeeded (SELECT 1).")
    except Exception as conn_err:
        messages.append(f"[FAIL] Database connection failed: {conn_err}")
        return False, messages

    # Inspect tables
    try:
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
        missing_tables = [t for t in REQUIRED_TABLES if t not in existing_tables]

        if not missing_tables:
            messages.append(f"[OK] All {len(REQUIRED_TABLES)} required tables exist: {', '.join(REQUIRED_TABLES)}.")
            # Check ad_date column on listings table
            columns = {col["name"] for col in inspector.get_columns("listings")}
            if "ad_date" in columns:
                messages.append("[OK] Schema contains 'ad_date' column on 'listings' table for historical preservation.")
            else:
                messages.append("[WARN] Schema missing 'ad_date' column on 'listings' table.")
            return True, messages

        if init_tables:
            messages.append(f"Initializing missing tables safely: {', '.join(missing_tables)}...")
            Base.metadata.create_all(bind=engine)
            # Re-inspect to confirm creation
            re_inspector = inspect(engine)
            re_existing = set(re_inspector.get_table_names())
            still_missing = [t for t in REQUIRED_TABLES if t not in re_existing]
            if still_missing:
                messages.append(f"[FAIL] Failed to initialize tables: {', '.join(still_missing)}.")
                return False, messages
            messages.append(f"[OK] Successfully initialized tables: {', '.join(missing_tables)}.")
            return True, messages
        else:
            messages.append(
                f"[FAIL] Missing database tables: {', '.join(missing_tables)}. "
                f"Run preflight with '--init-db' to safely initialize required tables without dropping existing data."
            )
            return False, messages

    except Exception as inspect_err:
        messages.append(f"[FAIL] Table inspection failed: {inspect_err}")
        return False, messages


def run_preflight(
    db_url: Optional[str] = None,
    init_tables: bool = False,
    raw_dir: Optional[Path] = None,
    cfg: Optional[Any] = None,
) -> Dict[str, Any]:
    """Runs all preflight checks and returns a summary dict."""
    dep_ok, dep_msgs = check_dependencies()
    cfg_ok, cfg_msgs = check_scraper_config(cfg)
    dir_ok, dir_msgs = check_output_directory(raw_dir)
    db_ok, db_msgs = check_database(db_url, init_tables)

    all_passed = dep_ok and cfg_ok and dir_ok and db_ok

    return {
        "all_passed": all_passed,
        "dependencies": {"ok": dep_ok, "messages": dep_msgs},
        "config": {"ok": cfg_ok, "messages": cfg_msgs},
        "output_dir": {"ok": dir_ok, "messages": dir_msgs},
        "database": {"ok": db_ok, "messages": db_msgs},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Production PostgreSQL Configuration & Preflight Validation"
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        default=False,
        help="Safely create any missing database tables without dropping existing data.",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Override database connection URL for preflight validation.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" PRODUCTION PREFLIGHT VALIDATION (NON-DESTRUCTIVE)")
    print("=" * 70)
    print("Safety Check: This tool does NOT scrape Riyasewana.")
    print("Safety Check: This tool does NOT insert test records into the database.")
    print("=" * 70)

    results = run_preflight(
        db_url=args.db_url,
        init_tables=args.init_db,
    )

    print("\n1. PYTHON DEPENDENCIES:")
    for msg in results["dependencies"]["messages"]:
        print(f"  {msg}")

    print("\n2. SCRAPER SAFETY CONFIGURATION:")
    for msg in results["config"]["messages"]:
        print(f"  {msg}")

    print("\n3. OUTPUT DIRECTORY:")
    for msg in results["output_dir"]["messages"]:
        print(f"  {msg}")

    print("\n4. DATABASE CONNECTIVITY & SCHEMA:")
    for msg in results["database"]["messages"]:
        print(f"  {msg}")

    print("\n" + "=" * 70)
    if results["all_passed"]:
        print(" [PASS] PREFLIGHT VALIDATION SUCCEEDED: Ready for production data collection.")
        print("=" * 70)
        return 0
    else:
        print(" [FAIL] PREFLIGHT VALIDATION FAILED: Resolve issues above before collecting data.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

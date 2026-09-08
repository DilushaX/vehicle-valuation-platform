from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, inspect

from database.connection import Base
from database.models import Listing, Vehicle, ScrapeRun, ListingObservation, PriceHistory
from scripts.preflight_check import (
    REQUIRED_PACKAGES,
    REQUIRED_TABLES,
    check_database,
    check_dependencies,
    check_output_directory,
    check_scraper_config,
    mask_db_url,
    run_preflight,
)


def test_dependencies_check_succeeds():
    """Verifies that all required production dependencies are detected."""
    ok, messages = check_dependencies()
    assert ok is True
    assert len(messages) == len(REQUIRED_PACKAGES)
    assert all("[OK]" in msg for msg in messages)


def test_mask_db_url():
    """Verifies credentials are securely masked in logs and command outputs."""
    url = "postgresql+psycopg2://admin_user:SuperSecretPassword123@db.internal:5432/prod_vehicles"
    masked = mask_db_url(url)
    assert "SuperSecretPassword123" not in masked
    assert "admin_user:***@db.internal" in masked


def test_scraper_config_valid():
    """Verifies conservative scraper configuration passes validation."""
    cfg = SimpleNamespace(
        REQUEST_DELAY=1.5,
        MAX_RETRIES=3,
        RETRY_BACKOFF=2.0,
        REQUEST_TIMEOUT=20.0,
    )
    ok, messages = check_scraper_config(cfg)
    assert ok is True
    assert any("safe conservative rate limit" in m for m in messages)


def test_scraper_config_dangerously_low_delay():
    """Verifies that dangerously low delays (<1.0s) fail preflight check."""
    cfg = SimpleNamespace(
        REQUEST_DELAY=0.2,
        MAX_RETRIES=3,
        RETRY_BACKOFF=2.0,
        REQUEST_TIMEOUT=20.0,
    )
    ok, messages = check_scraper_config(cfg)
    assert ok is False
    assert any("dangerously low" in m for m in messages)


def test_scraper_config_invalid_retries_and_timeout():
    """Verifies invalid retry and timeout thresholds are rejected."""
    cfg_retries = SimpleNamespace(
        REQUEST_DELAY=1.5,
        MAX_RETRIES=0,
        RETRY_BACKOFF=2.0,
        REQUEST_TIMEOUT=20.0,
    )
    ok, messages = check_scraper_config(cfg_retries)
    assert ok is False
    assert any("MAX_RETRIES" in m for m in messages)

    cfg_timeout = SimpleNamespace(
        REQUEST_DELAY=1.5,
        MAX_RETRIES=3,
        RETRY_BACKOFF=2.0,
        REQUEST_TIMEOUT=2.0,
    )
    ok, messages = check_scraper_config(cfg_timeout)
    assert ok is False
    assert any("REQUEST_TIMEOUT" in m for m in messages)


def test_output_directory_check(tmp_path: Path):
    """Verifies output directory existence and write capability."""
    ok, messages = check_output_directory(tmp_path)
    assert ok is True
    assert any("exists and is writable" in m for m in messages)
    # Verify no persistent temporary test file remains
    assert not (tmp_path / ".preflight_write_test").exists()


def test_database_check_empty_url():
    """Verifies failure when DATABASE_URL is missing or empty."""
    ok, messages = check_database(db_url="")
    assert ok is False
    assert any("not set or is empty" in m for m in messages)


def test_database_check_missing_tables_fails_without_init(tmp_path: Path):
    """Verifies database check reports missing tables and fails when init_tables is False."""
    db_file = tmp_path / "empty_test.db"
    db_url = f"sqlite:///{db_file}"

    ok, messages = check_database(db_url=db_url, init_tables=False)
    assert ok is False
    assert any("Missing database tables" in m for m in messages)
    assert any("--init-db" in m for m in messages)


def test_database_check_with_init_tables(tmp_path: Path):
    """Verifies --init-db initializes tables and confirms ad_date column presence."""
    db_file = tmp_path / "init_test.db"
    db_url = f"sqlite:///{db_file}"

    ok, messages = check_database(db_url=db_url, init_tables=True)
    assert ok is True
    assert any("Successfully initialized tables" in m for m in messages)

    # Inspect initialized tables
    engine = create_engine(db_url)
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    for req in REQUIRED_TABLES:
        assert req in existing

    cols = {c["name"] for c in inspector.get_columns("listings")}
    assert "ad_date" in cols


def test_preflight_does_not_insert_test_data(tmp_path: Path):
    """Verifies that preflight validation does NOT insert test records into the database."""
    db_file = tmp_path / "clean_test.db"
    db_url = f"sqlite:///{db_file}"

    results = run_preflight(
        db_url=db_url,
        init_tables=True,
        raw_dir=tmp_path / "raw",
    )
    assert results["all_passed"] is True

    engine = create_engine(db_url)
    with engine.connect() as conn:
        for table in REQUIRED_TABLES:
            count = conn.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").scalar()
            assert count == 0, f"Table {table} must have 0 rows after preflight!"


def test_preflight_does_not_scrape_riyasewana(tmp_path: Path):
    """Verifies that running preflight validation makes ZERO network calls to Riyasewana."""
    db_file = tmp_path / "network_test.db"
    db_url = f"sqlite:///{db_file}"

    with patch("scraper.client.RiyasewanaClient.get") as mock_get:
        results = run_preflight(
            db_url=db_url,
            init_tables=True,
            raw_dir=tmp_path / "raw",
        )
        assert results["all_passed"] is True
        mock_get.assert_not_called()

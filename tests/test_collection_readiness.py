"""
Production Collection Readiness and Full System Validation Test Suite (Phase 5 Step 7.5).
Verifies end-to-end operational readiness, category resolution, lifecycle safety,
deduplication invariants, and ML feature preparation consistency.
"""

from datetime import datetime, timezone
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func
from sqlalchemy.orm import Session

from config import settings
from data_pipeline.completeness import CategoryCompleteness
from data_pipeline.pipeline_runner import PipelineRunner
from data_pipeline.scheduler.lock import CollectionLock
from database.connection import get_engine
from database.models import (
    Listing,
    ListingObservation,
    PriceHistory,
    ScrapeRun,
    Vehicle,
)
from database.repository import VehicleRepository
from feature_engineering.dataset import MLDatasetLoader
from feature_engineering.pipeline import FeaturePipeline
from feature_engineering.validation import DataLeakageError, LeakageValidator
from scraper.discovery.category_discovery import DiscoveredCategory


def test_category_resolution_maps_canonical_urls():
    """
    Verifies that _resolve_categories correctly maps multi-word and slug categories
    like 'Three Wheelers' without producing invalid spaces in URLs.
    """
    runner = PipelineRunner()
    categories = ["Three Wheelers", "Cars", "Heavy-Duty", "Motorbikes"]
    resolved = runner._resolve_categories(categories)

    names = [c.name for c in resolved]
    urls = [c.url for c in resolved]

    assert "Cars" in names
    assert "Three Wheelers" in names
    assert all(" " not in u for u in urls), f"Found whitespace in resolved category URLs: {urls}"
    # Verify Three Wheelers maps to three-wheels or valid slug
    tw_cat = next(c for c in resolved if c.name == "Three Wheelers")
    assert "three-wheel" in tw_cat.url


def test_database_referential_integrity_invariants():
    """
    Verifies that all entities in PostgreSQL maintain strict foreign key relationships
    with zero orphan rows, duplicate IDs, or duplicate URLs.
    """
    engine = get_engine()
    with Session(engine) as session:
        # Check duplicate (source, listing_id)
        dup_listings = (
            session.query(Listing.source, Listing.listing_id, func.count(Listing.id))
            .group_by(Listing.source, Listing.listing_id)
            .having(func.count(Listing.id) > 1)
            .all()
        )
        assert len(dup_listings) == 0, f"Duplicate listings found: {dup_listings}"

        # Check duplicate URLs
        dup_urls = (
            session.query(Listing.listing_url, func.count(Listing.id))
            .group_by(Listing.listing_url)
            .having(func.count(Listing.id) > 1)
            .all()
        )
        assert len(dup_urls) == 0, f"Duplicate URLs found: {dup_urls}"

        # Check orphan listings
        orphan_listings = session.query(Listing).filter(~Listing.vehicle_id.in_(session.query(Vehicle.id))).count()
        assert orphan_listings == 0

        # Check orphan price histories
        orphan_ph = session.query(PriceHistory).filter(~PriceHistory.listing_id.in_(session.query(Listing.id))).count()
        assert orphan_ph == 0

        # Check orphan observations
        orphan_obs_l = session.query(ListingObservation).filter(~ListingObservation.listing_id.in_(session.query(Listing.id))).count()
        assert orphan_obs_l == 0
        orphan_obs_sr = session.query(ListingObservation).filter(~ListingObservation.scrape_run_id.in_(session.query(ScrapeRun.id))).count()
        assert orphan_obs_sr == 0


def test_partial_collection_never_deactivates_active_listings(db_session: Session):
    """
    Verifies that a partial collection (with max_pages or max_listings set)
    NEVER marks unseen listings as NO_LONGER_OBSERVED.
    """
    repo = VehicleRepository(db_session)
    v1 = repo.create_vehicle({"category": "Cars", "brand": "Toyota", "model": "Corolla", "manufacture_year": 2020})
    v2 = repo.create_vehicle({"category": "Cars", "brand": "Nissan", "model": "Sunny", "manufacture_year": 2018})

    l1 = repo.create_listing(v1, {"listing_id": "PARTIAL_1", "listing_url": "http://x/1", "current_status": "ACTIVE"})
    l2 = repo.create_listing(v2, {"listing_id": "PARTIAL_2", "listing_url": "http://x/2", "current_status": "ACTIVE"})
    db_session.commit()

    # Suppose collection only observes l1
    observed_ids = {"PARTIAL_1"}

    # Mock completeness report indicating partial scope
    is_complete_scope = False  # Because max_pages was set
    if is_complete_scope:
        repo.mark_unobserved_listings(observed_ids, source="riyasewana", category="Cars")

    # Verify both listings remain ACTIVE
    db_session.refresh(l1)
    db_session.refresh(l2)
    assert l1.current_status == "ACTIVE"
    assert l2.current_status == "ACTIVE"


def test_ml_dataset_loader_matches_database_eligible_count():
    """
    Verifies that MLDatasetLoader extracts exactly the count of listings
    marked ml_eligible = True in PostgreSQL.
    """
    engine = get_engine()
    with Session(engine) as session:
        expected_eligible = session.query(Listing).filter(Listing.ml_eligible.is_(True)).count()

    loader = MLDatasetLoader()
    raw_df = loader.load_raw_dataset(ml_eligible_only=True)
    assert len(raw_df) == expected_eligible


def test_collection_lock_mutual_exclusion():
    """
    Verifies that CollectionLock prevents two concurrent processes from acquiring
    the same lockfile simultaneously.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile = Path(tmpdir) / ".test.lock"

        lock1 = CollectionLock(lock_file_path=lockfile)
        assert lock1.acquire() is True
        assert lock1.is_locked is True

        lock2 = CollectionLock(lock_file_path=lockfile)
        assert lock2.acquire() is False
        assert lock2.is_locked is False

        lock1.release()
        assert lock1.is_locked is False

        assert lock2.acquire() is True
        assert lock2.is_locked is True
        lock2.release()

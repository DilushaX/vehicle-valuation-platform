"""
scripts/apply_phase4_indexes.py

Safely applies Phase 4 Step 1 indexes to the PostgreSQL production database
and non-destructively aligns Vehicle.category based on listing title descriptors.
"""
import logging
import sys
from pathlib import Path

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, text
from database.connection import get_engine, get_sessionmaker
from database.models import Vehicle, Listing

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("apply_phase4_indexes")

NEW_INDEXES = [
    # Vehicle indexes
    "CREATE INDEX IF NOT EXISTS idx_vehicle_category ON vehicles (category);",
    "CREATE INDEX IF NOT EXISTS idx_vehicle_brand_model ON vehicles (brand, model);",
    # Listing indexes
    "CREATE INDEX IF NOT EXISTS idx_listing_district ON listings (district);",
    "CREATE INDEX IF NOT EXISTS idx_listing_first_seen ON listings (first_seen_at);",
    # PriceHistory composite index
    "CREATE INDEX IF NOT EXISTS idx_price_history_listing_date ON price_history (listing_id, observed_at);",
    # ListingObservation composite index
    "CREATE INDEX IF NOT EXISTS idx_observation_listing_date ON listing_observations (listing_id, observed_at);",
    # ScrapeRun indexes
    "CREATE INDEX IF NOT EXISTS idx_scrape_run_category ON scrape_runs (category);",
    "CREATE INDEX IF NOT EXISTS idx_scrape_run_started ON scrape_runs (started_at);",
]

CATEGORY_PATTERNS = [
    ("heavy-duty", "Heavy-Duty"),
    ("three wheel", "Three Wheel"),
    ("motorbike", "Motorbike"),
    ("motorcycle", "Motorbike"),
    ("lorry", "Lorry"),
    ("pickup", "Pickup"),
    ("suv", "SUV"),
    ("van", "Van"),
    ("car", "Car"),
    ("bus", "Bus"),
]


def apply_indexes():
    engine = get_engine()
    logger.info("Applying non-locking B-tree indexes to PostgreSQL...")
    with engine.connect() as conn:
        for stmt in NEW_INDEXES:
            logger.info(f"Executing: {stmt.strip()}")
            conn.execute(text(stmt))
        conn.commit()
    logger.info("All Phase 4 indexes created/verified successfully.")


def align_vehicle_categories():
    SessionLocal = get_sessionmaker()
    session = SessionLocal()
    try:
        logger.info("Checking Vehicle.category alignment across existing listings...")
        listings = session.scalars(select(Listing).order_by(Listing.id)).all()
        updated_count = 0
        for l in listings:
            v = l.vehicle
            if not v:
                continue
            title_lower = (l.title or "").lower()
            detected_cat = None
            for key, cat_name in CATEGORY_PATTERNS:
                if key in title_lower:
                    detected_cat = cat_name
                    break

            if detected_cat and v.category != detected_cat:
                logger.info(f"Aligning vehicle #{v.id} ('{l.title[:40]}'): '{v.category}' -> '{detected_cat}'")
                v.category = detected_cat
                updated_count += 1

        if updated_count > 0:
            session.commit()
            logger.info(f"Successfully aligned category for {updated_count} vehicles.")
        else:
            logger.info("All vehicle categories are already aligned.")

    finally:
        session.close()


def main():
    apply_indexes()
    align_vehicle_categories()
    logger.info("Phase 4 Step 1 database alignment complete.")


if __name__ == "__main__":
    main()

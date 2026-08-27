"""
Database Repository Module
Encapsulates CRUD operations, delta tracking, price history tracking,
status transition management, and scrape run lifecycle.
"""
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_
from database.models import (
    Vehicle,
    Listing,
    PriceHistory,
    ListingStatusHistory,
    ScrapeRun,
    DataQualityRecord,
    ModelVersion,
    PredictionLog,
    utc_now
)

class VehicleRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_or_create_vehicle(
        self,
        category: str,
        make: str,
        model: str,
        yom: int,
        fuel_type: Optional[str] = None,
        transmission: Optional[str] = None,
        engine_cc: Optional[int] = None
    ) -> Vehicle:
        """Finds existing canonical vehicle entity or creates a new one."""
        query = self.db.query(Vehicle).filter(
            func.lower(Vehicle.category) == category.lower(),
            func.lower(Vehicle.make) == make.lower(),
            func.lower(Vehicle.model) == model.lower(),
            Vehicle.yom == yom
        )
        if fuel_type:
            query = query.filter(func.lower(Vehicle.fuel_type) == fuel_type.lower())
        if transmission:
            query = query.filter(func.lower(Vehicle.transmission) == transmission.lower())

        vehicle = query.first()
        if not vehicle:
            vehicle = Vehicle(
                category=category.capitalize(),
                make=make.title(),
                model=model.title(),
                yom=yom,
                fuel_type=fuel_type.capitalize() if fuel_type else None,
                transmission=transmission.capitalize() if transmission else None,
                engine_cc=engine_cc
            )
            self.db.add(vehicle)
            self.db.flush()
        return vehicle

    def upsert_listing(
        self,
        listing_data: Dict[str, Any],
        quality_status: str = "VALID",
        quality_score: float = 100.0,
        quality_issues: Optional[List[str]] = None,
        is_outlier: bool = False
    ) -> Tuple[Listing, str]:
        """
        Upserts a vehicle listing with delta tracking.
        Returns (Listing, action_taken) where action_taken is:
        'NEW', 'PRICE_CHANGED', 'UPDATED', or 'UNCHANGED'.
        """
        listing_id = str(listing_data["listing_id"])
        existing = self.db.query(Listing).filter(Listing.listing_id == listing_id).first()

        now = utc_now()
        vehicle = self.get_or_create_vehicle(
            category=listing_data.get("category", "Cars"),
            make=listing_data.get("make", "Unknown"),
            model=listing_data.get("model", "Unknown"),
            yom=int(listing_data.get("yom", 2015)),
            fuel_type=listing_data.get("fuel_type"),
            transmission=listing_data.get("transmission"),
            engine_cc=listing_data.get("engine_cc")
        )

        new_price = float(listing_data["price_rs"])
        action = "UNCHANGED"

        if not existing:
            # Create brand new listing
            action = "NEW"
            listing = Listing(
                listing_id=listing_id,
                source=listing_data.get("source", "riyasewana"),
                url=listing_data.get("url"),
                title=listing_data.get("title"),
                vehicle_id=vehicle.id,
                category=listing_data.get("category", "Cars").capitalize(),
                make=listing_data.get("make", "Unknown").title(),
                model=listing_data.get("model", "Unknown").title(),
                yom=int(listing_data.get("yom", 2015)),
                condition=listing_data.get("condition", "Used"),
                mileage_km=listing_data.get("mileage_km"),
                fuel_type=listing_data.get("fuel_type"),
                transmission=listing_data.get("transmission"),
                engine_cc=listing_data.get("engine_cc"),
                district=listing_data.get("district", "Colombo").title() if listing_data.get("district") else None,
                current_price=new_price,
                current_status="ACTIVE",
                data_quality_status=quality_status,
                quality_score=quality_score,
                first_seen_date=now,
                last_seen_date=now,
                posted_date=listing_data.get("posted_date", now)
            )
            self.db.add(listing)
            self.db.flush()

            # Record initial price history entry
            price_entry = PriceHistory(
                listing_id=listing.id,
                price=new_price,
                recorded_at=now,
                change_amount=0.0,
                change_percent=0.0
            )
            self.db.add(price_entry)

            # Record initial status
            status_entry = ListingStatusHistory(
                listing_id=listing.id,
                old_status=None,
                new_status="ACTIVE",
                changed_at=now,
                notes="Initial observation"
            )
            self.db.add(status_entry)

        else:
            listing = existing
            listing.last_seen_date = now
            listing.data_quality_status = quality_status
            listing.quality_score = quality_score

            # Reactivate if it was previously marked as NO_LONGER_OBSERVED
            if listing.current_status == "NO_LONGER_OBSERVED":
                status_entry = ListingStatusHistory(
                    listing_id=listing.id,
                    old_status=listing.current_status,
                    new_status="ACTIVE",
                    changed_at=now,
                    notes="Re-observed on platform"
                )
                self.db.add(status_entry)
                listing.current_status = "ACTIVE"
                action = "UPDATED"

            # Check for Price Change
            if abs(listing.current_price - new_price) > 1.0:
                action = "PRICE_CHANGED"
                old_price = listing.current_price
                diff_amount = new_price - old_price
                diff_pct = (diff_amount / old_price) * 100.0 if old_price > 0 else 0.0

                listing.current_price = new_price
                price_entry = PriceHistory(
                    listing_id=listing.id,
                    price=new_price,
                    recorded_at=now,
                    change_amount=diff_amount,
                    change_percent=diff_pct
                )
                self.db.add(price_entry)

            # Update structured fields if they were refreshed
            for field in ["condition", "mileage_km", "fuel_type", "transmission", "district", "engine_cc", "url", "title"]:
                val = listing_data.get(field)
                if val is not None:
                    setattr(listing, field, val)

        # Record Data Quality audit record
        import json
        dq_record = DataQualityRecord(
            listing_id=listing.id,
            status=quality_status,
            issues_json=json.dumps(quality_issues or []),
            is_outlier=is_outlier,
            score=quality_score,
            evaluated_at=now
        )
        self.db.add(dq_record)

        self.db.commit()
        return listing, action

    def mark_unseen_as_inactive(
        self,
        category: Optional[str],
        seen_listing_ids: List[str],
        run_start_time: datetime
    ) -> int:
        """
        Marks listings that were active before run_start_time but NOT observed in seen_listing_ids
        as NO_LONGER_OBSERVED.
        """
        query = self.db.query(Listing).filter(
            Listing.current_status == "ACTIVE",
            Listing.last_seen_date < run_start_time
        )
        if category:
            query = query.filter(func.lower(Listing.category) == category.lower())

        if seen_listing_ids:
            query = query.filter(~Listing.listing_id.in_(seen_listing_ids))

        inactive_listings = query.all()
        now = utc_now()
        count = 0
        for item in inactive_listings:
            item.current_status = "NO_LONGER_OBSERVED"
            status_entry = ListingStatusHistory(
                listing_id=item.id,
                old_status="ACTIVE",
                new_status="NO_LONGER_OBSERVED",
                changed_at=now,
                notes="Listing was no longer found on source website"
            )
            self.db.add(status_entry)
            count += 1

        self.db.commit()
        return count

    def get_listings(
        self,
        category: Optional[str] = None,
        make: Optional[str] = None,
        model: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        district: Optional[str] = None,
        status: Optional[str] = "ACTIVE",
        quality_status: Optional[str] = "VALID",
        limit: int = 500,
        offset: int = 0
    ) -> List[Listing]:
        """Query listings with flexible filtering."""
        query = self.db.query(Listing)
        if category:
            query = query.filter(func.lower(Listing.category) == category.lower())
        if make:
            query = query.filter(func.lower(Listing.make) == make.lower())
        if model:
            query = query.filter(func.lower(Listing.model) == model.lower())
        if min_year:
            query = query.filter(Listing.yom >= min_year)
        if max_year:
            query = query.filter(Listing.yom <= max_year)
        if min_price:
            query = query.filter(Listing.current_price >= min_price)
        if max_price:
            query = query.filter(Listing.current_price <= max_price)
        if district:
            query = query.filter(func.lower(Listing.district) == district.lower())
        if status:
            query = query.filter(Listing.current_status == status)
        if quality_status:
            query = query.filter(Listing.data_quality_status == quality_status)

        return query.order_by(desc(Listing.last_seen_date)).offset(offset).limit(limit).all()

    def get_all_clean_training_data(self, category: Optional[str] = None) -> List[Listing]:
        """Fetches all valid listings (including active and historical) for ML training."""
        query = self.db.query(Listing).filter(
            Listing.data_quality_status == "VALID",
            Listing.current_price > 0,
            Listing.yom > 0,
            Listing.mileage_km.isnot(None)
        )
        if category:
            query = query.filter(func.lower(Listing.category) == category.lower())
        return query.all()

    def start_scrape_run(self, run_id: str, category: Optional[str] = None) -> ScrapeRun:
        """Starts and logs a new scrape execution run."""
        run = ScrapeRun(
            run_id=run_id,
            category=category,
            started_at=utc_now(),
            status="RUNNING"
        )
        self.db.add(run)
        self.db.commit()
        return run

    def finish_scrape_run(
        self,
        run_id: str,
        total: int,
        new_c: int,
        updated_c: int,
        price_c: int,
        inactive_c: int,
        duplicate_c: int,
        invalid_c: int,
        status: str = "COMPLETED",
        error_msg: Optional[str] = None
    ) -> ScrapeRun:
        """Completes a scrape run record with aggregate telemetry."""
        run = self.db.query(ScrapeRun).filter(ScrapeRun.run_id == run_id).first()
        if run:
            run.ended_at = utc_now()
            run.total_scraped = total
            run.new_count = new_c
            run.updated_count = updated_c
            run.price_change_count = price_c
            run.no_longer_observed_count = inactive_c
            run.duplicate_count = duplicate_c
            run.invalid_count = invalid_c
            run.status = status
            run.error_message = error_msg
            self.db.commit()
        return run

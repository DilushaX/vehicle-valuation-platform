import json
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import (
    Vehicle,
    Listing,
    PriceHistory,
    ScrapeRun,
    ListingObservation,
)


class VehicleRepository:

    def __init__(self, db: Session):
        self.db = db

    def find_listing(
        self,
        listing_id: str,
        source: str = "riyasewana",
    ) -> Listing | None:
        statement = select(Listing).where(
            Listing.source == source,
            Listing.listing_id == listing_id,
        )
        return self.db.scalar(statement)

    def get_validation_issues(self, listing: Listing) -> list:
        if not listing.validation_issues:
            return []
        try:
            return json.loads(listing.validation_issues)
        except (json.JSONDecodeError, TypeError):
            return []

    def create_vehicle(self, data: dict) -> Vehicle:
        vehicle = Vehicle(
            category=data.get("category"),
            brand=data.get("brand"),
            model=data.get("model"),
            manufacture_year=data.get("manufacture_year"),
            registration_year=data.get("registration_year"),
            fuel_type=data.get("fuel_type"),
            transmission=data.get("transmission"),
            engine_cc=data.get("engine_cc"),
            condition=data.get("condition"),
        )
        self.db.add(vehicle)
        self.db.flush()
        return vehicle

    def create_listing(
        self,
        vehicle: Vehicle,
        data: dict,
        observed_at: datetime | None = None,
    ) -> Listing:
        validation_issues_val = data.get("validation_issues", [])
        if isinstance(validation_issues_val, (list, dict)):
            validation_issues_json = json.dumps(validation_issues_val)
        elif validation_issues_val is None:
            validation_issues_json = json.dumps([])
        else:
            try:
                json.loads(validation_issues_val)
                validation_issues_json = str(validation_issues_val)
            except (ValueError, TypeError):
                validation_issues_json = json.dumps([str(validation_issues_val)])

        ts = observed_at or datetime.now(timezone.utc)
        listing = Listing(
            listing_id=str(data["listing_id"]).strip(),
            vehicle_id=vehicle.id,
            listing_url=data["listing_url"],
            title=data.get("title"),
            description=data.get("description"),
            location=data.get("location"),
            district=data.get("district"),
            ad_date=data.get("ad_date"),
            source=data.get("source", "riyasewana"),
            current_status="ACTIVE",
            first_seen_at=ts,
            last_seen_at=ts,
            validation_issues=validation_issues_json,
            ml_eligible=data.get("is_valid", False),
        )
        self.db.add(listing)
        self.db.flush()
        return listing

    def add_price_history(
        self,
        listing: Listing,
        price: int | None,
        observed_at: datetime | None = None,
    ) -> PriceHistory:
        history = PriceHistory(
            listing=listing,
            price=price,
            observed_at=observed_at or datetime.now(timezone.utc),
        )
        self.db.add(history)
        self.db.flush()
        return history

    def get_latest_price(self, listing: Listing) -> int | None:
        if listing.price_history:
            sorted_ph = sorted(
                listing.price_history,
                key=lambda ph: (
                    ph.observed_at.replace(tzinfo=timezone.utc)
                    if ph.observed_at and ph.observed_at.tzinfo is None
                    else (ph.observed_at or datetime.min.replace(tzinfo=timezone.utc)),
                    ph.id or 0,
                ),
                reverse=True,
            )
            return sorted_ph[0].price

        statement = (
            select(PriceHistory)
            .where(PriceHistory.listing_id == listing.id)
            .order_by(PriceHistory.observed_at.desc(), PriceHistory.id.desc())
            .limit(1)
        )
        latest = self.db.scalar(statement)
        return latest.price if latest else None

    def add_observation(
        self,
        listing: Listing,
        scrape_run: ScrapeRun,
        price: int | None,
        mileage: int | None,
        observed_at: datetime | None = None,
        availability: str = "AVAILABLE",
    ) -> ListingObservation:
        observation = ListingObservation(
            listing=listing,
            scrape_run=scrape_run,
            observed_price=price,
            observed_mileage=mileage,
            observed_at=observed_at or datetime.now(timezone.utc),
            availability=availability,
        )
        self.db.add(observation)
        self.db.flush()
        return observation

    def sync_listing(
        self,
        data: dict,
        scrape_run: ScrapeRun | None = None,
        observed_at: datetime | None = None,
    ) -> tuple[Listing, bool]:
        """
        Idempotently synchronizes a scraped vehicle listing.
        - First observation: creates Vehicle, Listing, initial PriceHistory, and ListingObservation.
        - Subsequent observations: updates last_seen_at, sets current_status=ACTIVE, refreshes data,
          records ListingObservation, and records PriceHistory ONLY if asking price changed.
        Returns: (listing, is_new)
        """
        ts = observed_at or datetime.now(timezone.utc)
        source = data.get("source", "riyasewana")
        listing_id = str(data["listing_id"]).strip()
        price = data.get("price")
        mileage = data.get("mileage")

        listing = self.find_listing(listing_id=listing_id, source=source)

        if listing is None:
            # First observation
            vehicle = self.create_vehicle(data)
            listing = self.create_listing(vehicle, data, observed_at=ts)
            listing._was_reactivated = False
            listing._price_changed = False
            if price is not None:
                self.add_price_history(listing, price, observed_at=ts)
            if scrape_run is not None:
                self.add_observation(
                    listing, scrape_run, price, mileage, observed_at=ts
                )
            return listing, True
        else:
            # Subsequent observation
            was_no_longer_observed = (listing.current_status == "NO_LONGER_OBSERVED")
            listing._was_reactivated = was_no_longer_observed
            listing.last_seen_at = ts
            listing.current_status = "ACTIVE"

            if data.get("title"):
                listing.title = data.get("title")
            if data.get("description"):
                listing.description = data.get("description")
            if data.get("location"):
                listing.location = data.get("location")
            if data.get("district"):
                listing.district = data.get("district")
            if data.get("ad_date") and not listing.ad_date:
                listing.ad_date = data.get("ad_date")
            if "is_valid" in data:
                listing.ml_eligible = data.get("is_valid", False)
            if "validation_issues" in data:
                issues_val = data.get("validation_issues", [])
                if isinstance(issues_val, (list, dict)):
                    listing.validation_issues = json.dumps(issues_val)
                else:
                    listing.validation_issues = str(issues_val)

            # Update condition on vehicle if it was missing and now provided
            if listing.vehicle and not listing.vehicle.condition and data.get("condition"):
                listing.vehicle.condition = data.get("condition")

            # Update price history only if price changed
            price_changed = False
            if price is not None:
                latest_price = self.get_latest_price(listing)
                if latest_price is None or price != latest_price:
                    self.add_price_history(listing, price, observed_at=ts)
                    price_changed = True
            listing._price_changed = price_changed

            # Record daily observation
            if scrape_run is not None:
                self.add_observation(
                    listing, scrape_run, price, mileage, observed_at=ts
                )

            self.db.flush()
            return listing, False

    def mark_unobserved_listings(
        self,
        observed_listing_ids: set[str] | list[str],
        source: str = "riyasewana",
        category: str | None = None,
    ) -> list[Listing]:
        """
        Identifies previously active listings not observed in the current scrape run,
        and marks them as NO_LONGER_OBSERVED without deleting historical data.
        Never marks listings as SOLD.
        """
        observed_set = {str(lid).strip() for lid in observed_listing_ids}
        statement = select(Listing).where(
            Listing.source == source,
            Listing.current_status == "ACTIVE",
        )
        if category:
            cat_clean = category.strip()
            cat_singular = cat_clean.rstrip("s")
            statement = statement.join(Vehicle).where(
                (Vehicle.category == cat_clean)
                | (Vehicle.category == cat_singular)
                | (Vehicle.category.ilike(f"{cat_singular}%"))
            )

        active_listings = self.db.scalars(statement).all()
        marked = []
        for listing in active_listings:
            if listing.listing_id not in observed_set:
                listing.current_status = "NO_LONGER_OBSERVED"
                marked.append(listing)

        if marked:
            self.db.flush()
        return marked

    def mark_listing_no_longer_observed(self, listing: Listing) -> Listing:
        listing.current_status = "NO_LONGER_OBSERVED"
        self.db.flush()
        return listing

    def create_scrape_run(
        self,
        category: str | None = None,
        source: str = "riyasewana",
        pages_requested: int = 0,
    ) -> ScrapeRun:
        scrape_run = ScrapeRun(
            source=source,
            category=category,
            pages_requested=pages_requested,
            status="RUNNING",
            started_at=datetime.now(timezone.utc),
        )
        self.db.add(scrape_run)
        self.db.flush()
        return scrape_run

    def complete_scrape_run(
        self,
        scrape_run: ScrapeRun,
        pages_requested: int | None = None,
        pages_scraped: int | None = None,
        failed_pages: int = 0,
        listings_found: int = 0,
        new_listings: int = 0,
        updated_listings: int = 0,
        errors: str | None = None,
        failed_listings: int = 0,
        price_changes: int = 0,
        disappeared_listings: int = 0,
        reactivated_listings: int = 0,
        observations_created: int = 0,
    ) -> ScrapeRun:
        scrape_run.completed_at = datetime.now(timezone.utc)
        if pages_requested is not None:
            scrape_run.pages_requested = pages_requested
        if pages_scraped is not None:
            scrape_run.pages_scraped = pages_scraped
        scrape_run.failed_pages = failed_pages
        scrape_run.listings_found = listings_found
        scrape_run.new_listings = new_listings
        scrape_run.updated_listings = updated_listings
        if errors:
            scrape_run.errors = errors

        req = scrape_run.pages_requested
        scr = scrape_run.pages_scraped
        if scr == 0 and req > 0:
            scrape_run.status = "FAILED"
        elif failed_pages > 0 or failed_listings > 0 or (req > 0 and scr < req):
            scrape_run.status = "INCOMPLETE"
        else:
            scrape_run.status = "COMPLETED"

        self.db.flush()
        return scrape_run

    def fail_scrape_run(
        self,
        scrape_run: ScrapeRun,
        errors: str,
    ) -> ScrapeRun:
        scrape_run.completed_at = datetime.now(timezone.utc)
        scrape_run.status = "FAILED"
        scrape_run.errors = errors
        self.db.flush()
        return scrape_run

    def commit(self):
        self.db.commit()

    def rollback(self):
        self.db.rollback()

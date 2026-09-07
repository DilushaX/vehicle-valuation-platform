import json
from datetime import datetime

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

    def find_listing(self, listing_id: str):
        statement = select(Listing).where(
            Listing.listing_id == listing_id
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
        )

        self.db.add(vehicle)
        self.db.flush()

        return vehicle

    def create_listing(
        self,
        vehicle: Vehicle,
        data: dict,
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

        listing = Listing(
            listing_id=data["listing_id"],
            vehicle_id=vehicle.id,
            listing_url=data["listing_url"],
            title=data.get("title"),
            description=data.get("description"),
            location=data.get("location"),
            district=data.get("district"),
            source=data.get("source", "riyasewana"),
            current_status="ACTIVE",
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
    ):

        history = PriceHistory(
            listing_id=listing.id,
            price=price,
            observed_at=datetime.utcnow(),
        )

        self.db.add(history)

    def add_observation(
        self,
        listing: Listing,
        scrape_run: ScrapeRun,
        price: int | None,
        mileage: int | None,
    ):

        observation = ListingObservation(
            listing_id=listing.id,
            scrape_run_id=scrape_run.id,
            observed_price=price,
            observed_mileage=mileage,
            availability="AVAILABLE",
        )

        self.db.add(observation)

    def create_scrape_run(
        self,
        category: str | None = None,
    ) -> ScrapeRun:

        scrape_run = ScrapeRun(
            category=category,
            status="RUNNING",
        )

        self.db.add(scrape_run)
        self.db.flush()

        return scrape_run

    def commit(self):
        self.db.commit()

    def rollback(self):
        self.db.rollback()

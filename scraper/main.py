import json
import logging
from scraper.spiders.riyasewana_spider import RiyasewanaSpider
from scraper.validators.listing_validator import ListingValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TEST_URL = (
    "https://riyasewana.com/buy/"
    "toyota-corolla-axio-sale-homagama-12217083"
)


def main():
    print("=" * 60)
    print(" PHASE 1 — RIYASEWANA SINGLE LISTING PIPELINE ")
    print("=" * 60)
    print(f"Target URL: {TEST_URL}\n")

    with RiyasewanaSpider() as spider:
        validator = ListingValidator()

        print("[1/3] Fetching and Extracting Vehicle Data...")
        vehicle = spider.scrape_listing(TEST_URL)

        print("[2/3] Validating Data Quality...")
        validated_vehicle = validator.validate(vehicle)

        print("[3/3] Structured Vehicle Record Output:")
        print("=" * 60)

        # Pretty print structured fields (excluding raw_data for concise terminal display)
        display_dict = {
            k: v for k, v in validated_vehicle.items() if k != "raw_data"
        }
        print(json.dumps(display_dict, indent=2, ensure_ascii=False))

        print("\n" + "=" * 60)
        print(f"Validation Status : {'VALID' if validated_vehicle['is_valid'] else 'SUSPICIOUS/INVALID'}")
        print(f"Validation Issues : {validated_vehicle['validation_issues']}")
        print("=" * 60)


if __name__ == "__main__":
    main()

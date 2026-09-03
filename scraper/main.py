from scraper.spiders.riyasewana_spider import RiyasewanaSpider
from scraper.validators.listing_validator import ListingValidator


TEST_URL = (
    "https://riyasewana.com/buy/"
    "toyota-corolla-axio-sale-homagama-12217083"
)


def main():
    spider = RiyasewanaSpider()
    validator = ListingValidator()

    print("Fetching Riyasewana listing...")

    vehicle = spider.scrape_listing(TEST_URL)

    vehicle = validator.validate(vehicle)

    print("\n" + "=" * 50)
    print("SCRAPED VEHICLE")
    print("=" * 50)

    for key, value in vehicle.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

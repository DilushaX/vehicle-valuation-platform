"""
Mock Data Generator
Generates realistic Sri Lankan vehicle market data covering all 25 districts,
popular makes/models, realistic depreciation curves, price ranges, and intentional anomalies for quality engine testing.
"""
import random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

DISTRICTS = [
    "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "Nuwara Eliya",
    "Galle", "Matara", "Hambantota", "Jaffna", "Kilinochchi", "Mannar",
    "Vavuniya", "Mullaitivu", "Batticaloa", "Ampara", "Trincomalee",
    "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa", "Badulla",
    "Monaragala", "Ratnapura", "Kegalle"
]

DISTRICT_WEIGHTS = [
    0.35, 0.18, 0.08, 0.09, 0.02, 0.01,
    0.04, 0.03, 0.01, 0.02, 0.005, 0.005,
    0.005, 0.005, 0.01, 0.01, 0.01,
    0.05, 0.02, 0.02, 0.01, 0.01,
    0.005, 0.02, 0.02
]

# Base Price (Rs. in LKR) for year 2020 at 50,000 km
VEHICLE_CATALOG = {
    "Cars": [
        {"make": "Toyota", "model": "Aqua", "fuel": "Hybrid", "trans": "Automatic", "engine": 1500, "base_price": 8_200_000, "depreciation_per_year": 250_000},
        {"make": "Toyota", "model": "Prius", "fuel": "Hybrid", "trans": "Automatic", "engine": 1800, "base_price": 9_500_000, "depreciation_per_year": 300_000},
        {"make": "Toyota", "model": "Axio", "fuel": "Hybrid", "trans": "Automatic", "engine": 1500, "base_price": 9_800_000, "depreciation_per_year": 280_000},
        {"make": "Toyota", "model": "Premio", "fuel": "Petrol", "trans": "Automatic", "engine": 1500, "base_price": 13_500_000, "depreciation_per_year": 400_000},
        {"make": "Toyota", "model": "Vitz", "fuel": "Petrol", "trans": "Automatic", "engine": 1000, "base_price": 6_800_000, "depreciation_per_year": 220_000},
        {"make": "Toyota", "model": "Corolla 121", "fuel": "Petrol", "trans": "Automatic", "engine": 1300, "base_price": 4_600_000, "depreciation_per_year": 120_000},
        {"make": "Honda", "model": "Vezel", "fuel": "Hybrid", "trans": "Automatic", "engine": 1500, "base_price": 10_200_000, "depreciation_per_year": 320_000},
        {"make": "Honda", "model": "Fit GP5", "fuel": "Hybrid", "trans": "Automatic", "engine": 1500, "base_price": 7_600_000, "depreciation_per_year": 240_000},
        {"make": "Honda", "model": "Grace", "fuel": "Hybrid", "trans": "Automatic", "engine": 1500, "base_price": 9_200_000, "depreciation_per_year": 280_000},
        {"make": "Suzuki", "model": "Wagon R FX", "fuel": "Hybrid", "trans": "Automatic", "engine": 650, "base_price": 5_600_000, "depreciation_per_year": 180_000},
        {"make": "Suzuki", "model": "Wagon R FZ", "fuel": "Hybrid", "trans": "Automatic", "engine": 650, "base_price": 6_100_000, "depreciation_per_year": 190_000},
        {"make": "Suzuki", "model": "Alto", "fuel": "Petrol", "trans": "Manual", "engine": 800, "base_price": 3_200_000, "depreciation_per_year": 110_000},
        {"make": "Suzuki", "model": "Spacia", "fuel": "Hybrid", "trans": "Automatic", "engine": 650, "base_price": 6_400_000, "depreciation_per_year": 190_000},
        {"make": "Nissan", "model": "Leaf", "fuel": "Electric", "trans": "Automatic", "engine": 0, "base_price": 4_800_000, "depreciation_per_year": 350_000},
        {"make": "Nissan", "model": "Dayz", "fuel": "Petrol", "trans": "Automatic", "engine": 650, "base_price": 5_400_000, "depreciation_per_year": 170_000},
        {"make": "Hyundai", "model": "Grand i10", "fuel": "Petrol", "trans": "Automatic", "engine": 1000, "base_price": 4_900_000, "depreciation_per_year": 160_000},
        {"make": "Kia", "model": "Picanto", "fuel": "Petrol", "trans": "Automatic", "engine": 1000, "base_price": 5_100_000, "depreciation_per_year": 160_000}
    ],
    "SUVs": [
        {"make": "Toyota", "model": "Land Cruiser Prado", "fuel": "Diesel", "trans": "Automatic", "engine": 2800, "base_price": 48_000_000, "depreciation_per_year": 1_200_000},
        {"make": "Toyota", "model": "RAV4", "fuel": "Hybrid", "trans": "Automatic", "engine": 2500, "base_price": 22_000_000, "depreciation_per_year": 700_000},
        {"make": "Toyota", "model": "Rush", "fuel": "Petrol", "trans": "Automatic", "engine": 1500, "base_price": 12_500_000, "depreciation_per_year": 380_000},
        {"make": "Mitsubishi", "model": "Montero", "fuel": "Diesel", "trans": "Automatic", "engine": 3200, "base_price": 28_000_000, "depreciation_per_year": 800_000},
        {"make": "Nissan", "model": "X-Trail", "fuel": "Hybrid", "trans": "Automatic", "engine": 2000, "base_price": 14_000_000, "depreciation_per_year": 450_000},
        {"make": "Honda", "model": "CR-V", "fuel": "Petrol", "trans": "Automatic", "engine": 1500, "base_price": 18_500_000, "depreciation_per_year": 550_000}
    ],
    "Vans": [
        {"make": "Toyota", "model": "KDH 201", "fuel": "Diesel", "trans": "Automatic", "engine": 3000, "base_price": 16_500_000, "depreciation_per_year": 400_000},
        {"make": "Toyota", "model": "Hiace Super GL", "fuel": "Diesel", "trans": "Automatic", "engine": 3000, "base_price": 19_000_000, "depreciation_per_year": 450_000},
        {"make": "Nissan", "model": "Caravan NV350", "fuel": "Diesel", "trans": "Automatic", "engine": 2500, "base_price": 13_500_000, "depreciation_per_year": 350_000},
        {"make": "Suzuki", "model": "Every DA64V", "fuel": "Petrol", "trans": "Automatic", "engine": 650, "base_price": 4_200_000, "depreciation_per_year": 130_000}
    ],
    "Three-Wheel": [
        {"make": "Bajaj", "model": "RE 4-Stroke", "fuel": "Petrol", "trans": "Manual", "engine": 200, "base_price": 1_650_000, "depreciation_per_year": 60_000},
        {"make": "Bajaj", "model": "RE Compact", "fuel": "Petrol", "trans": "Manual", "engine": 200, "base_price": 1_850_000, "depreciation_per_year": 65_000},
        {"make": "TVS", "model": "King", "fuel": "Petrol", "trans": "Manual", "engine": 200, "base_price": 1_550_000, "depreciation_per_year": 55_000},
        {"make": "Piaggio", "model": "Ape City", "fuel": "Diesel", "trans": "Manual", "engine": 400, "base_price": 1_400_000, "depreciation_per_year": 50_000}
    ],
    "Motorbikes": [
        {"make": "Bajaj", "model": "Pulsar 150", "fuel": "Petrol", "trans": "Manual", "engine": 150, "base_price": 650_000, "depreciation_per_year": 25_000},
        {"make": "Yamaha", "model": "FZ-S V3", "fuel": "Petrol", "trans": "Manual", "engine": 150, "base_price": 880_000, "depreciation_per_year": 30_000},
        {"make": "Honda", "model": "Dio", "fuel": "Petrol", "trans": "Automatic", "engine": 110, "base_price": 580_000, "depreciation_per_year": 22_000},
        {"make": "TVS", "model": "Apache RTR 160", "fuel": "Petrol", "trans": "Manual", "engine": 160, "base_price": 720_000, "depreciation_per_year": 28_000}
    ],
    "Lorries": [
        {"make": "Tata", "model": "Dimo Batta (Ace)", "fuel": "Diesel", "trans": "Manual", "engine": 700, "base_price": 2_400_000, "depreciation_per_year": 90_000},
        {"make": "Isuzu", "model": "Elf NHR", "fuel": "Diesel", "trans": "Manual", "engine": 2800, "base_price": 7_200_000, "depreciation_per_year": 200_000},
        {"make": "Mitsubishi", "model": "Canter", "fuel": "Diesel", "trans": "Manual", "engine": 3900, "base_price": 8_500_000, "depreciation_per_year": 220_000}
    ],
    "Buses": [
        {"make": "Ashok Leyland", "model": "Viking 42 Seater", "fuel": "Diesel", "trans": "Manual", "engine": 5700, "base_price": 14_000_000, "depreciation_per_year": 400_000},
        {"make": "Toyota", "model": "Coaster", "fuel": "Diesel", "trans": "Manual", "engine": 4200, "base_price": 18_000_000, "depreciation_per_year": 500_000}
    ]
}

def generate_realistic_dataset(count: int = 1500, inject_anomalies: bool = True) -> List[Dict[str, Any]]:
    """
    Generates a rich, realistic dataset representing Sri Lankan vehicle listings.
    Optionally injects controlled anomalies to test data quality pipelines.
    """
    records = []
    start_id = 8400000

    categories = list(VEHICLE_CATALOG.keys())
    # Weighted category choices (Cars and SUVs are dominant in listings)
    cat_weights = [0.55, 0.15, 0.12, 0.08, 0.05, 0.03, 0.02]

    base_time = datetime.now(timezone.utc)

    for i in range(count):
        cat = random.choices(categories, weights=cat_weights, k=1)[0]
        vehicle_template = random.choice(VEHICLE_CATALOG[cat])

        # Realistic Year of Manufacture (2005 - 2024, peaked around 2015-2019)
        yom = int(random.triangular(2005, 2024, 2017))
        age = 2024 - yom

        # Realistic Mileage: ~10,000 to 18,000 km per year
        avg_km_per_year = random.uniform(10_000, 16_000)
        mileage = int(max(1_000, (age * avg_km_per_year) + random.gauss(0, 15_000)))

        # Condition
        if age <= 1 and mileage < 10_000:
            condition = random.choice(["Brand New", "Reconditioned", "Used"])
        elif age <= 4:
            condition = random.choice(["Reconditioned", "Used", "Used"])
        else:
            condition = "Used"

        # District
        district = random.choices(DISTRICTS, weights=DISTRICT_WEIGHTS, k=1)[0]
        # Colombo / Gampaha slightly higher price baseline (+3-5%)
        district_premium = 1.03 if district in ["Colombo", "Gampaha"] else 1.0

        # Calculate realistic Price based on baseline, year decay, mileage decay
        base = vehicle_template["base_price"]
        dep_year = vehicle_template["depreciation_per_year"]
        
        # Year adjustment (relative to 2020)
        year_diff = yom - 2020
        calculated_price = base + (year_diff * dep_year)
        
        # Mileage adjustment (Rs. 10 per km deviation from standard 50k)
        mileage_diff = mileage - 50_000
        calculated_price -= (mileage_diff * 8.0)

        # Apply market fluctuation / noise (±6%)
        noise = random.uniform(0.94, 1.06)
        price_rs = round(max(300_000, calculated_price * district_premium * noise), -4)

        listing_id = f"riyasewana_{start_id + i}"
        posted_days_ago = random.randint(0, 45)
        posted_date = base_time - timedelta(days=posted_days_ago)

        record = {
            "listing_id": listing_id,
            "source": "riyasewana",
            "url": f"https://riyasewana.com/buy/{vehicle_template['make'].lower()}-{vehicle_template['model'].lower().replace(' ', '-')}-sale-{district.lower()}-{start_id + i}",
            "title": f"{vehicle_template['make']} {vehicle_template['model']} {yom}",
            "category": cat,
            "make": vehicle_template["make"],
            "model": vehicle_template["model"],
            "yom": yom,
            "condition": condition,
            "mileage_km": mileage,
            "fuel_type": vehicle_template["fuel"],
            "transmission": vehicle_template["trans"],
            "engine_cc": vehicle_template["engine"],
            "district": district,
            "price_rs": price_rs,
            "posted_date": posted_date,
            "is_anomaly": False,
            "anomaly_type": None
        }

        # Inject controlled anomalies in ~4% of records for Data Quality Engine testing
        if inject_anomalies and random.random() < 0.04:
            anomaly_kind = random.choice([
                "suspicious_mileage_pattern",
                "suspicious_price_pattern",
                "missing_fuel",
                "negative_mileage",
                "extreme_price_outlier"
            ])
            record["is_anomaly"] = True
            record["anomaly_type"] = anomaly_kind

            if anomaly_kind == "suspicious_mileage_pattern":
                record["mileage_km"] = random.choice([123, 111111, 123456, 1])
            elif anomaly_kind == "suspicious_price_pattern":
                record["price_rs"] = random.choice([123.0, 111111.0, 50.0])
            elif anomaly_kind == "missing_fuel":
                record["fuel_type"] = None
            elif anomaly_kind == "negative_mileage":
                record["mileage_km"] = -500
            elif anomaly_kind == "extreme_price_outlier":
                record["price_rs"] = record["price_rs"] * 10.0  # 10x market price

        records.append(record)

    return records

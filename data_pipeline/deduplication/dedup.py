"""
Deduplication Engine Module
Identifies duplicate vehicle listings posted under different URLs or seller accounts.
"""
import hashlib
from typing import Dict, Any, List, Set

class DeduplicationEngine:
    @staticmethod
    def generate_fingerprint(record: Dict[str, Any]) -> str:
        """Generates a stable SHA256 fingerprint for a vehicle listing based on core physical & financial attributes."""
        make = str(record.get("make", "")).strip().lower()
        model = str(record.get("model", "")).strip().lower()
        yom = str(record.get("yom", ""))
        fuel = str(record.get("fuel_type", "")).strip().lower()
        trans = str(record.get("transmission", "")).strip().lower()
        mileage = str(record.get("mileage_km", ""))
        price = str(int(record.get("price_rs", 0)))
        district = str(record.get("district", "")).strip().lower()

        raw_str = f"{make}|{model}|{yom}|{fuel}|{trans}|{mileage}|{price}|{district}"
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

    @staticmethod
    def detect_duplicates(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flags records with duplicate fingerprints within a batch."""
        seen_fingerprints: Set[str] = set()
        deduped_records = []

        for r in records:
            fp = DeduplicationEngine.generate_fingerprint(r)
            if fp in seen_fingerprints:
                r["is_duplicate"] = True
            else:
                seen_fingerprints.add(fp)
                r["is_duplicate"] = False
            deduped_records.append(r)

        return deduped_records

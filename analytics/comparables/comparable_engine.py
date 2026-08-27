"""
Comparable Vehicle Engine Module
Finds and ranks matching historical and active vehicle listings
using a multi-attribute weighted similarity scoring algorithm.
"""
from typing import Dict, Any, List, Optional
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func
from database.models import Listing

class ComparableEngine:
    def __init__(self, db: Session):
        self.db = db

    def find_comparables(
        self,
        category: str,
        make: str,
        model: str,
        yom: int,
        mileage_km: int,
        fuel_type: Optional[str] = None,
        transmission: Optional[str] = None,
        district: Optional[str] = None,
        top_k: int = 6
    ) -> Dict[str, Any]:
        """
        Searches the database for comparable vehicles and computes similarity scores.
        Returns top-k matching listings along with aggregate market evidence stats.
        """
        # Query potential candidate listings
        candidates = self.db.query(Listing).filter(
            func.lower(Listing.category) == category.lower(),
            func.lower(Listing.make) == make.lower(),
            Listing.data_quality_status == "VALID",
            Listing.current_price > 0,
            Listing.yom.between(yom - 4, yom + 4)
        ).all()

        if not candidates:
            # Broader query if exact model/make is sparse
            candidates = self.db.query(Listing).filter(
                func.lower(Listing.category) == category.lower(),
                Listing.data_quality_status == "VALID",
                Listing.current_price > 0,
                Listing.yom.between(yom - 5, yom + 5)
            ).limit(100).all()

        scored_candidates = []

        for item in candidates:
            # 1. Model Match Score (35 pts)
            model_match = 1.0 if item.model.lower() == model.lower() else (
                0.5 if model.lower() in item.model.lower() or item.model.lower() in model.lower() else 0.1
            )
            score_model = model_match * 35.0

            # 2. Year Proximity Score (25 pts)
            year_diff = abs(item.yom - yom)
            year_factor = max(0.0, 1.0 - (year_diff * 0.20))
            score_year = year_factor * 25.0

            # 3. Mileage Proximity Score (15 pts)
            item_mileage = item.mileage_km or mileage_km
            km_diff = abs(item_mileage - mileage_km)
            mileage_factor = max(0.0, 1.0 - (km_diff / 75_000.0))
            score_mileage = mileage_factor * 15.0

            # 4. Fuel Type Match (10 pts)
            fuel_factor = 1.0 if (item.fuel_type or "").lower() == (fuel_type or "").lower() else 0.2
            score_fuel = fuel_factor * 10.0

            # 5. Transmission Match (8 pts)
            trans_factor = 1.0 if (item.transmission or "").lower() == (transmission or "").lower() else 0.3
            score_trans = trans_factor * 8.0

            # 6. District Proximity (7 pts)
            if district and item.district:
                dist_factor = 1.0 if item.district.lower() == district.lower() else 0.5
            else:
                dist_factor = 0.8
            score_dist = dist_factor * 7.0

            total_similarity = round(score_model + score_year + score_mileage + score_fuel + score_trans + score_dist, 1)
            total_similarity = min(99.9, max(10.0, total_similarity))

            scored_candidates.append({
                "listing_id": item.listing_id,
                "title": item.title or f"{item.make} {item.model} {item.yom}",
                "url": item.url,
                "make": item.make,
                "model": item.model,
                "yom": item.yom,
                "mileage_km": item.mileage_km,
                "fuel_type": item.fuel_type,
                "transmission": item.transmission,
                "district": item.district,
                "condition": item.condition,
                "price_rs": item.current_price,
                "similarity_score": total_similarity,
                "status": item.current_status
            })

        # Sort descending by similarity
        scored_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_matches = scored_candidates[:top_k]

        if not top_matches:
            return {
                "comparables": [],
                "count": 0,
                "median_price": 0.0,
                "avg_price": 0.0,
                "p25_price": 0.0,
                "p75_price": 0.0,
                "min_price": 0.0,
                "max_price": 0.0
            }

        prices = [m["price_rs"] for m in top_matches]
        return {
            "comparables": top_matches,
            "count": len(top_matches),
            "median_price": float(np.median(prices)),
            "avg_price": float(np.mean(prices)),
            "p25_price": float(np.percentile(prices, 25)),
            "p75_price": float(np.percentile(prices, 75)),
            "min_price": float(min(prices)),
            "max_price": float(max(prices))
        }

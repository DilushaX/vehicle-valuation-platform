"""
Comparable Vehicle Search Engine.

Retrieves and ranks similar market listings from the PostgreSQL database using
a weighted multi-attribute similarity scoring model. Prioritizes strong market
identity signals (Category, Brand, Model) followed by physical specifications
(Manufacture Year / Age, Mileage, Engine CC, Fuel, Transmission, District, Condition).

IMPORTANT SEMANTIC DEFINITION:
The similarity score represents multi-attribute specification distance between
the requested vehicle and candidate listings based on implemented feature weights.
It is NOT:
- prediction confidence
- valuation confidence
- probability of correctness
- price accuracy
- likelihood of being the same physical vehicle
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from data_pipeline.cleaning.cleaners import VehicleCleaner
from database.connection import get_sessionmaker
from feature_engineering.dataset import MLDatasetLoader

logger = logging.getLogger(__name__)


@dataclass
class ComparableVehicle:
    """
    Representation of a comparable vehicle retrieved from market listings.
    
    The similarity score represents specification alignment relative to the requested vehicle
    based on weighted feature distances. It does NOT represent prediction confidence, price
    accuracy, or identity probability.
    """
    listing_id: str
    category: str
    brand: str
    model: str
    manufacture_year: Optional[int]
    mileage: Optional[float]
    engine_cc: Optional[float]
    fuel_type: Optional[str]
    transmission: Optional[str]
    district: Optional[str]
    condition: Optional[str]
    asking_price: float
    similarity_score: float
    similarity_percentage: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "category": self.category,
            "brand": self.brand,
            "model": self.model,
            "manufacture_year": self.manufacture_year,
            "mileage": self.mileage,
            "engine_cc": self.engine_cc,
            "fuel_type": self.fuel_type,
            "transmission": self.transmission,
            "district": self.district,
            "condition": self.condition,
            "asking_price": round(float(self.asking_price), 2),
            "similarity_score": round(float(self.similarity_score), 4),
            "similarity_percentage": round(float(self.similarity_percentage), 1),
        }


class ComparableVehicleEngine:
    """
    Retrieves and ranks comparable vehicle listings for valuation reference.
    
    The similarity score measures feature-level specification alignment based on
    implemented attribute weights. It does NOT represent prediction confidence,
    valuation accuracy, or probability of correctness.
    
    Attribute Weights:
    - Brand (Make): 0.25
    - Model: 0.25
    - Manufacture Year / Age: 0.14
    - Mileage: 0.10
    - Engine CC: 0.08
    - Transmission: 0.06
    - Fuel Type: 0.04
    - District / Location: 0.04
    - Condition: 0.04
    Total: 1.00
    """

    # Weights summing to 1.00
    WEIGHT_BRAND = 0.25
    WEIGHT_MODEL = 0.25
    WEIGHT_YEAR = 0.14
    WEIGHT_MILEAGE = 0.10
    WEIGHT_CC = 0.08
    WEIGHT_TRANSMISSION = 0.06
    WEIGHT_FUEL = 0.04
    WEIGHT_DISTRICT = 0.04
    WEIGHT_CONDITION = 0.04

    def __init__(
        self,
        session: Optional[Session] = None,
        loader: Optional[MLDatasetLoader] = None,
    ):
        self.session = session
        self.loader = loader or MLDatasetLoader()

    def find_comparables(
        self,
        query: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        top_k: int = 5,
        match_category_strictly: bool = True,
        exclude_listing_id: Optional[str] = None,
        candidate_pool: Optional[pd.DataFrame] = None,
    ) -> List[ComparableVehicle]:
        """
        Finds the top-K most similar vehicles in the verified dataset.
        
        Args:
            query: Target vehicle features (dict or Series).
            top_k: Number of comparables to return (default: 5).
            match_category_strictly: If True, only considers vehicles within the same canonical category.
            exclude_listing_id: Optional listing ID to exclude (e.g. self-match).
            candidate_pool: Optional pre-loaded DataFrame of candidate listings.
            
        Returns:
            List of ComparableVehicle objects sorted by similarity score descending.
        """
        if isinstance(query, pd.DataFrame):
            if query.empty:
                return []
            q = query.iloc[0].to_dict()
        elif isinstance(query, pd.Series):
            q = query.to_dict()
        elif isinstance(query, dict):
            q = query
        else:
            raise ValueError(f"Unsupported query type: {type(query)}")

        # Clean query values
        q_cat_raw = q.get("category")
        q_cat = VehicleCleaner.canonicalize_category(str(q_cat_raw).strip()) if q_cat_raw else None
        q_brand = str(q.get("brand", "")).strip().lower()
        q_model = str(q.get("model", "")).strip().lower()
        q_year = self._extract_year(q)
        q_mileage = self._to_float(q.get("mileage"))
        q_cc = self._to_float(q.get("engine_cc"))
        q_fuel = str(q.get("fuel_type", "")).strip().lower()
        q_trans = str(q.get("transmission", "")).strip().lower()
        q_district = str(q.get("district", "")).strip().lower()
        q_cond = str(q.get("condition", "")).strip().lower()

        # Load candidate pool
        if candidate_pool is not None:
            candidates_df = candidate_pool.copy()
        else:
            candidates_df = self.loader.load_raw_dataset(
                session=self.session,
                ml_eligible_only=True,
                category=q_cat if match_category_strictly else None,
            )

        if candidates_df.empty:
            logger.warning("No candidate listings found for comparable search.")
            return []

        # Filters:
        # 1. Valid asking price (> 0 and not null)
        candidates_df = candidates_df[
            candidates_df["asking_price"].notna() & (candidates_df["asking_price"] > 0)
        ]

        # 2. ML eligible only
        if "ml_eligible" in candidates_df.columns:
            candidates_df = candidates_df[candidates_df["ml_eligible"] == True]

        # 3. Exclude query listing itself if given
        if exclude_listing_id:
            candidates_df = candidates_df[candidates_df["listing_id"] != str(exclude_listing_id)]

        # 4. Canonical category strict matching
        if match_category_strictly and q_cat:
            candidates_df = candidates_df[
                candidates_df["category"].apply(
                    lambda c: VehicleCleaner.canonicalize_category(str(c)) == q_cat if pd.notna(c) else False
                )
            ]

        if candidates_df.empty:
            return []

        # Deduplicate candidates by listing_id
        candidates_df = candidates_df.drop_duplicates(subset=["listing_id"]).copy()

        # Compute similarity score for each candidate
        scores: List[float] = []
        for _, row in candidates_df.iterrows():
            score = self._compute_similarity(
                q_brand=q_brand,
                q_model=q_model,
                q_year=q_year,
                q_mileage=q_mileage,
                q_cc=q_cc,
                q_fuel=q_fuel,
                q_trans=q_trans,
                q_district=q_district,
                q_cond=q_cond,
                row=row,
            )
            scores.append(score)

        candidates_df["similarity_score"] = scores
        candidates_df["similarity_percentage"] = [s * 100.0 for s in scores]

        # Sort by similarity descending, then by listing recency
        top_candidates = candidates_df.sort_values(
            by=["similarity_score", "asking_price"], ascending=[False, True]
        ).head(top_k)

        results: List[ComparableVehicle] = []
        for _, row in top_candidates.iterrows():
            c_year = int(row["manufacture_year"]) if pd.notna(row.get("manufacture_year")) else None
            c_mileage = float(row["mileage"]) if pd.notna(row.get("mileage")) else None
            c_cc = float(row["engine_cc"]) if pd.notna(row.get("engine_cc")) else None

            results.append(
                ComparableVehicle(
                    listing_id=str(row["listing_id"]),
                    category=str(row.get("category", "")),
                    brand=str(row.get("brand", "")),
                    model=str(row.get("model", "")),
                    manufacture_year=c_year,
                    mileage=c_mileage,
                    engine_cc=c_cc,
                    fuel_type=str(row.get("fuel_type", "")),
                    transmission=str(row.get("transmission", "")),
                    district=str(row.get("district", "")),
                    condition=str(row.get("condition", "")),
                    asking_price=float(row["asking_price"]),
                    similarity_score=float(row["similarity_score"]),
                    similarity_percentage=float(row["similarity_percentage"]),
                )
            )

        return results

    def _compute_similarity(
        self,
        q_brand: str,
        q_model: str,
        q_year: Optional[int],
        q_mileage: Optional[float],
        q_cc: Optional[float],
        q_fuel: str,
        q_trans: str,
        q_district: str,
        q_cond: str,
        row: pd.Series,
    ) -> float:
        """Calculates normalized multi-attribute similarity between 0.0 and 1.0."""
        score = 0.0

        # 1. Brand match (0.25)
        c_brand = str(row.get("brand", "")).strip().lower()
        if q_brand and c_brand and q_brand == c_brand:
            score += self.WEIGHT_BRAND
        elif q_brand in c_brand or c_brand in q_brand:
            score += self.WEIGHT_BRAND * 0.5

        # 2. Model match (0.25)
        c_model = str(row.get("model", "")).strip().lower()
        if q_model and c_model:
            if q_model == c_model:
                score += self.WEIGHT_MODEL
            elif q_model in c_model or c_model in q_model:
                score += self.WEIGHT_MODEL * 0.7
            else:
                # Token overlap check (e.g. "Fit GP5" vs "Fit")
                q_tokens = set(q_model.split())
                c_tokens = set(c_model.split())
                if q_tokens & c_tokens:
                    score += self.WEIGHT_MODEL * 0.5

        # 3. Manufacture Year decay (0.14)
        c_year = self._extract_year(row)
        if q_year is not None and c_year is not None:
            diff_years = abs(q_year - c_year)
            # Full match = 1.0, 5 years diff = 0.5, >= 10 years diff = 0.0
            year_sim = max(0.0, 1.0 - (diff_years / 10.0))
            score += self.WEIGHT_YEAR * year_sim
        else:
            score += self.WEIGHT_YEAR * 0.5  # Neutral assumption if missing

        # 4. Mileage distance (0.10)
        c_mileage = self._to_float(row.get("mileage"))
        if q_mileage is not None and c_mileage is not None:
            diff_km = abs(q_mileage - c_mileage)
            # Full match = 1.0, 50,000 km diff = 0.5, >= 100,000 km diff = 0.0
            mileage_sim = max(0.0, 1.0 - (diff_km / 100_000.0))
            score += self.WEIGHT_MILEAGE * mileage_sim
        else:
            score += self.WEIGHT_MILEAGE * 0.5

        # 5. Engine CC (0.08)
        c_cc = self._to_float(row.get("engine_cc"))
        if q_cc is not None and c_cc is not None:
            diff_cc = abs(q_cc - c_cc)
            # Full match = 1.0, 500 cc diff = 0.5, >= 1000 cc diff = 0.0
            cc_sim = max(0.0, 1.0 - (diff_cc / 1000.0))
            score += self.WEIGHT_CC * cc_sim
        else:
            score += self.WEIGHT_CC * 0.5

        # 6. Transmission (0.06)
        c_trans = str(row.get("transmission", "")).strip().lower()
        if q_trans and c_trans and q_trans == c_trans:
            score += self.WEIGHT_TRANSMISSION

        # 7. Fuel Type (0.04)
        c_fuel = str(row.get("fuel_type", "")).strip().lower()
        if q_fuel and c_fuel and q_fuel == c_fuel:
            score += self.WEIGHT_FUEL

        # 8. District (0.04)
        c_district = str(row.get("district", "")).strip().lower()
        if q_district and c_district and q_district == c_district:
            score += self.WEIGHT_DISTRICT

        # 9. Condition (0.04)
        c_cond = str(row.get("condition", "")).strip().lower()
        if q_cond and c_cond and q_cond == c_cond:
            score += self.WEIGHT_CONDITION

        return float(min(score, 1.0))

    def _extract_year(self, record: Union[Dict[str, Any], pd.Series]) -> Optional[int]:
        if "manufacture_year" in record and pd.notna(record.get("manufacture_year")):
            try:
                return int(record["manufacture_year"])
            except (ValueError, TypeError):
                pass
        if "vehicle_age" in record and pd.notna(record.get("vehicle_age")):
            try:
                return 2026 - int(record["vehicle_age"])
            except (ValueError, TypeError):
                pass
        return None

    def _to_float(self, val: Any) -> Optional[float]:
        if val is None or pd.isna(val):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

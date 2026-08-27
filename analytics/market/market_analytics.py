"""
Market Analytics Engine Module
Aggregates and computes market-level statistics:
Brand volume & price distribution, model rankings, age depreciation,
mileage curves, fuel & transmission comparisons, and geographic pricing.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func
from database.models import Listing

class MarketAnalyticsEngine:
    def __init__(self, db: Session):
        self.db = db

    def _load_dataframe(self, category: Optional[str] = None, only_valid: bool = True) -> pd.DataFrame:
        """Loads listings from DB into a Pandas DataFrame for high-performance aggregations."""
        query = self.db.query(
            Listing.id,
            Listing.category,
            Listing.make,
            Listing.model,
            Listing.yom,
            Listing.condition,
            Listing.mileage_km,
            Listing.fuel_type,
            Listing.transmission,
            Listing.engine_cc,
            Listing.district,
            Listing.current_price,
            Listing.current_status,
            Listing.data_quality_status,
            Listing.first_seen_date,
            Listing.last_seen_date
        )
        if category:
            query = query.filter(func.lower(Listing.category) == category.lower())
        if only_valid:
            query = query.filter(Listing.data_quality_status.in_(["VALID", "SUSPICIOUS"]))

        results = query.all()
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                "id": r[0],
                "category": r[1],
                "make": r[2],
                "model": r[3],
                "yom": r[4],
                "condition": r[5],
                "mileage_km": r[6],
                "fuel_type": r[7],
                "transmission": r[8],
                "engine_cc": r[9],
                "district": r[10],
                "current_price": r[11],
                "current_status": r[12],
                "data_quality_status": r[13],
                "first_seen_date": r[14],
                "last_seen_date": r[15],
                "vehicle_age": 2024 - (r[4] or 2015)
            }
            for r in results
        ])
        return df

    def get_market_overview(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Calculates global high-level summary KPIs."""
        df = self._load_dataframe(category=category, only_valid=False)
        if df.empty:
            return {
                "total_listings": 0, "active_listings": 0, "median_price": 0,
                "avg_price": 0, "valid_listings": 0, "top_brands": [], "top_models": []
            }

        valid_df = df[df["data_quality_status"] == "VALID"]
        prices = valid_df["current_price"].dropna()

        active_count = int((df["current_status"] == "ACTIVE").sum())
        valid_count = len(valid_df)

        brand_counts = valid_df["make"].value_counts().head(5).to_dict()
        model_counts = (valid_df["make"] + " " + valid_df["model"]).value_counts().head(5).to_dict()

        return {
            "total_listings": len(df),
            "active_listings": active_count,
            "valid_listings": valid_count,
            "median_price": float(prices.median()) if not prices.empty else 0.0,
            "avg_price": float(prices.mean()) if not prices.empty else 0.0,
            "min_price": float(prices.min()) if not prices.empty else 0.0,
            "max_price": float(prices.max()) if not prices.empty else 0.0,
            "top_brands": [{"make": k, "count": int(v)} for k, v in brand_counts.items()],
            "top_models": [{"model": k, "count": int(v)} for k, v in model_counts.items()]
        }

    def get_brand_analysis(self, category: Optional[str] = None, top_n: int = 10) -> List[Dict[str, Any]]:
        """Analyzes top automotive brands by listing volume and price statistics."""
        df = self._load_dataframe(category=category, only_valid=True)
        if df.empty:
            return []

        grouped = df.groupby("make").agg(
            listing_count=("id", "count"),
            median_price=("current_price", "median"),
            avg_price=("current_price", "mean"),
            min_price=("current_price", "min"),
            max_price=("current_price", "max"),
            avg_mileage=("mileage_km", "mean")
        ).reset_index()

        total_listings = len(df)
        grouped["market_share_pct"] = (grouped["listing_count"] / total_listings) * 100.0
        grouped = grouped.sort_values(by="listing_count", ascending=False).head(top_n)

        return grouped.to_dict(orient="records")

    def get_model_analysis(self, make: Optional[str] = None, category: Optional[str] = None, top_n: int = 15) -> List[Dict[str, Any]]:
        """Analyzes popular models, average pricing, and mileage distribution."""
        df = self._load_dataframe(category=category, only_valid=True)
        if df.empty:
            return []

        if make:
            df = df[df["make"].str.lower() == make.lower()]

        if df.empty:
            return []

        grouped = df.groupby(["make", "model"]).agg(
            listing_count=("id", "count"),
            median_price=("current_price", "median"),
            avg_price=("current_price", "mean"),
            min_price=("current_price", "min"),
            max_price=("current_price", "max"),
            avg_year=("yom", "mean"),
            avg_mileage=("mileage_km", "mean")
        ).reset_index()

        grouped = grouped.sort_values(by="listing_count", ascending=False).head(top_n)
        return grouped.to_dict(orient="records")

    def get_depreciation_analysis(self, make: Optional[str] = None, model: Optional[str] = None, category: Optional[str] = "Cars") -> List[Dict[str, Any]]:
        """Analyzes vehicle age vs median asking price to compute depreciation curves."""
        df = self._load_dataframe(category=category, only_valid=True)
        if df.empty:
            return []

        if make:
            df = df[df["make"].str.lower() == make.lower()]
        if model:
            df = df[df["model"].str.lower() == model.lower()]

        if df.empty:
            return []

        grouped = df.groupby("yom").agg(
            listing_count=("id", "count"),
            median_price=("current_price", "median"),
            avg_price=("current_price", "mean"),
            p25_price=("current_price", lambda x: np.percentile(x, 25)),
            p75_price=("current_price", lambda x: np.percentile(x, 75))
        ).reset_index()

        grouped = grouped.sort_values(by="yom", ascending=True)
        return grouped.to_dict(orient="records")

    def get_fuel_and_transmission_breakdown(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Calculates price distributions by Fuel Type and Transmission."""
        df = self._load_dataframe(category=category, only_valid=True)
        if df.empty:
            return {"fuel": [], "transmission": []}

        fuel_grp = df.groupby("fuel_type").agg(
            count=("id", "count"),
            median_price=("current_price", "median"),
            avg_price=("current_price", "mean")
        ).reset_index().to_dict(orient="records")

        trans_grp = df.groupby("transmission").agg(
            count=("id", "count"),
            median_price=("current_price", "median"),
            avg_price=("current_price", "mean")
        ).reset_index().to_dict(orient="records")

        return {"fuel": fuel_grp, "transmission": trans_grp}

    def get_district_pricing_heatmap(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Calculates geographic listing distribution and average price per district."""
        df = self._load_dataframe(category=category, only_valid=True)
        if df.empty:
            return []

        grouped = df.groupby("district").agg(
            listing_count=("id", "count"),
            median_price=("current_price", "median"),
            avg_price=("current_price", "mean")
        ).reset_index().sort_values(by="listing_count", ascending=False)

        return grouped.to_dict(orient="records")

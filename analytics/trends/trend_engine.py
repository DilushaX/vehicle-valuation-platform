"""
Market Trend Engine Module
Analyzes historical time-series trends:
Price reductions, price increases, listing volume dynamics,
and brand/model price movement over time.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from database.models import PriceHistory, ListingStatusHistory, Listing

class MarketTrendEngine:
    def __init__(self, db: Session):
        self.db = db

    def get_price_movement_summary(self, days: int = 30) -> Dict[str, Any]:
        """Analyzes recent price reductions vs price hikes."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        history_records = self.db.query(
            PriceHistory.price,
            PriceHistory.change_amount,
            PriceHistory.change_percent,
            PriceHistory.recorded_at,
            Listing.make,
            Listing.model,
            Listing.category
        ).join(Listing, PriceHistory.listing_id == Listing.id).filter(
            PriceHistory.recorded_at >= cutoff,
            PriceHistory.change_amount.isnot(None),
            PriceHistory.change_amount != 0
        ).all()

        if not history_records:
            return {
                "period_days": days,
                "total_price_changes": 0,
                "price_drops_count": 0,
                "price_hikes_count": 0,
                "avg_price_drop_amount": 0.0,
                "avg_price_hike_amount": 0.0,
                "recent_changes": []
            }

        df = pd.DataFrame([
            {
                "price": r[0],
                "change_amount": r[1],
                "change_percent": r[2],
                "recorded_at": r[3],
                "make": r[4],
                "model": r[5],
                "category": r[6]
            }
            for r in history_records
        ])

        drops = df[df["change_amount"] < 0]
        hikes = df[df["change_amount"] > 0]

        recent = df.sort_values(by="recorded_at", ascending=False).head(10).to_dict(orient="records")

        return {
            "period_days": days,
            "total_price_changes": len(df),
            "price_drops_count": len(drops),
            "price_hikes_count": len(hikes),
            "avg_price_drop_amount": float(drops["change_amount"].abs().mean()) if not drops.empty else 0.0,
            "avg_price_hike_amount": float(hikes["change_amount"].mean()) if not hikes.empty else 0.0,
            "recent_changes": recent
        }

    def get_listing_volume_trends(self) -> List[Dict[str, Any]]:
        """Aggregates weekly listing counts across observed history."""
        listings = self.db.query(Listing.first_seen_date, Listing.category).all()
        if not listings:
            return []

        df = pd.DataFrame([
            {"date": r[0], "category": r[1]}
            for r in listings if r[0] is not None
        ])

        df["date"] = pd.to_datetime(df["date"])
        df["period"] = df["date"].dt.to_period("W").astype(str)

        grouped = df.groupby(["period", "category"]).size().reset_index(name="listing_count")
        return grouped.to_dict(orient="records")

    def get_model_price_trend(self, make: str, model: str) -> List[Dict[str, Any]]:
        """Tracks the historical average asking price for a specific Make & Model."""
        listings = self.db.query(
            Listing.yom,
            Listing.current_price,
            Listing.first_seen_date
        ).filter(
            func.lower(Listing.make) == make.lower(),
            func.lower(Listing.model) == model.lower(),
            Listing.data_quality_status == "VALID"
        ).all()

        if not listings:
            return []

        df = pd.DataFrame([
            {"yom": r[0], "price": r[1], "date": r[2]}
            for r in listings
        ])

        grouped = df.groupby("yom").agg(
            sample_count=("price", "count"),
            avg_price=("price", "mean"),
            median_price=("price", "median"),
            min_price=("price", "min"),
            max_price=("price", "max")
        ).reset_index().sort_values(by="yom")

        return grouped.to_dict(orient="records")

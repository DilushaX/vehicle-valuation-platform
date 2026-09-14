"""
Historical Price and Lifecycle Observation Analysis Engine.
Analyzes PriceHistory movements, daily scrape observations, and enforces historical depth
limits to prevent false trend inferences from short observation timeframes.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

INSUFFICIENT_DEPTH_MESSAGE = (
    "Insufficient historical depth for reliable monthly market trend inference."
)


class HistoricalAnalyzer:
    """
    Evaluates historical asking-price revisions, daily scrape observation coverage,
    and long-term market trend validity.
    """

    @staticmethod
    def analyze_price_history(
        df_price_history: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Analyzes PriceHistory records:
        - total price events
        - unique listings tracked
        - price changes vs unchanged prices
        - direction of price movement (reductions vs increases)
        """
        if df_price_history.empty or "price" not in df_price_history.columns:
            return {
                "total_price_records": 0,
                "unique_listings_tracked": 0,
                "listings_with_price_changes": 0,
                "price_reductions": 0,
                "price_increases": 0,
                "unchanged_listings": 0,
                "avg_price_change_lkr": 0.0,
            }

        total_records = len(df_price_history)
        unique_listings = df_price_history["listing_id"].nunique()

        reductions = 0
        increases = 0
        unchanged = 0
        price_diffs: List[float] = []

        grouped = df_price_history.groupby("listing_id")

        for lid, group in grouped:
            # Sort by observed_at if available
            if "observed_at" in group.columns:
                sorted_grp = group.sort_values(by="observed_at", ascending=True)
            else:
                sorted_grp = group

            prices = pd.to_numeric(sorted_grp["price"], errors="coerce").dropna().values
            if len(prices) > 1:
                initial_price = prices[0]
                latest_price = prices[-1]
                diff = latest_price - initial_price
                price_diffs.append(diff)

                if diff < 0:
                    reductions += 1
                elif diff > 0:
                    increases += 1
                else:
                    unchanged += 1
            else:
                unchanged += 1

        changed_count = reductions + increases
        avg_diff = float(np.mean(price_diffs)) if price_diffs else 0.0

        return {
            "total_price_records": total_records,
            "unique_listings_tracked": unique_listings,
            "listings_with_price_changes": changed_count,
            "price_reductions": reductions,
            "price_increases": increases,
            "unchanged_listings": unchanged,
            "avg_price_change_lkr": round(avg_diff, 2),
        }

    @staticmethod
    def analyze_observations(
        df_obs: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Analyzes scrape observation records:
        - total observations
        - listings observed
        - observation frequency and availability breakdown
        """
        if df_obs.empty:
            return {
                "total_observations": 0,
                "unique_listings_observed": 0,
                "avg_observations_per_listing": 0.0,
                "availability_breakdown": {},
            }

        total_obs = len(df_obs)
        unique_listings = df_obs["listing_id"].nunique() if "listing_id" in df_obs else 0
        avg_obs = round(total_obs / unique_listings, 2) if unique_listings > 0 else 0.0

        avail = (
            dict(df_obs["availability"].value_counts())
            if "availability" in df_obs
            else {}
        )

        return {
            "total_observations": total_obs,
            "unique_listings_observed": unique_listings,
            "avg_observations_per_listing": avg_obs,
            "availability_breakdown": avail,
        }

    @staticmethod
    def evaluate_historical_depth(
        df_obs: pd.DataFrame,
        df_price_history: pd.DataFrame,
        min_days_for_monthly_trends: int = 60,
    ) -> Dict[str, Any]:
        """
        Evaluates whether dataset possesses sufficient historical depth for monthly trend analysis.
        Strictly enforces reporting disclaimer when timeframe is insufficient.
        """
        obs_dates = []
        if not df_obs.empty and "observed_at" in df_obs.columns:
            obs_dates.extend(pd.to_datetime(df_obs["observed_at"], errors="coerce").dropna().tolist())

        if not df_price_history.empty and "observed_at" in df_price_history.columns:
            obs_dates.extend(pd.to_datetime(df_price_history["observed_at"], errors="coerce").dropna().tolist())

        if not obs_dates:
            return {
                "has_sufficient_depth": False,
                "message": INSUFFICIENT_DEPTH_MESSAGE,
                "total_days_span": 0,
                "min_date": None,
                "max_date": None,
            }

        min_date = min(obs_dates)
        max_date = max(obs_dates)
        delta_days = (max_date - min_date).total_seconds() / 86400.0

        has_sufficient_depth = delta_days >= min_days_for_monthly_trends

        return {
            "has_sufficient_depth": has_sufficient_depth,
            "message": (
                "Historical depth sufficient for reliable monthly market trend analysis."
                if has_sufficient_depth
                else INSUFFICIENT_DEPTH_MESSAGE
            ),
            "total_days_span": round(delta_days, 1),
            "min_date": min_date.isoformat() if hasattr(min_date, "isoformat") else str(min_date),
            "max_date": max_date.isoformat() if hasattr(max_date, "isoformat") else str(max_date),
            "threshold_required_days": min_days_for_monthly_trends,
        }

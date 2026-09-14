"""
Descriptive Statistics Engine for EDA.
Computes dataset overview metrics and 5-number + IQR summaries for numerical variables
across the vehicle market dataset.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database.connection import get_sessionmaker
from database.models import (
    Listing,
    ListingObservation,
    PriceHistory,
    Vehicle,
)

NUMERICAL_VARIABLES = [
    "asking_price",
    "mileage",
    "manufacture_year",
    "registration_year",
    "vehicle_age",
    "engine_cc",
]


class DescriptiveAnalyzer:
    """
    Computes dataset-level metrics and parametric/non-parametric summaries
    for numerical variables.
    """

    @staticmethod
    def compute_dataset_overview(
        df: pd.DataFrame,
        session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        Computes high-level dataset counts and attribute cardinalities.
        If a database session is provided, database-level totals (including raw tables)
        are queried; otherwise counts are derived from the DataFrame.
        """
        total_listings = len(df)
        if total_listings == 0:
            return {
                "total_listings": 0,
                "total_vehicles": 0,
                "ml_eligible_listings": 0,
                "ml_ineligible_listings": 0,
                "ml_eligibility_rate_pct": 0.0,
                "active_listings": 0,
                "no_longer_observed_listings": 0,
                "total_price_history_events": 0,
                "total_observations": 0,
                "distinct_categories": 0,
                "distinct_brands": 0,
                "distinct_models": 0,
                "distinct_fuel_types": 0,
                "distinct_transmissions": 0,
                "distinct_districts": 0,
            }

        # Status and eligibility counts from df
        ml_eligible_count = int(df["ml_eligible"].sum()) if "ml_eligible" in df else 0
        ml_ineligible_count = total_listings - ml_eligible_count
        ml_rate = round((ml_eligible_count / total_listings) * 100.0, 2)

        active_count = (
            int((df["current_status"] == "ACTIVE").sum())
            if "current_status" in df
            else 0
        )
        no_longer_obs_count = (
            int((df["current_status"] == "NO_LONGER_OBSERVED").sum())
            if "current_status" in df
            else 0
        )

        # Attribute cardinalities
        distinct_cats = int(df["canonical_category"].nunique()) if "canonical_category" in df else int(df["category"].nunique())
        distinct_brands = int(df["brand"].dropna().nunique()) if "brand" in df else 0
        distinct_models = int(df["model"].dropna().nunique()) if "model" in df else 0
        distinct_fuels = int(df["fuel_type"].dropna().nunique()) if "fuel_type" in df else 0
        distinct_trans = int(df["transmission"].dropna().nunique()) if "transmission" in df else 0
        distinct_districts = int(df["district"].dropna().nunique()) if "district" in df else 0

        # Optional DB-level counts
        total_vehicles = total_listings
        total_price_history = int(df["price_history_count"].sum()) if "price_history_count" in df else 0
        total_observations = int(df["observation_count"].sum()) if "observation_count" in df else 0

        if session is not None:
            try:
                db_v = session.scalar(select(func.count(Vehicle.id)))
                if db_v is not None:
                    total_vehicles = db_v
                db_ph = session.scalar(select(func.count(PriceHistory.id)))
                if db_ph is not None:
                    total_price_history = db_ph
                db_obs = session.scalar(select(func.count(ListingObservation.id)))
                if db_obs is not None:
                    total_observations = db_obs
            except Exception:
                pass

        return {
            "total_listings": total_listings,
            "total_vehicles": total_vehicles,
            "ml_eligible_listings": ml_eligible_count,
            "ml_ineligible_listings": ml_ineligible_count,
            "ml_eligibility_rate_pct": ml_rate,
            "active_listings": active_count,
            "no_longer_observed_listings": no_longer_obs_count,
            "total_price_history_events": total_price_history,
            "total_observations": total_observations,
            "distinct_categories": distinct_cats,
            "distinct_brands": distinct_brands,
            "distinct_models": distinct_models,
            "distinct_fuel_types": distinct_fuels,
            "distinct_transmissions": distinct_trans,
            "distinct_districts": distinct_districts,
        }

    @staticmethod
    def compute_numerical_summary(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Calculates parametric and non-parametric summary statistics for numerical variables:
        count, missing_count, mean, std, min, Q1 (25%), median (50%), Q3 (75%), max, IQR.
        """
        cols = columns or [c for c in NUMERICAL_VARIABLES if c in df.columns]
        summary_rows = []

        total_rows = len(df)

        for col in cols:
            if col not in df.columns:
                continue

            series = pd.to_numeric(df[col], errors="coerce").dropna()
            valid_count = len(series)
            missing_count = total_rows - valid_count

            if valid_count > 0:
                mean_val = float(series.mean())
                std_val = float(series.std(ddof=1)) if valid_count > 1 else 0.0
                min_val = float(series.min())
                q1_val = float(series.quantile(0.25))
                median_val = float(series.median())
                q3_val = float(series.quantile(0.75))
                max_val = float(series.max())
                iqr_val = q3_val - q1_val
            else:
                mean_val = np.nan
                std_val = np.nan
                min_val = np.nan
                q1_val = np.nan
                median_val = np.nan
                q3_val = np.nan
                max_val = np.nan
                iqr_val = np.nan

            summary_rows.append(
                {
                    "variable": col,
                    "count": valid_count,
                    "missing_count": missing_count,
                    "missing_pct": round((missing_count / total_rows * 100.0), 2) if total_rows > 0 else 0.0,
                    "mean": round(mean_val, 2) if not np.isnan(mean_val) else None,
                    "std": round(std_val, 2) if not np.isnan(std_val) else None,
                    "min": round(min_val, 2) if not np.isnan(min_val) else None,
                    "q1": round(q1_val, 2) if not np.isnan(q1_val) else None,
                    "median": round(median_val, 2) if not np.isnan(median_val) else None,
                    "q3": round(q3_val, 2) if not np.isnan(q3_val) else None,
                    "max": round(max_val, 2) if not np.isnan(max_val) else None,
                    "iqr": round(iqr_val, 2) if not np.isnan(iqr_val) else None,
                }
            )

        return pd.DataFrame(summary_rows)

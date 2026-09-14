"""
Categorical Market Analysis Engine.
Analyzes distributions and central tendencies across categories, brands, models,
fuel types, transmissions, and geographic districts.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from config import settings

CANONICAL_CATEGORIES = settings.SUPPORTED_CATEGORIES


class CategoricalAnalyzer:
    """
    Analyzes categorical distributions across vehicle market dimensions
    with sample size protections and statistical summaries.
    """

    @staticmethod
    def analyze_categories(df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes vehicle categories across all canonical categories:
        - listing count & percentage
        - ML eligible count & percentage
        - median and mean asking price
        - median mileage
        - median manufacture year (YOM)
        """
        if df.empty or "canonical_category" not in df.columns:
            return pd.DataFrame(
                columns=[
                    "category",
                    "listing_count",
                    "pct_of_total",
                    "ml_eligible_count",
                    "ml_eligible_pct",
                    "median_asking_price",
                    "mean_asking_price",
                    "median_mileage",
                    "median_yom",
                ]
            )

        total_listings = len(df)
        rows: List[Dict[str, Any]] = []

        # Group by canonical category
        grouped = df.groupby("canonical_category")

        # Get all observed categories, plus ensure canonical order
        observed_cats = set(df["canonical_category"].dropna().unique())
        ordered_cats = [c for c in CANONICAL_CATEGORIES if c in observed_cats]
        # Include any remaining categories not in the canonical list
        remaining_cats = sorted(list(observed_cats - set(ordered_cats)))
        all_cats = ordered_cats + remaining_cats

        for cat in all_cats:
            group = grouped.get_group(cat) if cat in grouped.groups else pd.DataFrame()
            cnt = len(group)
            if cnt == 0:
                continue

            pct_total = round((cnt / total_listings) * 100.0, 2)
            el_count = int(group["ml_eligible"].sum()) if "ml_eligible" in group else 0
            el_pct = round((el_count / cnt) * 100.0, 2)

            prices = pd.to_numeric(group["asking_price"], errors="coerce").dropna()
            med_price = float(prices.median()) if not prices.empty else np.nan
            mean_price = float(prices.mean()) if not prices.empty else np.nan

            mileages = pd.to_numeric(group["mileage"], errors="coerce").dropna()
            med_mileage = float(mileages.median()) if not mileages.empty else np.nan

            yoms = pd.to_numeric(group["manufacture_year"], errors="coerce").dropna()
            med_yom = float(yoms.median()) if not yoms.empty else np.nan

            rows.append(
                {
                    "category": cat,
                    "listing_count": cnt,
                    "pct_of_total": pct_total,
                    "ml_eligible_count": el_count,
                    "ml_eligible_pct": el_pct,
                    "median_asking_price": round(med_price, 2) if not np.isnan(med_price) else None,
                    "mean_asking_price": round(mean_price, 2) if not np.isnan(mean_price) else None,
                    "median_mileage": round(med_mileage, 2) if not np.isnan(med_mileage) else None,
                    "median_yom": int(med_yom) if not np.isnan(med_yom) else None,
                }
            )

        res_df = pd.DataFrame(rows)
        return res_df

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

    @staticmethod
    def analyze_brands(
        df: pd.DataFrame,
        min_sample: int = 5,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Analyzes vehicle brands with minimum sample size protections.
        Flags brands below min_sample as low-sample to prevent misleading rankings.
        """
        cols = [
            "brand",
            "listing_count",
            "pct_of_total",
            "ml_eligible_count",
            "ml_eligible_pct",
            "median_asking_price",
            "mean_asking_price",
            "min_asking_price",
            "max_asking_price",
            "is_low_sample",
            "sample_flag",
        ]
        if df.empty or "brand" not in df.columns:
            return pd.DataFrame(columns=cols)

        total_listings = len(df)
        valid_df = df.dropna(subset=["brand"]).copy()
        if valid_df.empty:
            return pd.DataFrame(columns=cols)

        rows: List[Dict[str, Any]] = []
        grouped = valid_df.groupby("brand")

        for brand_name, group in grouped:
            cnt = len(group)
            pct_total = round((cnt / total_listings) * 100.0, 2)
            el_count = int(group["ml_eligible"].sum()) if "ml_eligible" in group else 0
            el_pct = round((el_count / cnt) * 100.0, 2)

            prices = pd.to_numeric(group["asking_price"], errors="coerce").dropna()
            med_price = float(prices.median()) if not prices.empty else np.nan
            mean_price = float(prices.mean()) if not prices.empty else np.nan
            min_price = float(prices.min()) if not prices.empty else np.nan
            max_price = float(prices.max()) if not prices.empty else np.nan

            is_low = cnt < min_sample
            flag = f"Low Sample (<{min_sample})" if is_low else "Sufficient Sample"

            rows.append(
                {
                    "brand": str(brand_name).strip(),
                    "listing_count": cnt,
                    "pct_of_total": pct_total,
                    "ml_eligible_count": el_count,
                    "ml_eligible_pct": el_pct,
                    "median_asking_price": round(med_price, 2) if not np.isnan(med_price) else None,
                    "mean_asking_price": round(mean_price, 2) if not np.isnan(mean_price) else None,
                    "min_asking_price": round(min_price, 2) if not np.isnan(min_price) else None,
                    "max_asking_price": round(max_price, 2) if not np.isnan(max_price) else None,
                    "is_low_sample": is_low,
                    "sample_flag": flag,
                }
            )

        res_df = pd.DataFrame(rows)
        res_df = res_df.sort_values(
            by=["listing_count", "median_asking_price"],
            ascending=[False, False],
        ).reset_index(drop=True)

        if top_n is not None and top_n > 0:
            res_df = res_df.head(top_n)

        return res_df

    @staticmethod
    def analyze_models(
        df: pd.DataFrame,
        brand: Optional[str] = None,
        min_sample: int = 3,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Analyzes vehicle models with minimum sample size protections.
        Computes price, mileage, and YOM central tendencies and spreads.
        """
        cols = [
            "brand",
            "model",
            "listing_count",
            "pct_of_total",
            "median_asking_price",
            "mean_asking_price",
            "median_mileage",
            "median_yom",
            "min_yom",
            "max_yom",
            "is_low_sample",
            "sample_flag",
        ]
        if df.empty or "model" not in df.columns:
            return pd.DataFrame(columns=cols)

        valid_df = df.dropna(subset=["model"]).copy()
        if brand:
            valid_df = valid_df[valid_df["brand"].astype(str).str.lower() == brand.strip().lower()]

        if valid_df.empty:
            return pd.DataFrame(columns=cols)

        total_listings = len(df)
        rows: List[Dict[str, Any]] = []

        # If brand column is missing, fill with "Unknown"
        if "brand" not in valid_df.columns:
            valid_df["brand"] = "Unknown"

        grouped = valid_df.groupby(["brand", "model"])

        for (b, m), group in grouped:
            cnt = len(group)
            pct_total = round((cnt / total_listings) * 100.0, 2)

            prices = pd.to_numeric(group["asking_price"], errors="coerce").dropna()
            med_price = float(prices.median()) if not prices.empty else np.nan
            mean_price = float(prices.mean()) if not prices.empty else np.nan

            mileages = pd.to_numeric(group["mileage"], errors="coerce").dropna()
            med_mileage = float(mileages.median()) if not mileages.empty else np.nan

            yoms = pd.to_numeric(group["manufacture_year"], errors="coerce").dropna()
            med_yom = float(yoms.median()) if not yoms.empty else np.nan
            min_yom = int(yoms.min()) if not yoms.empty else None
            max_yom = int(yoms.max()) if not yoms.empty else None

            is_low = cnt < min_sample
            flag = f"Low Sample (<{min_sample})" if is_low else "Sufficient Sample"

            rows.append(
                {
                    "brand": str(b).strip(),
                    "model": str(m).strip(),
                    "listing_count": cnt,
                    "pct_of_total": pct_total,
                    "median_asking_price": round(med_price, 2) if not np.isnan(med_price) else None,
                    "mean_asking_price": round(mean_price, 2) if not np.isnan(mean_price) else None,
                    "median_mileage": round(med_mileage, 2) if not np.isnan(med_mileage) else None,
                    "median_yom": int(med_yom) if not np.isnan(med_yom) else None,
                    "min_yom": min_yom,
                    "max_yom": max_yom,
                    "is_low_sample": is_low,
                    "sample_flag": flag,
                }
            )

        res_df = pd.DataFrame(rows)
        res_df = res_df.sort_values(
            by=["listing_count", "median_asking_price"],
            ascending=[False, False],
        ).reset_index(drop=True)

        if top_n is not None and top_n > 0:
            res_df = res_df.head(top_n)

        return res_df

    @staticmethod
    def analyze_fuel_types(
        df: pd.DataFrame,
        min_sample: int = 3,
    ) -> pd.DataFrame:
        """
        Analyzes fuel types (Petrol, Diesel, Hybrid, Electric, Gas/LPG).
        Computes counts, proportions, and central tendency of asking prices.
        """
        cols = [
            "fuel_type",
            "listing_count",
            "pct_of_total",
            "median_asking_price",
            "mean_asking_price",
            "is_low_sample",
            "sample_flag",
        ]
        if df.empty or "fuel_type" not in df.columns:
            return pd.DataFrame(columns=cols)

        total_listings = len(df)
        valid_df = df.dropna(subset=["fuel_type"]).copy()
        if valid_df.empty:
            return pd.DataFrame(columns=cols)

        rows: List[Dict[str, Any]] = []
        for fuel, group in valid_df.groupby("fuel_type"):
            cnt = len(group)
            pct_total = round((cnt / total_listings) * 100.0, 2)

            prices = pd.to_numeric(group["asking_price"], errors="coerce").dropna()
            med_price = float(prices.median()) if not prices.empty else np.nan
            mean_price = float(prices.mean()) if not prices.empty else np.nan

            is_low = cnt < min_sample
            flag = f"Low Sample (<{min_sample})" if is_low else "Sufficient Sample"

            rows.append(
                {
                    "fuel_type": str(fuel).strip(),
                    "listing_count": cnt,
                    "pct_of_total": pct_total,
                    "median_asking_price": round(med_price, 2) if not np.isnan(med_price) else None,
                    "mean_asking_price": round(mean_price, 2) if not np.isnan(mean_price) else None,
                    "is_low_sample": is_low,
                    "sample_flag": flag,
                }
            )

        res_df = pd.DataFrame(rows)
        return res_df.sort_values(by=["listing_count", "median_asking_price"], ascending=[False, False]).reset_index(drop=True)

    @staticmethod
    def analyze_transmissions(
        df: pd.DataFrame,
        min_sample: int = 3,
    ) -> pd.DataFrame:
        """
        Analyzes transmission types (Automatic, Manual).
        Computes counts, proportions, and median asking prices.
        """
        cols = [
            "transmission",
            "listing_count",
            "pct_of_total",
            "median_asking_price",
            "mean_asking_price",
            "is_low_sample",
            "sample_flag",
        ]
        if df.empty or "transmission" not in df.columns:
            return pd.DataFrame(columns=cols)

        total_listings = len(df)
        valid_df = df.dropna(subset=["transmission"]).copy()
        if valid_df.empty:
            return pd.DataFrame(columns=cols)

        rows: List[Dict[str, Any]] = []
        for trans, group in valid_df.groupby("transmission"):
            cnt = len(group)
            pct_total = round((cnt / total_listings) * 100.0, 2)

            prices = pd.to_numeric(group["asking_price"], errors="coerce").dropna()
            med_price = float(prices.median()) if not prices.empty else np.nan
            mean_price = float(prices.mean()) if not prices.empty else np.nan

            is_low = cnt < min_sample
            flag = f"Low Sample (<{min_sample})" if is_low else "Sufficient Sample"

            rows.append(
                {
                    "transmission": str(trans).strip(),
                    "listing_count": cnt,
                    "pct_of_total": pct_total,
                    "median_asking_price": round(med_price, 2) if not np.isnan(med_price) else None,
                    "mean_asking_price": round(mean_price, 2) if not np.isnan(mean_price) else None,
                    "is_low_sample": is_low,
                    "sample_flag": flag,
                }
            )

        res_df = pd.DataFrame(rows)
        return res_df.sort_values(by=["listing_count", "median_asking_price"], ascending=[False, False]).reset_index(drop=True)

    @staticmethod
    def analyze_districts(
        df: pd.DataFrame,
        min_sample: int = 3,
    ) -> pd.DataFrame:
        """
        Analyzes geographic listing distribution across Sri Lankan administrative districts.
        Reports observed price differences by district without asserting causal price claims.
        """
        cols = [
            "district",
            "listing_count",
            "pct_of_total",
            "median_asking_price",
            "mean_asking_price",
            "is_low_sample",
            "sample_flag",
        ]
        if df.empty or "district" not in df.columns:
            return pd.DataFrame(columns=cols)

        total_listings = len(df)
        valid_df = df.dropna(subset=["district"]).copy()
        # Filter out empty or whitespace-only district strings
        valid_df = valid_df[valid_df["district"].astype(str).str.strip() != ""]

        if valid_df.empty:
            return pd.DataFrame(columns=cols)

        rows: List[Dict[str, Any]] = []
        for dist, group in valid_df.groupby("district"):
            cnt = len(group)
            pct_total = round((cnt / total_listings) * 100.0, 2)

            prices = pd.to_numeric(group["asking_price"], errors="coerce").dropna()
            med_price = float(prices.median()) if not prices.empty else np.nan
            mean_price = float(prices.mean()) if not prices.empty else np.nan

            is_low = cnt < min_sample
            flag = f"Low Sample (<{min_sample})" if is_low else "Sufficient Sample"

            rows.append(
                {
                    "district": str(dist).strip(),
                    "listing_count": cnt,
                    "pct_of_total": pct_total,
                    "median_asking_price": round(med_price, 2) if not np.isnan(med_price) else None,
                    "mean_asking_price": round(mean_price, 2) if not np.isnan(mean_price) else None,
                    "is_low_sample": is_low,
                    "sample_flag": flag,
                }
            )

        res_df = pd.DataFrame(rows)
        return res_df.sort_values(by=["listing_count", "median_asking_price"], ascending=[False, False]).reset_index(drop=True)

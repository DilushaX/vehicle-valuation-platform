"""
Market Analytics Engine for Market Intelligence Dashboard (Phase 10.3).

Provides pure, testable dataset filtering, option extraction, and descriptive
aggregations for the Market Intelligence Dashboard.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from eda.categorical import CategoricalAnalyzer
from eda.distributions import DistributionAnalyzer
from eda.relationships import RelationshipAnalyzer


def get_district_summary(
    df: pd.DataFrame,
    min_sample: int = 1,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Analyzes geographic listing distribution across administrative districts.
    Reports observed price differences by district without asserting causal claims.
    Reuses CategoricalAnalyzer.analyze_districts.
    """
    if df.empty or "district" not in df.columns:
        return pd.DataFrame(
            columns=[
                "district",
                "listing_count",
                "pct_of_total",
                "median_asking_price",
                "mean_asking_price",
                "is_low_sample",
                "sample_flag",
            ]
        )
    res = CategoricalAnalyzer.analyze_districts(df, min_sample=min_sample)
    if top_n is not None and top_n > 0:
        res = res.head(top_n)
    return res


def get_price_distribution_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes parametric and non-parametric asking price distribution metrics.
    Reuses DistributionAnalyzer.analyze_price_distribution.
    """
    if df.empty or "asking_price" not in df.columns:
        return {
            "count": 0,
            "missing_count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q1": None,
            "median": None,
            "q3": None,
            "max": None,
            "iqr": None,
        }
    return DistributionAnalyzer.analyze_price_distribution(df)


def get_price_relationships(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Computes bivariate correlations (Pearson & Spearman) between asking price
    and numerical attributes (vehicle age, mileage, engine CC).
    Reuses RelationshipAnalyzer.compute_bivariate_relationships.
    """
    if df.empty:
        return {}
    return RelationshipAnalyzer.compute_bivariate_relationships(df)


def get_fuel_summary(
    df: pd.DataFrame,
    min_sample: int = 1,
) -> pd.DataFrame:
    """
    Analyzes vehicle fuel types (Petrol, Diesel, Hybrid, etc.).
    Descriptive comparison only. Does not make causal or superiority claims.
    Reuses CategoricalAnalyzer.analyze_fuel_types.
    """
    if df.empty or "fuel_type" not in df.columns:
        return pd.DataFrame(
            columns=[
                "fuel_type",
                "listing_count",
                "pct_of_total",
                "median_asking_price",
                "mean_asking_price",
                "is_low_sample",
                "sample_flag",
            ]
        )
    return CategoricalAnalyzer.analyze_fuel_types(df, min_sample=min_sample)


def get_transmission_summary(
    df: pd.DataFrame,
    min_sample: int = 1,
) -> pd.DataFrame:
    """
    Analyzes transmission types (Automatic, Manual).
    Descriptive comparison only.
    Reuses CategoricalAnalyzer.analyze_transmissions.
    """
    if df.empty or "transmission" not in df.columns:
        return pd.DataFrame(
            columns=[
                "transmission",
                "listing_count",
                "pct_of_total",
                "median_asking_price",
                "mean_asking_price",
                "is_low_sample",
                "sample_flag",
            ]
        )
    return CategoricalAnalyzer.analyze_transmissions(df, min_sample=min_sample)


def get_vehicle_age_distribution_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes vehicle age and manufacture year distributions.
    Reuses DistributionAnalyzer.analyze_yom_and_age_distribution.
    """
    if df.empty or "manufacture_year" not in df.columns:
        return {"count": 0, "missing_count": 0}
    return DistributionAnalyzer.analyze_yom_and_age_distribution(df)


def get_mileage_distribution_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes odometer mileage distributions and bracket frequencies.
    Reuses DistributionAnalyzer.analyze_mileage_distribution.
    """
    if df.empty or "mileage" not in df.columns:
        return {"count": 0, "missing_count": 0}
    return DistributionAnalyzer.analyze_mileage_distribution(df)


def get_mileage_bracket_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes listing counts and median asking prices across standard mileage brackets.
    """
    brackets = [
        ("< 25,000 km", 0, 25_000),
        ("25,000 - 50,000 km", 25_000, 50_000),
        ("50,000 - 100,000 km", 50_000, 100_000),
        ("100,000 - 150,000 km", 100_000, 150_000),
        ("150,000 - 200,000 km", 150_000, 200_000),
        ("> 200,000 km", 200_000, float("inf")),
    ]
    cols = ["bracket", "listing_count", "median_asking_price", "mean_asking_price"]
    if df.empty or "mileage" not in df.columns:
        return pd.DataFrame(columns=cols)

    valid = df.dropna(subset=["mileage"]).copy()
    valid["mileage_num"] = pd.to_numeric(valid["mileage"], errors="coerce")
    valid = valid.dropna(subset=["mileage_num"])

    rows = []
    for label, low, high in brackets:
        if high == float("inf"):
            sub = valid[valid["mileage_num"] >= low]
        else:
            sub = valid[(valid["mileage_num"] >= low) & (valid["mileage_num"] < high)]
        cnt = len(sub)
        prices = pd.to_numeric(sub.get("asking_price", pd.Series()), errors="coerce").dropna()
        prices = prices[prices > 0]
        med = float(prices.median()) if not prices.empty else None
        mean_p = float(prices.mean()) if not prices.empty else None
        rows.append({
            "bracket": label,
            "listing_count": cnt,
            "median_asking_price": med,
            "mean_asking_price": mean_p,
        })
    return pd.DataFrame(rows)


def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes category-level market summary statistics:
    listing count, market share percentage, ML eligible count/rate,
    median asking price, mean asking price, median mileage, and median manufacture year.
    Reuses CategoricalAnalyzer.analyze_categories.
    """
    if df.empty:
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
    return CategoricalAnalyzer.analyze_categories(df)


def get_brand_summary(
    df: pd.DataFrame,
    min_sample: int = 1,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Analyzes vehicle brands with configurable minimum sample size and top-N ranking.
    Descriptive statistics only (not a brand value or quality ranking).
    Reuses CategoricalAnalyzer.analyze_brands.
    """
    if df.empty or "brand" not in df.columns:
        return pd.DataFrame(
            columns=[
                "brand",
                "listing_count",
                "pct_of_total",
                "median_asking_price",
                "mean_asking_price",
                "min_asking_price",
                "max_asking_price",
                "is_low_sample",
                "sample_flag",
            ]
        )
    return CategoricalAnalyzer.analyze_brands(df, min_sample=min_sample, top_n=top_n)


def get_model_summary(
    df: pd.DataFrame,
    brand: Optional[str] = None,
    min_sample: int = 1,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Analyzes vehicle models respecting category and brand constraints.
    Computes asking price and vehicle attribute medians with sample flags.
    Reuses CategoricalAnalyzer.analyze_models.
    """
    if df.empty or "model" not in df.columns:
        return pd.DataFrame(
            columns=[
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
        )
    return CategoricalAnalyzer.analyze_models(
        df, brand=brand, min_sample=min_sample, top_n=top_n
    )


def compute_market_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes high-level market overview metrics from a listings DataFrame.

    Distinguishes listings from vehicle records:
    - total_listings: Total advertised market listing records in the filtered view.
    - total_vehicles: Unique physical vehicle specifications associated with the listings.
    - asking price statistics: Advertised seller prices (not confirmed sale prices).
    """
    if df.empty:
        return {
            "total_listings": 0,
            "total_vehicles": 0,
            "distinct_categories": 0,
            "distinct_brands": 0,
            "distinct_models": 0,
            "median_asking_price": None,
            "average_asking_price": None,
            "min_asking_price": None,
            "max_asking_price": None,
            "active_listings": 0,
            "ml_eligible_listings": 0,
            "ml_eligible_pct": 0.0,
        }

    total_listings = len(df)
    
    # Check if vehicle_id exists in df, otherwise treat vehicle spec count
    if "vehicle_id" in df.columns:
        total_vehicles = int(df["vehicle_id"].nunique())
    else:
        # In current schema, each listing corresponds to its own vehicle specification record
        total_vehicles = total_listings

    cat_col = "canonical_category" if "canonical_category" in df.columns else "category"
    distinct_categories = int(df[cat_col].dropna().nunique()) if cat_col in df.columns else 0
    distinct_brands = int(df["brand"].dropna().nunique()) if "brand" in df.columns else 0
    distinct_models = int(df["model"].dropna().nunique()) if "model" in df.columns else 0

    # Asking price metrics
    valid_prices = pd.to_numeric(df["asking_price"], errors="coerce").dropna()
    valid_prices = valid_prices[valid_prices > 0]

    if not valid_prices.empty:
        median_price = float(valid_prices.median())
        mean_price = float(valid_prices.mean())
        min_price = float(valid_prices.min())
        max_price = float(valid_prices.max())
    else:
        median_price, mean_price, min_price, max_price = None, None, None, None

    active_cnt = (
        int((df["current_status"] == "ACTIVE").sum())
        if "current_status" in df.columns
        else 0
    )
    ml_eligible_cnt = (
        int(df["ml_eligible"].sum())
        if "ml_eligible" in df.columns
        else 0
    )
    ml_eligible_pct = round((ml_eligible_cnt / total_listings) * 100.0, 1) if total_listings > 0 else 0.0

    return {
        "total_listings": total_listings,
        "total_vehicles": total_vehicles,
        "distinct_categories": distinct_categories,
        "distinct_brands": distinct_brands,
        "distinct_models": distinct_models,
        "median_asking_price": median_price,
        "average_asking_price": mean_price,
        "min_asking_price": min_price,
        "max_asking_price": max_price,
        "active_listings": active_cnt,
        "ml_eligible_listings": ml_eligible_cnt,
        "ml_eligible_pct": ml_eligible_pct,
    }


def get_filter_options(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extracts unique filter options and min/max ranges available in the dataset.
    Safely handles empty datasets and missing values.
    """
    if df.empty:
        return {
            "categories": [],
            "brands": [],
            "models": [],
            "districts": [],
            "fuel_types": [],
            "transmissions": [],
            "conditions": [],
            "year_min": 1990,
            "year_max": 2026,
            "price_min": 0,
            "price_max": 100_000_000,
        }

    # Canonical categories or raw categories
    cat_col = "canonical_category" if "canonical_category" in df.columns else "category"
    categories = sorted([str(c) for c in df[cat_col].dropna().unique() if str(c).strip()])

    brands = sorted([str(b) for b in df["brand"].dropna().unique() if str(b).strip()]) if "brand" in df.columns else []
    models = sorted([str(m) for m in df["model"].dropna().unique() if str(m).strip()]) if "model" in df.columns else []
    districts = sorted([str(d) for d in df["district"].dropna().unique() if str(d).strip()]) if "district" in df.columns else []
    fuel_types = sorted([str(f) for f in df["fuel_type"].dropna().unique() if str(f).strip()]) if "fuel_type" in df.columns else []
    transmissions = sorted([str(t) for t in df["transmission"].dropna().unique() if str(t).strip()]) if "transmission" in df.columns else []
    conditions = sorted([str(c) for c in df["condition"].dropna().unique() if str(c).strip()]) if "condition" in df.columns else []

    # Years
    if "manufacture_year" in df.columns:
        valid_years = pd.to_numeric(df["manufacture_year"], errors="coerce").dropna()
        year_min = int(valid_years.min()) if not valid_years.empty else 1990
        year_max = int(valid_years.max()) if not valid_years.empty else 2026
    else:
        year_min, year_max = 1990, 2026

    # Prices
    if "asking_price" in df.columns:
        valid_prices = pd.to_numeric(df["asking_price"], errors="coerce").dropna()
        valid_prices = valid_prices[valid_prices > 0]
        price_min = int(valid_prices.min()) if not valid_prices.empty else 0
        price_max = int(valid_prices.max()) if not valid_prices.empty else 50_000_000
    else:
        price_min, price_max = 0, 50_000_000

    return {
        "categories": categories,
        "brands": brands,
        "models": models,
        "districts": districts,
        "fuel_types": fuel_types,
        "transmissions": transmissions,
        "conditions": conditions,
        "year_min": year_min,
        "year_max": year_max,
        "price_min": price_min,
        "price_max": price_max,
    }


def apply_filters(
    df: pd.DataFrame,
    categories: Optional[Sequence[str]] = None,
    brands: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    districts: Optional[Sequence[str]] = None,
    fuel_types: Optional[Sequence[str]] = None,
    transmissions: Optional[Sequence[str]] = None,
    conditions: Optional[Sequence[str]] = None,
    year_range: Optional[Tuple[int, int]] = None,
    price_range: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    Applies global filters to a vehicle listings DataFrame in-memory.
    Returns a filtered copy without modifying the original DataFrame.
    """
    if df.empty:
        return df.copy()

    filtered = df.copy()
    cat_col = "canonical_category" if "canonical_category" in filtered.columns else "category"

    # Category filter
    if categories:
        cat_set = {str(c).strip().lower() for c in categories}
        filtered = filtered[
            filtered[cat_col].fillna("").astype(str).str.strip().str.lower().isin(cat_set)
        ]

    # Brand filter
    if brands and "brand" in filtered.columns:
        brand_set = {str(b).strip().lower() for b in brands}
        filtered = filtered[
            filtered["brand"].fillna("").astype(str).str.strip().str.lower().isin(brand_set)
        ]

    # Model filter
    if models and "model" in filtered.columns:
        model_set = {str(m).strip().lower() for m in models}
        filtered = filtered[
            filtered["model"].fillna("").astype(str).str.strip().str.lower().isin(model_set)
        ]

    # District filter
    if districts and "district" in filtered.columns:
        dist_set = {str(d).strip().lower() for d in districts}
        filtered = filtered[
            filtered["district"].fillna("").astype(str).str.strip().str.lower().isin(dist_set)
        ]

    # Fuel Type filter
    if fuel_types and "fuel_type" in filtered.columns:
        fuel_set = {str(f).strip().lower() for f in fuel_types}
        filtered = filtered[
            filtered["fuel_type"].fillna("").astype(str).str.strip().str.lower().isin(fuel_set)
        ]

    # Transmission filter
    if transmissions and "transmission" in filtered.columns:
        trans_set = {str(t).strip().lower() for t in transmissions}
        filtered = filtered[
            filtered["transmission"].fillna("").astype(str).str.strip().str.lower().isin(trans_set)
        ]

    # Condition filter
    if conditions and "condition" in filtered.columns:
        cond_set = {str(c).strip().lower() for c in conditions}
        filtered = filtered[
            filtered["condition"].fillna("").astype(str).str.strip().str.lower().isin(cond_set)
        ]

    # Manufacture Year range
    if year_range and "manufacture_year" in filtered.columns:
        y_min, y_max = year_range
        years = pd.to_numeric(filtered["manufacture_year"], errors="coerce")
        # Keep records matching range, or include records where year is within range
        filtered = filtered[(years >= y_min) & (years <= y_max)]

    # Asking Price range
    if price_range and "asking_price" in filtered.columns:
        p_min, p_max = price_range
        prices = pd.to_numeric(filtered["asking_price"], errors="coerce")
        filtered = filtered[(prices >= p_min) & (prices <= p_max)]

    return filtered.reset_index(drop=True)

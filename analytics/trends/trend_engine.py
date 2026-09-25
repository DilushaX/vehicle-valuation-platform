"""
Historical Market Trend Engine (Phase 10.3).

Computes longitudinal asking price dynamics, monthly activity levels,
and enforces minimum historical depth gates to prevent premature or misleading
trend inferences from limited observation windows.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from eda.historical import HistoricalAnalyzer, INSUFFICIENT_DEPTH_MESSAGE


def evaluate_trend_readiness(
    df_obs: pd.DataFrame,
    df_price_history: pd.DataFrame,
    min_days: int = 60,
) -> Dict[str, Any]:
    """
    Evaluates whether the dataset contains sufficient observation depth
    for statistically defensible monthly trend inference.
    """
    return HistoricalAnalyzer.evaluate_historical_depth(
        df_obs=df_obs,
        df_price_history=df_price_history,
        min_days_for_monthly_trends=min_days,
    )


def compute_price_movement_summary(df_price_history: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarizes asking price revisions across tracked listing histories.
    """
    return HistoricalAnalyzer.analyze_price_history(df_price_history)


def compute_monthly_trends(
    df: pd.DataFrame,
    df_price_history: Optional[pd.DataFrame] = None,
    min_days: int = 60,
) -> Dict[str, Any]:
    """
    Aggregates monthly listing activity and asking price dynamics.

    Strictly gates on historical depth:
    If date span is below min_days (default 60 days), returns has_sufficient_depth=False
    and refuses to extrapolate or fabricate historical trends.
    """
    empty_result = {
        "has_sufficient_depth": False,
        "message": "Insufficient historical observations for this analysis.",
        "days_span": 0.0,
        "min_date": None,
        "max_date": None,
        "monthly_activity": pd.DataFrame(
            columns=["year_month", "listing_count", "median_asking_price", "mean_asking_price"]
        ),
        "category_monthly": pd.DataFrame(
            columns=["year_month", "category", "listing_count", "median_asking_price"]
        ),
    }

    if df.empty:
        return empty_result

    # Determine date column for time series
    date_col = None
    if "first_seen_at" in df.columns:
        date_col = "first_seen_at"
    elif "ad_date" in df.columns:
        date_col = "ad_date"

    if not date_col:
        return empty_result

    # Evaluate depth using first_seen_at / price_history
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if df_price_history is not None and not df_price_history.empty and "observed_at" in df_price_history.columns:
        ph_dates = pd.to_datetime(df_price_history["observed_at"], errors="coerce").dropna()
        all_dates = pd.concat([dates, ph_dates])
    else:
        all_dates = dates

    if all_dates.empty:
        return empty_result

    min_d = all_dates.min()
    max_d = all_dates.max()
    days_span = (max_d - min_d).total_seconds() / 86400.0

    if days_span < min_days:
        return {
            "has_sufficient_depth": False,
            "message": (
                f"Insufficient historical observations for this analysis. "
                f"Current observations span {days_span:.1f} days (minimum {min_days} days required for reliable monthly trend analysis). "
                f"Asking prices and listing volumes are not fabricated or extrapolated."
            ),
            "days_span": round(days_span, 1),
            "min_date": min_d.strftime("%Y-%m-%d") if hasattr(min_d, "strftime") else str(min_d),
            "max_date": max_d.strftime("%Y-%m-%d") if hasattr(max_d, "strftime") else str(max_d),
            "monthly_activity": empty_result["monthly_activity"],
            "category_monthly": empty_result["category_monthly"],
        }

    # Sufficient depth: aggregate monthly metrics
    df_valid = df.copy()
    df_valid["_dt"] = pd.to_datetime(df_valid[date_col], errors="coerce")
    df_valid = df_valid.dropna(subset=["_dt"])
    df_valid["year_month"] = df_valid["_dt"].dt.to_period("M").astype(str)

    monthly_rows = []
    for ym, group in df_valid.groupby("year_month"):
        cnt = len(group)
        prices = pd.to_numeric(group["asking_price"], errors="coerce").dropna()
        prices = prices[prices > 0]
        med_p = float(prices.median()) if not prices.empty else None
        mean_p = float(prices.mean()) if not prices.empty else None
        monthly_rows.append({
            "year_month": ym,
            "listing_count": cnt,
            "median_asking_price": med_p,
            "mean_asking_price": mean_p,
        })
    monthly_df = pd.DataFrame(monthly_rows).sort_values(by="year_month").reset_index(drop=True)

    # Category monthly
    cat_col = "canonical_category" if "canonical_category" in df_valid.columns else "category"
    cat_monthly_rows = []
    if cat_col in df_valid.columns:
        for (ym, cat), group in df_valid.groupby(["year_month", cat_col]):
            cnt = len(group)
            prices = pd.to_numeric(group["asking_price"], errors="coerce").dropna()
            prices = prices[prices > 0]
            med_p = float(prices.median()) if not prices.empty else None
            cat_monthly_rows.append({
                "year_month": ym,
                "category": str(cat),
                "listing_count": cnt,
                "median_asking_price": med_p,
            })
    cat_monthly_df = pd.DataFrame(cat_monthly_rows).sort_values(by=["year_month", "category"]).reset_index(drop=True)

    return {
        "has_sufficient_depth": True,
        "message": "Historical observation depth is sufficient for monthly trend analysis.",
        "days_span": round(days_span, 1),
        "min_date": min_d.strftime("%Y-%m-%d"),
        "max_date": max_d.strftime("%Y-%m-%d"),
        "monthly_activity": monthly_df,
        "category_monthly": cat_monthly_df,
    }

"""
Distribution and Visualization Analysis Engine.
Analyzes asking price, manufacture year (YOM), vehicle age, mileage, and engine CC
distributions across categories and generates publication-grade visualizations using Matplotlib.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


class DistributionAnalyzer:
    """
    Computes statistical distribution properties and renders clean, professional
    visualizations for univariate vehicle market variables.
    """

    @staticmethod
    def analyze_price_distribution(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes asking price distributions, separating valid, missing, and suspicious values."""
        if df.empty or "asking_price" not in df.columns:
            return {"count": 0, "missing_count": 0}

        total = len(df)
        prices_series = pd.to_numeric(df["asking_price"], errors="coerce")
        missing_count = int(prices_series.isna().sum())
        valid_prices = prices_series.dropna()

        # Suspicious prices from validation_issues if available
        suspicious_count = 0
        if "validation_issues" in df.columns:
            for issues in df["validation_issues"]:
                if isinstance(issues, list) and any("price" in str(iss).lower() for iss in issues):
                    suspicious_count += 1

        if valid_prices.empty:
            return {
                "count": 0,
                "missing_count": missing_count,
                "suspicious_count": suspicious_count,
            }

        q1 = float(valid_prices.quantile(0.25))
        median = float(valid_prices.median())
        q3 = float(valid_prices.quantile(0.75))
        iqr = q3 - q1

        return {
            "count": len(valid_prices),
            "missing_count": missing_count,
            "suspicious_count": suspicious_count,
            "mean": round(float(valid_prices.mean()), 2),
            "std": round(float(valid_prices.std(ddof=1)), 2) if len(valid_prices) > 1 else 0.0,
            "min": float(valid_prices.min()),
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": float(valid_prices.max()),
            "iqr": iqr,
        }

    @staticmethod
    def analyze_yom_and_age_distribution(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes manufacture year (YOM) and vehicle age distributions."""
        if df.empty or "manufacture_year" not in df.columns:
            return {"count": 0, "missing_count": 0}

        yom_series = pd.to_numeric(df["manufacture_year"], errors="coerce").dropna()
        if yom_series.empty:
            return {"count": 0, "missing_count": len(df)}

        age_series = pd.to_numeric(df.get("vehicle_age", pd.Series()), errors="coerce").dropna()

        return {
            "count": len(yom_series),
            "missing_count": len(df) - len(yom_series),
            "yom_median": int(yom_series.median()),
            "yom_min": int(yom_series.min()),
            "yom_max": int(yom_series.max()),
            "yom_q1": int(yom_series.quantile(0.25)),
            "yom_q3": int(yom_series.quantile(0.75)),
            "age_median": float(age_series.median()) if not age_series.empty else None,
            "age_mean": round(float(age_series.mean()), 2) if not age_series.empty else None,
            "age_min": float(age_series.min()) if not age_series.empty else None,
            "age_max": float(age_series.max()) if not age_series.empty else None,
            "age_q1": float(age_series.quantile(0.25)) if not age_series.empty else None,
            "age_q3": float(age_series.quantile(0.75)) if not age_series.empty else None,
            "age_iqr": float(age_series.quantile(0.75) - age_series.quantile(0.25)) if not age_series.empty else None,
        }

    @staticmethod
    def analyze_mileage_distribution(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes odometer mileage, distinguishing valid vs suspicious readings."""
        if df.empty or "mileage" not in df.columns:
            return {"count": 0, "missing_count": 0}

        mileage_series = pd.to_numeric(df["mileage"], errors="coerce")
        missing_count = int(mileage_series.isna().sum())
        valid_mileage = mileage_series.dropna()

        suspicious_count = 0
        if "validation_issues" in df.columns:
            for issues in df["validation_issues"]:
                if isinstance(issues, list) and any("mileage" in str(iss).lower() for iss in issues):
                    suspicious_count += 1

        if valid_mileage.empty:
            return {
                "count": 0,
                "missing_count": missing_count,
                "suspicious_count": suspicious_count,
            }

        q1 = float(valid_mileage.quantile(0.25))
        median = float(valid_mileage.median())
        q3 = float(valid_mileage.quantile(0.75))
        iqr = q3 - q1

        # Mileage bracket categorization
        brackets = {
            "< 25,000 km": int((valid_mileage < 25000).sum()),
            "25,000 - 50,000 km": int(((valid_mileage >= 25000) & (valid_mileage < 50000)).sum()),
            "50,000 - 100,000 km": int(((valid_mileage >= 50000) & (valid_mileage < 100000)).sum()),
            "100,000 - 150,000 km": int(((valid_mileage >= 100000) & (valid_mileage < 150000)).sum()),
            "150,000 - 200,000 km": int(((valid_mileage >= 150000) & (valid_mileage < 200000)).sum()),
            "> 200,000 km": int((valid_mileage >= 200000).sum()),
        }

        return {
            "count": len(valid_mileage),
            "missing_count": missing_count,
            "suspicious_count": suspicious_count,
            "mean": round(float(valid_mileage.mean()), 2),
            "std": round(float(valid_mileage.std(ddof=1)), 2) if len(valid_mileage) > 1 else 0.0,
            "min": float(valid_mileage.min()),
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": float(valid_mileage.max()),
            "iqr": iqr,
            "brackets": brackets,
        }

    @staticmethod
    def analyze_engine_cc_distribution(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes engine displacement, handling category-specific variations and EV exemptions."""
        if df.empty or "engine_cc" not in df.columns:
            return {"count": 0, "missing_count": 0}

        cc_series = pd.to_numeric(df["engine_cc"], errors="coerce")
        missing_count = int(cc_series.isna().sum())
        valid_cc = cc_series.dropna()

        if valid_cc.empty:
            return {"count": 0, "missing_count": missing_count}

        by_cat = {}
        if "canonical_category" in df.columns:
            for cat, group in df.groupby("canonical_category"):
                g_cc = pd.to_numeric(group["engine_cc"], errors="coerce").dropna()
                if not g_cc.empty:
                    by_cat[str(cat)] = {
                        "count": len(g_cc),
                        "median": float(g_cc.median()),
                        "mean": round(float(g_cc.mean()), 2),
                        "min": float(g_cc.min()),
                        "max": float(g_cc.max()),
                    }

        return {
            "count": len(valid_cc),
            "missing_count": missing_count,
            "mean": round(float(valid_cc.mean()), 2),
            "std": round(float(valid_cc.std(ddof=1)), 2) if len(valid_cc) > 1 else 0.0,
            "min": float(valid_cc.min()),
            "median": float(valid_cc.median()),
            "q1": float(valid_cc.quantile(0.25)),
            "q3": float(valid_cc.quantile(0.75)),
            "max": float(valid_cc.max()),
            "by_category": by_cat,
        }

    # ==========================================================================
    # VISUALIZATION GENERATORS
    # ==========================================================================

    @staticmethod
    def plot_price_distribution(
        df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Generates asking price histogram and box plot (Linear and Log scales)."""
        valid_prices = pd.to_numeric(df.get("asking_price", pd.Series()), errors="coerce").dropna()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        if not valid_prices.empty:
            prices_millions = valid_prices / 1_000_000.0

            # Histogram
            ax1.hist(prices_millions, bins=25, color="#2b5c8f", edgecolor="black", alpha=0.75)
            ax1.set_title("Observed Asking Price Distribution (Linear Scale)", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Asking Price (Million LKR / Rs.)", fontsize=10)
            ax1.set_ylabel("Listing Count", fontsize=10)
            ax1.grid(True, linestyle="--", alpha=0.5)

            # Boxplot with log-scale
            ax2.boxplot(valid_prices, orientation="vertical", patch_artist=True,
                        boxprops=dict(facecolor="#4a90e2", color="black"),
                        medianprops=dict(color="red", linewidth=2))
            ax2.set_yscale("log")
            ax2.set_title("Asking Price Spread (Log Scale)", fontsize=12, fontweight="bold")
            ax2.set_ylabel("Asking Price (LKR / Rs. Log Scale)", fontsize=10)
            ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"Rs. {y:,.0f}"))
            ax2.grid(True, linestyle="--", alpha=0.5)
        else:
            ax1.text(0.5, 0.5, "No Asking Price Data Available", ha="center", va="center")
            ax2.text(0.5, 0.5, "No Asking Price Data Available", ha="center", va="center")

        plt.tight_layout()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return fig

    @staticmethod
    def plot_category_price_comparison(
        df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Generates asking price comparison across all vehicle categories."""
        fig, ax = plt.subplots(figsize=(12, 6))

        valid_df = df.dropna(subset=["asking_price", "canonical_category"]).copy()
        if not valid_df.empty:
            categories = []
            price_groups = []
            for cat, grp in valid_df.groupby("canonical_category"):
                prices_m = grp["asking_price"].astype(float) / 1_000_000.0
                if len(prices_m) > 0:
                    categories.append(f"{cat}\n(n={len(prices_m)})")
                    price_groups.append(prices_m.values)

            if price_groups:
                box = ax.boxplot(price_groups, tick_labels=categories, patch_artist=True,
                                 medianprops=dict(color="red", linewidth=2))
                for patch in box["boxes"]:
                    patch.set_facecolor("#5c93c4")
                ax.set_title("Observed Asking Price Comparison by Vehicle Category", fontsize=13, fontweight="bold")
                ax.set_ylabel("Asking Price (Million LKR / Rs.)", fontsize=11)
                ax.set_xlabel("Vehicle Category", fontsize=11)
                ax.grid(True, linestyle="--", alpha=0.5)
        else:
            ax.text(0.5, 0.5, "No Data for Category Comparison", ha="center", va="center")

        plt.tight_layout()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return fig

    @staticmethod
    def plot_yom_and_age_distribution(
        df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Generates Manufacture Year (YOM) and Vehicle Age distribution plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        yom_series = pd.to_numeric(df.get("manufacture_year", pd.Series()), errors="coerce").dropna()
        age_series = pd.to_numeric(df.get("vehicle_age", pd.Series()), errors="coerce").dropna()

        if not yom_series.empty:
            ax1.hist(yom_series, bins=min(25, yom_series.nunique() or 10), color="#2e7d32", edgecolor="black", alpha=0.75)
            ax1.set_title("Manufacture Year (YOM) Distribution", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Year of Manufacture", fontsize=10)
            ax1.set_ylabel("Listing Count", fontsize=10)
            ax1.grid(True, linestyle="--", alpha=0.5)

        if not age_series.empty:
            ax2.hist(age_series, bins=min(20, age_series.nunique() or 10), color="#ef6c00", edgecolor="black", alpha=0.75)
            ax2.set_title("Vehicle Age Distribution", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Vehicle Age (Years)", fontsize=10)
            ax2.set_ylabel("Listing Count", fontsize=10)
            ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return fig

    @staticmethod
    def plot_mileage_distribution(
        df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Generates mileage distribution histogram and box plot with extreme-value handling."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        mileage_series = pd.to_numeric(df.get("mileage", pd.Series()), errors="coerce").dropna()

        if not mileage_series.empty:
            mileage_k = mileage_series / 1_000.0
            has_extreme = (mileage_k.max() > 500.0) or (len(mileage_k) >= 10 and mileage_k.max() > mileage_k.quantile(0.95) * 3)

            # Histogram (Ax1)
            if has_extreme:
                cutoff_km = max(float(mileage_k.quantile(0.98)), 300.0)
                norm_mk = mileage_k[mileage_k <= cutoff_km]
                ax1.hist(norm_mk, bins=25, color="#5e35b1", edgecolor="black", alpha=0.75)
                ax1.set_title(
                    f"Odometer Mileage Distribution (<= {cutoff_km:,.0f}k km)\nExtreme Tail Retained in Dataset",
                    fontsize=11,
                    fontweight="bold",
                )
            else:
                ax1.hist(mileage_k, bins=min(25, mileage_k.nunique() or 10), color="#5e35b1", edgecolor="black", alpha=0.75)
                ax1.set_title("Odometer Mileage Distribution", fontsize=11, fontweight="bold")

            ax1.set_xlabel("Mileage ('000 km)", fontsize=10)
            ax1.set_ylabel("Listing Count", fontsize=10)
            ax1.grid(True, linestyle="--", alpha=0.5)

            # Boxplot (Ax2) - Log scale if wide dynamic range / extreme values
            if has_extreme or (mileage_k.max() > 0 and (mileage_k.max() / max(float(mileage_k.quantile(0.25)), 1.0) > 50)):
                # Use log scale with clipped lower bound for zero mileage display
                disp_mk = mileage_k.clip(lower=0.1)
                ax2.boxplot(
                    disp_mk,
                    orientation="vertical",
                    patch_artist=True,
                    boxprops=dict(facecolor="#7e57c2", color="black"),
                    medianprops=dict(color="red", linewidth=2),
                )
                ax2.set_yscale("log")
                ax2.set_title("Mileage Spread & Dispersion (Log Scale)\nAll Observations Retained", fontsize=11, fontweight="bold")
                ax2.set_ylabel("Mileage ('000 km, Log Scale)", fontsize=10)
            else:
                ax2.boxplot(
                    mileage_k,
                    orientation="vertical",
                    patch_artist=True,
                    boxprops=dict(facecolor="#7e57c2", color="black"),
                    medianprops=dict(color="red", linewidth=2),
                )
                ax2.set_title("Mileage Spread & Dispersion", fontsize=11, fontweight="bold")
                ax2.set_ylabel("Mileage ('000 km)", fontsize=10)

            ax2.grid(True, linestyle="--", alpha=0.5)
        else:
            ax1.text(0.5, 0.5, "No Mileage Data Available", ha="center", va="center")
            ax2.text(0.5, 0.5, "No Mileage Data Available", ha="center", va="center")

        plt.tight_layout()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return fig

    @staticmethod
    def plot_fuel_and_transmission_distribution(
        df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Generates bar charts for fuel type and transmission distributions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        fuel_counts = df["fuel_type"].dropna().value_counts() if "fuel_type" in df else pd.Series()
        trans_counts = df["transmission"].dropna().value_counts() if "transmission" in df else pd.Series()

        if not fuel_counts.empty:
            fuel_counts.plot(kind="bar", ax=ax1, color="#00897b", edgecolor="black", alpha=0.85)
            ax1.set_title("Fuel Type Distribution", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Fuel Type", fontsize=10)
            ax1.set_ylabel("Listing Count", fontsize=10)
            ax1.tick_params(axis="x", rotation=30)
            ax1.grid(True, linestyle="--", alpha=0.5)

        if not trans_counts.empty:
            trans_counts.plot(kind="bar", ax=ax2, color="#3949ab", edgecolor="black", alpha=0.85)
            ax2.set_title("Transmission Distribution", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Transmission", fontsize=10)
            ax2.set_ylabel("Listing Count", fontsize=10)
            ax2.tick_params(axis="x", rotation=0)
            ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return fig

    @staticmethod
    def plot_district_distribution(
        df: pd.DataFrame,
        top_n: int = 10,
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Generates geographic listing distribution bar chart for top districts."""
        fig, ax = plt.subplots(figsize=(12, 6))

        valid_dist = df["district"].dropna() if "district" in df else pd.Series()
        valid_dist = valid_dist[valid_dist.astype(str).str.strip() != ""]

        if not valid_dist.empty:
            counts = valid_dist.value_counts().head(top_n)
            counts.plot(kind="bar", ax=ax, color="#d81b60", edgecolor="black", alpha=0.85)
            ax.set_title(f"Observed Listings by District (Top {top_n})", fontsize=13, fontweight="bold")
            ax.set_xlabel("District", fontsize=11)
            ax.set_ylabel("Listing Count", fontsize=11)
            ax.tick_params(axis="x", rotation=30)
            ax.grid(True, linestyle="--", alpha=0.5)
        else:
            ax.text(0.5, 0.5, "No District Data Available", ha="center", va="center")

        plt.tight_layout()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return fig

"""
Relationship and Correlation Analysis Engine.
Analyzes bivariate associations between asking price, mileage, manufacture year,
vehicle age, and engine capacity using Pearson and Spearman rank correlations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CORR_VARS = [
    "asking_price",
    "mileage",
    "manufacture_year",
    "vehicle_age",
    "engine_cc",
]

NON_CAUSALITY_DISCLAIMER = (
    "STATISTICAL NOTICE: Correlation measures observed statistical association and does "
    "not imply causality. External market factors, condition, and optional equipment influence asking prices."
)


class RelationshipAnalyzer:
    """
    Computes parametric and non-parametric correlation matrices, bivariate relationship
    diagnostics, and scatter visualizations.
    """

    @staticmethod
    def compute_correlations(
        df: pd.DataFrame,
        variables: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Computes Pearson (linear) and Spearman (rank/monotonic) correlation matrices.
        Only includes numeric columns with at least 2 valid data points.
        """
        vars_to_use = variables or [v for v in DEFAULT_CORR_VARS if v in df.columns]
        if df.empty or len(vars_to_use) < 2:
            empty_df = pd.DataFrame(index=vars_to_use, columns=vars_to_use)
            return {"pearson": empty_df, "spearman": empty_df}

        numeric_df = df[vars_to_use].apply(pd.to_numeric, errors="coerce")

        pearson_mat = numeric_df.corr(method="pearson").round(4)
        spearman_mat = numeric_df.corr(method="spearman").round(4)

        return {
            "pearson": pearson_mat,
            "spearman": spearman_mat,
        }

    @staticmethod
    def compute_bivariate_relationships(
        df: pd.DataFrame,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyzes core bivariate pairs involving asking price.
        Calculates Pearson and Spearman coefficients and sample sizes.
        """
        pairs = [
            ("asking_price", "mileage", "Asking Price vs Odometer Mileage"),
            ("asking_price", "manufacture_year", "Asking Price vs Manufacture Year (YOM)"),
            ("asking_price", "vehicle_age", "Asking Price vs Vehicle Age"),
            ("asking_price", "engine_cc", "Asking Price vs Engine Capacity (CC)"),
        ]

        results = {}

        for var1, var2, label in pairs:
            if var1 not in df.columns or var2 not in df.columns:
                continue

            valid = df[[var1, var2]].dropna().apply(pd.to_numeric, errors="coerce").dropna()
            n = len(valid)

            if n >= 3:
                p_corr = float(valid[var1].corr(valid[var2], method="pearson"))
                s_corr = float(valid[var1].corr(valid[var2], method="spearman"))
            else:
                p_corr = np.nan
                s_corr = np.nan

            results[f"{var1}_vs_{var2}"] = {
                "label": label,
                "sample_size": n,
                "pearson_r": round(p_corr, 4) if not np.isnan(p_corr) else None,
                "spearman_rho": round(s_corr, 4) if not np.isnan(s_corr) else None,
                "disclaimer": NON_CAUSALITY_DISCLAIMER,
            }

        return results

    @staticmethod
    def compute_category_correlations(
        df: pd.DataFrame,
        min_sample: int = 5,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Computes price correlations stratified by vehicle category.
        Ensures comparisons acknowledge differing category mechanics (e.g. motorbikes vs lorries).
        """
        if df.empty or "canonical_category" not in df.columns:
            return {}

        cat_results = {}
        for cat, group in df.groupby("canonical_category"):
            valid = group[["asking_price", "mileage", "manufacture_year", "vehicle_age", "engine_cc"]].dropna(
                subset=["asking_price"]
            )
            n = len(valid)
            if n < min_sample:
                cat_results[str(cat)] = {
                    "sample_size": n,
                    "status": f"Sample below threshold ({min_sample})",
                }
                continue

            # Mileage corr
            m_valid = valid[["asking_price", "mileage"]].dropna()
            m_rho = (
                float(m_valid["asking_price"].corr(m_valid["mileage"], method="spearman"))
                if len(m_valid) >= 3
                else None
            )

            # Age corr
            a_valid = valid[["asking_price", "vehicle_age"]].dropna()
            a_rho = (
                float(a_valid["asking_price"].corr(a_valid["vehicle_age"], method="spearman"))
                if len(a_valid) >= 3
                else None
            )

            # CC corr
            c_valid = valid[["asking_price", "engine_cc"]].dropna()
            c_rho = (
                float(c_valid["asking_price"].corr(c_valid["engine_cc"], method="spearman"))
                if len(c_valid) >= 3
                else None
            )

            cat_results[str(cat)] = {
                "sample_size": n,
                "spearman_price_mileage": round(m_rho, 4) if m_rho is not None and not np.isnan(m_rho) else None,
                "spearman_price_age": round(a_rho, 4) if a_rho is not None and not np.isnan(a_rho) else None,
                "spearman_price_engine_cc": round(c_rho, 4) if c_rho is not None and not np.isnan(c_rho) else None,
            }

        return cat_results

    # ==========================================================================
    # VISUALIZATION GENERATORS
    # ==========================================================================

    @staticmethod
    def plot_correlation_matrix(
        df: pd.DataFrame,
        method: str = "spearman",
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Renders correlation matrix heatmap using clean Matplotlib styling."""
        fig, ax = plt.subplots(figsize=(8, 7))

        corrs = RelationshipAnalyzer.compute_correlations(df)
        mat = corrs.get(method.lower(), corrs["spearman"])

        if not mat.empty and mat.dropna(how="all").shape[0] > 1:
            cax = ax.matshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
            fig.colorbar(cax)

            ticks = range(len(mat.columns))
            labels = [col.replace("_", " ").title() for col in mat.columns]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(labels, rotation=45, ha="left", fontsize=10)
            ax.set_yticklabels(labels, fontsize=10)

            # Add numerical labels in cells
            for i in range(len(mat.index)):
                for j in range(len(mat.columns)):
                    val = mat.iloc[i, j]
                    txt = f"{val:.2f}" if not pd.isna(val) else "N/A"
                    color = "white" if not pd.isna(val) and abs(val) > 0.5 else "black"
                    ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)

            ax.set_title(
                f"{method.capitalize()} Rank Correlation Matrix\n(Values: -1.0 to +1.0)",
                fontsize=12,
                fontweight="bold",
                pad=20,
            )
        else:
            ax.text(0.5, 0.5, "Insufficient Data for Correlation Matrix", ha="center", va="center")

        plt.tight_layout()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return fig

    @staticmethod
    def plot_price_vs_mileage(
        df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Generates Asking Price vs Mileage scatter plot with category color coding."""
        fig, ax = plt.subplots(figsize=(10, 6))

        valid = df.dropna(subset=["asking_price", "mileage"]).copy()
        if not valid.empty:
            p_m = valid["asking_price"].astype(float) / 1_000_000.0
            m_k = valid["mileage"].astype(float) / 1_000.0

            ax.scatter(m_k, p_m, color="#1976d2", alpha=0.7, edgecolors="k", s=50)
            ax.set_title("Observed Asking Price vs Odometer Mileage", fontsize=12, fontweight="bold")
            ax.set_xlabel("Odometer Mileage ('000 km)", fontsize=10)
            ax.set_ylabel("Asking Price (Million LKR / Rs.)", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)

            # Fit linear trend line if enough points
            if len(valid) >= 5:
                z = np.polyfit(m_k, p_m, 1)
                p = np.poly1d(z)
                x_vals = np.linspace(m_k.min(), m_k.max(), 100)
                ax.plot(x_vals, p(x_vals), color="red", linestyle="--", label="OLS Trend")
                ax.legend()
        else:
            ax.text(0.5, 0.5, "No Price vs Mileage Data Available", ha="center", va="center")

        plt.tight_layout()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return fig

    @staticmethod
    def plot_price_vs_age(
        df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Generates Asking Price vs Vehicle Age scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        valid = df.dropna(subset=["asking_price", "vehicle_age"]).copy()
        if not valid.empty:
            p_m = valid["asking_price"].astype(float) / 1_000_000.0
            age = valid["vehicle_age"].astype(float)

            ax.scatter(age, p_m, color="#388e3c", alpha=0.7, edgecolors="k", s=50)
            ax.set_title("Observed Asking Price vs Vehicle Age", fontsize=12, fontweight="bold")
            ax.set_xlabel("Vehicle Age (Years)", fontsize=10)
            ax.set_ylabel("Asking Price (Million LKR / Rs.)", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)

            if len(valid) >= 5:
                z = np.polyfit(age, p_m, 1)
                p = np.poly1d(z)
                x_vals = np.linspace(age.min(), age.max(), 100)
                ax.plot(x_vals, p(x_vals), color="red", linestyle="--", label="OLS Trend")
                ax.legend()
        else:
            ax.text(0.5, 0.5, "No Price vs Age Data Available", ha="center", va="center")

        plt.tight_layout()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return fig

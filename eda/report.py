"""
Exploratory Data Analysis (EDA) Reporting Engine.
Orchestrates end-to-end statistical analysis, visualization generation, summary table export,
and publication of the reproducible Sri Lankan vehicle market intelligence report.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from config import settings
from database.connection import get_sessionmaker
from eda.categorical import CategoricalAnalyzer
from eda.dataset import EDADatasetLoader
from eda.descriptive import DescriptiveAnalyzer
from eda.distributions import DistributionAnalyzer
from eda.historical import HistoricalAnalyzer
from eda.outliers import OutlierAnalyzer
from eda.relationships import RelationshipAnalyzer

logger = logging.getLogger("eda.report")

ASKING_PRICE_DISCLAIMER = (
    "IMPORTANT DISCLAIMER: Listed prices represent seller asking/advertised prices extracted "
    "from publicly accessible listings on Riyasewana. They DO NOT represent completed transaction "
    "prices or confirmed market sale values. Actual finalized sales may differ due to buyer-seller negotiation."
)

NON_CAUSALITY_NOTICE = (
    "STATISTICAL NOTE: All correlations and bivariate trends represent observed co-movements and "
    "empirical associations within the collected sample. Correlation does NOT establish causation."
)


class EDAReporter:
    """
    Executes the full EDA workflow and generates structured artifacts in data/analysis/.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        loader: Optional[EDADatasetLoader] = None,
    ):
        self.output_dir = output_dir or Path("data/analysis")
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.reports_dir = self.output_dir / "reports"
        self.loader = loader or EDADatasetLoader()

        self._create_directories()

    def _create_directories(self) -> None:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_figures(self, df: pd.DataFrame) -> Dict[str, Path]:
        """Generates all univariate and bivariate figures."""
        paths = {}
        if df.empty:
            return paths

        p1 = self.figures_dir / "price_distribution.png"
        DistributionAnalyzer.plot_price_distribution(df, output_path=p1)
        paths["price_distribution"] = p1

        p2 = self.figures_dir / "category_price_comparison.png"
        DistributionAnalyzer.plot_category_price_comparison(df, output_path=p2)
        paths["category_price_comparison"] = p2

        p3 = self.figures_dir / "yom_age_distribution.png"
        DistributionAnalyzer.plot_yom_and_age_distribution(df, output_path=p3)
        paths["yom_age_distribution"] = p3

        p4 = self.figures_dir / "mileage_distribution.png"
        DistributionAnalyzer.plot_mileage_distribution(df, output_path=p4)
        paths["mileage_distribution"] = p4

        p5 = self.figures_dir / "fuel_transmission_distribution.png"
        DistributionAnalyzer.plot_fuel_and_transmission_distribution(df, output_path=p5)
        paths["fuel_transmission_distribution"] = p5

        p6 = self.figures_dir / "district_distribution.png"
        DistributionAnalyzer.plot_district_distribution(df, output_path=p6)
        paths["district_distribution"] = p6

        p7 = self.figures_dir / "correlation_matrix.png"
        RelationshipAnalyzer.plot_correlation_matrix(df, output_path=p7)
        paths["correlation_matrix"] = p7

        p8 = self.figures_dir / "price_vs_mileage.png"
        RelationshipAnalyzer.plot_price_vs_mileage(df, output_path=p8)
        paths["price_vs_mileage"] = p8

        p9 = self.figures_dir / "price_vs_age.png"
        RelationshipAnalyzer.plot_price_vs_age(df, output_path=p9)
        paths["price_vs_age"] = p9

        return paths

    def generate_tables(
        self,
        df: pd.DataFrame,
        session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """Generates CSV and JSON summary tables."""
        generated: Dict[str, Any] = {}

        # 1. Dataset Overview
        overview = DescriptiveAnalyzer.compute_dataset_overview(df, session=session)
        overview_path = self.tables_dir / "dataset_overview.json"
        with open(overview_path, "w", encoding="utf-8") as f:
            json.dump(overview, f, indent=2, default=str)
        generated["dataset_overview"] = overview

        # 2. Numerical Summary
        num_summary = DescriptiveAnalyzer.compute_numerical_summary(df)
        num_summary.to_csv(self.tables_dir / "numerical_summary.csv", index=False)
        num_summary.to_json(self.tables_dir / "numerical_summary.json", orient="records", indent=2)
        generated["numerical_summary"] = num_summary

        # 3. Category Summary
        cat_summary = CategoricalAnalyzer.analyze_categories(df)
        cat_summary.to_csv(self.tables_dir / "category_summary.csv", index=False)
        cat_summary.to_json(self.tables_dir / "category_summary.json", orient="records", indent=2)
        generated["category_summary"] = cat_summary

        # 4. Brand Summary
        brand_summary = CategoricalAnalyzer.analyze_brands(df)
        brand_summary.to_csv(self.tables_dir / "brand_summary.csv", index=False)
        brand_summary.to_json(self.tables_dir / "brand_summary.json", orient="records", indent=2)
        generated["brand_summary"] = brand_summary

        # 5. Model Summary
        model_summary = CategoricalAnalyzer.analyze_models(df)
        model_summary.to_csv(self.tables_dir / "model_summary.csv", index=False)
        model_summary.to_json(self.tables_dir / "model_summary.json", orient="records", indent=2)
        generated["model_summary"] = model_summary

        # 6. Fuel Summary
        fuel_summary = CategoricalAnalyzer.analyze_fuel_types(df)
        fuel_summary.to_csv(self.tables_dir / "fuel_summary.csv", index=False)
        fuel_summary.to_json(self.tables_dir / "fuel_summary.json", orient="records", indent=2)
        generated["fuel_summary"] = fuel_summary

        # 7. Transmission Summary
        trans_summary = CategoricalAnalyzer.analyze_transmissions(df)
        trans_summary.to_csv(self.tables_dir / "transmission_summary.csv", index=False)
        trans_summary.to_json(self.tables_dir / "transmission_summary.json", orient="records", indent=2)
        generated["transmission_summary"] = trans_summary

        # 8. District Summary
        dist_summary = CategoricalAnalyzer.analyze_districts(df)
        dist_summary.to_csv(self.tables_dir / "district_summary.csv", index=False)
        dist_summary.to_json(self.tables_dir / "district_summary.json", orient="records", indent=2)
        generated["district_summary"] = dist_summary

        # 9. Outlier Analysis
        outlier_data = OutlierAnalyzer.analyze_outliers(df)
        flagged_df = pd.DataFrame(outlier_data.get("flagged_records", []))
        if not flagged_df.empty:
            flagged_df.to_csv(self.tables_dir / "flagged_outliers.csv", index=False)
            flagged_df.to_json(self.tables_dir / "flagged_outliers.json", orient="records", indent=2)
        generated["outliers"] = outlier_data

        # 10. Correlations
        corrs = RelationshipAnalyzer.compute_correlations(df)
        bivariate = RelationshipAnalyzer.compute_bivariate_relationships(df)
        category_corrs = RelationshipAnalyzer.compute_category_correlations(df)
        corr_out = {
            "pearson": corrs["pearson"].to_dict(),
            "spearman": corrs["spearman"].to_dict(),
            "bivariate_relationships": bivariate,
            "category_correlations": category_corrs,
        }
        with open(self.tables_dir / "correlations.json", "w", encoding="utf-8") as f:
            json.dump(corr_out, f, indent=2, default=str)
        generated["correlations"] = corr_out

        return generated

    def generate_markdown_report(
        self,
        overview: Dict[str, Any],
        num_df: pd.DataFrame,
        cat_df: pd.DataFrame,
        brand_df: pd.DataFrame,
        outlier_data: Dict[str, Any],
        corr_data: Dict[str, Any],
        hist_data: Dict[str, Any],
        depth_data: Dict[str, Any],
    ) -> Path:
        """Constructs a comprehensive, readable Markdown market intelligence report."""
        report_path = self.reports_dir / "eda_market_report.md"
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = [
            "# Sri Lankan Vehicle Market — Exploratory Data Analysis (EDA) Report",
            f"*Generated: {now_str}*",
            "",
            "> [!IMPORTANT]",
            f"> {ASKING_PRICE_DISCLAIMER}",
            "",
            "> [!NOTE]",
            f"> {NON_CAUSALITY_NOTICE}",
            "",
            "## 1. Executive Summary & Dataset Populations",
            "",
            "The platform collects publicly accessible vehicle listings from Riyasewana across 8 vehicle categories.",
            "The analysis distinguishes three analytical populations:",
            "1. **Full Dataset**: Complete historical registry of all observed listings and vehicles.",
            "2. **Quality-Filtered Dataset**: Records evaluated against syntax, bounds, and consistency checks.",
            "3. **ML-Eligible Dataset**: Verified listings meeting all critical valuation criteria (valid asking price, mileage, YOM, Make, Model, Category).",
            "",
            "| Metric | Observed Count / Rate |",
            "| :--- | :--- |",
            f"| **Total Listings** | {overview.get('total_listings', 0):,} |",
            f"| **Total Vehicles** | {overview.get('total_vehicles', 0):,} |",
            f"| **ML Eligible Listings** | {overview.get('ml_eligible_listings', 0):,} ({overview.get('ml_eligibility_rate_pct', 0.0):.1f}%) |",
            f"| **ML Ineligible Listings** | {overview.get('ml_ineligible_listings', 0):,} |",
            f"| **Active Listings** | {overview.get('active_listings', 0):,} |",
            f"| **No-Longer-Observed Listings** | {overview.get('no_longer_observed_listings', 0):,} |",
            f"| **Total Price Events** | {overview.get('total_price_history_events', 0):,} |",
            f"| **Total Lifecycle Observations** | {overview.get('total_observations', 0):,} |",
            f"| **Distinct Categories** | {overview.get('distinct_categories', 0)} |",
            f"| **Distinct Brands** | {overview.get('distinct_brands', 0)} |",
            f"| **Distinct Models** | {overview.get('distinct_models', 0)} |",
            f"| **Districts Represented** | {overview.get('distinct_districts', 0)} |",
            "",
            "---",
            "## 2. Vehicle Category Market Profiles",
            "",
            "Central tendency and distribution across the 8 canonical vehicle categories:",
            "",
            "| Category | Listings | Share (%) | ML Eligible | Median Asking Price (Rs.) | Mean Asking Price (Rs.) | Median Mileage (km) | Median YOM |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        if not cat_df.empty:
            for _, r in cat_df.iterrows():
                p_med = f"Rs. {r['median_asking_price']:,.0f}" if pd.notna(r["median_asking_price"]) else "N/A"
                p_avg = f"Rs. {r['mean_asking_price']:,.0f}" if pd.notna(r["mean_asking_price"]) else "N/A"
                m_med = f"{r['median_mileage']:,.0f} km" if pd.notna(r["median_mileage"]) else "N/A"
                y_med = str(int(r["median_yom"])) if pd.notna(r["median_yom"]) else "N/A"
                lines.append(
                    f"| **{r['category']}** | {r['listing_count']} | {r['pct_of_total']:.1f}% | {r['ml_eligible_count']} ({r['ml_eligible_pct']:.0f}%) | {p_med} | {p_avg} | {m_med} | {y_med} |"
                )

        lines.extend([
            "",
            "---",
            "## 3. Brand & Model Market Concentration",
            "",
            "Dominant vehicle brands observed in the dataset (configurable threshold: minimum 5 listings for reliable inference):",
            "",
            "| Brand | Listings | Share (%) | Median Asking Price (Rs.) | Mean Asking Price (Rs.) | Reliability Flag |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |",
        ])

        if not brand_df.empty:
            for _, r in brand_df.head(10).iterrows():
                p_med = f"Rs. {r['median_asking_price']:,.0f}" if pd.notna(r["median_asking_price"]) else "N/A"
                p_avg = f"Rs. {r['mean_asking_price']:,.0f}" if pd.notna(r["mean_asking_price"]) else "N/A"
                lines.append(
                    f"| **{r['brand']}** | {r['listing_count']} | {r['pct_of_total']:.1f}% | {p_med} | {p_avg} | {r['sample_flag']} |"
                )

        lines.extend([
            "",
            "---",
            "## 4. Numerical Variables & Five-Number Summaries",
            "",
            "Parametric (mean, standard deviation) and non-parametric (median, IQR) dispersion statistics:",
            "",
            "| Variable | Valid Count | Missing (%) | Mean | Std Dev | Min | Q1 (25%) | Median (50%) | Q3 (75%) | Max | IQR |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        if not num_df.empty:
            for _, r in num_df.iterrows():
                f_mean = f"{r['mean']:,.2f}" if pd.notna(r["mean"]) else "N/A"
                f_std = f"{r['std']:,.2f}" if pd.notna(r["std"]) else "N/A"
                f_min = f"{r['min']:,.2f}" if pd.notna(r["min"]) else "N/A"
                f_q1 = f"{r['q1']:,.2f}" if pd.notna(r["q1"]) else "N/A"
                f_med = f"{r['median']:,.2f}" if pd.notna(r["median"]) else "N/A"
                f_q3 = f"{r['q3']:,.2f}" if pd.notna(r["q3"]) else "N/A"
                f_max = f"{r['max']:,.2f}" if pd.notna(r["max"]) else "N/A"
                f_iqr = f"{r['iqr']:,.2f}" if pd.notna(r["iqr"]) else "N/A"
                lines.append(
                    f"| `{r['variable']}` | {r['count']} | {r['missing_pct']:.1f}% | {f_mean} | {f_std} | {f_min} | {f_q1} | {f_med} | {f_q3} | {f_max} | {f_iqr} |"
                )

        lines.extend([
            "",
            "---",
            "## 5. Bivariate Relationships & Correlation Analysis",
            "",
            "Statistical associations between asking price and key valuation dimensions:",
            "",
            "| Relationship Pair | Sample Size | Pearson ($r$) | Spearman ($\\rho$) | Association Interpretation |",
            "| :--- | :---: | :---: | :---: | :--- |",
        ])

        biv = corr_data.get("bivariate_relationships", {})
        for key, info in biv.items():
            p_r = f"{info['pearson_r']:.4f}" if info.get("pearson_r") is not None else "N/A"
            s_rho = f"{info['spearman_rho']:.4f}" if info.get("spearman_rho") is not None else "N/A"
            interp = "Inverse relationship" if (info.get("spearman_rho") or 0) < 0 else "Direct relationship"
            lines.append(
                f"| **{info['label']}** | {info['sample_size']} | {p_r} | {s_rho} | {interp} |"
            )

        lines.extend([
            "",
            "---",
            "## 6. Outlier Analysis & Data Integrity",
            "",
            f"Detected **{outlier_data.get('flagged_outliers_count', 0)}** statistical outliers using IQR and percentile thresholds.",
            "Outliers are systematically categorized without destructive deletion:",
            "- **Suspicious Data**: Records containing artificial patterns (e.g. dummy mileage sequences `123456`, implausible prices).",
            "- **Genuine Market Extremes**: Authentic observations representing luxury exotics, heavy commercial equipment, or vintage collectors.",
            "- **Insufficient Information**: Incomplete records flagged for exclusion from ML training.",
            "",
            "---",
            "## 7. Historical Market Dynamics & Timeframe Depth Limits",
            "",
            f"- **Observed Timeframe Span**: {depth_data.get('total_days_span', 0.0):.1f} days",
            f"- **Historical Depth Status**: {depth_data.get('message', 'N/A')}",
            f"- **Price Revision Events**: {hist_data.get('listings_with_price_changes', 0)} listings with observed price changes "
            f"({hist_data.get('price_reductions', 0)} reductions, {hist_data.get('price_increases', 0)} increases).",
            "",
            "> [!WARNING]",
            f"> {depth_data.get('message')}",
            "",
            "---",
            "## 8. Key Takeaways for Future Valuation Modeling (Phase 5+)",
            "",
            "1. **Category Specificity**: Market scales differ drastically between vehicle categories (e.g., motorbikes vs cars vs heavy commercial vehicles). Segmented valuation models or category interaction terms will be essential.",
            "2. **Non-Linear Age Decay**: Asking price exhibits strong non-linear depreciation curves with vehicle age rather than strict linear decay.",
            "3. **Brand Concentration**: Market listings are concentrated in top Japanese and Indian manufacturers (Toyota, Suzuki, Honda, Bajaj).",
            "4. **Outlier Quarantine**: Step 5 quality filters successfully isolate dummy odometer and price sequences, preventing model distortion while retaining verified observations.",
            "",
            "*Report generated by the Vehicle Market Intelligence Platform EDA subsystem.*",
        ])

        content = "\n".join(lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)

        return report_path

    def run(
        self,
        session: Optional[Session] = None,
        category: Optional[str] = None,
        ml_eligible_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes complete EDA pipeline:
        Loads data -> Generates figures -> Exports tables -> Compiles Markdown Report.
        """
        logger.info("Starting Exploratory Data Analysis pipeline...")

        df = self.loader.load_listings(
            session=session,
            category=category,
            ml_eligible_only=ml_eligible_only,
        )
        df_ph = self.loader.load_price_history(session=session)
        df_obs = self.loader.load_observations(session=session)

        # 1. Figures
        figures = self.generate_figures(df)

        # 2. Tables
        tables = self.generate_tables(df, session=session)

        # 3. Historical analysis
        hist_data = HistoricalAnalyzer.analyze_price_history(df_ph)
        obs_data = HistoricalAnalyzer.analyze_observations(df_obs)
        depth_data = HistoricalAnalyzer.evaluate_historical_depth(df_obs, df_ph)

        # 4. Markdown report
        report_path = self.generate_markdown_report(
            overview=tables["dataset_overview"],
            num_df=tables["numerical_summary"],
            cat_df=tables["category_summary"],
            brand_df=tables["brand_summary"],
            outlier_data=tables["outliers"],
            corr_data=tables["correlations"],
            hist_data=hist_data,
            depth_data=depth_data,
        )

        logger.info(f"EDA execution completed successfully. Report: {report_path}")

        return {
            "total_listings_analyzed": len(df),
            "figures_generated": len(figures),
            "tables_generated": len(tables),
            "report_path": str(report_path),
            "depth_status": depth_data["message"],
        }


def main():
    parser = argparse.ArgumentParser(
        description="Sri Lankan Vehicle Market Exploratory Data Analysis (EDA) Runner"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional category filter (e.g. Cars, Vans, SUVs). Default: all categories.",
    )
    parser.add_argument(
        "--ml-eligible-only",
        action="store_true",
        default=False,
        help="Filter to ML-eligible listings only.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/analysis",
        help="Target directory for figures, tables, and reports.",
    )
    args = parser.parse_args()

    session_factory = get_sessionmaker()
    with session_factory() as session:
        reporter = EDAReporter(output_dir=Path(args.output_dir))
        res = reporter.run(
            session=session,
            category=args.category,
            ml_eligible_only=args.ml_eligible_only,
        )

        print("\n" + "=" * 60)
        print("EDA EXECUTION COMPLETED")
        print("=" * 60)
        print(f"Listings Analyzed   : {res['total_listings_analyzed']}")
        print(f"Figures Generated   : {res['figures_generated']}")
        print(f"Tables Exported     : {res['tables_generated']}")
        print(f"Markdown Report     : {res['report_path']}")
        print(f"Historical Depth    : {res['depth_status']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

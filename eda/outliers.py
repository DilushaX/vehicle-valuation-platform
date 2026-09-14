"""
Outlier Analysis Engine.
Identifies and classifies statistical outliers in asking price, mileage, and engine capacity
using defensible IQR and percentile methods without mutating or deleting dataset records.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class OutlierAnalyzer:
    """
    Detects and classifies numerical outliers into genuine observations, suspicious entries,
    or insufficient information, integrating with Step 5 validation issues.
    """

    @staticmethod
    def detect_iqr_outliers(
        series: pd.Series,
        multiplier: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Calculates Tukey's IQR bounds (Q1 - 1.5*IQR, Q3 + 1.5*IQR).
        Returns lower/upper bounds, outlier counts, and indices.
        """
        s_clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(s_clean) < 4:
            return {
                "count": len(s_clean),
                "q1": None,
                "q3": None,
                "iqr": None,
                "lower_bound": None,
                "upper_bound": None,
                "low_outlier_count": 0,
                "high_outlier_count": 0,
                "total_outliers": 0,
            }

        q1 = float(s_clean.quantile(0.25))
        q3 = float(s_clean.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - (multiplier * iqr)
        upper_bound = q3 + (multiplier * iqr)

        low_mask = s_clean < lower_bound
        high_mask = s_clean > upper_bound

        return {
            "count": len(s_clean),
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "low_outlier_count": int(low_mask.sum()),
            "high_outlier_count": int(high_mask.sum()),
            "total_outliers": int(low_mask.sum() + high_mask.sum()),
            "low_indices": list(s_clean[low_mask].index),
            "high_indices": list(s_clean[high_mask].index),
        }

    @staticmethod
    def detect_percentile_outliers(
        series: pd.Series,
        low_p: float = 0.01,
        high_p: float = 0.99,
    ) -> Dict[str, Any]:
        """Calculates percentile boundaries (default: 1st and 99th percentiles)."""
        s_clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(s_clean) < 10:
            return {
                "count": len(s_clean),
                "low_threshold": None,
                "high_threshold": None,
                "total_outliers": 0,
            }

        low_thresh = float(s_clean.quantile(low_p))
        high_thresh = float(s_clean.quantile(high_p))

        low_mask = s_clean < low_thresh
        high_mask = s_clean > high_thresh

        return {
            "count": len(s_clean),
            "low_threshold": round(low_thresh, 2),
            "high_threshold": round(high_thresh, 2),
            "low_outlier_count": int(low_mask.sum()),
            "high_outlier_count": int(high_mask.sum()),
            "total_outliers": int(low_mask.sum() + high_mask.sum()),
            "low_indices": list(s_clean[low_mask].index),
            "high_indices": list(s_clean[high_mask].index),
        }

    @classmethod
    def analyze_outliers(
        cls,
        df: pd.DataFrame,
        variables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes numerical variables for outliers and classifies each outlier into:
        - 'suspicious_data' (if flagged by validation engine with dummy/invalid issue)
        - 'possible_genuine_market_observation' (valid record representing genuine market tail)
        - 'insufficient_information' (unverified or boundary anomaly)
        """
        vars_to_check = variables or ["asking_price", "mileage", "engine_cc"]
        analysis_by_var = {}
        flagged_records: List[Dict[str, Any]] = []

        for var in vars_to_check:
            if var not in df.columns:
                continue

            series = df[var]
            iqr_res = cls.detect_iqr_outliers(series)
            pct_res = cls.detect_percentile_outliers(series)

            analysis_by_var[var] = {
                "iqr_bounds": {
                    "lower": iqr_res["lower_bound"],
                    "upper": iqr_res["upper_bound"],
                    "iqr": iqr_res["iqr"],
                    "outliers_count": iqr_res["total_outliers"],
                },
                "percentile_bounds": {
                    "p01": pct_res["low_threshold"],
                    "p99": pct_res["high_threshold"],
                    "outliers_count": pct_res["total_outliers"],
                },
            }

            # Classify outlier records
            all_outlier_indices = set(iqr_res.get("low_indices", []) + iqr_res.get("high_indices", []))

            for idx in all_outlier_indices:
                row = df.loc[idx]
                val = row[var]
                issues = row.get("validation_issues", [])
                if not isinstance(issues, list):
                    issues = []

                # Classification logic
                has_var_issue = any(var in str(iss).lower() for iss in issues)
                has_critical_issue = any(
                    str(iss) in {"suspicious_price_pattern", "suspicious_mileage_pattern", "invalid_price", "invalid_mileage"}
                    for iss in issues
                )

                if has_critical_issue or has_var_issue:
                    classification = "suspicious_data"
                elif bool(row.get("ml_eligible")) is True:
                    classification = "possible_genuine_market_observation"
                else:
                    classification = "insufficient_information"

                outlier_type = "high_outlier" if val > (iqr_res["upper_bound"] or 0) else "low_outlier"

                flagged_records.append(
                    {
                        "listing_id": row.get("listing_id"),
                        "category": row.get("canonical_category", row.get("category")),
                        "brand": row.get("brand"),
                        "model": row.get("model"),
                        "variable": var,
                        "value": val,
                        "outlier_type": outlier_type,
                        "classification": classification,
                        "validation_issues": issues,
                    }
                )

        return {
            "summary_by_variable": analysis_by_var,
            "flagged_outliers_count": len(flagged_records),
            "flagged_records": flagged_records,
        }

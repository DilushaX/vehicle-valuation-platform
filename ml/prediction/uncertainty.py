"""
Model-Based Valuation Prediction Range and Uncertainty Estimation.

Calculates indicative prediction intervals reflecting model disagreement
across individual decision trees in a Random Forest ensemble or residual variance.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


@dataclass
class ValuationRange:
    """
    Structured model-based valuation range.
    
    Note:
    This is an indicative prediction range representing tree dispersion
    or model uncertainty. It is NOT a guaranteed statistical confidence interval.
    """
    estimate: float
    lower_bound: float
    upper_bound: float
    spread: float
    percentile_lower: int
    percentile_upper: int
    method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimate": round(float(self.estimate), 2),
            "lower": round(float(self.lower_bound), 2),
            "upper": round(float(self.upper_bound), 2),
            "spread": round(float(self.spread), 2),
            "percentile_lower": self.percentile_lower,
            "percentile_upper": self.percentile_upper,
            "method": self.method,
        }

    def format_summary(self) -> str:
        return (
            f"Estimated Asking Price: Rs. {self.estimate:,.0f} "
            f"(Indicative Range: Rs. {self.lower_bound:,.0f} – Rs. {self.upper_bound:,.0f})"
        )


class UncertaintyEstimator:
    """
    Estimates prediction uncertainty and constructs model-based valuation ranges.
    """

    def __init__(
        self,
        model: Any,
        metadata: Optional[Dict[str, Any]] = None,
        default_percentile_lower: int = 10,
        default_percentile_upper: int = 90,
    ):
        self.model = model
        self.metadata = metadata or {}
        self.default_percentile_lower = default_percentile_lower
        self.default_percentile_upper = default_percentile_upper
        self._underlying_rf: Optional[RandomForestRegressor] = None
        self._preprocessor = None
        self._is_log_target = True
        self._init_components()

    def _init_components(self) -> None:
        """Extracts pipeline preprocessor and regressor components."""
        pipeline = self.model
        if isinstance(self.model, TransformedTargetRegressor):
            pipeline = self.model.regressor_
            self._is_log_target = getattr(self.model, "func", None) in [np.log1p, np.log]

        if hasattr(pipeline, "named_steps"):
            self._preprocessor = pipeline.named_steps.get("preprocessor")
            regressor = pipeline.named_steps.get("regressor")
            if isinstance(regressor, RandomForestRegressor):
                self._underlying_rf = regressor
        elif hasattr(pipeline, "steps"):
            self._preprocessor = pipeline.steps[0][1]
            regressor = pipeline.steps[-1][1]
            if isinstance(regressor, RandomForestRegressor):
                self._underlying_rf = regressor

    def estimate_range(
        self,
        formatted_df: pd.DataFrame,
        point_estimate: float,
        percentile_lower: Optional[int] = None,
        percentile_upper: Optional[int] = None,
    ) -> ValuationRange:
        """
        Calculates indicative lower and upper bounds for a formatted vehicle listing.
        
        Args:
            formatted_df: 1-row DataFrame pre-validated by predictor.
            point_estimate: Point asking price estimate in LKR.
            percentile_lower: Lower percentile (default: 10).
            percentile_upper: Upper percentile (default: 90).
            
        Returns:
            ValuationRange instance with lower, upper, and central estimate.
        """
        p_low = percentile_lower if percentile_lower is not None else self.default_percentile_lower
        p_high = percentile_upper if percentile_upper is not None else self.default_percentile_upper

        if p_low >= p_high:
            raise ValueError(f"percentile_lower ({p_low}) must be strictly less than percentile_upper ({p_high}).")

        # Approach 1: Random Forest tree-dispersion (preferred)
        if self._underlying_rf is not None and self._preprocessor is not None:
            return self._range_via_rf_trees(formatted_df, point_estimate, p_low, p_high)
        else:
            return self._range_via_metadata_rmse(point_estimate, p_low, p_high)

    def _range_via_rf_trees(
        self,
        formatted_df: pd.DataFrame,
        point_estimate: float,
        p_low: int,
        p_high: int,
    ) -> ValuationRange:
        """Computes quantiles across all decision trees in the Random Forest."""
        X_trans = self._preprocessor.transform(formatted_df.iloc[[0]])
        tree_preds_raw = np.array([tree.predict(X_trans)[0] for tree in self._underlying_rf.estimators_])

        if self._is_log_target:
            tree_preds_lkr = np.expm1(tree_preds_raw)
        else:
            tree_preds_lkr = tree_preds_raw

        # Compute empirical percentiles across tree predictions
        lower = float(np.percentile(tree_preds_lkr, p_low))
        upper = float(np.percentile(tree_preds_lkr, p_high))

        # Enforce mathematical ordering: lower <= point_estimate <= upper
        lower = min(lower, point_estimate)
        upper = max(upper, point_estimate)

        # Enforce physical positivity constraint (asking price >= 10,000 LKR)
        lower = max(lower, 10_000.0)

        spread = upper - lower

        return ValuationRange(
            estimate=point_estimate,
            lower_bound=lower,
            upper_bound=upper,
            spread=spread,
            percentile_lower=p_low,
            percentile_upper=p_high,
            method=f"RandomForest {len(self._underlying_rf.estimators_)}-tree empirical dispersion ({p_low}th–{p_high}th percentiles)",
        )

    def _range_via_metadata_rmse(
        self,
        point_estimate: float,
        p_low: int,
        p_high: int,
    ) -> ValuationRange:
        """Fallback uncertainty estimation using reported test RMSE from metadata."""
        test_metrics = self.metadata.get("test_metrics", {})
        rmse = float(test_metrics.get("rmse", 1_500_000.0))

        # Approximate bounds using normal multiplier for given percentiles
        z_multiplier = 1.28  # ~80% coverage (10th to 90th)
        lower = max(point_estimate - z_multiplier * rmse, 10_000.0)
        upper = max(point_estimate + z_multiplier * rmse, point_estimate)

        # Enforce ordering
        lower = min(lower, point_estimate)
        spread = upper - lower

        return ValuationRange(
            estimate=point_estimate,
            lower_bound=lower,
            upper_bound=upper,
            spread=spread,
            percentile_lower=p_low,
            percentile_upper=p_high,
            method=f"Normal approximation using test RMSE (LKR {rmse:,.0f})",
        )

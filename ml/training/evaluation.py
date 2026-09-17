"""
Evaluation Metrics, Cross-Validation Scoring, and Error Analysis.
Provides rigorous evaluation of vehicle asking-price regression models
strictly on the original Sri Lankan Rupee (LKR) scale.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)

logger = logging.getLogger(__name__)


@dataclass
class RegressionMetrics:
    """
    Evaluation metrics computed on the original asking price scale (LKR).
    """
    mae: float
    rmse: float
    r2: float
    medae: float
    mape: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        d = {
            "mae": float(self.mae),
            "rmse": float(self.rmse),
            "r2": float(self.r2),
            "medae": float(self.medae),
        }
        if self.mape is not None:
            d["mape"] = float(self.mape)
        return d

    def format_summary(self) -> str:
        """Returns human-readable formatted summary string."""
        return (
            f"MAE: LKR {self.mae:,.0f} | "
            f"RMSE: LKR {self.rmse:,.0f} | "
            f"R²: {self.r2:.4f} | "
            f"MedAE: LKR {self.medae:,.0f}"
        )


def evaluate_predictions(
    y_true: Union[pd.Series, np.ndarray, Sequence[float]],
    y_pred: Union[pd.Series, np.ndarray, Sequence[float]],
) -> RegressionMetrics:
    """
    Computes regression evaluation metrics between actual and predicted asking prices.
    
    IMPORTANT:
    y_true and y_pred MUST be on the original LKR scale (not log-transformed).
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true has {len(yt)} items, y_pred has {len(yp)} items.")
    if len(yt) == 0:
        raise ValueError("Cannot evaluate empty predictions array.")
    if np.any(np.isnan(yt)) or np.any(np.isnan(yp)):
        raise ValueError("y_true or y_pred contains NaN values.")

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(root_mean_squared_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    medae = float(median_absolute_error(yt, yp))

    # Calculate MAPE safely (avoiding zero division)
    valid_mask = yt > 0
    if valid_mask.any():
        mape = float(np.mean(np.abs((yt[valid_mask] - yp[valid_mask]) / yt[valid_mask])) * 100.0)
    else:
        mape = None

    return RegressionMetrics(
        mae=mae,
        rmse=rmse,
        r2=r2,
        medae=medae,
        mape=mape,
    )


@dataclass
class CrossValidationResult:
    """
    Summary of k-fold cross-validation performance across folds.
    """
    model_name: str
    target_transform: str
    cv_mae_mean: float
    cv_mae_std: float
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_r2_mean: float
    cv_r2_std: float
    cv_medae_mean: float
    cv_medae_std: float
    fold_metrics: List[RegressionMetrics]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "target_transform": self.target_transform,
            "cv_mae_mean": self.cv_mae_mean,
            "cv_mae_std": self.cv_mae_std,
            "cv_rmse_mean": self.cv_rmse_mean,
            "cv_rmse_std": self.cv_rmse_std,
            "cv_r2_mean": self.cv_r2_mean,
            "cv_r2_std": self.cv_r2_std,
            "cv_medae_mean": self.cv_medae_mean,
            "cv_medae_std": self.cv_medae_std,
            "num_folds": len(self.fold_metrics),
        }


class ErrorAnalyzer:
    """
    Performs detailed error analysis on regression predictions without
    exposing seller private contact information.
    """

    def __init__(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame] = None,
    ):
        self.y_true = np.asarray(y_true, dtype=np.float64)
        self.y_pred = np.asarray(y_pred, dtype=np.float64)
        self.X = X.reset_index(drop=True)
        self.meta = meta.reset_index(drop=True) if meta is not None else None

        abs_error = np.abs(self.y_true - self.y_pred)
        pct_error = np.where(self.y_true > 0, (abs_error / self.y_true) * 100.0, np.nan)
        residual = self.y_pred - self.y_true

        df = pd.DataFrame({
            "actual_price": self.y_true,
            "predicted_price": self.y_pred,
            "residual": residual,
            "absolute_error": abs_error,
            "percentage_error": pct_error,
        })

        if self.meta is not None and "listing_id" in self.meta.columns:
            df.insert(0, "listing_id", self.meta["listing_id"])

        for col in ["category", "brand", "model", "vehicle_age", "mileage"]:
            if col in self.X.columns:
                df[col] = self.X[col]

        self.error_df = df

    def get_largest_absolute_errors(self, n: int = 5) -> pd.DataFrame:
        """Returns records with largest absolute prediction errors."""
        return self.error_df.sort_values(by="absolute_error", ascending=False).head(n)

    def get_largest_percentage_errors(self, n: int = 5) -> pd.DataFrame:
        """Returns records with largest percentage prediction errors."""
        return self.error_df.sort_values(by="percentage_error", ascending=False).head(n)

    def summarize_by_category(self) -> pd.DataFrame:
        """Computes error aggregates by category."""
        if "category" not in self.error_df.columns:
            return pd.DataFrame()
        return (
            self.error_df.groupby("category")
            .agg(
                count=("actual_price", "count"),
                median_actual=("actual_price", "median"),
                median_predicted=("predicted_price", "median"),
                mean_mae=("absolute_error", "mean"),
                median_mae=("absolute_error", "median"),
                median_pct_error=("percentage_error", "median"),
            )
            .reset_index()
            .sort_values(by="count", ascending=False)
        )

    def summarize_by_price_tier(self) -> pd.DataFrame:
        """Computes error aggregates across price brackets."""
        df = self.error_df.copy()
        bins = [0, 2_000_000, 5_000_000, 10_000_000, 25_000_000, float("inf")]
        labels = ["< 2M", "2M - 5M", "5M - 10M", "10M - 25M", "> 25M"]
        df["price_tier"] = pd.cut(df["actual_price"], bins=bins, labels=labels, right=False)
        return (
            df.groupby("price_tier", observed=False)
            .agg(
                count=("actual_price", "count"),
                mean_mae=("absolute_error", "mean"),
                median_mae=("absolute_error", "median"),
                median_pct_error=("percentage_error", "median"),
            )
            .reset_index()
        )

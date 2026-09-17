"""
Baseline Regression Models for Vehicle Valuation.
Provides Mean and Median baselines that predict constant training-set summary statistics
without using vehicle features, establishing performance benchmarks.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class MeanBaselineRegressor(BaseEstimator, RegressorMixin):
    """
    Predicts the mean of the training target variable for every sample.
    Does not use input features X.
    """

    def __init__(self):
        self.mean_value_: Optional[float] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Fits baseline by computing training target mean."""
        y_arr = np.asarray(y, dtype=np.float64)
        if len(y_arr) == 0:
            raise ValueError("Cannot fit baseline on empty target.")
        self.mean_value_ = float(np.mean(y_arr))
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts constant training mean for all samples in X."""
        if self.mean_value_ is None:
            raise RuntimeError("Baseline regressor is not fitted.")
        n_samples = len(X)
        return np.full(shape=(n_samples,), fill_value=self.mean_value_, dtype=np.float64)


class MedianBaselineRegressor(BaseEstimator, RegressorMixin):
    """
    Predicts the median of the training target variable for every sample.
    Does not use input features X.
    """

    def __init__(self):
        self.median_value_: Optional[float] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Fits baseline by computing training target median."""
        y_arr = np.asarray(y, dtype=np.float64)
        if len(y_arr) == 0:
            raise ValueError("Cannot fit baseline on empty target.")
        self.median_value_ = float(np.median(y_arr))
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts constant training median for all samples in X."""
        if self.median_value_ is None:
            raise RuntimeError("Baseline regressor is not fitted.")
        n_samples = len(X)
        return np.full(shape=(n_samples,), fill_value=self.median_value_, dtype=np.float64)

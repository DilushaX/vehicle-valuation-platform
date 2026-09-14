"""
Exploratory Data Analysis (EDA) Package for Sri Lankan Vehicle Valuation Platform.
Provides modular, read-only analytical tools for understanding market distributions,
relationships, outliers, and category trends without modifying database records.
"""

from eda.dataset import EDADatasetLoader

__all__ = ["EDADatasetLoader"]

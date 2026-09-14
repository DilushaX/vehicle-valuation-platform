"""
Exploratory Data Analysis (EDA) Package for Sri Lankan Vehicle Valuation Platform.
Provides modular, read-only analytical tools for understanding market distributions,
relationships, outliers, and category trends without modifying database records.
"""

from eda.categorical import CategoricalAnalyzer
from eda.dataset import EDADatasetLoader
from eda.descriptive import DescriptiveAnalyzer

__all__ = [
    "EDADatasetLoader",
    "DescriptiveAnalyzer",
    "CategoricalAnalyzer",
]

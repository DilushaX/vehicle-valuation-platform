"""
Exploratory Data Analysis (EDA) Package for Sri Lankan Vehicle Valuation Platform.
Provides modular, read-only analytical tools for understanding market distributions,
relationships, outliers, and category trends without modifying database records.
"""

from eda.categorical import CategoricalAnalyzer
from eda.dataset import EDADatasetLoader
from eda.descriptive import DescriptiveAnalyzer
from eda.distributions import DistributionAnalyzer
from eda.historical import HistoricalAnalyzer
from eda.outliers import OutlierAnalyzer
from eda.relationships import RelationshipAnalyzer

__all__ = [
    "EDADatasetLoader",
    "DescriptiveAnalyzer",
    "CategoricalAnalyzer",
    "DistributionAnalyzer",
    "RelationshipAnalyzer",
    "OutlierAnalyzer",
    "HistoricalAnalyzer",
    "EDAReporter",
]


def __getattr__(name: str):
    if name == "EDAReporter":
        from eda.report import EDAReporter
        return EDAReporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

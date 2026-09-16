"""
Feature Engineering and ML-Ready Feature Preparation Subsystem.
Provides modular components for feature derivation, categorical handling,
leakage validation, and reusable preprocessor pipelines for the Sri Lankan
vehicle market valuation platform.
"""

from feature_engineering.dataset import MLDatasetLoader

__all__ = [
    "MLDatasetLoader",
]

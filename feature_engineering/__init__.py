"""
Feature Engineering and ML-Ready Feature Preparation Subsystem.
Provides modular components for feature derivation, categorical handling,
leakage validation, and reusable preprocessor pipelines for the Sri Lankan
vehicle market valuation platform.
"""

from feature_engineering.categorical import (
    CategoricalFeatureEngineer,
    RareCategoryGrouper,
)
from feature_engineering.dataset import MLDatasetLoader
from feature_engineering.numerical import NumericalFeatureEngineer
from feature_engineering.pipeline import (
    FeaturePipeline,
    FeaturePipelineConfig,
)
from feature_engineering.schema import (
    FEATURE_CATALOG,
    FeatureMetadata,
    FeatureSchema,
)
from feature_engineering.target import TargetTransformer
from feature_engineering.validation import DataLeakageError, LeakageValidator

__all__ = [
    "MLDatasetLoader",
    "TargetTransformer",
    "NumericalFeatureEngineer",
    "CategoricalFeatureEngineer",
    "RareCategoryGrouper",
    "LeakageValidator",
    "DataLeakageError",
    "FeaturePipeline",
    "FeaturePipelineConfig",
    "FeatureMetadata",
    "FeatureSchema",
    "FEATURE_CATALOG",
]

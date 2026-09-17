"""
Model Training & Evaluation Package.
Provides baseline models, candidate regressors, cross-validation, and metrics evaluation.
"""

from ml.training.evaluation import (
    CrossValidationResult,
    ErrorAnalyzer,
    RegressionMetrics,
    evaluate_predictions,
)

__all__ = [
    "RegressionMetrics",
    "CrossValidationResult",
    "ErrorAnalyzer",
    "evaluate_predictions",
]

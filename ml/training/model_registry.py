"""
Candidate Model Registry and Pipeline Factory.
Builds scikit-learn compatible regression pipelines integrating feature preprocessors,
baseline estimators, linear regression, random forest, and gradient boosting models.
"""

from typing import Callable, Dict, Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from ml.training.baselines import MeanBaselineRegressor, MedianBaselineRegressor

TargetTransformType = Literal["raw", "log1p"]


def create_model_pipeline(
    estimator: BaseEstimator,
    preprocessor: Optional[ColumnTransformer] = None,
    target_transform: TargetTransformType = "raw",
) -> BaseEstimator:
    """
    Constructs an end-to-end regression pipeline combining preprocessor,
    estimator, and optional target transformation (raw vs log1p).
    
    When target_transform == 'log1p', the pipeline is wrapped in a
    TransformedTargetRegressor so that fit() transforms the target using log1p
    and predict() automatically transforms predictions back using expm1 to
    the original LKR scale.
    """
    if preprocessor is not None:
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", estimator),
        ])
    else:
        pipe = Pipeline(steps=[
            ("regressor", estimator),
        ])

    if target_transform == "log1p":
        return TransformedTargetRegressor(
            regressor=pipe,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    elif target_transform == "raw":
        return pipe
    else:
        raise ValueError(f"Unsupported target_transform '{target_transform}'. Must be 'raw' or 'log1p'.")


def get_candidate_models(
    preprocessor_factory: Callable[[], ColumnTransformer],
    target_transform: TargetTransformType = "raw",
    random_state: int = 42,
) -> Dict[str, BaseEstimator]:
    """
    Returns the dictionary of candidate regression models and baselines
    to be evaluated in cross-validation and testing.
    
    Configurations:
    - MeanBaseline: predicts training mean
    - MedianBaseline: predicts training median
    - LinearRegression: ordinary least squares
    - RandomForestRegressor: n_estimators=300, min_samples_leaf=2, random_state=42
    - HistGradientBoostingRegressor: max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_reg=1.0, random_state=42
    """
    models: Dict[str, BaseEstimator] = {}

    # 1. Baselines (no preprocessor needed since features are ignored)
    models["MeanBaseline"] = create_model_pipeline(
        estimator=MeanBaselineRegressor(),
        preprocessor=None,
        target_transform=target_transform,
    )
    models["MedianBaseline"] = create_model_pipeline(
        estimator=MedianBaselineRegressor(),
        preprocessor=None,
        target_transform=target_transform,
    )

    # 2. Linear Regression
    models["LinearRegression"] = create_model_pipeline(
        estimator=LinearRegression(),
        preprocessor=preprocessor_factory(),
        target_transform=target_transform,
    )

    # 3. Random Forest Regressor
    models["RandomForestRegressor"] = create_model_pipeline(
        estimator=RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        preprocessor=preprocessor_factory(),
        target_transform=target_transform,
    )

    # 4. HistGradientBoostingRegressor
    models["HistGradientBoostingRegressor"] = create_model_pipeline(
        estimator=HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=random_state,
        ),
        preprocessor=preprocessor_factory(),
        target_transform=target_transform,
    )

    return models

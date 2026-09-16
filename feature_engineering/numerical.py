"""
Numerical Feature Engineering and Transformations.
Handles derived vehicle age, YOM redundancy policy, registration year indicator,
and skewness transformations for numerical variables (mileage, engine_cc).
"""

import logging
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_REFERENCE_YEAR = 2026
AgeRepresentationType = Literal["vehicle_age", "manufacture_year"]
MileageTransformType = Literal["none", "log1p"]


class NumericalFeatureEngineer:
    """
    Computes derived numerical features and manages redundancy policies.
    
    Guarantees:
    1. Vehicle age derived via explicit reference_year (prevents temporal data leakage).
    2. Negative vehicle ages are strictly rejected.
    3. Manufacture year and vehicle age are NOT both included in active features
       (eliminates exact collinearity rho = -1.00).
    4. Missing registration year is never converted to 0; missing indicator is created.
    5. Missing mileage/engine_cc are never filled with 0.
    6. Mileage log1p transformation does not overwrite original mileage.
    """

    def __init__(
        self,
        reference_year: int = DEFAULT_REFERENCE_YEAR,
        age_representation: AgeRepresentationType = "vehicle_age",
        include_registration_year: bool = False,
        mileage_transform: MileageTransformType = "none",
    ):
        if age_representation not in ("vehicle_age", "manufacture_year"):
            raise ValueError(f"Invalid age_representation '{age_representation}'. Must be 'vehicle_age' or 'manufacture_year'.")
        if mileage_transform not in ("none", "log1p"):
            raise ValueError(f"Invalid mileage_transform '{mileage_transform}'. Must be 'none' or 'log1p'.")

        self.reference_year = reference_year
        self.age_representation = age_representation
        self.include_registration_year = include_registration_year
        self.mileage_transform = mileage_transform

    def compute_vehicle_age(
        self,
        df: pd.DataFrame,
        manufacture_year_col: str = "manufacture_year",
        reference_year: Optional[int] = None,
    ) -> pd.Series:
        """
        Computes vehicle age as: reference_year - manufacture_year.
        
        Args:
            df: Input DataFrame containing manufacture year.
            manufacture_year_col: Column name for manufacture year.
            reference_year: Optional explicit reference year (defaults to self.reference_year).
            
        Returns:
            pd.Series containing computed vehicle age (float).
            
        Raises:
            ValueError: If negative vehicle age is encountered.
            KeyError: If manufacture_year_col is missing.
        """
        ref_year = reference_year or self.reference_year
        if manufacture_year_col not in df.columns:
            raise KeyError(f"Column '{manufacture_year_col}' not found in DataFrame.")

        yom = pd.to_numeric(df[manufacture_year_col], errors="coerce")
        age = ref_year - yom

        # Check for negative age
        negative_mask = age < 0
        if negative_mask.any():
            invalid_indices = df[negative_mask].index.tolist()
            invalid_yoms = yom[negative_mask].tolist()
            raise ValueError(
                f"Negative vehicle age detected for {len(invalid_indices)} records with YOM > {ref_year}: "
                f"{invalid_yoms[:5]}. Future manufacture years are invalid."
            )

        return age.astype(float)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives all numerical features on a copy of the input DataFrame.
        
        Derivations:
        - vehicle_age: derived from reference_year - manufacture_year
        - registration_year_missing: binary indicator (1 if missing, 0 if present)
        - mileage_log1p: np.log1p(mileage) if mileage_transform == 'log1p'
        
        Args:
            df: Input DataFrame.
            
        Returns:
            New DataFrame with engineered numerical features added.
        """
        df_out = df.copy()

        # Derive vehicle_age
        df_out["vehicle_age"] = self.compute_vehicle_age(df_out)

        # Registration year missing indicator
        if "registration_year" in df_out.columns:
            reg_year_numeric = pd.to_numeric(df_out["registration_year"], errors="coerce")
            df_out["registration_year_missing"] = reg_year_numeric.isna().astype(int)
        else:
            df_out["registration_year_missing"] = 1

        # Skewed mileage transformation
        if "mileage" in df_out.columns:
            mileage_num = pd.to_numeric(df_out["mileage"], errors="coerce")
            if (mileage_num < 0).any():
                raise ValueError("Negative mileage values encountered in dataset.")
            if self.mileage_transform == "log1p":
                df_out["mileage_log1p"] = np.log1p(mileage_num)

        return df_out

    def get_active_numerical_features(self) -> list[str]:
        """
        Returns the list of numerical feature names to be fed to ML preprocessors
        under the configured policy.
        """
        features: list[str] = []

        # Age representation policy (strictly excludes the redundant one)
        if self.age_representation == "vehicle_age":
            features.append("vehicle_age")
        else:
            features.append("manufacture_year")

        # Mileage features
        if self.mileage_transform == "log1p":
            features.append("mileage_log1p")
        else:
            features.append("mileage")

        # Engine capacity
        features.append("engine_cc")

        # Registration year
        if self.include_registration_year:
            features.append("registration_year")
            features.append("registration_year_missing")

        return features

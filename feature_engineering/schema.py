"""
Feature Schema and Metadata Management.
Provides structured descriptions, data types, transformations, default usage policies,
and leakage risk assessments for every feature in the valuation pipeline.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """
    Metadata specification for a single feature or target variable.
    """
    name: str
    source: str
    data_type: str
    feature_type: str  # "numerical", "categorical", "target", "traceability"
    transformation: str
    used_by_default: bool
    leakage_risk: str
    explanation: str


FEATURE_CATALOG: List[FeatureMetadata] = [
    FeatureMetadata(
        name="asking_price",
        source="price_history / observations",
        data_type="float",
        feature_type="target",
        transformation="raw (or log1p if configured)",
        used_by_default=True,
        leakage_risk="High - Separated as y, strictly forbidden in X",
        explanation="Observed seller asking price in LKR. Must never enter feature matrix X.",
    ),
    FeatureMetadata(
        name="vehicle_age",
        source="manufacture_year",
        data_type="float",
        feature_type="numerical",
        transformation="reference_year (2026) - manufacture_year",
        used_by_default=True,
        leakage_risk="None",
        explanation="Vehicle age in years relative to dataset reference year (2026). Replaces manufacture_year to avoid collinearity.",
    ),
    FeatureMetadata(
        name="manufacture_year",
        source="vehicles.manufacture_year",
        data_type="integer",
        feature_type="traceability",
        transformation="retained in metadata, excluded from active features",
        used_by_default=False,
        leakage_risk="None (Collinearity rho = -1.00 with vehicle_age)",
        explanation="Year of manufacture. Excluded by default because vehicle_age is perfectly collinear (-1.00 Spearman).",
    ),
    FeatureMetadata(
        name="mileage",
        source="listing_observations.observed_mileage",
        data_type="float",
        feature_type="numerical",
        transformation="median imputation, natural scale (or log1p if configured)",
        used_by_default=True,
        leakage_risk="None",
        explanation="Odometer mileage at observation time. Right-skewed distribution.",
    ),
    FeatureMetadata(
        name="mileage_log1p",
        source="listing_observations.observed_mileage",
        data_type="float",
        feature_type="numerical",
        transformation="np.log1p(mileage)",
        used_by_default=False,
        leakage_risk="None",
        explanation="Log-transformed mileage for models sensitive to high variance and extreme mileage values.",
    ),
    FeatureMetadata(
        name="engine_cc",
        source="vehicles.engine_cc",
        data_type="float",
        feature_type="numerical",
        transformation="median imputation",
        used_by_default=True,
        leakage_risk="None",
        explanation="Engine displacement in cubic centimeters (CC).",
    ),
    FeatureMetadata(
        name="registration_year",
        source="vehicles.registration_year",
        data_type="float",
        feature_type="numerical",
        transformation="optional median imputation; accompanied by missing indicator",
        used_by_default=False,
        leakage_risk="None",
        explanation="Year of official registration. Excluded by default due to ~70% missingness in Sri Lankan listings.",
    ),
    FeatureMetadata(
        name="registration_year_missing",
        source="vehicles.registration_year",
        data_type="integer",
        feature_type="numerical",
        transformation="binary missing indicator (1 if missing, 0 if present)",
        used_by_default=False,
        leakage_risk="None",
        explanation="Explicit binary flag indicating whether vehicle has a recorded registration year.",
    ),
    FeatureMetadata(
        name="category",
        source="vehicles.category",
        data_type="string",
        feature_type="categorical",
        transformation="canonicalized string -> OneHotEncoder(handle_unknown='ignore')",
        used_by_default=True,
        leakage_risk="None",
        explanation="Canonical vehicle category (e.g. Cars, Vans, SUVs). Segregates distinct price regimes.",
    ),
    FeatureMetadata(
        name="brand",
        source="vehicles.brand",
        data_type="string",
        feature_type="categorical",
        transformation="rare grouping (min_freq=5) -> OneHotEncoder(handle_unknown='ignore')",
        used_by_default=True,
        leakage_risk="None",
        explanation="Vehicle manufacturer brand. Infrequent brands grouped to 'Other' to prevent high-cardinality overfitting.",
    ),
    FeatureMetadata(
        name="model",
        source="vehicles.model",
        data_type="string",
        feature_type="categorical",
        transformation="rare grouping (min_freq=5) -> OneHotEncoder(handle_unknown='ignore')",
        used_by_default=True,
        leakage_risk="None",
        explanation="Vehicle model name. Infrequent models grouped to 'Other'.",
    ),
    FeatureMetadata(
        name="brand_model",
        source="brand + model",
        data_type="string",
        feature_type="categorical",
        transformation="concatenation -> rare grouping (min_freq=5) -> OneHotEncoder",
        used_by_default=True,
        leakage_risk="None",
        explanation="Domain interaction combining brand and model hierarchy (e.g. Toyota_Corolla).",
    ),
    FeatureMetadata(
        name="fuel_type",
        source="vehicles.fuel_type",
        data_type="string",
        feature_type="categorical",
        transformation="normalized string -> OneHotEncoder(handle_unknown='ignore')",
        used_by_default=True,
        leakage_risk="None",
        explanation="Normalized fuel type (Petrol, Diesel, Hybrid, Electric).",
    ),
    FeatureMetadata(
        name="transmission",
        source="vehicles.transmission",
        data_type="string",
        feature_type="categorical",
        transformation="normalized string -> OneHotEncoder(handle_unknown='ignore')",
        used_by_default=True,
        leakage_risk="None",
        explanation="Normalized gearbox transmission (Automatic, Manual, Tiptronic).",
    ),
    FeatureMetadata(
        name="district",
        source="listings.district",
        data_type="string",
        feature_type="categorical",
        transformation="imputation ('missing') -> OneHotEncoder(handle_unknown='ignore')",
        used_by_default=True,
        leakage_risk="None",
        explanation="Administrative district in Sri Lanka where listing is located.",
    ),
    FeatureMetadata(
        name="condition",
        source="vehicles.condition",
        data_type="string",
        feature_type="categorical",
        transformation="imputation ('missing') -> OneHotEncoder(handle_unknown='ignore')",
        used_by_default=True,
        leakage_risk="None",
        explanation="Vehicle condition (Registered (Used), Unregistered, Brand New).",
    ),
    FeatureMetadata(
        name="listing_id",
        source="listings.listing_id",
        data_type="string",
        feature_type="traceability",
        transformation="retained in metadata only, excluded from X",
        used_by_default=False,
        leakage_risk="None (Excluded from model features)",
        explanation="Riyasewana listing ID for auditability and joins.",
    ),
    FeatureMetadata(
        name="vehicle_id",
        source="vehicles.id",
        data_type="integer",
        feature_type="traceability",
        transformation="retained in metadata only, excluded from X",
        used_by_default=False,
        leakage_risk="None (Excluded from model features)",
        explanation="Internal vehicle surrogate key for database traceability.",
    ),
]


class FeatureSchema:
    """
    Catalog and exporter of feature metadata.
    """

    def __init__(self, catalog: Optional[List[FeatureMetadata]] = None):
        self.catalog = catalog or list(FEATURE_CATALOG)

    def to_dict(self) -> List[Dict[str, Any]]:
        """Converts catalog to list of dictionaries."""
        return [asdict(f) for f in self.catalog]

    def to_json(self, filepath: Optional[Path | str] = None, indent: int = 2) -> str:
        """Serializes catalog to JSON string or writes to file."""
        json_str = json.dumps(self.to_dict(), indent=indent)
        if filepath:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json_str, encoding="utf-8")
        return json_str

    def to_markdown_table(self) -> str:
        """Formats the feature catalog as a GitHub Flavored Markdown table."""
        headers = ["Feature", "Type", "Source", "Transformation", "Default", "Leakage Risk", "Explanation"]
        rows = [
            f"| {' | '.join(headers)} |",
            f"| {' | '.join(['---'] * len(headers))} |",
        ]
        for f in self.catalog:
            default_str = "Yes" if f.used_by_default else "No"
            row = [
                f"`{f.name}`",
                f.feature_type,
                f.source,
                f.transformation,
                default_str,
                f.leakage_risk,
                f.explanation,
            ]
            rows.append(f"| {' | '.join(row)} |")
        return "\n".join(rows)

    def get_features_by_type(self, feature_type: str) -> List[FeatureMetadata]:
        """Returns features matching given feature_type."""
        return [f for f in self.catalog if f.feature_type == feature_type]

    def get_default_features(self) -> List[str]:
        """Returns names of all features enabled by default in active X."""
        return [f.name for f in self.catalog if f.used_by_default and f.feature_type != "target"]

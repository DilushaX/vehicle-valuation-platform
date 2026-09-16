"""
ML Dataset Preparation CLI Script.
Extracts ML-eligible vehicle listings from PostgreSQL, applies feature engineering
and leakage protection, and exports reproducible ML dataset artifacts to disk.

Usage:
    python scripts/prepare_ml_dataset.py [OPTIONS]

Options:
    --target-transform {raw,log1p}
    --age-representation {vehicle_age,manufacture_year}
    --include-registration-year
    --mileage-transform {none,log1p}
    --min-frequency INT
    --output-dir DIR

Guarantees:
- PostgreSQL remains completely unchanged (read-only queries).
- Generates reproducible artifacts in data/analysis/ml/.
- Enforces strict leakage validation before dataset export.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict

import pandas as pd

# Add workspace root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_engineering.dataset import MLDatasetLoader
from feature_engineering.pipeline import FeaturePipeline, FeaturePipelineConfig
from feature_engineering.schema import FeatureSchema
from feature_engineering.validation import DataLeakageError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prepare_ml_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ML-ready vehicle valuation dataset from PostgreSQL."
    )
    parser.add_argument(
        "--target-transform",
        choices=["raw", "log1p"],
        default="raw",
        help="Target transformation to apply to observed asking price (default: raw).",
    )
    parser.add_argument(
        "--age-representation",
        choices=["vehicle_age", "manufacture_year"],
        default="vehicle_age",
        help="Age feature policy: use vehicle_age (default) or manufacture_year.",
    )
    parser.add_argument(
        "--include-registration-year",
        action="store_true",
        default=False,
        help="Include registration_year and missing indicator in feature matrix.",
    )
    parser.add_argument(
        "--mileage-transform",
        choices=["none", "log1p"],
        default="none",
        help="Transformation policy for mileage: none (default) or log1p.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=5,
        help="Minimum frequency threshold for grouping rare categorical levels to 'Other' (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/analysis/ml",
        help="Directory to save generated ML dataset and schema artifacts (default: data/analysis/ml).",
    )
    return parser.parse_args()


def prepare_and_export_dataset(
    target_transform: str = "raw",
    age_representation: str = "vehicle_age",
    include_registration_year: bool = False,
    mileage_transform: str = "none",
    min_frequency: int = 5,
    output_dir: str = "data/analysis/ml",
) -> Dict[str, Any]:
    """
    Executes the end-to-end dataset extraction, feature engineering, and export workflow.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1/6: Loading ML-eligible records from PostgreSQL (read-only)...")
    loader = MLDatasetLoader()
    raw_df = loader.load_raw_dataset(ml_eligible_only=True)
    logger.info(f"Loaded {len(raw_df)} ML-eligible listing records.")

    if raw_df.empty:
        raise ValueError("No ML-eligible records returned from database. Cannot proceed.")

    logger.info("Step 2/6: Initializing FeaturePipeline and configuration...")
    config = FeaturePipelineConfig(
        target_transform=target_transform,  # type: ignore
        age_representation=age_representation,  # type: ignore
        include_registration_year=include_registration_year,
        mileage_transform=mileage_transform,  # type: ignore
        min_frequency=min_frequency,
    )
    pipeline = FeaturePipeline(config=config)

    logger.info("Step 3/6: Engineering features and separating target (X, y, meta)...")
    X, y, meta = pipeline.prepare_features(raw_df)
    logger.info(f"Prepared feature matrix X shape: {X.shape}, target y length: {len(y)}")

    logger.info("Step 4/6: Validating zero data leakage...")
    pipeline.leakage_validator.validate_matrix_against_target(X, meta["raw_asking_price"])
    logger.info("Leakage validation PASSED successfully. Target and private fields strictly decoupled.")

    logger.info("Step 5/6: Generating combined ML-ready dataset artifact...")
    # Assemble complete dataset with traceability metadata and target
    ml_df = pd.concat([meta, X, y], axis=1)

    csv_path = out_path / "ml_dataset.csv"
    ml_df.to_csv(csv_path, index=False)
    logger.info(f"Saved ML dataset artifact: {csv_path} ({len(ml_df)} rows, {len(ml_df.columns)} columns)")

    logger.info("Step 6/6: Generating feature schema and dataset summary...")
    schema = FeatureSchema()
    schema_path = out_path / "feature_schema.json"
    schema.to_json(schema_path)
    logger.info(f"Saved feature schema: {schema_path}")

    # Compute summary statistics
    target_series = meta["raw_asking_price"]
    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_of_truth": "PostgreSQL (read-only extraction)",
        "target_definition": {
            "name": y.name,
            "transform": target_transform,
            "description": "Seller advertised/asking price in LKR (NOT confirmed transaction price)",
            "min": float(target_series.min()),
            "max": float(target_series.max()),
            "median": float(target_series.median()),
            "mean": float(target_series.mean()),
            "q25": float(target_series.quantile(0.25)),
            "q75": float(target_series.quantile(0.75)),
        },
        "dataset_statistics": {
            "total_records": len(ml_df),
            "features_in_X": len(X.columns),
            "active_feature_names": list(X.columns),
            "metadata_columns": list(meta.columns),
            "categories": X["category"].value_counts().to_dict() if "category" in X.columns else {},
        },
        "configuration": {
            "target_transform": target_transform,
            "age_representation": age_representation,
            "include_registration_year": include_registration_year,
            "mileage_transform": mileage_transform,
            "min_frequency": min_frequency,
            "reference_year": config.reference_year,
        },
    }

    summary_path = out_path / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Saved dataset summary: {summary_path}")

    return summary


def main() -> None:
    args = parse_args()
    try:
        summary = prepare_and_export_dataset(
            target_transform=args.target_transform,
            age_representation=args.age_representation,
            include_registration_year=args.include_registration_year,
            mileage_transform=args.mileage_transform,
            min_frequency=args.min_frequency,
            output_dir=args.output_dir,
        )
        print("\n=== ML DATASET PREPARATION COMPLETED SUCCESSFULLY ===")
        print(f"Total ML-ready records: {summary['dataset_statistics']['total_records']}")
        print(f"Features in X: {summary['dataset_statistics']['features_in_X']}")
        print(f"Target variable: {summary['target_definition']['name']} ({summary['target_definition']['transform']})")
        print(f"Target median: LKR {summary['target_definition']['median']:,.0f}")
        print(f"Categories: {summary['dataset_statistics']['categories']}")
    except (DataLeakageError, ValueError, KeyError) as e:
        logger.error(f"Preparation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Model Training & Evaluation CLI.
Executes leakage-safe training, cross-validation, model comparison, error analysis,
and artifact export for vehicle valuation models.

Usage:
    python scripts/train_valuation_model.py [OPTIONS]

Options:
    --target-transform {raw,log1p,compare-both}
    --test-size FLOAT
    --random-state INT
    --cv-folds INT
    --output-dir DIR
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.trainer import ModelTrainer, TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_valuation_model")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate regression models for vehicle valuation."
    )
    parser.add_argument(
        "--target-transform",
        choices=["raw", "log1p", "compare-both"],
        default="raw",
        help="Target transformation policy: raw, log1p, or compare-both (default: raw).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Holdout test partition fraction (default: 0.20).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds on training set (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/analysis/ml",
        help="Directory to save artifacts, figures, and reports (default: data/analysis/ml).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    transforms_to_run = ["raw", "log1p"] if args.target_transform == "compare-both" else [args.target_transform]

    for transform in transforms_to_run:
        logger.info("=" * 70)
        logger.info(f" EXECUTING MODEL TRAINING PIPELINE (Target Transform: {transform})")
        logger.info("=" * 70)

        config = TrainerConfig(
            test_size=args.test_size,
            random_state=args.random_state,
            cv_folds=args.cv_folds,
            target_transform=transform,  # type: ignore
            output_dir=Path(args.output_dir),
        )

        trainer = ModelTrainer(config=config)
        results = trainer.train_and_evaluate()

        best_name = results["best_model_name"]
        metrics = results["best_test_metrics"]

        print("\n" + "=" * 70)
        print(f" TRAINING PIPELINE COMPLETED SUCCESSFULLY [{transform.upper()}]")
        print("=" * 70)
        print(f"Selected Best Model : {best_name}")
        print(f"Training Samples    : {results['train_size']}")
        print(f"Holdout Test Samples: {results['test_size']}")
        print(f"Holdout Test MAE    : LKR {metrics.mae:,.0f}")
        print(f"Holdout Test RMSE   : LKR {metrics.rmse:,.0f}")
        print(f"Holdout Test R²     : {metrics.r2:.4f}")
        print(f"Holdout Test MedAE  : LKR {metrics.medae:,.0f}")
        print(f"Saved Artifact      : {results['model_path']}")
        print(f"Evaluation Report   : {results['report_path']}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

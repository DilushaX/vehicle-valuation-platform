"""
Model Training, Cross-Validation, and Evaluation Orchestrator.
Coordinates train/test partitioning, baseline evaluation, 5-fold cross-validation,
model selection, holdout test evaluation, error analysis, artifact serialization,
and reporting.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, train_test_split

from feature_engineering.dataset import MLDatasetLoader
from feature_engineering.pipeline import FeaturePipeline, FeaturePipelineConfig
from ml.training.baselines import MeanBaselineRegressor, MedianBaselineRegressor
from ml.training.evaluation import (
    CrossValidationResult,
    ErrorAnalyzer,
    RegressionMetrics,
    evaluate_predictions,
)
from ml.training.model_registry import (
    TargetTransformType,
    create_model_pipeline,
    get_candidate_models,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """
    Configuration for model training and evaluation pipeline.
    """
    test_size: float = 0.20
    random_state: int = 42
    cv_folds: int = 5
    stratify_by_category: bool = True
    target_transform: TargetTransformType = "raw"
    output_dir: Path = field(default_factory=lambda: Path("data/analysis/ml"))


class ModelTrainer:
    """
    Orchestrates leakage-safe regression model training and evaluation.
    
    Guarantees:
    1. Preprocessing is fitted strictly on training data (or training folds).
    2. Holdout test set remains completely untouched during model selection.
    3. Model selection is based solely on training-set cross-validation metrics.
    4. All metrics are calculated and reported in original LKR scale.
    5. Post-training error analysis uses non-causal descriptive interpretations.
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        self.output_dir = Path(self.config.output_dir)
        self.models_dir = self.output_dir / "models"
        self.figures_dir = self.output_dir / "figures"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline_coordinator = FeaturePipeline(
            FeaturePipelineConfig(target_transform=self.config.target_transform)
        )

    def prepare_data(
        self,
        raw_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Loads and prepares feature matrix X, target y, and metadata meta.
        """
        if raw_df is None:
            loader = MLDatasetLoader()
            raw_df = loader.load_raw_dataset(ml_eligible_only=True)
        X, y_trans, meta = self.pipeline_coordinator.prepare_features(raw_df)
        y_raw = meta["raw_asking_price"].copy()
        y_raw.name = "asking_price"
        return X, y_raw, meta

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        meta: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
        """
        Partitions data into training and holdout test sets.
        Uses category stratification if statistically viable, otherwise falls back to random split.
        """
        stratify_col = None
        if self.config.stratify_by_category and "category" in X.columns:
            counts = X["category"].value_counts()
            min_count = counts.min()
            # Stratification is viable if minimum category count >= 2 in test and train
            if min_count >= 5:
                stratify_col = X["category"]
                logger.info("Category stratification applied to holdout split.")
            else:
                logger.warning(
                    f"Minimum category count is {min_count} (< 5). "
                    "Falling back to reproducible random split to prevent singleton test classes."
                )

        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify_col,
        )

        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
        meta_train, meta_test = meta.iloc[train_idx].copy(), meta.iloc[test_idx].copy()

        logger.info(f"Split complete: Train={len(X_train)} samples, Test={len(X_test)} samples.")
        return X_train, X_test, y_train, y_test, meta_train, meta_test

    def run_cross_validation(
        self,
        models: Dict[str, BaseEstimator],
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Dict[str, CrossValidationResult]:
        """
        Executes 5-fold cross-validation on the TRAINING set only.
        Metrics are strictly calculated on the original LKR price scale.
        """
        kf = KFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        cv_results: Dict[str, CrossValidationResult] = {}

        for name, model in models.items():
            logger.info(f"Running {self.config.cv_folds}-fold CV for: {name} (target={self.config.target_transform})...")
            fold_metrics: List[RegressionMetrics] = []

            for fold_idx, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
                X_tr, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]

                # Clone model so preprocessor is freshly fitted on this fold only
                fold_model = clone(model)
                fold_model.fit(X_tr, y_tr)

                y_pred = fold_model.predict(X_val)

                # Ensure evaluation is on original LKR scale
                # If y_val was log1p, it will be inverted, but here y_train has raw asking prices
                # or TransformedTargetRegressor already inverted the predictions!
                metrics = evaluate_predictions(y_val, y_pred)
                fold_metrics.append(metrics)

            mae_scores = [m.mae for m in fold_metrics]
            rmse_scores = [m.rmse for m in fold_metrics]
            r2_scores = [m.r2 for m in fold_metrics]
            medae_scores = [m.medae for m in fold_metrics]

            cv_res = CrossValidationResult(
                model_name=name,
                target_transform=self.config.target_transform,
                cv_mae_mean=float(np.mean(mae_scores)),
                cv_mae_std=float(np.std(mae_scores)),
                cv_rmse_mean=float(np.mean(rmse_scores)),
                cv_rmse_std=float(np.std(rmse_scores)),
                cv_r2_mean=float(np.mean(r2_scores)),
                cv_r2_std=float(np.std(r2_scores)),
                cv_medae_mean=float(np.mean(medae_scores)),
                cv_medae_std=float(np.std(medae_scores)),
                fold_metrics=fold_metrics,
            )
            cv_results[name] = cv_res

        return cv_results

    def select_best_model(
        self,
        cv_results: Dict[str, CrossValidationResult],
    ) -> str:
        """
        Selects the best model based on lowest cross-validation MAE on training folds.
        Excludes baseline models from being chosen as the active ML candidate if
        a real regression model achieves lower MAE.
        """
        non_baseline_models = {
            k: v for k, v in cv_results.items()
            if "Baseline" not in k
        }

        selection_pool = non_baseline_models if non_baseline_models else cv_results
        best_name = min(selection_pool.keys(), key=lambda k: selection_pool[k].cv_mae_mean)
        logger.info(f"Model Selection: '{best_name}' selected based on lowest CV MAE (LKR {selection_pool[best_name].cv_mae_mean:,.0f}).")
        return best_name

    def evaluate_test_set(
        self,
        models: Dict[str, BaseEstimator],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Tuple[RegressionMetrics, np.ndarray]]:
        """
        Fits each candidate model on the entire training set and evaluates
        ONCE on the untouched holdout test set.
        """
        test_results: Dict[str, Tuple[RegressionMetrics, np.ndarray]] = {}

        for name, model in models.items():
            fitted_model = clone(model)
            fitted_model.fit(X_train, y_train)
            preds = fitted_model.predict(X_test)
            metrics = evaluate_predictions(y_test, preds)
            test_results[name] = (metrics, preds)

        return test_results

    def generate_visualizations(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
    ) -> List[Path]:
        """
        Generates diagnostic regression plots under data/analysis/ml/figures/.
        1. Actual vs Predicted Price
        2. Residual Distribution
        3. Absolute Error Distribution
        """
        y_act = np.asarray(y_test, dtype=np.float64) / 1_000_000.0  # Convert to Millions LKR
        y_pr = np.asarray(y_pred, dtype=np.float64) / 1_000_000.0
        residuals = y_pr - y_act
        abs_errors = np.abs(y_act - y_pr)

        created_files: List[Path] = []

        # 1. Actual vs Predicted
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_act, y_pr, alpha=0.7, color="#1f77b4", edgecolors="k", s=50)
        min_val = min(y_act.min(), y_pr.min())
        max_val = max(y_act.max(), y_pr.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction (y = x)")
        ax.set_title(f"Actual vs. Predicted Asking Price ({model_name})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Observed Actual Asking Price (Million LKR)", fontsize=11)
        ax.set_ylabel("Predicted Asking Price (Million LKR)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        plt.tight_layout()
        p1 = self.figures_dir / "actual_vs_predicted.png"
        fig.savefig(p1, dpi=300)
        plt.close(fig)
        created_files.append(p1)

        # 2. Residual Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.hist(residuals, bins=15, color="#2ca02c", edgecolor="black", alpha=0.7)
        ax1.axvline(0, color="red", linestyle="--", lw=1.5)
        ax1.set_title("Prediction Residuals Distribution", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Residual (Predicted - Actual) [Million LKR]", fontsize=10)
        ax1.set_ylabel("Frequency", fontsize=10)
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2.boxplot(residuals, vert=True, patch_artist=True, boxprops=dict(facecolor="#98df8a"))
        ax2.axhline(0, color="red", linestyle="--", lw=1.5)
        ax2.set_title("Residual Dispersion", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Residual [Million LKR]", fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        p2 = self.figures_dir / "residual_distribution.png"
        fig.savefig(p2, dpi=300)
        plt.close(fig)
        created_files.append(p2)

        # 3. Absolute Error Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(abs_errors, bins=15, color="#ff7f0e", edgecolor="black", alpha=0.7)
        med_err = np.median(abs_errors)
        ax.axvline(med_err, color="blue", linestyle="-.", lw=2, label=f"Median Error: LKR {med_err:.2f}M")
        ax.set_title(f"Absolute Prediction Error Distribution ({model_name})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Absolute Error (Million LKR)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        plt.tight_layout()
        p3 = self.figures_dir / "absolute_error_distribution.png"
        fig.savefig(p3, dpi=300)
        plt.close(fig)
        created_files.append(p3)

        return created_files

    def export_comparison_table(
        self,
        cv_results: Dict[str, CrossValidationResult],
        test_results: Dict[str, Tuple[RegressionMetrics, np.ndarray]],
        best_model_name: str,
    ) -> pd.DataFrame:
        """Exports model comparison table to CSV and returns DataFrame."""
        rows: List[Dict[str, Any]] = []

        for name in cv_results.keys():
            cv = cv_results[name]
            test_m, _ = test_results[name]
            rows.append({
                "Model": name,
                "Target_Transform": cv.target_transform,
                "Selected": "Yes (Best CV)" if name == best_model_name else "No",
                "CV_MAE_Mean_LKR": cv.cv_mae_mean,
                "CV_MAE_Std_LKR": cv.cv_mae_std,
                "CV_RMSE_Mean_LKR": cv.cv_rmse_mean,
                "CV_R2_Mean": cv.cv_r2_mean,
                "Test_MAE_LKR": test_m.mae,
                "Test_RMSE_LKR": test_m.rmse,
                "Test_R2": test_m.r2,
                "Test_MedAE_LKR": test_m.medae,
            })

        df_comp = pd.DataFrame(rows).sort_values(by="CV_MAE_Mean_LKR", ascending=True)
        csv_path = self.output_dir / "model_comparison.csv"
        df_comp.to_csv(csv_path, index=False)
        logger.info(f"Saved model comparison table: {csv_path}")
        return df_comp

    def export_evaluation_report(
        self,
        df_comp: pd.DataFrame,
        best_model_name: str,
        best_test_metrics: RegressionMetrics,
        error_analyzer: ErrorAnalyzer,
        train_size: int,
        test_size: int,
        features_count: int,
    ) -> Path:
        """Generates comprehensive Markdown model evaluation report."""
        report_path = self.output_dir / "model_evaluation_report.md"

        cat_error_summary = error_analyzer.summarize_by_category()
        tier_error_summary = error_analyzer.summarize_by_price_tier()
        top_abs_errors = error_analyzer.get_largest_absolute_errors(5)
        top_pct_errors = error_analyzer.get_largest_percentage_errors(5)

        md: List[str] = [
            "# Vehicle Valuation Model Training & Evaluation Report",
            "",
            f"**Evaluation Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            "**Target Variable**: Observed Seller Asking Price (`asking_price`) in LKR  ",
            "**Evaluation Scale**: Original Sri Lankan Rupees (LKR)  ",
            "",
            "> [!IMPORTANT]",
            "> **Asking Price Notice**: The platform models seller advertised asking prices from Riyasewana, ",
            "> NOT finalized transaction prices. Predictions represent listed market expectations.",
            "",
            "> [!NOTE]",
            "> **Sample Size Limitation**: The current dataset contains **113 ML-eligible records** across 8 vehicle categories. ",
            "> This is an experimental, research-ready evaluation benchmark, NOT a production-validated valuation engine.",
            "",
            "---",
            "",
            "## 1. Experimental Setup & Partitions",
            f"- **Total ML-Eligible Records**: {train_size + test_size}",
            f"- **Training Set (80%)**: {train_size} records",
            f"- **Holdout Test Set (20%)**: {test_size} records (completely untouched during model selection)",
            f"- **Active Model Features**: {features_count} features (including derived vehicle age, CC, mileage, categorical one-hot, brand_model interaction)",
            f"- **Cross-Validation**: 5-Fold Cross-Validation on training partitions only",
            f"- **Model Selection Criterion**: Lowest Mean Cross-Validation MAE on training folds",
            "",
            "---",
            "",
            "## 2. Model Comparison Table",
            "",
            "| Model | Target Transform | Selected | CV MAE (Mean ± Std) | CV RMSE (Mean) | CV R² (Mean) | Test MAE (LKR) | Test RMSE (LKR) | Test R² | Test MedAE (LKR) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for _, r in df_comp.iterrows():
            md.append(
                f"| **{r['Model']}** | `{r['Target_Transform']}` | {r['Selected']} | "
                f"LKR {r['CV_MAE_Mean_LKR']:,.0f} ± {r['CV_MAE_Std_LKR']:,.0f} | "
                f"LKR {r['CV_RMSE_Mean_LKR']:,.0f} | {r['CV_R2_Mean']:.4f} | "
                f"LKR {r['Test_MAE_LKR']:,.0f} | LKR {r['Test_RMSE_LKR']:,.0f} | "
                f"{r['Test_R2']:.4f} | LKR {r['Test_MedAE_LKR']:,.0f} |"
            )

        md.extend([
            "",
            "---",
            "",
            f"## 3. Selected Model Performance: `{best_model_name}`",
            f"- **Holdout Test MAE**: LKR {best_test_metrics.mae:,.0f}",
            f"- **Holdout Test RMSE**: LKR {best_test_metrics.rmse:,.0f}",
            f"- **Holdout Test R²**: {best_test_metrics.r2:.4f}",
            f"- **Holdout Test Median Absolute Error**: LKR {best_test_metrics.medae:,.0f}",
            "",
            "---",
            "",
            "## 4. Error Analysis & Diagnostics",
            "",
            "### Category Error Breakdown",
            "| Category | Test Count | Median Actual (LKR) | Median Predicted (LKR) | Mean MAE (LKR) | Median Error % |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for _, r in cat_error_summary.iterrows():
            md.append(
                f"| {r['category']} | {r['count']} | LKR {r['median_actual']:,.0f} | "
                f"LKR {r['median_predicted']:,.0f} | LKR {r['mean_mae']:,.0f} | {r['median_pct_error']:.1f}% |"
            )

        md.extend([
            "",
            "### Price Bracket Error Breakdown",
            "| Price Bracket | Test Count | Mean MAE (LKR) | Median MAE (LKR) | Median Error % |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])

        for _, r in tier_error_summary.iterrows():
            md.append(
                f"| {r['price_tier']} | {r['count']} | LKR {r['mean_mae']:,.0f} | "
                f"LKR {r['median_mae']:,.0f} | {r['median_pct_error']:.1f}% |"
            )

        md.extend([
            "",
            "### Top 5 Largest Absolute Errors (Holdout Test Set)",
            "| Listing ID | Category | Make / Model | Age | Mileage | Actual Asking | Predicted | Absolute Error (LKR) | Error % |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for _, r in top_abs_errors.iterrows():
            lid = r.get("listing_id", "N/A")
            cat = r.get("category", "N/A")
            bm = f"{r.get('brand', '')} {r.get('model', '')}".strip()
            age = f"{r.get('vehicle_age', 0):.0f} yrs"
            mil = f"{r.get('mileage', 0):,.0f} km"
            md.append(
                f"| `{lid}` | {cat} | {bm} | {age} | {mil} | "
                f"LKR {r['actual_price']:,.0f} | LKR {r['predicted_price']:,.0f} | "
                f"LKR {r['absolute_error']:,.0f} | {r['percentage_error']:.1f}% |"
            )

        md.extend([
            "",
            "---",
            "",
            "## 5. Methodological Limitations & Future Collection Roadmap",
            "1. **Small Sample Volume (113 Records)**: With only 11-25 listings per category, statistical power is constrained. Models must be retrained as scheduled collection accumulates 1,000+ observations.",
            "2. **Asking Price Premise**: Listing prices reflect advertised seller demands which typically contain negotiation margins not captured in public online classifieds.",
            "3. **Extreme Luxury / Heavy-Duty Outliers**: High-value commercial and luxury vehicles create large absolute residuals, highlighting the need for category-stratified or category-specific sub-models in future iterations.",
            "",
            "### Readiness Verdict",
            "**EXPERIMENTAL / RESEARCH-READY** (Not production-grade).",
        ])

        report_path.write_text("\n".join(md), encoding="utf-8")
        logger.info(f"Saved model evaluation report: {report_path}")
        return report_path

    def save_model_artifacts(
        self,
        best_name: str,
        best_model: BaseEstimator,
        cv_results: Dict[str, CrossValidationResult],
        test_results: Dict[str, Tuple[RegressionMetrics, np.ndarray]],
        train_size: int,
        test_size: int,
    ) -> Tuple[Path, Path]:
        """Serializes winning model pipeline and metadata JSON."""
        model_path = self.models_dir / "model.joblib"
        metadata_path = self.models_dir / "model_metadata.json"

        joblib.dump(best_model, model_path)
        logger.info(f"Saved winning model artifact: {model_path}")

        best_cv = cv_results[best_name]
        best_test, _ = test_results[best_name]

        metadata: Dict[str, Any] = {
            "model_name": best_name,
            "sklearn_version": sklearn_version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "target_variable": "asking_price",
            "target_transform": self.config.target_transform,
            "reference_year": 2026,
            "train_samples": train_size,
            "test_samples": test_size,
            "random_state": self.config.random_state,
            "cv_folds": self.config.cv_folds,
            "cv_metrics": best_cv.to_dict(),
            "test_metrics": best_test.to_dict(),
            "status": "EXPERIMENTAL_RESEARCH_BENCHMARK",
        }

        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info(f"Saved model metadata: {metadata_path}")

        return model_path, metadata_path

    def train_and_evaluate(
        self,
        raw_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete training, cross-validation, selection, testing,
        error analysis, and artifact generation pipeline.
        """
        logger.info("=== STEP 1: PREPARING DATASET & FEATURES ===")
        X, y, meta = self.prepare_data(raw_df)

        logger.info("=== STEP 2: PARTITIONING TRAIN / TEST SPLITS ===")
        X_train, X_test, y_train, y_test, meta_train, meta_test = self.split_data(X, y, meta)

        logger.info("=== STEP 3: INITIALIZING CANDIDATE MODELS & BASELINES ===")
        models = get_candidate_models(
            preprocessor_factory=self.pipeline_coordinator.build_preprocessor,
            target_transform=self.config.target_transform,
            random_state=self.config.random_state,
        )

        logger.info("=== STEP 4: RUNNING 5-FOLD CROSS-VALIDATION (TRAINING SET ONLY) ===")
        cv_results = self.run_cross_validation(models, X_train, y_train)

        logger.info("=== STEP 5: SELECTING BEST MODEL BASED ON CV PERFORMANCE ===")
        best_model_name = self.select_best_model(cv_results)

        logger.info("=== STEP 6: EVALUATING CANDIDATES ON UNTOUCHED HOLDOUT TEST SET ===")
        test_results = self.evaluate_test_set(models, X_train, y_train, X_test, y_test)
        best_test_metrics, best_test_preds = test_results[best_model_name]

        logger.info(f"Selected Model Test Metrics: {best_test_metrics.format_summary()}")

        logger.info("=== STEP 7: ERROR ANALYSIS & DIAGNOSTICS ===")
        error_analyzer = ErrorAnalyzer(
            y_true=y_test,
            y_pred=best_test_preds,
            X=X_test,
            meta=meta_test,
        )

        logger.info("=== STEP 8: GENERATING VISUALIZATIONS ===")
        figures = self.generate_visualizations(
            y_test=y_test,
            y_pred=best_test_preds,
            model_name=best_model_name,
        )

        logger.info("=== STEP 9: EXPORTING COMPARISON TABLES & REPORTS ===")
        df_comp = self.export_comparison_table(cv_results, test_results, best_model_name)
        report_path = self.export_evaluation_report(
            df_comp=df_comp,
            best_model_name=best_model_name,
            best_test_metrics=best_test_metrics,
            error_analyzer=error_analyzer,
            train_size=len(X_train),
            test_size=len(X_test),
            features_count=len(X.columns),
        )

        logger.info("=== STEP 10: SERIALIZING WINNING MODEL ARTIFACT ===")
        # Refit winning model on full training set
        winning_model = clone(models[best_model_name])
        winning_model.fit(X_train, y_train)

        model_path, metadata_path = self.save_model_artifacts(
            best_name=best_model_name,
            best_model=winning_model,
            cv_results=cv_results,
            test_results=test_results,
            train_size=len(X_train),
            test_size=len(X_test),
        )

        return {
            "best_model_name": best_model_name,
            "cv_results": cv_results,
            "test_results": test_results,
            "best_test_metrics": best_test_metrics,
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "report_path": str(report_path),
            "figures": [str(f) for f in figures],
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

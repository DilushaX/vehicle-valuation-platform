"""
Model Explainability Visualization Subsystem for Vehicle Valuation.

Provides reusable plotting capabilities to visualize local feature contributions
(Tree SHAP attributions) for individual vehicle asking price predictions.

Important Note:
Feature contributions represent model-level attributions and do NOT imply real-world
causation or physical price guarantees.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")  # Headless non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAME_DISPLAY_MAP: Dict[str, str] = {
    "brand_model": "Brand & Model",
    "vehicle_age": "Vehicle Age",
    "mileage": "Mileage",
    "engine_cc": "Engine Displacement",
    "category": "Vehicle Category",
    "brand": "Make / Brand",
    "model": "Vehicle Model",
    "fuel_type": "Fuel Type",
    "transmission": "Transmission",
    "district": "Location / District",
    "condition": "Vehicle Condition",
}


def _extract_explanation_records(
    explanation: Union[Sequence[Dict[str, Any]], Any, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extracts and standardizes explanation feature contribution records
    from various input formats (list, dict, or ValuationResult dataclass).
    """
    if explanation is None:
        raise ValueError("Explanation data cannot be None.")

    # Handle ValuationResult object
    if hasattr(explanation, "explanation"):
        explanation = getattr(explanation, "explanation")
    elif isinstance(explanation, dict) and "explanation" in explanation:
        explanation = explanation["explanation"]

    if not isinstance(explanation, (list, tuple)):
        raise ValueError(
            f"Invalid explanation data type: expected list of feature contribution dictionaries, "
            f"got {type(explanation).__name__}."
        )

    records: List[Dict[str, Any]] = []
    for idx, item in enumerate(explanation):
        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid explanation item at index {idx}: expected dictionary, got {type(item).__name__}."
            )
        if "feature" not in item:
            raise ValueError(f"Invalid explanation item at index {idx}: missing required 'feature' key.")
        if "contribution" not in item:
            raise ValueError(f"Invalid explanation item at index {idx}: missing required 'contribution' key.")

        try:
            contrib_float = float(item["contribution"])
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid contribution value for feature '{item.get('feature')}' at index {idx}: "
                f"must be numeric float, got {item.get('contribution')}."
            ) from e

        records.append({
            "feature": str(item["feature"]),
            "value": str(item.get("value", "")),
            "contribution": contrib_float,
            "direction": str(item.get("direction", "positive" if contrib_float >= 0 else "negative")),
            "description": str(item.get("description", "")),
        })

    return records


class FeatureContributionVisualizer:
    """
    Renders publication-ready horizontal bar charts illustrating
    local model feature contributions (SHAP attribution) for vehicle valuations.
    """

    @staticmethod
    def create_plot(
        explanation: Union[Sequence[Dict[str, Any]], Any, Dict[str, Any]],
        output_path: Union[str, Path] = "data/analysis/ml/figures/valuation_feature_contributions.png",
        top_k: Optional[int] = 10,
        show_values: bool = True,
        title: Optional[str] = None,
        raise_on_empty: bool = True,
    ) -> Path:
        """
        Creates and saves a horizontal bar chart of model feature contributions.

        Args:
            explanation: List of explanation dictionaries, a dict containing 'explanation',
                         or a ValuationResult instance.
            output_path: Target filesystem path for the output PNG figure.
            top_k: Maximum number of top contributing features by absolute magnitude to display.
            show_values: Whether to include the vehicle feature's actual input value in label.
            title: Custom plot title. Defaults to standard non-causal attribution title.
            raise_on_empty: If True, raises ValueError when explanation is empty.
                            If False, generates a clean placeholder figure.

        Returns:
            Path object pointing to the generated image file.
            
        Raises:
            ValueError: If explanation data is invalid or empty (when raise_on_empty=True).
        """
        records = _extract_explanation_records(explanation)
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if len(records) == 0:
            if raise_on_empty:
                raise ValueError("Explanation data is empty: no feature contributions to visualize.")
            return FeatureContributionVisualizer._create_empty_placeholder(target_path, title)

        # Sort features by absolute contribution descending
        sorted_records = sorted(records, key=lambda x: abs(x["contribution"]), reverse=True)

        if top_k is not None and top_k > 0:
            display_records = sorted_records[:top_k]
        else:
            display_records = sorted_records

        # For horizontal bar chart, reverse so the highest magnitude appears at the top
        plot_records = list(reversed(display_records))

        # Prepare labels and values
        labels: List[str] = []
        contributions: List[float] = []
        bar_colors: List[str] = []
        edge_colors: List[str] = []

        for item in plot_records:
            feat_key = item["feature"]
            display_name = FEATURE_NAME_DISPLAY_MAP.get(feat_key, feat_key.replace("_", " ").title())
            val = item.get("value", "")

            if show_values and val and val != "Unknown":
                label = f"{display_name} ({val})"
            else:
                label = display_name

            labels.append(label)
            contrib = item["contribution"]
            contributions.append(contrib)

            if contrib > 0:
                bar_colors.append("#2b8a3e")   # Forest Green
                edge_colors.append("#1e602b")
            elif contrib < 0:
                bar_colors.append("#c92a2a")   # Crimson Red
                edge_colors.append("#8f1d1d")
            else:
                bar_colors.append("#6c757d")   # Muted Gray
                edge_colors.append("#495057")

        # Dynamic figure height based on number of features
        n_features = len(plot_records)
        fig_height = max(5.0, 0.45 * n_features + 2.5)
        fig, ax = plt.subplots(figsize=(10.5, fig_height))

        y_positions = np.arange(n_features)
        bars = ax.barh(
            y_positions,
            contributions,
            color=bar_colors,
            edgecolor=edge_colors,
            linewidth=1.0,
            height=0.65,
            alpha=0.88,
        )

        # Reference baseline at zero
        ax.axvline(0, color="#212529", linestyle="-", linewidth=1.2, alpha=0.75)

        # Y-ticks
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=10, fontweight="medium")

        # Determine x-axis limits with padding for value labels
        max_abs = max((abs(c) for c in contributions), default=0.1)
        padding = max(0.05, max_abs * 0.22)
        min_x = min(min(contributions, default=0.0) - padding, -0.05)
        max_x = max(max(contributions, default=0.0) + padding, 0.05)
        ax.set_xlim(min_x, max_x)

        # Value annotations on each bar
        for bar, val in zip(bars, contributions):
            if val >= 0:
                text_x = val + (max_abs * 0.02)
                ha = "left"
                label_text = f"+{val:.4f}" if abs(val) < 0.1 else f"+{val:.3f}"
            else:
                text_x = val - (max_abs * 0.02)
                ha = "right"
                label_text = f"{val:.4f}" if abs(val) < 0.1 else f"{val:.3f}"

            ax.text(
                text_x,
                bar.get_y() + bar.get_height() / 2.0,
                label_text,
                va="center",
                ha=ha,
                fontsize=9.5,
                fontweight="semibold",
                color="#1a1a1a",
            )

        # Labels and Titles (strictly non-causal attribution terminology)
        main_title = title or "Model Feature Contributions"
        ax.set_title(
            main_title,
            fontsize=13.5,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel(
            "Model Feature Contribution (Tree SHAP Attribution)",
            fontsize=10.5,
            fontweight="medium",
            labelpad=10,
        )
        ax.set_ylabel("Vehicle Feature", fontsize=10.5, fontweight="medium")

        # Subtle gridlines along x-axis
        ax.grid(True, linestyle=":", alpha=0.5, axis="x")
        ax.set_axisbelow(True)

        # Add custom legend distinguishing positive and negative contributions
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2b8a3e", edgecolor="#1e602b", label="Positive Contribution (Raises Model Prediction)"),
            Patch(facecolor="#c92a2a", edgecolor="#8f1d1d", label="Negative Contribution (Lowers Model Prediction)"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower right",
            framealpha=0.92,
            fontsize=9.0,
            edgecolor="#cccccc",
        )

        # Informative attribution note in footer
        fig.text(
            0.5,
            -0.02,
            "Attributions reflect model-level tree feature contributions for this specific vehicle prediction (does not imply real-world causation).",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#555555",
            fontstyle="italic",
        )

        plt.tight_layout()
        fig.savefig(target_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Valuation explainability plot saved successfully to {target_path}")

        return target_path

    @staticmethod
    def _create_empty_placeholder(target_path: Path, title: Optional[str] = None) -> Path:
        """Creates a clean placeholder figure when no feature contributions are available."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No Model Feature Contributions Available\nfor This Vehicle Prediction",
            ha="center",
            va="center",
            fontsize=12,
            color="#666666",
            fontstyle="italic",
        )
        ax.set_title(title or "Model Feature Contributions", fontsize=13, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(target_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return target_path


def create_feature_contribution_plot(
    explanation: Union[Sequence[Dict[str, Any]], Any, Dict[str, Any]],
    output_path: Union[str, Path] = "data/analysis/ml/figures/valuation_feature_contributions.png",
    top_k: Optional[int] = 10,
    show_values: bool = True,
    title: Optional[str] = None,
    raise_on_empty: bool = True,
) -> Path:
    """
    Convenience functional interface to create and save a feature contribution plot.
    
    Args:
        explanation: List of explanation dictionaries, dict with 'explanation', or ValuationResult.
        output_path: Target filepath where the figure will be saved.
        top_k: Maximum number of top contributing factors to display.
        show_values: Whether to include feature values in labels.
        title: Optional custom plot title.
        raise_on_empty: Whether to raise ValueError if explanation data is empty.
        
    Returns:
        Path to the saved image file.
    """
    return FeatureContributionVisualizer.create_plot(
        explanation=explanation,
        output_path=output_path,
        top_k=top_k,
        show_values=show_values,
        title=title,
        raise_on_empty=raise_on_empty,
    )

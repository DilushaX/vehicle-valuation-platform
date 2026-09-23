"""
Human-Readable Vehicle Valuation Report Formatter.

Formats unified valuation results into clean, professional text and markdown
summaries with sensible financial rounding and mandatory market disclaimers.
"""

from typing import Any, Dict, List, Optional, Union

from ml.valuation.valuation_service import ValuationResult


def round_sensible_lkr(amount: float) -> int:
    """
    Rounds asking prices to the nearest 10,000 LKR to avoid misleading micro-precision.
    """
    if amount <= 0:
        return 0
    return int(round(amount / 10_000.0) * 10_000)


class ValuationReportFormatter:
    """
    Generates human-readable plain text and Markdown reports from ValuationResult.
    """

    @staticmethod
    def format_text(
        result: Union[ValuationResult, Dict[str, Any]],
        vehicle_spec: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Formats valuation output as a plain-text terminal/email summary.
        """
        data = result.to_dict() if isinstance(result, ValuationResult) else result
        spec = vehicle_spec or {}

        # Round figures sensibly
        est_raw = data.get("estimated_asking_price_lkr", 0.0)
        est_rounded = round_sensible_lkr(est_raw)

        range_dict = data.get("prediction_range_lkr", {})
        lower_rounded = round_sensible_lkr(range_dict.get("lower", est_raw * 0.85))
        upper_rounded = round_sensible_lkr(range_dict.get("upper", est_raw * 1.15))

        lines: List[str] = [
            "============================================================",
            "                VEHICLE VALUATION REPORT                   ",
            "============================================================",
            "",
            "VEHICLE SPECIFICATION:",
        ]

        # Vehicle details
        brand = spec.get("brand") or spec.get("make") or "Vehicle"
        model = spec.get("model") or ""
        lines.append(f"  Make & Model : {brand} {model}".rstrip())

        year = spec.get("manufacture_year")
        age = spec.get("vehicle_age")
        if year:
            lines.append(f"  Year         : {year} ({age:.0f} yrs old)" if age is not None else f"  Year         : {year}")
        elif age is not None:
            lines.append(f"  Vehicle Age  : {age:.0f} years")

        if spec.get("engine_cc"):
            lines.append(f"  Engine CC    : {spec['engine_cc']:,.0f} cc")
        if spec.get("fuel_type"):
            lines.append(f"  Fuel Type    : {spec['fuel_type']}")
        if spec.get("transmission"):
            lines.append(f"  Transmission : {spec['transmission']}")
        if spec.get("mileage") is not None:
            lines.append(f"  Mileage      : {spec['mileage']:,.0f} km")
        if spec.get("district"):
            lines.append(f"  Location     : {spec['district']}")
        if spec.get("condition"):
            lines.append(f"  Condition    : {spec['condition']}")

        lines.extend([
            "",
            "------------------------------------------------------------",
            "VALUATION ESTIMATE (Sri Lankan Rupees - LKR):",
            f"  Estimated Market Asking Price : Rs. {est_rounded:,.0f}",
            f"  Indicative Model-Based Range  : Rs. {lower_rounded:,.0f} – Rs. {upper_rounded:,.0f}",
            "------------------------------------------------------------",
            "",
            "KEY MODEL CONTRIBUTING FACTORS:",
        ])

        # Factors
        explanations = data.get("explanation", [])
        if explanations:
            for idx, factor in enumerate(explanations[:5], start=1):
                feat_name = factor.get("feature", "").replace("_", " ").title()
                feat_val = factor.get("value", "")
                direction = factor.get("direction", "neutral")
                contrib_sign = "+" if direction == "positive" else "-"
                lines.append(f"  {idx}. {feat_name} ({feat_val}): {direction.upper()} influence ({contrib_sign})")
        else:
            lines.append("  (No feature attribution available)")

        # Comparables
        comparables = data.get("comparables", [])
        lines.extend([
            "",
            "COMPARABLE MARKET LISTINGS (Riyasewana):",
        ])
        if comparables:
            for idx, comp in enumerate(comparables[:4], start=1):
                c_id = comp.get("listing_id", "")
                c_brand = comp.get("brand", "")
                c_model = comp.get("model", "")
                c_year = comp.get("manufacture_year") or "N/A"
                c_km = f"{comp['mileage']:,.0f} km" if comp.get("mileage") else "N/A"
                c_price = round_sensible_lkr(comp.get("asking_price", 0.0))
                c_dist = comp.get("district") or ""
                c_sim = comp.get("similarity_percentage", 0.0)
                lines.append(
                    f"  {idx}. [#{c_id}] {c_brand} {c_model} ({c_year}, {c_km}, {c_dist}) "
                    f"— Asking: Rs. {c_price:,.0f} (Similarity: {c_sim:.0f}%)"
                )
        else:
            lines.append("  (No close comparables found in database)")

        # Data Quality
        dq = data.get("data_quality", {})
        status_val = dq.get("status", "COMPLETE")
        prov_val = dq.get("provided_features", 11)
        exp_val = dq.get("expected_features", 11)
        missing_val = dq.get("missing_features", [])
        missing_str = ", ".join(missing_val) if missing_val else "None"
        comps_count = dq.get("comparable_count", len(comparables))

        lines.extend([
            "",
            "------------------------------------------------------------",
            "VALUATION DATA QUALITY:",
            f"  Status              : {status_val}",
            f"  Input Features      : {prov_val} / {exp_val}",
            f"  Missing Features    : {missing_str}",
            f"  Comparable Listings : {comps_count}",
            "------------------------------------------------------------",
        ])

        # Limitations
        lines.extend([
            "",
            "============================================================",
            "IMPORTANT NOTICE & DISCLAIMERS:",
            "  * This valuation estimates market asking prices observed on Riyasewana.",
            "  * It is NOT a confirmed transaction or final selling price.",
            "  * Actual negotiated selling prices may differ from advertised asking prices.",
            "  * Indicative model-based range reflects tree dispersion, not a guaranteed appraisal.",
            "  * Actual vehicle value may differ due to factors not captured by the dataset, including physical condition, accident history, mechanical condition, battery/engine health, and registration documentation.",
            "============================================================",
        ])

        return "\n".join(lines)

    @staticmethod
    def format_markdown(
        result: Union[ValuationResult, Dict[str, Any]],
        vehicle_spec: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Formats valuation output as structured GitHub/Streamlit Markdown.
        """
        data = result.to_dict() if isinstance(result, ValuationResult) else result
        spec = vehicle_spec or {}

        est_raw = data.get("estimated_asking_price_lkr", 0.0)
        est_rounded = round_sensible_lkr(est_raw)

        range_dict = data.get("prediction_range_lkr", {})
        lower_rounded = round_sensible_lkr(range_dict.get("lower", est_raw * 0.85))
        upper_rounded = round_sensible_lkr(range_dict.get("upper", est_raw * 1.15))

        brand = spec.get("brand") or spec.get("make") or "Vehicle"
        model = spec.get("model") or ""

        md: List[str] = [
            f"# Vehicle Valuation: {brand} {model}".strip(),
            "",
            "### 🏷️ Estimated Market Asking Price",
            f"## **Rs. {est_rounded:,.0f}**",
            f"*Indicative Model-Based Range: **Rs. {lower_rounded:,.0f} – Rs. {upper_rounded:,.0f}***  ",
            "",
            "> [!IMPORTANT]",
            "> **Asking Price Notice**: This valuation estimates market asking prices observed on Riyasewana. ",
            "> It is **not** a confirmed transaction or final selling price. Actual negotiated selling prices may differ from advertised asking prices because the collected dataset does not contain verified final transaction prices. ",
            "> Actual vehicle value may differ due to factors not captured by the dataset, including physical condition, accident history, mechanical condition, battery/engine health, and registration documentation.",
            "",
            "---",
            "",
            "### 🔍 Key Model Factors",
            "| Factor | Specification | Influence | Description |",
            "| :--- | :--- | :--- | :--- |",
        ]

        for factor in data.get("explanation", [])[:5]:
            feat = factor.get("feature", "").replace("_", " ").title()
            val = factor.get("value", "")
            direction = factor.get("direction", "").capitalize()
            desc = factor.get("description", "")
            badge = "🟢 Positive" if direction.lower() == "positive" else "🔴 Negative"
            md.append(f"| **{feat}** | {val} | {badge} | {desc} |")

        md.extend([
            "",
            "---",
            "",
            "### 🚗 Comparable Market Listings",
            "| Listing ID | Vehicle | Specs | Location | Asking Price (LKR) | Similarity |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for comp in data.get("comparables", [])[:5]:
            lid = comp.get("listing_id", "")
            veh = f"{comp.get('brand', '')} {comp.get('model', '')}"
            yr = comp.get("manufacture_year") or "—"
            km = f"{comp['mileage']:,.0f} km" if comp.get("mileage") else "—"
            dist = comp.get("district", "—")
            price = round_sensible_lkr(comp.get("asking_price", 0.0))
            sim = comp.get("similarity_percentage", 0.0)
            md.append(f"| `{lid}` | **{veh}** | {yr}, {km} | {dist} | Rs. {price:,.0f} | **{sim:.0f}%** |")

        # Data Quality Section
        dq = data.get("data_quality", {})
        status_val = dq.get("status", "COMPLETE")
        prov_val = dq.get("provided_features", 11)
        exp_val = dq.get("expected_features", 11)
        missing_val = dq.get("missing_features", [])
        missing_str = ", ".join(missing_val) if missing_val else "None"
        comps_count = dq.get("comparable_count", len(data.get("comparables", [])))
        badge = "🟢 COMPLETE" if status_val == "COMPLETE" else "🟡 PARTIAL"

        md.extend([
            "",
            "---",
            "",
            "### 📋 Valuation Data Quality",
            f"- **Status**: {badge}",
            f"- **Input Features**: {prov_val} / {exp_val}",
            f"- **Missing Features**: {missing_str}",
            f"- **Comparable Listings**: {comps_count}",
            "",
            "---",
            "",
            "> [!NOTE]",
            "> **Dataset Limitation**: The underlying Random Forest model was evaluated on a verified research benchmark dataset of 113 records across 8 categories.",
        ])

        return "\n".join(md)


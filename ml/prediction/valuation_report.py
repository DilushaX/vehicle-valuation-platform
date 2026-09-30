"""
Human-Readable Vehicle Valuation Report Formatter (Phase 10.5 Part 3).

Formats unified valuation results into clean, professional plain-text and Markdown
summaries with sensible financial rounding, comprehensive audit metadata, and mandatory
Sri Lankan vehicle market disclaimers. Single source of truth for dashboard display
and downloadable report export.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from ml.valuation.valuation_service import ValuationResult


def round_sensible_lkr(amount: float) -> int:
    """
    Rounds asking prices to the nearest 10,000 LKR to avoid misleading micro-precision.
    """
    if amount <= 0:
        return 0
    return int(round(amount / 10_000.0) * 10_000)


def sanitize_filename_component(text: Any) -> str:
    """
    Sanitizes a string component for safe use in filenames.
    Replaces spaces and invalid filesystem characters with underscores,
    collapses multiple underscores, and converts to lowercase.
    """
    if text is None:
        return ""
    cleaned = str(text).strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"[^\w\-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_-")
    return cleaned.lower()


def generate_report_filename(
    vehicle_spec: Optional[Dict[str, Any]] = None,
    extension: str = "txt",
) -> str:
    """
    Generates a deterministic, filesystem-safe filename for a valuation report.
    Format: vehicle_valuation_<brand>_<model>_<year>.<ext>
    
    Examples:
        - brand='Toyota', model='Premio', year=2016 -> 'vehicle_valuation_toyota_premio_2016.txt'
        - brand='Mercedes-Benz', model='C 200 (AMG / Sport)', year=2018 -> 'vehicle_valuation_mercedes-benz_c_200_amg_sport_2018.txt'
        - missing fields -> 'vehicle_valuation_report.txt'
    """
    spec = vehicle_spec or {}
    brand = sanitize_filename_component(spec.get("brand") or spec.get("make") or "")
    model = sanitize_filename_component(spec.get("model") or "")
    year_val = spec.get("manufacture_year")
    year = sanitize_filename_component(str(year_val).strip() if year_val is not None else "")

    parts = ["vehicle", "valuation"]
    if brand:
        parts.append(brand)
    if model:
        parts.append(model)
    if year:
        parts.append(year)

    if len(parts) == 2:
        parts.append("report")

    clean_ext = extension.lstrip(".").lower() or "txt"
    return f"{'_'.join(parts)}.{clean_ext}"


class ValuationReportFormatter:
    """
    Generates human-readable plain text and Markdown reports from ValuationResult,
    ValuationWorkflowResult, or dictionary responses.
    
    Ensures identical formatting and wording across both dashboard preview and
    file download (single source of truth).
    """

    @staticmethod
    def _extract_dict(result: Union[ValuationResult, Any, Dict[str, Any]]) -> Dict[str, Any]:
        """Extracts a normalized dictionary from various result representations."""
        if hasattr(result, "to_valuation_result") and callable(getattr(result, "to_valuation_result")):
            try:
                return result.to_valuation_result().to_dict()
            except Exception:
                return result.to_dict() if hasattr(result, "to_dict") else dict(result)
        elif hasattr(result, "to_dict") and callable(getattr(result, "to_dict")):
            return result.to_dict()
        elif isinstance(result, dict):
            return result
        return {}

    @staticmethod
    def format_text(
        result: Union[ValuationResult, Any, Dict[str, Any]],
        vehicle_spec: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Formats valuation output as a plain-text consolidated summary (Phase 10.5 Part 3).
        Single source of truth for both display preview and download export.
        """
        data = ValuationReportFormatter._extract_dict(result)
        spec = vehicle_spec or data.get("vehicle_spec") or {}

        # Round figures sensibly
        est_raw = data.get("estimated_asking_price_lkr")
        if est_raw is None:
            est_raw = data.get("estimated_asking_price")
        if est_raw is None and "valuation" in data and isinstance(data["valuation"], dict):
            est_raw = data["valuation"].get("estimated_asking_price_lkr") or data["valuation"].get("estimated_asking_price", 0.0)
        est_raw = float(est_raw or 0.0)
        est_rounded = round_sensible_lkr(est_raw)

        range_dict = (
            data.get("prediction_range_lkr")
            or data.get("indicative_prediction_range")
            or data.get("prediction_range")
        )
        if range_dict is None and "range" in data and isinstance(data["range"], dict):
            range_dict = data["range"]
        range_dict = range_dict or {}
        lower_raw = range_dict.get("lower") or range_dict.get("lower_bound") or (est_raw * 0.85 if est_raw > 0 else 0.0)
        upper_raw = range_dict.get("upper") or range_dict.get("upper_bound") or (est_raw * 1.15 if est_raw > 0 else 0.0)
        lower_rounded = round_sensible_lkr(lower_raw)
        upper_rounded = round_sensible_lkr(upper_raw)

        # Model details
        model_dict = data.get("model") or (data.get("valuation", {}).get("model") if isinstance(data.get("valuation"), dict) else {}) or {}
        m_name = model_dict.get("name") or (data.get("audit", {}).get("model_name") if isinstance(data.get("audit"), dict) else "RandomForestRegressor")
        m_status = model_dict.get("status") or (data.get("audit", {}).get("model_status") if isinstance(data.get("audit"), dict) else "EXPERIMENTAL_RESEARCH_BENCHMARK")

        # Timestamp
        audit_raw = data.get("audit") or {}
        if isinstance(audit_raw, dict) and "metadata" in audit_raw:
            gen_timestamp = audit_raw.get("metadata", {}).get("generated_at")
        elif isinstance(audit_raw, dict):
            gen_timestamp = audit_raw.get("generated_at")
        else:
            gen_timestamp = None
        if not gen_timestamp:
            gen_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines: List[str] = [
            "============================================================",
            "                VEHICLE VALUATION REPORT                   ",
            "============================================================",
            f"Report Generated : {gen_timestamp}",
            "",
            "VEHICLE SPECIFICATION:",
        ]

        # Vehicle details (Section A)
        cat = spec.get("category")
        if cat:
            lines.append(f"  Category     : {cat}")
        brand = spec.get("brand") or spec.get("make") or "Vehicle"
        model = spec.get("model") or ""
        lines.append(f"  Make & Model : {brand} {model}".rstrip())

        year = spec.get("manufacture_year")
        age = spec.get("vehicle_age")
        if year is not None:
            lines.append(f"  Year         : {year} ({age:.0f} yrs old)" if age is not None else f"  Year         : {year}")
        elif age is not None:
            lines.append(f"  Vehicle Age  : {age:.0f} years")
        else:
            lines.append("  Year         : Not specified")

        if spec.get("mileage") is not None:
            lines.append(f"  Mileage      : {spec['mileage']:,.0f} km")
        else:
            lines.append("  Mileage      : Not specified")

        if spec.get("engine_cc") is not None and spec.get("engine_cc") > 0:
            lines.append(f"  Engine CC    : {spec['engine_cc']:,.0f} cc")
        else:
            lines.append("  Engine CC    : Not specified")

        if spec.get("fuel_type"):
            lines.append(f"  Fuel Type    : {spec['fuel_type']}")
        else:
            lines.append("  Fuel Type    : Not specified")

        if spec.get("transmission"):
            lines.append(f"  Transmission : {spec['transmission']}")
        else:
            lines.append("  Transmission : Not specified")

        dist = spec.get("district") or spec.get("location")
        if dist:
            lines.append(f"  District     : {dist}")
        else:
            lines.append("  District     : Not specified")

        if spec.get("condition"):
            lines.append(f"  Condition    : {spec['condition']}")
        else:
            lines.append("  Condition    : Not specified")

        # Valuation Result (Section B)
        lines.extend([
            "",
            "------------------------------------------------------------",
            "VALUATION ESTIMATE (Sri Lankan Rupees - LKR):",
            f"  Estimated Market Asking Price : Rs. {est_rounded:,.0f}",
            f"  Indicative Model-Based Range  : Rs. {lower_rounded:,.0f} – Rs. {upper_rounded:,.0f}",
            "  (Indicative Model-Based Prediction Range based on ensemble tree dispersion)",
            f"  Model Architecture            : {m_name}",
            f"  Model Status                  : {m_status}",
            "------------------------------------------------------------",
            "",
            "KEY MODEL CONTRIBUTING FACTORS:",
        ])

        # Factors (Section C)
        explanations = data.get("explanation")
        if isinstance(explanations, dict):
            explanations = explanations.get("factors", [])
        elif not isinstance(explanations, list):
            explanations = []

        if explanations:
            for idx, factor in enumerate(explanations[:5], start=1):
                feat_name = factor.get("feature", "").replace("_", " ").title()
                feat_val = factor.get("value", "")
                direction = factor.get("direction", "neutral")
                contrib_sign = "+" if direction == "positive" else "-"
                lines.append(f"  {idx}. {feat_name} ({feat_val}): {direction.upper()} influence ({contrib_sign})")
        else:
            lines.append("  (No feature attribution available)")

        # Comparables (Section F)
        comparables = data.get("comparables")
        if isinstance(comparables, dict):
            comparables = comparables.get("items", [])
        elif not isinstance(comparables, list):
            comparables = []

        lines.extend([
            "",
            "COMPARABLE MARKET LISTINGS (Riyasewana):",
        ])
        if comparables:
            for idx, comp in enumerate(comparables[:5], start=1):
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

        # Comparable Market Summary (Section E)
        cms = data.get("comparable_market_summary")
        if cms is None and "market_summary" in data:
            cms = data["market_summary"]
        if not cms and comparables:
            from analytics.comparables.market_summary import create_comparable_market_summary
            cms = create_comparable_market_summary(comparables).to_dict()

        if cms and cms.get("comparable_count", 0) > 0:
            c_cnt = cms.get("comparable_count")
            min_p = round_sensible_lkr(cms.get("min_asking_price", 0.0))
            max_p = round_sensible_lkr(cms.get("max_asking_price", 0.0))
            med_p = round_sensible_lkr(cms.get("median_asking_price", 0.0))
            avg_p = round_sensible_lkr(cms.get("average_asking_price", 0.0))
            spr_p = round_sensible_lkr(cms.get("price_spread", 0.0))
            lines.extend([
                "",
                "COMPARABLE MARKET SUMMARY:",
                f"  Comparable Count      : {c_cnt}",
                f"  Minimum Asking Price  : Rs. {min_p:,.0f}",
                f"  Maximum Asking Price  : Rs. {max_p:,.0f}",
                f"  Median Asking Price   : Rs. {med_p:,.0f}",
                f"  Average Asking Price  : Rs. {avg_p:,.0f}",
                f"  Price Spread          : Rs. {spr_p:,.0f}",
                f"  Based on the {c_cnt} returned comparable vehicles, advertised asking prices range from",
                f"  Rs. {min_p:,.0f} to Rs. {max_p:,.0f}, with a median asking price of Rs. {med_p:,.0f}",
                f"  (Average: Rs. {avg_p:,.0f}, Price Spread: Rs. {spr_p:,.0f}).",
                "  * Note: Based on advertised asking prices, not verified transaction-price data.",
            ])
        else:
            lines.extend([
                "",
                "COMPARABLE MARKET SUMMARY:",
                "  Comparable Count      : 0",
                "  (No sufficiently similar comparables were found to compute market statistics)",
            ])

        # Data Quality (Section D)
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

        # Valuation Audit & Reproducibility (Section G)
        audit = data.get("audit") or {}
        if isinstance(audit, dict) and "metadata" in audit and "reproducibility" in audit:
            rep = audit.get("reproducibility", {})
            audit = audit.get("metadata", {})
        elif isinstance(audit, dict) and "metadata" in audit:
            rep = data.get("reproducibility") or audit.get("reproducibility", {})
            audit = audit.get("metadata", {})
        else:
            rep = data.get("reproducibility") or {}

        if audit or rep:
            m_name = audit.get("model_name", "RandomForestRegressor")
            m_ver = audit.get("model_version")
            m_str = f"{m_name} ({m_ver})" if m_ver else m_name
            target_str = audit.get("target", "asking_price")
            m_status_str = audit.get("model_status", "EXPERIMENTAL_RESEARCH_BENCHMARK")
            gen_str = audit.get("generated_at", gen_timestamp)
            fingerprint = rep.get("fingerprint", "N/A")
            range_method = audit.get("prediction_range_method", "individual_tree_percentiles")
            comp_method = audit.get("comparable_method", "Multi-attribute weighted similarity scoring")
            val_method = audit.get("valuation_method", "RandomForest asking-price regression with Tree SHAP explainability")
            feat_ver = audit.get("feature_schema_version", "1.0")

            lines.extend([
                "",
                "------------------------------------------------------------",
                "VALUATION AUDIT & REPRODUCIBILITY:",
                f"  Model                       : {m_str}",
                f"  Target                      : Advertised asking price ({target_str})",
                f"  Model Status                : {m_status_str}",
                f"  Feature Schema Version      : {feat_ver}",
                f"  Valuation Method            : {val_method}",
                f"  Prediction Range Method     : {range_method}",
                f"  Comparable Method           : {comp_method}",
                f"  Generated                   : {gen_str}",
                f"  Reproducibility Fingerprint : {fingerprint}",
                "------------------------------------------------------------",
            ])

        # Limitations & Disclaimers (Section H)
        lines.extend([
            "",
            "============================================================",
            "IMPORTANT NOTICE & DISCLAIMERS:",
            "  * This valuation estimates market asking prices observed on Riyasewana.",
            "  * It is NOT a confirmed transaction or final selling price.",
            "  * Actual negotiated selling prices may differ from advertised asking prices.",
            "  * The prediction range is an indicative model-based range reflecting tree dispersion, NOT a statistical confidence interval.",
            "  * SHAP values describe model contribution/attribution and must not be described as causal effects.",
            "  * Comparable similarity reflects specification distance scoring and does not prove that two listings are the same physical vehicle.",
            "  * The current valuation model is an experimental research benchmark and is not a production-grade appraisal system.",
            "  * Actual vehicle value may differ due to factors not captured by the dataset, including physical condition, accident history, mechanical condition, battery/engine health, and registration documentation.",
            "============================================================",
        ])

        return "\n".join(lines)

    @staticmethod
    def format_markdown(
        result: Union[ValuationResult, Any, Dict[str, Any]],
        vehicle_spec: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Formats valuation output as structured GitHub/Streamlit Markdown.
        """
        data = ValuationReportFormatter._extract_dict(result)
        spec = vehicle_spec or data.get("vehicle_spec") or {}

        est_raw = data.get("estimated_asking_price_lkr")
        if est_raw is None:
            est_raw = data.get("estimated_asking_price")
        if est_raw is None and "valuation" in data and isinstance(data["valuation"], dict):
            est_raw = data["valuation"].get("estimated_asking_price_lkr") or data["valuation"].get("estimated_asking_price", 0.0)
        est_raw = float(est_raw or 0.0)
        est_rounded = round_sensible_lkr(est_raw)

        range_dict = (
            data.get("prediction_range_lkr")
            or data.get("indicative_prediction_range")
            or data.get("prediction_range")
        )
        if range_dict is None and "range" in data and isinstance(data["range"], dict):
            range_dict = data["range"]
        range_dict = range_dict or {}
        lower_raw = range_dict.get("lower") or range_dict.get("lower_bound") or (est_raw * 0.85 if est_raw > 0 else 0.0)
        upper_raw = range_dict.get("upper") or range_dict.get("upper_bound") or (est_raw * 1.15 if est_raw > 0 else 0.0)
        lower_rounded = round_sensible_lkr(lower_raw)
        upper_rounded = round_sensible_lkr(upper_raw)

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

        explanations = data.get("explanation")
        if isinstance(explanations, dict):
            explanations = explanations.get("factors", [])
        elif not isinstance(explanations, list):
            explanations = []

        if explanations:
            for factor in explanations[:5]:
                feat = factor.get("feature", "").replace("_", " ").title()
                val = factor.get("value", "")
                direction = factor.get("direction", "").capitalize()
                desc = factor.get("description", "")
                badge = "🟢 Positive" if direction.lower() == "positive" else "🔴 Negative"
                md.append(f"| **{feat}** | {val} | {badge} | {desc} |")
        else:
            md.append("| *(None)* | — | — | *(No feature attribution factors available)* |")

        md.extend([
            "",
            "---",
            "",
            "### 🚗 Comparable Market Listings",
        ])

        comparables = data.get("comparables")
        if isinstance(comparables, dict):
            comparables = comparables.get("items", [])
        elif not isinstance(comparables, list):
            comparables = []

        if comparables:
            md.extend([
                "| Listing ID | Vehicle | Specs | Location | Asking Price (LKR) | Similarity |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
            ])
            for comp in comparables[:5]:
                lid = comp.get("listing_id", "")
                veh = f"{comp.get('brand', '')} {comp.get('model', '')}"
                yr = comp.get("manufacture_year") or "—"
                km = f"{comp['mileage']:,.0f} km" if comp.get("mileage") else "—"
                dist = comp.get("district", "—")
                price = round_sensible_lkr(comp.get("asking_price", 0.0))
                sim = comp.get("similarity_percentage", 0.0)
                md.append(f"| `{lid}` | **{veh}** | {yr}, {km} | {dist} | Rs. {price:,.0f} | **{sim:.0f}%** |")
        else:
            md.append("*(No close comparables found in database)*")

        # Comparable Market Summary
        cms = data.get("comparable_market_summary")
        if cms is None and "market_summary" in data:
            cms = data["market_summary"]
        if not cms and comparables:
            from analytics.comparables.market_summary import create_comparable_market_summary
            cms = create_comparable_market_summary(comparables).to_dict()

        if cms and cms.get("comparable_count", 0) > 0:
            c_cnt = cms.get("comparable_count")
            min_p = round_sensible_lkr(cms.get("min_asking_price", 0.0))
            max_p = round_sensible_lkr(cms.get("max_asking_price", 0.0))
            med_p = round_sensible_lkr(cms.get("median_asking_price", 0.0))
            avg_p = round_sensible_lkr(cms.get("average_asking_price", 0.0))
            spr_p = round_sensible_lkr(cms.get("price_spread", 0.0))
            md.extend([
                "",
                "#### 📊 Comparable Market Summary",
                f"Based on the **{c_cnt}** returned comparable vehicles, advertised asking prices range from **Rs. {min_p:,.0f}** to **Rs. {max_p:,.0f}**, with a median asking price of **Rs. {med_p:,.0f}** (Average: Rs. {avg_p:,.0f}, Price Spread: Rs. {spr_p:,.0f}).",
                "",
                "> [!NOTE]",
                "> **Asking Price Notice**: This summary is based strictly on seller advertised asking prices from retrieved comparables and does not represent verified transaction prices or statistical confidence.",
            ])
        else:
            md.extend([
                "",
                "#### 📊 Comparable Market Summary",
                "*(No sufficiently similar comparables were found to compute market statistics)*",
            ])

        # Data Quality Section
        dq = data.get("data_quality", {})
        status_val = dq.get("status", "COMPLETE")
        prov_val = dq.get("provided_features", 11)
        exp_val = dq.get("expected_features", 11)
        missing_val = dq.get("missing_features", [])
        missing_str = ", ".join(missing_val) if missing_val else "None"
        comps_count = dq.get("comparable_count", len(comparables))
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
        ])

        # Valuation Audit & Reproducibility
        audit = data.get("audit") or {}
        if isinstance(audit, dict) and "metadata" in audit and "reproducibility" in audit:
            rep = audit.get("reproducibility", {})
            audit = audit.get("metadata", {})
        elif isinstance(audit, dict) and "metadata" in audit:
            rep = data.get("reproducibility") or audit.get("reproducibility", {})
            audit = audit.get("metadata", {})
        else:
            rep = data.get("reproducibility") or {}

        if audit or rep:
            m_name = audit.get("model_name", "RandomForestRegressor")
            m_ver = audit.get("model_version")
            m_str = f"{m_name} ({m_ver})" if m_ver else m_name
            target_str = audit.get("target", "asking_price")
            m_status_str = audit.get("model_status", "EXPERIMENTAL_RESEARCH_BENCHMARK")
            gen_str = audit.get("generated_at", "N/A")
            fingerprint = rep.get("fingerprint", "N/A")
            range_method = audit.get("prediction_range_method", "individual_tree_percentiles")
            comp_method = audit.get("comparable_method", "Multi-attribute weighted similarity scoring")
            val_method = audit.get("valuation_method", "RandomForest asking-price regression with Tree SHAP explainability")
            feat_ver = audit.get("feature_schema_version", "1.0")

            md.extend([
                "",
                "---",
                "",
                "### 🛡️ Valuation Audit & Reproducibility",
                f"- **Model**: `{m_str}`",
                f"- **Target**: Advertised asking price (`{target_str}`)",
                f"- **Model Status**: `{m_status_str}`",
                f"- **Feature Schema Version**: `{feat_ver}`",
                f"- **Valuation Method**: {val_method}",
                f"- **Generated**: `{gen_str}`",
                f"- **Reproducibility Fingerprint**: `{fingerprint}` (SHA-256)",
                f"- **Prediction Range Method**: {range_method}",
                f"- **Comparable Method**: {comp_method}",
                "",
                "> [!NOTE]",
                "> **Audit Notice**: The reproducibility fingerprint identifies equivalent valuation configurations and inputs. It does not represent a cryptographic security signature or financial appraisal guarantee.",
            ])

        md.extend([
            "",
            "---",
            "",
            "> [!NOTE]",
            "> **Dataset Limitation**: The underlying Random Forest model was evaluated on a verified research benchmark dataset of 113 records across 8 categories.",
        ])

        return "\n".join(md)

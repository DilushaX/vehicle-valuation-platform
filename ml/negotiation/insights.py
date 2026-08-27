"""
Negotiation Insights Engine Module
Generates optional data-driven negotiation perspectives based on observed listing distributions,
comparables, and price deviation from market range.
Explicitly avoids claiming guaranteed outcomes.
"""
from typing import Dict, Any, Optional

class NegotiationInsightEngine:
    @staticmethod
    def generate_negotiation_insights(
        estimated_value: float,
        range_low: float,
        range_high: float,
        seller_asking_price: Optional[float] = None,
        comparable_median_price: Optional[float] = None,
        comparable_count: int = 0
    ) -> Dict[str, Any]:
        """
        Generates structured data-based negotiation insights.
        """
        if not seller_asking_price or seller_asking_price <= 0:
            return {
                "available": False,
                "message": "Enter a seller asking price to view data-driven negotiation insights."
            }

        delta_rs = seller_asking_price - estimated_value
        delta_pct = (delta_rs / estimated_value) * 100.0

        insights = []
        bargaining_context = ""

        if seller_asking_price > range_high:
            bargaining_context = "ABOVE_ESTIMATED_MARKET_RANGE"
            premium_amount = seller_asking_price - range_high
            insights.append(
                f"The asking price is Rs. {premium_amount:,.0f} above the upper estimated market range (Rs. {range_high:,.0f})."
            )
            insights.append(
                f"Historical listing data suggests similar models typically list between Rs. {range_low:,.0f} and Rs. {range_high:,.0f}."
            )
            if comparable_median_price and comparable_median_price > 0:
                insights.append(
                    f"The median price among {comparable_count} closely comparable active listings is Rs. {comparable_median_price:,.0f}."
                )
            suggested_opening_range = f"Rs. {range_low:,.0f} – Rs. {estimated_value:,.0f}"

        elif seller_asking_price < range_low:
            bargaining_context = "BELOW_ESTIMATED_MARKET_RANGE"
            discount_amount = range_low - seller_asking_price
            insights.append(
                f"The asking price is Rs. {discount_amount:,.0f} below the lower estimated market range (Rs. {range_low:,.0f})."
            )
            insights.append(
                "Because this listing is already positioned below prevailing market asking rates, substantial seller discount headroom may be limited."
            )
            suggested_opening_range = f"Rs. {seller_asking_price * 0.96:,.0f} – Rs. {seller_asking_price:,.0f}"

        else:
            bargaining_context = "WITHIN_ESTIMATED_MARKET_RANGE"
            insights.append(
                f"The asking price (Rs. {seller_asking_price:,.0f}) sits comfortably within the observed market range (Rs. {range_low:,.0f} – Rs. {range_high:,.0f})."
            )
            if delta_rs > 0:
                insights.append(
                    f"It is slightly above the central market point by Rs. {delta_rs:,.0f} (+{delta_pct:.1f}%)."
                )
            else:
                insights.append(
                    f"It is slightly below the central market point by Rs. {abs(delta_rs):,.0f} ({delta_pct:.1f}%)."
                )
            suggested_opening_range = f"Rs. {range_low:,.0f} – Rs. {estimated_value:,.0f}"

        return {
            "available": True,
            "bargaining_context": bargaining_context,
            "seller_asking_price": seller_asking_price,
            "estimated_market_value": estimated_value,
            "market_range": {"low": range_low, "high": range_high},
            "price_difference_amount": delta_rs,
            "price_difference_percent": round(delta_pct, 1),
            "suggested_negotiation_reference_range": suggested_opening_range,
            "observations": insights,
            "disclaimer": (
                "Notice: Market Intelligence insights are based strictly on statistical observations of online asking prices. "
                "They do not guarantee transaction outcomes, actual vehicle condition, or final negotiated settlement prices."
            )
        }

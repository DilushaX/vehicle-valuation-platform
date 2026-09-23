"""
Comparable Vehicle Market Summary Subsystem.

Calculates descriptive market statistics (min, max, median, average, spread)
for comparable vehicles retrieved by the comparable search engine.

Important Methodological Notice:
- Summarizes advertised asking prices observed on market listings.
- Does NOT represent confirmed transaction or selling prices.
- Does NOT represent statistical confidence, prediction accuracy, or certainty.
- Depends strictly on the volume and attribute alignment of retrieved listings.
- Having more comparables does NOT automatically imply higher valuation accuracy.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ComparableMarketSummary:
    """
    Descriptive asking price summary across retrieved comparable vehicles.
    
    Attributes:
        comparable_count: Count of valid comparable listings included in the summary.
        min_asking_price: Lowest advertised asking price in LKR.
        max_asking_price: Highest advertised asking price in LKR.
        median_asking_price: Median advertised asking price in LKR.
        average_asking_price: Arithmetic mean advertised asking price in LKR.
        price_spread: Difference between maximum and minimum asking price in LKR.
    """
    comparable_count: int
    min_asking_price: Optional[float]
    max_asking_price: Optional[float]
    median_asking_price: Optional[float]
    average_asking_price: Optional[float]
    price_spread: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparable_count": int(self.comparable_count),
            "min_asking_price": round(float(self.min_asking_price), 2) if self.min_asking_price is not None else None,
            "max_asking_price": round(float(self.max_asking_price), 2) if self.max_asking_price is not None else None,
            "median_asking_price": round(float(self.median_asking_price), 2) if self.median_asking_price is not None else None,
            "average_asking_price": round(float(self.average_asking_price), 2) if self.average_asking_price is not None else None,
            "price_spread": round(float(self.price_spread), 2) if self.price_spread is not None else None,
        }


def _extract_asking_prices(
    comparables: Optional[Union[Sequence[Any], pd.DataFrame]],
) -> List[float]:
    """
    Extracts and validates positive numeric asking prices from comparable records.
    Safely ignores records with missing, non-numeric, or non-positive asking prices.
    """
    if comparables is None:
        return []

    if isinstance(comparables, pd.DataFrame):
        if "asking_price" not in comparables.columns or comparables.empty:
            return []
        series = pd.to_numeric(comparables["asking_price"], errors="coerce").dropna()
        return [float(p) for p in series if p > 0]

    prices: List[float] = []
    for item in comparables:
        val = None
        if hasattr(item, "asking_price"):
            val = getattr(item, "asking_price")
        elif isinstance(item, dict) and "asking_price" in item:
            val = item["asking_price"]

        if val is None:
            continue

        try:
            p_float = float(val)
            if not np.isnan(p_float) and not np.isinf(p_float) and p_float > 0:
                prices.append(p_float)
        except (ValueError, TypeError):
            continue

    return prices


def create_comparable_market_summary(
    comparables: Optional[Union[Sequence[Any], pd.DataFrame]] = None,
) -> ComparableMarketSummary:
    """
    Computes summary asking price statistics for comparable vehicles.

    Args:
        comparables: Sequence of ComparableVehicle objects, dictionaries, or DataFrame.

    Returns:
        ComparableMarketSummary with count, min, max, median, average, and spread.
        Returns empty/None fields safely if zero valid comparables are provided.
    """
    prices = _extract_asking_prices(comparables)

    if not prices:
        return ComparableMarketSummary(
            comparable_count=0,
            min_asking_price=None,
            max_asking_price=None,
            median_asking_price=None,
            average_asking_price=None,
            price_spread=None,
        )

    count = len(prices)
    min_val = float(np.min(prices))
    max_val = float(np.max(prices))
    median_val = float(np.median(prices))
    avg_val = float(np.mean(prices))
    spread_val = float(max_val - min_val)

    return ComparableMarketSummary(
        comparable_count=count,
        min_asking_price=round(min_val, 2),
        max_asking_price=round(max_val, 2),
        median_asking_price=round(median_val, 2),
        average_asking_price=round(avg_val, 2),
        price_spread=round(spread_val, 2),
    )

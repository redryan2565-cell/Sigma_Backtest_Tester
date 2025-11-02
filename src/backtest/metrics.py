from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import newton


def max_drawdown(series: pd.Series) -> float:
    """Compute Maximum Drawdown (MDD) from a value series.
    
    Series should already be filtered to start from first investment date.

    Args:
        series: Series of portfolio values (e.g., NAV) in chronological order.
               Should be filtered to start from first investment date.

    Returns:
        Maximum drawdown as a negative number (e.g., -0.35 for -35%).
        Returns 0.0 if series is empty or has fewer than 2 points.
    """
    if series.empty or len(series) < 2:
        return 0.0
    
    vals = series.astype(float).values
    
    # Remove any zeros or negative values that would cause issues
    if (vals <= 0).any():
        # Filter to positive values only
        positive_mask = vals > 0
        if not positive_mask.any():
            return 0.0
        vals = vals[positive_mask]
        if len(vals) < 2:
            return 0.0
    
    running_max = np.maximum.accumulate(vals)
    # Avoid division by zero
    drawdowns = (vals / np.where(running_max == 0, np.nan, running_max)) - 1.0
    drawdowns = np.nan_to_num(drawdowns, nan=0.0, posinf=0.0, neginf=-1.0)
    
    # Clip to valid range [-1, 0]
    mdd = float(np.clip(drawdowns.min(), -1.0, 0.0))
    return mdd


def cagr(start_value: float, end_value: float, periods_in_years: float) -> float:
    """Compound Annual Growth Rate.

    Args:
        start_value: Starting value (> 0).
        end_value: Ending value (> 0).
        periods_in_years: Duration in years (> 0).

    Returns:
        Annualized return as decimal. Returns 0.0 when inputs are invalid.
    """
    if start_value <= 0 or end_value <= 0 or periods_in_years <= 0:
        return 0.0
    return float((end_value / start_value) ** (1.0 / periods_in_years) - 1.0)


def xirr(cash_flows: Sequence[float], dates: Sequence[datetime]) -> float:
    """Extended IRR for irregular cash flow dates.

    Positive flows are inflows, negative are outflows. For an investment,
    contributions are typically negative and the final liquidation is positive.
    """
    if len(cash_flows) != len(dates) or len(cash_flows) == 0:
        return 0.0
    # convert to year fractions from the first date
    t0 = dates[0]
    years = np.array([(d - t0).days / 365.25 for d in dates], dtype=float)
    amounts = np.array(cash_flows, dtype=float)

    def npv(rate: float) -> float:
        return float(np.sum(amounts / (1.0 + rate) ** years))

    try:
        irr = newton(lambda r: npv(r), x0=0.1, tol=1e-7, maxiter=100)
    except Exception:  # pragma: no cover - rare convergence issues
        return 0.0
    return float(irr)



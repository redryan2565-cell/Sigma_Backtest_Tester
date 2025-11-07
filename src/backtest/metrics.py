from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime

import numpy as np
import pandas as pd


def _secant_root(
    func: Callable[[float], float],
    x0: float = 0.1,
    x1: float | None = None,
    tol: float = 1e-7,
    maxiter: int = 100,
) -> float:
    """Solve func(x)=0 using a secant-like iteration.

    Returns the final iterate; the caller should validate convergence.
    """

    if x1 is None:
        x1 = x0 + 0.05 if x0 != 0 else 0.1

    f0 = func(x0)
    f1 = func(x1)

    for _ in range(maxiter):
        denom = f1 - f0
        if abs(denom) < 1e-12:
            break
        x2 = x1 - f1 * (x1 - x0) / denom
        if not np.isfinite(x2):
            break
        if abs(x2 - x1) < tol:
            return float(x2)
        x0, f0 = x1, f1
        x1, f1 = x2, func(x2)

    return float(x1)


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
        if rate <= -0.999999:
            return np.inf
        denom = (1.0 + rate) ** years
        return float(np.sum(amounts / denom))

    try:
        irr = _secant_root(npv)
    except Exception:  # pragma: no cover - defensive fallback
        return 0.0

    if not np.isfinite(irr):
        return 0.0

    if abs(npv(irr)) > 1e-6:
        return 0.0

    return float(irr)


def sharpe_ratio(nav_series: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from NAV series.
    
    Args:
        nav_series: Series of NAV values in chronological order.
        risk_free_rate: Risk-free rate (annual, as decimal). Defaults to 0.0.
        
    Returns:
        Sharpe ratio (annualized). Returns 0.0 if insufficient data or invalid.
    """
    if nav_series.empty or len(nav_series) < 2:
        return 0.0

    # Calculate daily returns: NAV_t / NAV_{t-1} - 1
    nav_values = nav_series.values
    if (nav_values <= 0).any():
        return 0.0

    daily_returns = np.diff(nav_values) / nav_values[:-1]

    if len(daily_returns) == 0:
        return 0.0

    # Annualize
    mean_return = np.mean(daily_returns) * 252  # Trading days per year
    std_return = np.std(daily_returns, ddof=1) * np.sqrt(252)

    if std_return == 0:
        return 0.0

    sharpe = (mean_return - risk_free_rate) / std_return
    return float(sharpe) if np.isfinite(sharpe) else 0.0


def sortino_ratio(nav_series: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio from NAV series.
    
    Args:
        nav_series: Series of NAV values in chronological order.
        risk_free_rate: Risk-free rate (annual, as decimal). Defaults to 0.0.
        
    Returns:
        Sortino ratio (annualized). Returns 0.0 if insufficient data or invalid.
    """
    if nav_series.empty or len(nav_series) < 2:
        return 0.0

    # Calculate daily returns: NAV_t / NAV_{t-1} - 1
    nav_values = nav_series.values
    if (nav_values <= 0).any():
        return 0.0

    daily_returns = np.diff(nav_values) / nav_values[:-1]

    if len(daily_returns) == 0:
        return 0.0

    # Annualize mean return
    mean_return = np.mean(daily_returns) * 252  # Trading days per year

    # Downside deviation: std of negative returns only
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) == 0:
        # No downside, perfect Sortino
        downside_std = 0.0
    else:
        downside_std = np.std(downside_returns, ddof=1) * np.sqrt(252)

    if downside_std == 0:
        if mean_return > risk_free_rate:
            return np.inf
        return 0.0

    sortino = (mean_return - risk_free_rate) / downside_std
    return float(sortino) if np.isfinite(sortino) else 0.0


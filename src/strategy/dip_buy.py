from __future__ import annotations

from typing import Literal

import pandas as pd

AllocationMode = Literal["split", "first_hit"]


def generate_dip_signals(adj_close: pd.Series, threshold: float) -> pd.Series:
    """Generate dip signals where daily return <= threshold.

    Args:
        adj_close: Series of adjusted close prices indexed by date.
        threshold: Return threshold (e.g., -0.041 for -4.1%).

    Returns:
        Boolean Series indexed by date.
    """
    adj_close = adj_close.astype(float)
    rets = adj_close.pct_change()
    signals = rets <= float(threshold)
    signals.iloc[0] = False  # first day has no prior return
    return signals


def allocate_weekly_budget(
    signals: pd.Series,
    weekly_budget: float,
    mode: AllocationMode,
    carryover: bool,
    week_ending: str = "W-SUN",
) -> pd.Series:
    """Allocate weekly budget to signal days according to mode and carryover.

    Args:
        signals: Boolean Series of signals indexed by date.
        weekly_budget: Budget per week (currency units).
        mode: 'split' or 'first_hit'.
        carryover: If True, unused budget carries to next week.
        week_ending: Pandas weekly rule for period end (default 'W-SUN').

    Returns:
        Series of allocation amounts per day (0.0 when no allocation).
    """
    if weekly_budget < 0:
        raise ValueError("weekly_budget must be non-negative")
    if mode not in ("split", "first_hit"):
        raise ValueError("mode must be 'split' or 'first_hit'")

    # Convert to DatetimeIndex, removing timezone if present to avoid warning
    idx = pd.DatetimeIndex(signals.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    week_periods = idx.to_period(week_ending)
    alloc = pd.Series(0.0, index=idx)
    carry_pool = 0.0

    # iterate weeks in chronological order
    for week in pd.unique(week_periods):
        mask = week_periods == week
        week_idx = idx[mask]
        week_signals = signals.loc[week_idx]

        pool = weekly_budget + (carry_pool if carryover else 0.0)
        sig_days = week_signals[week_signals].index.tolist()

        spent = 0.0
        if mode == "split":
            if len(sig_days) > 0 and pool > 0:
                per = float(pool) / float(len(sig_days))
                for d in sig_days:
                    alloc.loc[d] = per
                spent = per * len(sig_days)
        else:  # first_hit
            if len(sig_days) > 0 and pool > 0:
                first = sig_days[0]
                alloc.loc[first] = pool
                spent = pool

        leftover = pool - spent
        carry_pool = leftover if carryover else 0.0

    return alloc


def allocate_shares_per_signal(
    signals: pd.Series,
    shares_per_signal: float,
    prices: pd.Series,
    slippage_rate: float,
    fee_rate: float,
) -> pd.Series:
    """Allocate fixed number of shares per signal occurrence.

    Returns the total cash outflow (BuyAmt + Fee) needed per day.
    This will be used by compute_ledger to calculate exact shares and fees.

    Args:
        signals: Boolean Series of signals indexed by date.
        shares_per_signal: Number of shares to buy per signal (must be > 0).
        prices: Series of prices indexed by date (used for calculating allocation amount).
        slippage_rate: Slippage rate applied to price.
        fee_rate: Fee rate applied multiplicatively.

    Returns:
        Series of total cost (BuyAmt + Fee) per day (0.0 when no signal).
        Note: In compute_ledger, this will be split into BuyAmt and Fee separately.
    """
    if shares_per_signal <= 0:
        raise ValueError("shares_per_signal must be positive")
    if len(signals) != len(prices):
        raise ValueError("signals and prices must have same length")

    # Convert to DatetimeIndex, removing timezone if present to avoid warning
    idx = pd.DatetimeIndex(signals.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    alloc = pd.Series(0.0, index=idx)

    # For each signal day, calculate the total cash needed:
    # exec_price = price × (1 + slippage_rate)
    # BuyAmt = shares × exec_price
    # Fee = BuyAmt × fee_rate
    # TotalCost = BuyAmt + Fee = shares × exec_price × (1 + fee_rate)
    signal_days = signals[signals].index

    for day in signal_days:
        if day in prices.index:
            price = float(prices.loc[day])
            exec_price = price * (1.0 + float(slippage_rate))
            buy_amt = exec_price * float(shares_per_signal)
            fee_amt = buy_amt * float(fee_rate)
            total_cost = buy_amt + fee_amt
            alloc.loc[day] = total_cost

    return alloc



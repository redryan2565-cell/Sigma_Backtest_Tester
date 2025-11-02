from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd

from ..backtest.metrics import cagr as cagr_func
from ..backtest.metrics import max_drawdown, xirr
from ..strategy.dip_buy import (
    AllocationMode,
    allocate_shares_per_signal,
    allocate_weekly_budget,
    generate_dip_signals,
)


@dataclass(frozen=True)
class BacktestParams:
    threshold: float
    weekly_budget: float | None = None
    mode: AllocationMode | None = None
    carryover: bool | None = None
    shares_per_signal: float | None = None
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005


def _compute_allocation(prices: pd.DataFrame, params: BacktestParams) -> pd.Series:
    signals = generate_dip_signals(prices["AdjClose"], params.threshold)
    
    # Determine which allocation mode to use
    if params.shares_per_signal is not None:
        # Shares-based allocation
        if params.shares_per_signal <= 0:
            raise ValueError("shares_per_signal must be positive")
        allocation = allocate_shares_per_signal(
            signals=signals,
            shares_per_signal=float(params.shares_per_signal),
            prices=prices["AdjClose"],
            slippage_rate=float(params.slippage_rate),
            fee_rate=float(params.fee_rate),
        )
    else:
        # Budget-based allocation (legacy)
        if params.weekly_budget is None or params.mode is None or params.carryover is None:
            raise ValueError(
                "For budget-based mode, weekly_budget, mode, and carryover must be provided"
            )
        allocation = allocate_weekly_budget(
            signals=signals,
            weekly_budget=float(params.weekly_budget),
            mode=params.mode,
            carryover=params.carryover,
            week_ending="W-SUN",
        )
    
    allocation = allocation.reindex(prices.index, fill_value=0.0)
    
    # Validate allocation (no NaN/Inf)
    if allocation.isna().any() or np.isinf(allocation).any():
        raise ValueError("Allocation contains invalid values (NaN or Inf)")
    
    return allocation


def compute_ledger(
    prices: pd.DataFrame,
    params: BacktestParams,
    allocation: pd.Series,
) -> pd.DataFrame:
    """Compute complete accounting ledger with all intermediate columns.
    
    Returns DataFrame with columns:
    DailyRet, Signal, Mode, AdjClose, PxExec, BuyAmt, Fee, SharesBought,
    CashFlow, CumCashFlow, CumShares, Equity, CashBalance, NAV,
    CumInvested, Drawdown
    """
    idx = prices.index
    adj = prices["AdjClose"].astype(float)
    
    # Daily returns and signals
    daily_ret = adj.pct_change()
    signals = generate_dip_signals(adj, params.threshold)
    
    # Execution price (with slippage)
    px_exec = adj * (1.0 + float(params.slippage_rate))
    
    # Determine mode
    is_shares_mode = params.shares_per_signal is not None
    
    # Initialize columns
    buy_amt = pd.Series(0.0, index=idx)
    fee = pd.Series(0.0, index=idx)
    shares_bought = pd.Series(0.0, index=idx)
    
    if is_shares_mode:
        # Shares-based mode
        shares_fixed = float(params.shares_per_signal)
        signal_days = signals[signals].index
        
        for day in signal_days:
            if day in idx:
                shares_bought.loc[day] = shares_fixed
                buy_amt.loc[day] = shares_fixed * px_exec.loc[day]
                fee.loc[day] = buy_amt.loc[day] * float(params.fee_rate)
    else:
        # Budget-based mode
        # allocation already contains BuyAmt (budget per signal)
        buy_amt = allocation.copy()
        # Shares = BuyAmt / PxExec (buy with gross amount, fee separate)
        mask = buy_amt > 0
        shares_bought.loc[mask] = buy_amt[mask] / px_exec[mask]
        fee.loc[mask] = buy_amt[mask] * float(params.fee_rate)
    
    # Cash flow: negative for buys (cash outflow)
    cash_flow = -(buy_amt + fee)
    
    # Cumulative values
    cum_shares = shares_bought.cumsum()
    cum_cash_flow = cash_flow.cumsum()
    
    # Cumulative invested (positive, monotone increasing)
    cum_invested = -np.minimum(cum_cash_flow, 0.0)

    # Equity (mark-to-market) and residual cash balance
    equity = cum_shares * adj
    cash_balance = cum_invested + cum_cash_flow

    # NAV represents total account value (equity + remaining cash)
    nav = equity + cash_balance
    
    # Drawdown: only compute after first investment
    drawdown = pd.Series(np.nan, index=idx)
    first_invest_idx = (cum_invested > 0)
    if first_invest_idx.any():
        first_invest_date = cum_invested[first_invest_idx].index[0]
        nav_post = nav.loc[first_invest_date:]
        if len(nav_post) > 1:
            running_max = nav_post.expanding().max()
            drawdown_post = (nav_post / running_max) - 1.0
            drawdown.loc[first_invest_date:] = drawdown_post
        else:
            drawdown.loc[first_invest_date:] = 0.0
    
    # Mode string
    mode_str = "shares" if is_shares_mode else (params.mode or "unknown")
    
    ledger = pd.DataFrame({
        "DailyRet": daily_ret,
        "Signal": signals.astype(int),
        "Mode": mode_str,
        "AdjClose": adj,
        "PxExec": px_exec,
        "BuyAmt": buy_amt,
        "Fee": fee,
        "SharesBought": shares_bought,
        "CashFlow": cash_flow,
        "CumCashFlow": cum_cash_flow,
        "CumShares": cum_shares,
        "Equity": equity,
        "CashBalance": cash_balance,
        "NAV": nav,
        "CumInvested": cum_invested,
        "Drawdown": drawdown,
    }, index=idx)
    
    # Validate invariants
    _validate_ledger(ledger)
    
    return ledger


def _validate_ledger(ledger: pd.DataFrame) -> None:
    """Validate accounting invariants."""
    cum_shares = ledger["CumShares"]
    cum_cash_flow = ledger["CumCashFlow"]
    nav = ledger["NAV"]
    equity = ledger["Equity"]
    cash_balance = ledger["CashBalance"]
    cum_invested = ledger["CumInvested"]
    drawdown = ledger["Drawdown"]
    
    # CumShares is non-decreasing
    if not (cum_shares.diff().fillna(0.0) >= -1e-10).all():
        raise ValueError("CumShares must be non-decreasing")
    
    # CumInvested is non-decreasing and equals -min(CumCashFlow, 0)
    expected_cum_invested = -np.minimum(cum_cash_flow, 0.0)
    if not np.allclose(cum_invested, expected_cum_invested, rtol=1e-8, atol=1e-8):
        raise ValueError(f"CumInvested must equal -min(CumCashFlow, 0); max diff: {np.abs(cum_invested - expected_cum_invested).max()}")
    
    cum_invested_diff = cum_invested.diff().fillna(0.0)
    if not (cum_invested_diff >= -1e-10).all():
        raise ValueError("CumInvested must be non-decreasing")
    
    # Cash balance should equal invested capital plus cash flows
    expected_cash_balance = cum_invested + cum_cash_flow
    if not np.allclose(cash_balance, expected_cash_balance, rtol=1e-8, atol=1e-8):
        max_diff = np.abs(cash_balance - expected_cash_balance).max()
        raise ValueError(f"CashBalance mismatch; max diff: {max_diff}")

    # NAV == Equity + CashBalance
    nav_computed = equity + cash_balance
    if not np.allclose(nav, nav_computed, rtol=1e-8, atol=1e-8):
        max_diff = np.abs(nav - nav_computed).max()
        raise ValueError(f"NAV must equal Equity + CashBalance; max diff: {max_diff}")
    
    # Drawdown in valid range (ignore NaN)
    drawdown_valid = drawdown.dropna()
    if len(drawdown_valid) > 0:
        min_dd = drawdown_valid.min()
        max_dd = drawdown_valid.max()
        if min_dd < -1.0 - 1e-9:
            raise ValueError(f"Drawdown too negative: {min_dd}")
        if max_dd > 1e-9:
            raise ValueError(f"Drawdown positive: {max_dd}")


def run_backtest(prices: pd.DataFrame, params: BacktestParams) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run dip-buy backtest on provided price data.

    Args:
        prices: DataFrame with columns ['Open','High','Low','Close','AdjClose','Volume'].
        params: Backtest parameters.

    Returns:
        Tuple of (daily_df, metrics_dict).
    """
    if prices is None or prices.empty:
        raise ValueError("Price data is empty")
    required = ["AdjClose"]
    for c in required:
        if c not in prices.columns:
            raise ValueError(f"Missing required column: {c}")

    prices = prices.sort_index().copy()
    
    # Get allocation (needed for budget-based mode to determine weekly budget splits)
    # For shares-based mode, allocation is computed but ledger recalculates from scratch
    allocation = _compute_allocation(prices, params)
    
    # Compute complete ledger (handles both modes correctly)
    ledger = compute_ledger(prices, params, allocation)
    
    idx = ledger.index
    nav = ledger["NAV"]
    equity = ledger["Equity"]
    cum_cash_flow = ledger["CumCashFlow"]
    cum_invested = ledger["CumInvested"]
    drawdown = ledger["Drawdown"]
    signals = ledger["Signal"].astype(bool)
    
    # Find first investment date
    first_invest_mask = cum_invested > 0
    if not first_invest_mask.any():
        # No trades executed
        metrics: Dict[str, float] = {
            "TotalInvested": 0.0,
            "EndingEquity": 0.0,
            "EndingNAV": 0.0,
            "CumulativeReturn": 0.0,
            "CAGR": 0.0,
            "MDD": 0.0,
            "Trades": 0.0,
            "HitDays": float(int(signals.sum())),
            "XIRR": 0.0,
        }
        return ledger, metrics
    
    first_invest_date = cum_invested[first_invest_mask].index[0]
    first_invest_nav = nav.loc[first_invest_date]
    first_invest_amount = cum_invested.loc[first_invest_date]
    
    # Metrics
    total_invested = float(cum_invested.iloc[-1])
    ending_equity = float(equity.iloc[-1])
    ending_nav = float(nav.iloc[-1])
    hit_days = int(signals.sum())
    trades = int((ledger["BuyAmt"] > 0).sum())
    
    # Cumulative return
    if total_invested > 1e-9:
        cumulative_return = (ending_nav / total_invested) - 1.0
    else:
        cumulative_return = 0.0
    
    # MDD: computed from first investment date
    nav_post = nav.loc[first_invest_date:]
    if len(nav_post) > 1:
        mdd_value = float(max_drawdown(nav_post))
    else:
        mdd_value = 0.0
    
    # CAGR: from first investment to end
    last_date = idx[-1]
    days = (last_date - first_invest_date).days if hasattr(last_date, 'days') else (pd.Timestamp(last_date) - pd.Timestamp(first_invest_date)).days
    if days > 0 and total_invested > 1e-9 and ending_nav > 0:
        years = days / 365.25
        cagr_value = cagr_func(total_invested, ending_nav, years)
    else:
        cagr_value = 0.0
    
    # XIRR: only non-zero cash flows
    cash_flows = ledger["CashFlow"]
    nonzero_cf_mask = np.abs(cash_flows) > 1e-10
    if nonzero_cf_mask.any():
        cf_dates = list(idx[nonzero_cf_mask].to_pydatetime())
        cf_values = list(cash_flows[nonzero_cf_mask].values)
        # Add terminal liquidation
        cf_dates.append(idx[-1].to_pydatetime())
        cf_values.append(float(nav.iloc[-1]))
        xirr_value = xirr(cf_values, cf_dates)
    else:
        xirr_value = 0.0
    
    # Validate MDD is reasonable
    if abs(mdd_value) > 1.2:
        import warnings
        warnings.warn(
            f"MDD value {mdd_value:.2%} seems extreme. "
            f"First invest NAV: {first_invest_nav:.2f}, Ending NAV: {ending_nav:.2f}"
        )
    
    metrics: Dict[str, float] = {
        "TotalInvested": round(total_invested, 6),
        "EndingEquity": round(ending_equity, 6),
        "EndingNAV": round(ending_nav, 6),
        "CumulativeReturn": float(cumulative_return),
        "CAGR": float(cagr_value),
        "MDD": float(mdd_value),
        "Trades": float(trades),
        "HitDays": float(hit_days),
        "XIRR": float(xirr_value),
    }
    
    return ledger, metrics

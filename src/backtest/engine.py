from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd

from .metrics import cagr as cagr_func
from .metrics import max_drawdown, xirr
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
    # Take-Profit / Stop-Loss parameters
    enable_tp_sl: bool = False
    tp_threshold: float | None = None  # Take-profit threshold as decimal (e.g., 0.30 for 30%)
    sl_threshold: float | None = None  # Stop-loss threshold as decimal (e.g., -0.25 for -25%)
    tp_sell_percentage: float = 1.0  # Fraction of shares to sell on TP trigger (0.25, 0.50, 0.75, or 1.0)
    sl_sell_percentage: float = 1.0  # Fraction of shares to sell on SL trigger (0.25, 0.50, 0.75, or 1.0)
    # Baseline reset and hysteresis/cooldown parameters
    reset_baseline_after_tp_sl: bool = True  # Reset ROI baseline after TP/SL trigger (default: ON)
    tp_hysteresis: float = 0.0  # TP hysteresis percentage (e.g., 0.10 for 10%, default: 0 = disabled)
    sl_hysteresis: float = 0.0  # SL hysteresis percentage (e.g., 0.10 for 10%, default: 0 = disabled)
    tp_cooldown_days: int = 0  # TP cooldown period in days (default: 0 = disabled)
    sl_cooldown_days: int = 0  # SL cooldown period in days (default: 0 = disabled)
    
    def __post_init__(self):
        """Validate TP/SL parameters."""
        if self.enable_tp_sl:
            if self.tp_threshold is None:
                raise ValueError("tp_threshold must be provided when enable_tp_sl is True")
            if self.sl_threshold is None:
                raise ValueError("sl_threshold must be provided when enable_tp_sl is True")
            if self.tp_threshold <= 0:
                raise ValueError("tp_threshold must be positive (e.g., 0.30 for 30%)")
            if self.sl_threshold >= 0:
                raise ValueError("sl_threshold must be negative (e.g., -0.25 for -25%)")
            if self.tp_sell_percentage <= 0 or self.tp_sell_percentage > 1.0:
                raise ValueError("tp_sell_percentage must be between 0 and 1.0 (e.g., 0.25 for 25%)")
            if self.sl_sell_percentage <= 0 or self.sl_sell_percentage > 1.0:
                raise ValueError("sl_sell_percentage must be between 0 and 1.0 (e.g., 0.25 for 25%)")
            if self.tp_hysteresis < 0:
                raise ValueError("tp_hysteresis must be non-negative")
            if self.sl_hysteresis < 0:
                raise ValueError("sl_hysteresis must be non-negative")
            if self.tp_cooldown_days < 0:
                raise ValueError("tp_cooldown_days must be non-negative")
            if self.sl_cooldown_days < 0:
                raise ValueError("sl_cooldown_days must be non-negative")


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
    
    Supports Take-Profit and Stop-Loss functionality when enabled.
    Processes days sequentially to handle dynamic sell events.
    
    Returns DataFrame with columns:
    DailyRet, Signal, Mode, AdjClose, PxExec, BuyAmt, Fee, SharesBought,
    CashFlow, CumCashFlow, CumShares, Equity, CashBalance, NAV,
    CumInvested, Drawdown, NAVReturn, PositionCost, TP_triggered, SL_triggered,
    SharesSold, SellPrice, GrossProceeds, SellFee, NetProceeds, RealizedPnl,
    PostSellCumShares
    """
    idx = prices.index
    adj = prices["AdjClose"].astype(float)
    
    # Daily returns and signals
    daily_ret = adj.pct_change()
    signals = generate_dip_signals(adj, params.threshold)
    
    # Execution price for buys (with slippage)
    px_exec_buy = adj * (1.0 + float(params.slippage_rate))
    # Execution price for sells (with slippage deducted)
    px_exec_sell = adj * (1.0 - float(params.slippage_rate))
    
    # Convert to numpy arrays for faster indexing (performance optimization)
    adj_arr = adj.values
    px_exec_buy_arr = px_exec_buy.values
    px_exec_sell_arr = px_exec_sell.values
    
    # Determine mode
    is_shares_mode = params.shares_per_signal is not None
    
    # Initialize arrays for all columns
    n = len(idx)
    buy_amt = np.zeros(n, dtype=float)
    fee = np.zeros(n, dtype=float)
    shares_bought = np.zeros(n, dtype=float)
    cash_flow = np.zeros(n, dtype=float)
    shares_sold = np.zeros(n, dtype=float)
    sell_price = np.zeros(n, dtype=float)
    gross_proceeds = np.zeros(n, dtype=float)
    sell_fee = np.zeros(n, dtype=float)
    net_proceeds = np.zeros(n, dtype=float)
    realized_pnl = np.zeros(n, dtype=float)
    tp_triggered = np.zeros(n, dtype=bool)
    sl_triggered = np.zeros(n, dtype=bool)
    
    # State variables (updated sequentially)
    cum_shares = 0.0
    cum_cash_flow = 0.0
    cum_invested = 0.0
    position_cost = 0.0  # Cost basis of current holdings
    roi_base: float | None = None  # Baseline NAV for TP/SL calculation (reset after TP/SL)
    last_tp_date_idx: int | None = None  # Last TP trigger date index (for cooldown)
    last_sl_date_idx: int | None = None  # Last SL trigger date index (for cooldown)
    tp_hysteresis_active = False  # TP hysteresis active state
    sl_hysteresis_active = False  # SL hysteresis active state
    
    # Arrays to store cumulative values
    cum_shares_arr = np.zeros(n, dtype=float)
    cum_cash_flow_arr = np.zeros(n, dtype=float)
    cum_invested_arr = np.zeros(n, dtype=float)
    position_cost_arr = np.zeros(n, dtype=float)
    equity_arr = np.zeros(n, dtype=float)
    cash_balance_arr = np.zeros(n, dtype=float)
    nav_arr = np.zeros(n, dtype=float)
    nav_return_arr = np.zeros(n, dtype=float)
    portfolio_return_arr = np.zeros(n, dtype=float)
    unrealized_pnl_arr = np.zeros(n, dtype=float)
    profit_arr = np.zeros(n, dtype=float)  # Profit = Equity + CumCashFlow
    nav_including_invested_arr = np.zeros(n, dtype=float)  # NAV_including_invested = CumInvested + Profit
    roi_base_arr = np.zeros(n, dtype=float)
    roi_base_arr[:] = np.nan  # Initialize as NaN (will be set when holdings exist)
    nav_return_global_arr = np.zeros(n, dtype=float)
    nav_return_baselined_arr = np.zeros(n, dtype=float)
    nav_return_baselined_arr[:] = np.nan  # Initialize as NaN (will be set when roi_base exists)
    post_sell_cum_shares_arr = np.zeros(n, dtype=float)
    
    # Process each day sequentially
    for i, day in enumerate(idx):
        # Use array indexing for performance (faster than .loc)
        adj_price = adj_arr[i]
        px_exec_buy_price = px_exec_buy_arr[i]
        px_exec_sell_price = px_exec_sell_arr[i]
        
        # Process buy logic first
        if is_shares_mode:
            # Shares-based mode
            if signals.iloc[i]:  # Use iloc for faster indexing
                shares_fixed = float(params.shares_per_signal)
                shares_bought[i] = shares_fixed
                buy_amt[i] = shares_fixed * px_exec_buy_price
                fee[i] = buy_amt[i] * float(params.fee_rate)
        else:
            # Budget-based mode
            alloc_val = allocation.iloc[i]  # Use iloc for faster indexing
            if alloc_val > 0:
                buy_amt[i] = alloc_val
                shares_bought[i] = buy_amt[i] / px_exec_buy_price
                fee[i] = buy_amt[i] * float(params.fee_rate)
        
        # Update state for buys
        if shares_bought[i] > 0:
            total_cost = buy_amt[i] + fee[i]
            cum_shares += shares_bought[i]
            position_cost += total_cost
            cum_invested += total_cost
            cash_flow[i] = -total_cost
        
        cum_cash_flow += cash_flow[i]
        
        # Calculate current equity and NAV
        equity = cum_shares * adj_price
        cash_balance = cum_invested + cum_cash_flow
        nav = equity + cash_balance
        
        # Calculate Portfolio Return (average cost basis vs current value)
        if position_cost > 1e-9 and cum_shares > 1e-9:
            portfolio_return = (equity / position_cost) - 1.0
        else:
            portfolio_return = 0.0
        
        # Calculate unrealized P/L (current position value - cost basis)
        unrealized_pnl = equity - position_cost
        
        # Calculate Profit = Equity + CumCashFlow (net profit/loss)
        profit = equity + cum_cash_flow
        
        # Calculate NAV_including_invested = CumInvested + Profit
        nav_including_invested = cum_invested + profit
        
        # Initialize ROI_base on first buy (after NAV calculation)
        if roi_base is None and cum_shares > 1e-9 and nav > 1e-9:
            roi_base = nav
        
        # Calculate NAVReturn_global (monitoring/reporting, kept for compatibility)
        if cum_invested > 1e-9:
            nav_return_global = (nav / cum_invested) - 1.0
        else:
            nav_return_global = 0.0
        
        # Calculate NAVReturn_baselined (for TP/SL trigger evaluation)
        # Only calculate if TP/SL is enabled or if we need it for output
        nav_return_baselined = 0.0
        if params.enable_tp_sl and roi_base is not None and roi_base > 1e-9:
            nav_return_baselined = (nav / roi_base) - 1.0
        
        # Calculate NAV Return from investment (for historical reference, same as global)
        nav_return = nav_return_global
        
        # Check TP/SL triggers (only if enabled and we have holdings and baseline)
        if params.enable_tp_sl and cum_shares > 0 and position_cost > 1e-9 and roi_base is not None and roi_base > 1e-9:
            trigger_fired = False
            
            # Check cooldown periods
            tp_cooldown_active = False
            sl_cooldown_active = False
            
            if params.tp_cooldown_days > 0 and last_tp_date_idx is not None:
                days_since_tp = i - last_tp_date_idx
                if days_since_tp < params.tp_cooldown_days:
                    tp_cooldown_active = True
            
            if params.sl_cooldown_days > 0 and last_sl_date_idx is not None:
                days_since_sl = i - last_sl_date_idx
                if days_since_sl < params.sl_cooldown_days:
                    sl_cooldown_active = True
            
            # Check Take-Profit first (based on baselined return)
            if params.tp_threshold is not None and not tp_cooldown_active:
                # Check hysteresis: if hysteresis is active, require return to drop below threshold - hysteresis first
                if tp_hysteresis_active:
                    # Require return to drop below threshold - hysteresis before re-arming
                    if nav_return_baselined < params.tp_threshold - params.tp_hysteresis:
                        tp_hysteresis_active = False  # Re-arm TP
                
                # Check TP trigger (only if not in hysteresis or already re-armed)
                if not tp_hysteresis_active and nav_return_baselined >= params.tp_threshold:
                    tp_triggered[i] = True
                    trigger_fired = True
                    # Activate hysteresis if configured
                    if params.tp_hysteresis > 0:
                        tp_hysteresis_active = True
            
            # Check Stop-Loss (only if TP didn't trigger, based on baselined return)
            if not trigger_fired and params.sl_threshold is not None and not sl_cooldown_active:
                # Check hysteresis: if hysteresis is active, require return to rise above threshold + hysteresis first
                if sl_hysteresis_active:
                    # Require return to rise above threshold + hysteresis before re-arming
                    # Note: sl_threshold is negative, so threshold + hysteresis means less negative
                    if nav_return_baselined > params.sl_threshold + params.sl_hysteresis:
                        sl_hysteresis_active = False  # Re-arm SL
                
                # Check SL trigger (only if not in hysteresis or already re-armed)
                if not sl_hysteresis_active and nav_return_baselined <= params.sl_threshold:
                    sl_triggered[i] = True
                    trigger_fired = True
                    # Activate hysteresis if configured
                    if params.sl_hysteresis > 0:
                        sl_hysteresis_active = True
            
            # Execute sell if triggered
            if trigger_fired:
                # Determine which trigger fired and use corresponding sell percentage
                if tp_triggered[i]:
                    sell_frac = float(params.tp_sell_percentage)
                else:  # SL triggered
                    sell_frac = float(params.sl_sell_percentage)
                
                # Calculate shares to sell
                # Rounding rule: round up if >= 0.5, round down if < 0.5
                # Minimum 1 share if rounding yields 0 but we have shares
                shares_to_sell_float = cum_shares * sell_frac
                shares_to_sell = int(shares_to_sell_float + 0.5)  # Standard rounding: 0.5 and above rounds up
                
                # Edge case: if rounding yields 0 but we have shares, force at least 1 share
                # But only if sell_percentage is large enough (e.g., > 0.01)
                if shares_to_sell == 0 and cum_shares > 0:
                    if sell_frac > 0.01:  # Only force if selling more than 1%
                        shares_to_sell = 1
                    else:
                        # Skip sale if rounding yields 0 and percentage is very small
                        shares_to_sell = 0
                
                # Cap at current holdings
                shares_to_sell = min(shares_to_sell, int(cum_shares))
                
                if shares_to_sell > 0:
                    # Calculate sell price and proceeds
                    sell_price[i] = px_exec_sell_price
                    gross_proceeds[i] = float(shares_to_sell) * px_exec_sell_price
                    sell_fee[i] = gross_proceeds[i] * float(params.fee_rate)
                    net_proceeds[i] = gross_proceeds[i] - sell_fee[i]
                    
                    # Calculate cost basis of shares sold (average cost method)
                    if cum_shares > 1e-9:
                        avg_cost_per_share = position_cost / cum_shares
                        cost_of_shares_sold = avg_cost_per_share * float(shares_to_sell)
                    else:
                        cost_of_shares_sold = position_cost  # If selling all shares, use all position cost
                    
                    # Calculate realized P/L
                    realized_pnl[i] = net_proceeds[i] - cost_of_shares_sold
                    
                    # Update state
                    shares_sold[i] = float(shares_to_sell)
                    cum_shares -= float(shares_to_sell)
                    position_cost -= cost_of_shares_sold
                    cum_cash_flow += net_proceeds[i]
                    cash_flow[i] += net_proceeds[i]  # Add sell proceeds to cash flow
                    
                    # Ensure position_cost doesn't go negative due to rounding
                    if position_cost < -1e-10:
                        position_cost = 0.0
                    
                    # Ensure cum_shares doesn't go negative
                    if cum_shares < -1e-10:
                        cum_shares = 0.0
                    
                    # Recalculate equity and NAV after sale
                    equity = cum_shares * adj_price
                    cash_balance = cum_invested + cum_cash_flow
                    nav = equity + cash_balance
                    
                    # Recalculate portfolio return after sale
                    if position_cost > 1e-9 and cum_shares > 1e-9:
                        portfolio_return = (equity / position_cost) - 1.0
                    else:
                        portfolio_return = 0.0
                    
                    # Recalculate unrealized P/L
                    unrealized_pnl = equity - position_cost
                    
                    # Recalculate Profit
                    profit = equity + cum_cash_flow
                    
                    # Recalculate NAV_including_invested
                    nav_including_invested = cum_invested + profit
                    
                    # Reset ROI baseline after TP/SL sale (if enabled)
                    if params.reset_baseline_after_tp_sl:
                        roi_base = nav  # Reset baseline to NAV after trade
                        # Reset hysteresis states when baseline resets
                        tp_hysteresis_active = False
                        sl_hysteresis_active = False
                    
                    # Update last trigger dates for cooldown
                    if tp_triggered[i]:
                        last_tp_date_idx = i
                    if sl_triggered[i]:
                        last_sl_date_idx = i
                    
                    # Recalculate NAVReturn_global
                    if cum_invested > 1e-9:
                        nav_return_global = (nav / cum_invested) - 1.0
                    else:
                        nav_return_global = 0.0
                    
                    # Recalculate NAVReturn_baselined (after baseline reset)
                    if roi_base is not None and roi_base > 1e-9:
                        nav_return_baselined = (nav / roi_base) - 1.0
                    else:
                        nav_return_baselined = 0.0
                    
                    # Recalculate NAV return (same as global)
                    nav_return = nav_return_global
        
        # Store values for this day
        cum_shares_arr[i] = cum_shares
        cum_cash_flow_arr[i] = cum_cash_flow
        cum_invested_arr[i] = cum_invested
        position_cost_arr[i] = position_cost
        equity_arr[i] = equity
        cash_balance_arr[i] = cash_balance
        nav_arr[i] = nav
        nav_return_arr[i] = nav_return_global  # Store global return (monitoring)
        portfolio_return_arr[i] = portfolio_return
        unrealized_pnl_arr[i] = unrealized_pnl
        profit_arr[i] = profit
        nav_including_invested_arr[i] = nav_including_invested
        if roi_base is not None:
            roi_base_arr[i] = roi_base
        nav_return_global_arr[i] = nav_return_global
        if params.enable_tp_sl and roi_base is not None and roi_base > 1e-9:
            nav_return_baselined_arr[i] = nav_return_baselined
        elif roi_base is not None and roi_base > 1e-9:
            # Calculate even if TP/SL disabled for output consistency
            nav_return_baselined_arr[i] = (nav / roi_base) - 1.0
        # else: remains NaN (initialized)
        post_sell_cum_shares_arr[i] = cum_shares
    
    # Calculate drawdown: only compute after first investment
    drawdown = pd.Series(np.nan, index=idx)
    first_invest_mask = cum_invested_arr > 1e-9
    if first_invest_mask.any():
        first_invest_date_idx = np.where(first_invest_mask)[0][0]
        nav_post = pd.Series(nav_arr[first_invest_date_idx:], index=idx[first_invest_date_idx:])
        if len(nav_post) > 1:
            running_max = nav_post.expanding().max()
            drawdown_post = (nav_post / running_max) - 1.0
            drawdown.loc[idx[first_invest_date_idx]:] = drawdown_post.values
        else:
            drawdown.loc[idx[first_invest_date_idx]:] = 0.0
    
    # Mode string
    mode_str = "shares" if is_shares_mode else (params.mode or "unknown")
    
    # Build DataFrame
    ledger = pd.DataFrame({
        "DailyRet": daily_ret,
        "Signal": signals.astype(int),
        "Mode": mode_str,
        "AdjClose": adj,
        "PxExec": px_exec_buy,
        "BuyAmt": pd.Series(buy_amt, index=idx),
        "Fee": pd.Series(fee, index=idx),
        "SharesBought": pd.Series(shares_bought, index=idx),
        "CashFlow": pd.Series(cash_flow, index=idx),
        "CumCashFlow": pd.Series(cum_cash_flow_arr, index=idx),
        "CumShares": pd.Series(cum_shares_arr, index=idx),
        "Equity": pd.Series(equity_arr, index=idx),
        "CashBalance": pd.Series(cash_balance_arr, index=idx),
        "NAV": pd.Series(nav_arr, index=idx),
        "CumInvested": pd.Series(cum_invested_arr, index=idx),
        "Drawdown": drawdown,
        "NAVReturn": pd.Series(nav_return_arr, index=idx),
        "NAVReturn_global": pd.Series(nav_return_global_arr, index=idx),
        "NAVReturn_baselined": pd.Series(nav_return_baselined_arr, index=idx),
        "ROI_base": pd.Series(roi_base_arr, index=idx),
        "PortfolioReturn": pd.Series(portfolio_return_arr, index=idx),
        "PositionCost": pd.Series(position_cost_arr, index=idx),
        "UnrealizedPnl": pd.Series(unrealized_pnl_arr, index=idx),
        "Profit": pd.Series(profit_arr, index=idx),
        "NAV_including_invested": pd.Series(nav_including_invested_arr, index=idx),
        "TP_triggered": pd.Series(tp_triggered, index=idx),
        "SL_triggered": pd.Series(sl_triggered, index=idx),
        "SharesSold": pd.Series(shares_sold, index=idx),
        "SellPrice": pd.Series(sell_price, index=idx),
        "GrossProceeds": pd.Series(gross_proceeds, index=idx),
        "SellFee": pd.Series(sell_fee, index=idx),
        "NetProceeds": pd.Series(net_proceeds, index=idx),
        "RealizedPnl": pd.Series(realized_pnl, index=idx),
        "PostSellCumShares": pd.Series(post_sell_cum_shares_arr, index=idx),
    }, index=idx)
    
    # Validate invariants
    _validate_ledger(ledger, params)
    
    return ledger


def _validate_ledger(ledger: pd.DataFrame, params: BacktestParams) -> None:
    """Validate accounting invariants including TP/SL logic."""
    cum_shares = ledger["CumShares"]
    cum_cash_flow = ledger["CumCashFlow"]
    nav = ledger["NAV"]
    equity = ledger["Equity"]
    cash_balance = ledger["CashBalance"]
    cum_invested = ledger["CumInvested"]
    drawdown = ledger["Drawdown"]
    
    # CumShares should never go negative
    if (cum_shares < -1e-10).any():
        raise ValueError("CumShares must never be negative")
    
    # CumShares can decrease on sells (TP/SL), so it's not strictly non-decreasing anymore
    # But we validate it never goes negative above
    
    # CumInvested validation
    # When TP/SL is enabled and sells occur, CumCashFlow can become positive
    # In that case, CumInvested should equal the maximum cumulative investment made
    # For no-sell case: CumInvested = -min(CumCashFlow, 0)
    # For sell case: CumInvested = max(-min(CumCashFlow, 0)) where max is over all days
    
    # Check if there are any sells (TP/SL enabled)
    has_sells = False
    if "SharesSold" in ledger.columns:
        has_sells = (ledger["SharesSold"] > 1e-9).any()
    
    if has_sells:
        # With sells, CumInvested should equal the maximum cumulative investment
        # This is the highest value that CumInvested ever reached
        max_cum_invested = cum_invested.max()
        # CumInvested should be non-decreasing and constant after the last investment
        # Actually, CumInvested should equal the cumulative sum of all buy costs
        # Let's validate: CumInvested should equal the cumulative sum of (BuyAmt + Fee) for all buys
        buy_costs = ledger["BuyAmt"] + ledger["Fee"]
        expected_cum_invested_from_buys = buy_costs.cumsum()
        
        # They should match (with small tolerance for rounding)
        if not np.allclose(cum_invested, expected_cum_invested_from_buys, rtol=1e-8, atol=1e-8):
            max_diff = np.abs(cum_invested - expected_cum_invested_from_buys).max()
            raise ValueError(f"CumInvested must equal cumulative sum of buy costs; max diff: {max_diff}")
    else:
        # No sells: original validation holds
        expected_cum_invested = -np.minimum(cum_cash_flow, 0.0)
        if not np.allclose(cum_invested, expected_cum_invested, rtol=1e-8, atol=1e-8):
            max_diff = np.abs(cum_invested - expected_cum_invested).max()
            raise ValueError(f"CumInvested must equal -min(CumCashFlow, 0); max diff: {max_diff}")
    
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
    
    # TP/SL specific validations
    if "NAVReturn" in ledger.columns:
        nav_return = ledger["NAVReturn"]
        position_cost = ledger["PositionCost"]
        shares_sold = ledger["SharesSold"]
        realized_pnl = ledger["RealizedPnl"]
        net_proceeds = ledger["NetProceeds"]
        
        # NAVReturn validation: NAVReturn = (NAV / CumInvested) - 1
        nav_return_computed = pd.Series(0.0, index=ledger.index)
        mask = cum_invested > 1e-9
        nav_return_computed.loc[mask] = (nav.loc[mask] / cum_invested.loc[mask]) - 1.0
        if not np.allclose(nav_return.loc[mask], nav_return_computed.loc[mask], rtol=1e-8, atol=1e-8):
            max_diff = np.abs(nav_return.loc[mask] - nav_return_computed.loc[mask]).max()
            raise ValueError(f"NAVReturn mismatch; max diff: {max_diff}")
        
        # NAVReturn_global validation: NAVReturn_global = (NAV / CumInvested) - 1
        if "NAVReturn_global" in ledger.columns:
            nav_return_global = ledger["NAVReturn_global"]
            nav_return_global_computed = pd.Series(0.0, index=ledger.index)
            nav_return_global_computed.loc[mask] = (nav.loc[mask] / cum_invested.loc[mask]) - 1.0
            if not np.allclose(nav_return_global.loc[mask], nav_return_global_computed.loc[mask], rtol=1e-8, atol=1e-8):
                max_diff = np.abs(nav_return_global.loc[mask] - nav_return_global_computed.loc[mask]).max()
                raise ValueError(f"NAVReturn_global mismatch; max diff: {max_diff}")
        
        # NAVReturn_baselined validation: NAVReturn_baselined = (NAV / ROI_base) - 1
        if "NAVReturn_baselined" in ledger.columns and "ROI_base" in ledger.columns:
            nav_return_baselined = ledger["NAVReturn_baselined"]
            roi_base = ledger["ROI_base"]
            
            nav_return_baselined_computed = pd.Series(np.nan, index=ledger.index)
            baseline_mask = (~roi_base.isna()) & (roi_base > 1e-9)
            nav_return_baselined_computed.loc[baseline_mask] = (nav.loc[baseline_mask] / roi_base.loc[baseline_mask]) - 1.0
            
            # Compare only where both are not NaN
            valid_mask = baseline_mask & (~nav_return_baselined.isna())
            if valid_mask.any():
                if not np.allclose(nav_return_baselined.loc[valid_mask], nav_return_baselined_computed.loc[valid_mask], rtol=1e-8, atol=1e-8):
                    max_diff = np.abs(nav_return_baselined.loc[valid_mask] - nav_return_baselined_computed.loc[valid_mask]).max()
                    raise ValueError(f"NAVReturn_baselined mismatch; max diff: {max_diff}")
        
        # ROI_base validation: ROI_base should be positive when holdings exist
        if "ROI_base" in ledger.columns:
            roi_base = ledger["ROI_base"]
            holdings_mask = cum_shares > 1e-9
            roi_base_when_holdings = roi_base.loc[holdings_mask]
            if roi_base_when_holdings.notna().any():
                if (roi_base_when_holdings[roi_base_when_holdings.notna()] <= 0).any():
                    raise ValueError("ROI_base must be positive when holdings exist")
        
        # PortfolioReturn validation: PortfolioReturn = (Equity / PositionCost) - 1
        if "PortfolioReturn" in ledger.columns:
            portfolio_return = ledger["PortfolioReturn"]
            equity = ledger["Equity"]
            
            portfolio_return_computed = pd.Series(0.0, index=ledger.index)
            portfolio_mask = position_cost > 1e-9
            portfolio_return_computed.loc[portfolio_mask] = (equity.loc[portfolio_mask] / position_cost.loc[portfolio_mask]) - 1.0
            if not np.allclose(portfolio_return.loc[portfolio_mask], portfolio_return_computed.loc[portfolio_mask], rtol=1e-8, atol=1e-8):
                max_diff = np.abs(portfolio_return.loc[portfolio_mask] - portfolio_return_computed.loc[portfolio_mask]).max()
                raise ValueError(f"PortfolioReturn mismatch; max diff: {max_diff}")
        
        # UnrealizedPnl validation: UnrealizedPnl = Equity - PositionCost
        if "UnrealizedPnl" in ledger.columns:
            unrealized_pnl = ledger["UnrealizedPnl"]
            unrealized_pnl_computed = equity - position_cost
            if not np.allclose(unrealized_pnl, unrealized_pnl_computed, rtol=1e-8, atol=1e-8):
                max_diff = np.abs(unrealized_pnl - unrealized_pnl_computed).max()
                raise ValueError(f"UnrealizedPnl mismatch; max diff: {max_diff}")
        
        # PositionCost validation
        if (position_cost < -1e-10).any():
            raise ValueError("PositionCost must never be negative")
        if (position_cost > cum_invested + 1e-8).any():
            raise ValueError("PositionCost cannot exceed CumInvested")
        
        # After full liquidation, PositionCost should be 0
        full_liquidation_mask = cum_shares < 1e-9
        if full_liquidation_mask.any():
            position_cost_after_liquidation = position_cost.loc[full_liquidation_mask]
            if not (position_cost_after_liquidation.abs() < 1e-6).all():
                max_cost = position_cost_after_liquidation.abs().max()
                raise ValueError(f"PositionCost should be 0 after full liquidation; max: {max_cost}")
        
        # SharesSold validation - ensure it doesn't exceed shares held at that moment
        # Note: This is checked during execution, but validate retroactively
        if (shares_sold < 0).any():
            raise ValueError("SharesSold must never be negative")
        
        # Validate that shares sold don't exceed what was held (check sequentially)
        # For each day with a sell, verify shares_sold <= CumShares_before_sell
        # Since we process sequentially, CumShares after buy should be >= shares_sold
        # We can't easily check this retroactively without tracking state, so we rely on execution-time checks
        
        # RealizedPnl validation: RealizedPnl = NetProceeds - CostBasis
        # Cost basis is calculated as: PositionCost_before / CumShares_before * SharesSold
        # This is validated by checking consistency
        sell_mask = shares_sold > 1e-9
        if sell_mask.any():
            # For each sell, verify P/L calculation
            for idx_val in ledger.index[sell_mask]:
                shares_sold_val = shares_sold.loc[idx_val]
                net_proceeds_val = net_proceeds.loc[idx_val]
                realized_pnl_val = realized_pnl.loc[idx_val]
                
                # Get state before sell (need to track this, but for validation we can approximate)
                # This is complex to validate retroactively, so we'll validate the accounting is consistent
                # The main check is that RealizedPnl matches the pattern
                if not np.isfinite(realized_pnl_val):
                    raise ValueError(f"RealizedPnl must be finite at {idx_val}")
        
        # Sum of SharesSold + final CumShares should equal total SharesBought
        total_shares_bought = ledger["SharesBought"].sum()
        total_shares_sold = shares_sold.sum()
        final_shares = cum_shares.iloc[-1]
        if not np.allclose(total_shares_bought, total_shares_sold + final_shares, rtol=1e-8, atol=1e-8):
            diff = abs(total_shares_bought - (total_shares_sold + final_shares))
            raise ValueError(f"Shares accounting mismatch: Bought={total_shares_bought:.6f}, Sold={total_shares_sold:.6f}, Final={final_shares:.6f}, Diff={diff:.6f}")
        
        # TP/SL trigger validation
        tp_triggered = ledger["TP_triggered"]
        sl_triggered = ledger["SL_triggered"]
        
        # Only one trigger per day
        both_triggered = tp_triggered & sl_triggered
        if both_triggered.any():
            raise ValueError("TP and SL cannot both trigger on the same day")
        
        # If TP/SL triggered, SharesSold should be > 0
        trigger_mask = tp_triggered | sl_triggered
        if trigger_mask.any():
            shares_sold_on_trigger = shares_sold.loc[trigger_mask]
            if (shares_sold_on_trigger < 1e-9).any():
                raise ValueError("SharesSold must be > 0 when TP/SL is triggered")


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
    ending_profit = float(ledger["Profit"].iloc[-1])
    ending_nav_including_invested = float(ledger["NAV_including_invested"].iloc[-1])
    hit_days = int(signals.sum())
    trades = int((ledger["BuyAmt"] > 0).sum())
    
    # TP/SL metrics
    num_take_profits = 0
    num_stop_losses = 0
    total_realized_gain = 0.0
    total_realized_loss = 0.0
    ending_position_cost = 0.0
    
    if "TP_triggered" in ledger.columns:
        num_take_profits = int(ledger["TP_triggered"].sum())
        num_stop_losses = int(ledger["SL_triggered"].sum())
        
        realized_pnl = ledger["RealizedPnl"]
        total_realized_gain = float(realized_pnl[realized_pnl > 0].sum())
        total_realized_loss = float(abs(realized_pnl[realized_pnl < 0].sum()))
        
        if "PositionCost" in ledger.columns:
            ending_position_cost = float(ledger["PositionCost"].iloc[-1])
    
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
        "Profit": round(ending_profit, 6),
        "NAV_including_invested": round(ending_nav_including_invested, 6),
        "CumulativeReturn": float(cumulative_return),
        "CAGR": float(cagr_value),
        "MDD": float(mdd_value),
        "Trades": float(trades),
        "HitDays": float(hit_days),
        "XIRR": float(xirr_value),
        "NumTakeProfits": float(num_take_profits),
        "NumStopLosses": float(num_stop_losses),
        "TotalRealizedGain": round(total_realized_gain, 6),
        "TotalRealizedLoss": round(total_realized_loss, 6),
        "NetRealizedPnl": round(total_realized_gain - total_realized_loss, 6),
        "EndingPositionCost": round(ending_position_cost, 6),
    }
    
    return ledger, metrics

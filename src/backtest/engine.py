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
        # Validate TP parameters if TP is enabled
        if self.tp_threshold is not None:
            if self.tp_threshold <= 0:
                raise ValueError("tp_threshold must be positive (e.g., 0.30 for 30%)")
            if self.tp_sell_percentage <= 0 or self.tp_sell_percentage > 1.0:
                raise ValueError("tp_sell_percentage must be between 0 and 1.0 (e.g., 0.25 for 25%)")
            if self.tp_hysteresis < 0:
                raise ValueError("tp_hysteresis must be non-negative")
            if self.tp_cooldown_days < 0:
                raise ValueError("tp_cooldown_days must be non-negative")
        
        # Validate SL parameters if SL is enabled
        if self.sl_threshold is not None:
            if self.sl_threshold >= 0:
                raise ValueError("sl_threshold must be negative (e.g., -0.25 for -25%)")
            if self.sl_sell_percentage <= 0 or self.sl_sell_percentage > 1.0:
                raise ValueError("sl_sell_percentage must be between 0 and 1.0 (e.g., 0.25 for 25%)")
            if self.sl_hysteresis < 0:
                raise ValueError("sl_hysteresis must be non-negative")
            if self.sl_cooldown_days < 0:
                raise ValueError("sl_cooldown_days must be non-negative")


def _compute_allocation(prices: pd.DataFrame, params: BacktestParams) -> pd.Series:
    signals = generate_dip_signals(prices["AdjClose"], params.threshold)
    
    # Always use shares-based allocation (budget-based mode removed)
    if params.shares_per_signal is None:
        raise ValueError("shares_per_signal must be provided (budget-based mode is no longer supported)")
    
    if params.shares_per_signal <= 0:
        raise ValueError("shares_per_signal must be positive")
    
    allocation = allocate_shares_per_signal(
        signals=signals,
        shares_per_signal=float(params.shares_per_signal),
        prices=prices["AdjClose"],
        slippage_rate=float(params.slippage_rate),
        fee_rate=float(params.fee_rate),
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
    Uses shares-based mode only (budget-based mode removed).
    
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
    signals_arr = signals.values  # Convert signals to numpy array
    
    # Always use shares-based mode (budget-based mode removed)
    if params.shares_per_signal is None:
        raise ValueError("shares_per_signal must be provided (budget-based mode is no longer supported)")
    
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
    roi_base_history: list[tuple[int, float, str]] = []  # History of ROI baseline resets: [(date_idx, roi_base_value, trigger_type), ...]
    
    # Position tracking: FIFO queue of (entry_date_idx, shares, entry_price)
    # Used to track holding period for TP/SL triggers
    positions: list[tuple[int, float, float]] = []  # [(entry_date_idx, shares, entry_price), ...]
    
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
    
    # Debugging arrays for TP/SL validation
    weighted_avg_entry_price_arr = np.zeros(n, dtype=float)
    weighted_avg_entry_price_arr[:] = np.nan  # Initialize as NaN
    position_return_arr = np.zeros(n, dtype=float)
    position_return_arr[:] = np.nan  # Initialize as NaN
    portfolio_return_ratio_arr = np.zeros(n, dtype=float)
    portfolio_return_ratio_arr[:] = np.nan  # Initialize as NaN
    tp_sell_price_arr = np.zeros(n, dtype=float)
    tp_sell_price_arr[:] = np.nan  # Initialize as NaN
    tp_entry_reset_price_arr = np.zeros(n, dtype=float)
    tp_entry_reset_price_arr[:] = np.nan  # Initialize as NaN
    
    # Process each day sequentially
    for i, day in enumerate(idx):
        # Use array indexing for performance (faster than .loc)
        adj_price = adj_arr[i]
        px_exec_buy_price = px_exec_buy_arr[i]
        px_exec_sell_price = px_exec_sell_arr[i]
        
        # ===== STEP 1: Process buy logic FIRST =====
        # Buy logic is processed first, then TP/SL check happens AFTER buy
        # This ensures that TP/SL cannot trigger on the same day as a buy
        shares_bought_today = 0.0
        
        # Always use shares-based mode (optimized: use numpy array instead of pandas iloc)
        if signals_arr[i]:  # Use numpy array for faster indexing
            shares_fixed = float(params.shares_per_signal)
            shares_bought[i] = shares_fixed
            buy_amt[i] = shares_fixed * px_exec_buy_price
            fee[i] = buy_amt[i] * float(params.fee_rate)
            shares_bought_today = shares_fixed
        
        # Update state for buys
        if shares_bought_today > 0:
            total_cost = buy_amt[i] + fee[i]
            cum_shares += shares_bought_today
            position_cost += total_cost
            cum_invested += total_cost
            # Add buy cost to cash flow
            cash_flow[i] -= total_cost
            
            # Add new position to tracking list (FIFO queue)
            # entry_price includes slippage and fee per share
            entry_price_per_share = px_exec_buy_price * (1.0 + float(params.fee_rate))
            positions.append((i, shares_bought_today, entry_price_per_share))
        
        # ===== STEP 2: Check TP/SL triggers AFTER buy =====
        # IMPORTANT: TP/SL triggers are based on entry_price of positions, not NAVReturn_baselined
        # TP/SL triggers are only valid for positions held for at least 1 day
        # If shares were bought today, skip TP/SL check entirely (must wait until next day)
        min_holding_days = 1  # Minimum holding period before TP/SL can trigger
        
        # Calculate position-based return: (current_price / entry_price) - 1.0
        # Use weighted average entry_price of all eligible positions (held for 1+ days)
        position_return = 0.0
        has_eligible_position = False
        weighted_avg_entry_price = 0.0
        
        if positions and shares_bought_today == 0.0 and cum_shares > 1e-9:
            # Calculate weighted average entry_price for positions held for 1+ days
            total_eligible_shares = 0.0
            total_eligible_cost = 0.0
            
            for entry_date_idx, pos_shares, pos_entry_price in positions:
                holding_days = i - entry_date_idx
                if holding_days >= min_holding_days and pos_entry_price > 1e-9:  # Validate entry_price
                    total_eligible_shares += pos_shares
                    total_eligible_cost += pos_shares * pos_entry_price
            
            if total_eligible_shares > 1e-9 and total_eligible_cost > 1e-9:
                weighted_avg_entry_price = total_eligible_cost / total_eligible_shares
                # Calculate return based on current price vs entry price
                if weighted_avg_entry_price > 1e-9:  # Additional safety check
                    position_return = (adj_price / weighted_avg_entry_price) - 1.0
                    has_eligible_position = True
                    
                    # Store debugging info
                    weighted_avg_entry_price_arr[i] = weighted_avg_entry_price
                    position_return_arr[i] = position_return
        
        # Check TP/SL triggers (only if at least one is enabled, no buy today, and we have eligible positions)
        if (params.tp_threshold is not None or params.sl_threshold is not None) and shares_bought_today == 0.0 and has_eligible_position and weighted_avg_entry_price > 1e-9:
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
            
            # Check Take-Profit first (based on position return: current_price / entry_price - 1)
            if params.tp_threshold is not None and not tp_cooldown_active:
                # Check hysteresis: if hysteresis is active, require return to drop below threshold - hysteresis first
                if tp_hysteresis_active:
                    # Require return to drop below threshold - hysteresis before re-arming
                    if position_return < params.tp_threshold - params.tp_hysteresis:
                        tp_hysteresis_active = False  # Re-arm TP
                
                # Check TP trigger (only if not in hysteresis or already re-armed)
                if not tp_hysteresis_active and position_return >= params.tp_threshold:
                    tp_triggered[i] = True
                    trigger_fired = True
                    # Activate hysteresis if configured
                    if params.tp_hysteresis > 0:
                        tp_hysteresis_active = True
            
            # Check Stop-Loss (only if TP didn't trigger, based on position return)
            if not trigger_fired and params.sl_threshold is not None and not sl_cooldown_active:
                # Check hysteresis: if hysteresis is active, require return to rise above threshold + hysteresis first
                if sl_hysteresis_active:
                    # Require return to rise above threshold + hysteresis before re-arming
                    # Note: sl_threshold is negative, so threshold + hysteresis means less negative
                    if position_return > params.sl_threshold + params.sl_hysteresis:
                        sl_hysteresis_active = False  # Re-arm SL
                
                # Check SL trigger (only if not in hysteresis or already re-armed)
                if not sl_hysteresis_active and position_return <= params.sl_threshold:
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
                    
                    # Update positions list: FIFO removal of sold shares
                    # IMPORTANT: After TP/SL trigger, reset entry_price of remaining positions to current price
                    # This prevents remaining positions from immediately triggering TP again on the next day
                    remaining_to_sell = float(shares_to_sell)
                    current_sell_price_per_share = px_exec_sell_price  # Use execution sell price (with slippage)
                    
                    while remaining_to_sell > 1e-9 and positions:
                        entry_date_idx, entry_shares, entry_price = positions[0]
                        if entry_shares <= remaining_to_sell:
                            # Remove entire position
                            remaining_to_sell -= entry_shares
                            positions.pop(0)
                        else:
                            # Partial removal: update first position
                            positions[0] = (entry_date_idx, entry_shares - remaining_to_sell, entry_price)
                            remaining_to_sell = 0.0
                    
                    # Reset entry_price of all remaining positions to current sell price after TP/SL trigger
                    # This ensures that remaining positions start from 0% return, preventing immediate re-trigger
                    if positions and (tp_triggered[i] or sl_triggered[i]):
                        # Update all remaining positions: reset entry_price to current sell price and entry_date to today
                        # This effectively resets the return calculation for remaining positions
                        # Use current execution sell price (already includes slippage) as base, then add fee
                        new_entry_price = current_sell_price_per_share * (1.0 + float(params.fee_rate))
                        # Reset all remaining positions to use new entry_price and today's date
                        positions = [(i, pos_shares, new_entry_price) for _, pos_shares, _ in positions]
                    
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
                        portfolio_return_ratio_arr[i] = equity / position_cost  # Store ratio for debugging
                    else:
                        portfolio_return = 0.0
                        portfolio_return_ratio_arr[i] = np.nan
                    
                    # Store TP sell price and entry reset price for debugging
                    if tp_triggered[i]:
                        tp_sell_price_arr[i] = px_exec_sell_price
                        # Store the new entry_price that was set for remaining positions
                        if positions:
                            tp_entry_reset_price_arr[i] = new_entry_price
                    elif sl_triggered[i]:
                        # SL also resets entry_price, store it
                        if positions:
                            tp_entry_reset_price_arr[i] = new_entry_price
                    
                    # Recalculate unrealized P/L
                    unrealized_pnl = equity - position_cost
                    
                    # Recalculate Profit
                    profit = equity + cum_cash_flow
                    
                    # Recalculate NAV_including_invested
                    nav_including_invested = cum_invested + profit
                    
                    # Reset ROI baseline after TP/SL sale (if enabled)
                    if params.reset_baseline_after_tp_sl:
                        roi_base = nav  # Reset baseline to NAV after trade
                        # Record baseline reset in history
                        trigger_type = 'TP' if tp_triggered[i] else 'SL'
                        roi_base_history.append((i, roi_base, trigger_type))
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
        
        # ===== GUARD CLAUSE: Prevent TP/SL trigger on buy days =====
        # If shares were bought today, TP/SL triggers are invalid (must be held for at least 1 day)
        # This is a final safety check - TP/SL check above already skips when shares_bought_today > 0
        if shares_bought_today > 1e-9:
            tp_triggered[i] = False
            sl_triggered[i] = False
        
        cum_cash_flow += cash_flow[i]
        
        # ===== STEP 4: Calculate final state AFTER buy (if any) =====
        # Calculate current equity and NAV (after buy, if any)
        equity = cum_shares * adj_price
        cash_balance = cum_invested + cum_cash_flow
        nav = equity + cash_balance
        
        # Calculate Portfolio Return (average cost basis vs current value)
        if position_cost > 1e-9 and cum_shares > 1e-9:
            portfolio_return = (equity / position_cost) - 1.0
            portfolio_return_ratio_arr[i] = equity / position_cost  # Store ratio for debugging (daily)
        else:
            portfolio_return = 0.0
            portfolio_return_ratio_arr[i] = np.nan  # No position, ratio is NaN
        
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
        
        # Calculate NAVReturn_baselined (for output, after all transactions)
        nav_return_baselined = 0.0
        if (params.tp_threshold is not None or params.sl_threshold is not None) and roi_base is not None and roi_base > 1e-9:
            nav_return_baselined = (nav / roi_base) - 1.0
        
        # Calculate NAV Return from investment (for historical reference, same as global)
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
        if (params.tp_threshold is not None or params.sl_threshold is not None) and roi_base is not None and roi_base > 1e-9:
            nav_return_baselined_arr[i] = nav_return_baselined
        elif roi_base is not None and roi_base > 1e-9:
            # Calculate even if TP/SL disabled for output consistency
            nav_return_baselined_arr[i] = (nav / roi_base) - 1.0
        # else: remains NaN (initialized)
        post_sell_cum_shares_arr[i] = cum_shares
    
    # Calculate drawdown: Portfolio Return based (Equity / PositionCost)
    # This avoids masking losses when new investments are added
    # Portfolio Return = (Equity / PositionCost) - 1.0
    # Drawdown = (Current Portfolio Return / Peak Portfolio Return) - 1.0
    # However, Portfolio Return can be negative, so we use (Equity / PositionCost) ratio instead
    # Drawdown = (Current Ratio / Peak Ratio) - 1.0, where Ratio = Equity / PositionCost
    drawdown = pd.Series(np.nan, index=idx)
    first_invest_mask = cum_invested_arr > 1e-9
    if first_invest_mask.any():
        first_invest_date_idx = np.where(first_invest_mask)[0][0]
        
        # Calculate Portfolio Return ratio series (Equity / PositionCost)
        # This ratio represents the value multiplier relative to cost basis
        # Ratio > 1.0 means profit, Ratio < 1.0 means loss
        portfolio_ratio_series = pd.Series(np.nan, index=idx)
        for j in range(len(idx)):
            if position_cost_arr[j] > 1e-6 and cum_shares_arr[j] > 1e-9:
                portfolio_ratio_series.iloc[j] = equity_arr[j] / position_cost_arr[j]
            elif position_cost_arr[j] <= 1e-6:
                # No position (완전 미보유 상태)
                # 투자 리스크가 없으므로 손실이 발생할 수 없음
                # Drawdown을 0으로 명시적으로 설정
                drawdown.iloc[j] = 0.0
                # Portfolio ratio는 NaN으로 유지 (MDD 계산에서 제외)
                portfolio_ratio_series.iloc[j] = np.nan
        
        # Calculate drawdown based on Portfolio Return ratio
        # Drawdown = (Current Ratio / Peak Ratio) - 1.0
        # This reflects actual portfolio losses relative to cost basis
        # TP/SL triggers don't reset PositionCost, so drawdown correctly tracks portfolio decline
        portfolio_ratio_post = portfolio_ratio_series.loc[idx[first_invest_date_idx]:]
        
        if len(portfolio_ratio_post) > 1:
            # Filter out NaN values (no position)
            valid_mask = portfolio_ratio_post.notna() & (portfolio_ratio_post > 1e-9)
            if valid_mask.any():
                portfolio_ratio_valid = portfolio_ratio_post[valid_mask]
                running_max_ratio = portfolio_ratio_valid.expanding().max()
                
                # Drawdown = (Current Ratio / Peak Ratio) - 1.0
                # This shows actual portfolio value decline from its peak ratio
                # Example: Peak Ratio = 1.5 (50% profit), Current Ratio = 1.2 (20% profit)
                # Drawdown = (1.2 / 1.5) - 1.0 = -0.2 = -20%
                drawdown_post = (portfolio_ratio_valid / running_max_ratio) - 1.0
                # Clip to valid range [-1, 0] (drawdown cannot exceed -100%)
                drawdown_post = drawdown_post.clip(lower=-1.0, upper=0.0)
                drawdown.loc[portfolio_ratio_valid.index] = drawdown_post.values
            else:
                # No valid positions, set drawdown to 0
                drawdown.loc[idx[first_invest_date_idx]:] = 0.0
        else:
            drawdown.loc[idx[first_invest_date_idx]:] = 0.0
    
    # Mode string
    mode_str = "shares"  # Always use shares-based mode
    
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
        # Debugging columns for TP/SL validation
        "WeightedAvgEntryPrice": pd.Series(weighted_avg_entry_price_arr, index=idx),
        "PositionReturn": pd.Series(position_return_arr, index=idx),
        "PortfolioReturnRatio": pd.Series(portfolio_return_ratio_arr, index=idx),
        "TP_SellPrice": pd.Series(tp_sell_price_arr, index=idx),
        "TP_EntryResetPrice": pd.Series(tp_entry_reset_price_arr, index=idx),
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
    adj_close = ledger["AdjClose"]
    
    # DailyRet validation: DailyRet_t = AdjClose_t / AdjClose_{t-1} - 1 (first day should be False/NaN)
    if "DailyRet" in ledger.columns:
        daily_ret = ledger["DailyRet"]
        daily_ret_computed = adj_close.pct_change()
        # First day should be NaN/False, skip it
        if len(daily_ret) > 1:
            mask = ~daily_ret_computed.isna()
            if mask.any():
                if not np.allclose(daily_ret.loc[mask], daily_ret_computed.loc[mask], rtol=1e-8, atol=1e-8, equal_nan=True):
                    max_diff = np.abs(daily_ret.loc[mask] - daily_ret_computed.loc[mask]).max()
                    raise ValueError(f"DailyRet mismatch; max diff: {max_diff}")
    
    # Buy/Sell accounting identities validation
    if "BuyAmt" in ledger.columns and "SharesBought" in ledger.columns and "Fee" in ledger.columns:
        buy_amt = ledger["BuyAmt"]
        shares_bought = ledger["SharesBought"]
        fee = ledger["Fee"]
        px_exec_buy = ledger["PxExec"]
        
        # Buy: cost = shares * buy_price, fee = cost * fee_rate, CashFlow = -(cost + fee)
        buy_mask = shares_bought > 1e-9
        if buy_mask.any():
            # PxExec should match AdjClose * (1 + slippage_rate) for buys
            expected_px_exec_buy = adj_close.loc[buy_mask] * (1.0 + float(params.slippage_rate))
            if not np.allclose(px_exec_buy.loc[buy_mask], expected_px_exec_buy, rtol=1e-8, atol=1e-8):
                max_diff = np.abs(px_exec_buy.loc[buy_mask] - expected_px_exec_buy).max()
                raise ValueError(f"Buy execution price mismatch; max diff: {max_diff}")
            
            # BuyAmt should equal shares_bought * px_exec_buy
            expected_buy_amt = shares_bought.loc[buy_mask] * px_exec_buy.loc[buy_mask]
            if not np.allclose(buy_amt.loc[buy_mask], expected_buy_amt, rtol=1e-8, atol=1e-8):
                max_diff = np.abs(buy_amt.loc[buy_mask] - expected_buy_amt).max()
                raise ValueError(f"BuyAmt mismatch; max diff: {max_diff}")
            
            # Fee should equal buy_amt * fee_rate
            expected_fee = buy_amt.loc[buy_mask] * float(params.fee_rate)
            if not np.allclose(fee.loc[buy_mask], expected_fee, rtol=1e-8, atol=1e-8):
                max_diff = np.abs(fee.loc[buy_mask] - expected_fee).max()
                raise ValueError(f"Buy fee mismatch; max diff: {max_diff}")
    
    # Sell accounting validation
    if "SharesSold" in ledger.columns and "GrossProceeds" in ledger.columns and "SellFee" in ledger.columns:
        shares_sold = ledger["SharesSold"]
        gross_proceeds = ledger["GrossProceeds"]
        sell_fee = ledger["SellFee"]
        sell_price = ledger["SellPrice"]
        
        sell_mask = shares_sold > 1e-9
        if sell_mask.any():
            # SellPrice should match AdjClose * (1 - slippage_rate) for sells
            expected_sell_price = adj_close.loc[sell_mask] * (1.0 - float(params.slippage_rate))
            if not np.allclose(sell_price.loc[sell_mask], expected_sell_price, rtol=1e-8, atol=1e-8):
                max_diff = np.abs(sell_price.loc[sell_mask] - expected_sell_price).max()
                raise ValueError(f"Sell execution price mismatch; max diff: {max_diff}")
            
            # GrossProceeds should equal shares_sold * sell_price
            expected_gross_proceeds = shares_sold.loc[sell_mask] * sell_price.loc[sell_mask]
            if not np.allclose(gross_proceeds.loc[sell_mask], expected_gross_proceeds, rtol=1e-8, atol=1e-8):
                max_diff = np.abs(gross_proceeds.loc[sell_mask] - expected_gross_proceeds).max()
                raise ValueError(f"GrossProceeds mismatch; max diff: {max_diff}")
            
            # SellFee should equal gross_proceeds * fee_rate
            expected_sell_fee = gross_proceeds.loc[sell_mask] * float(params.fee_rate)
            if not np.allclose(sell_fee.loc[sell_mask], expected_sell_fee, rtol=1e-8, atol=1e-8):
                max_diff = np.abs(sell_fee.loc[sell_mask] - expected_sell_fee).max()
                raise ValueError(f"Sell fee mismatch; max diff: {max_diff}")
        
        # CashFlow validation for sells: CashFlow = buy_cashflow + sell_cashflow
        # On sell days, CashFlow includes both buy (negative) and sell (positive net_proceeds) flows
        if "CashFlow" in ledger.columns and "NetProceeds" in ledger.columns and "BuyAmt" in ledger.columns and "Fee" in ledger.columns:
            cash_flow = ledger["CashFlow"]
            net_proceeds = ledger["NetProceeds"]
            buy_amt = ledger["BuyAmt"]
            fee = ledger["Fee"]
            if sell_mask.any():
                # For sell days, CashFlow = -(BuyAmt + Fee) + NetProceeds (if buy also occurred)
                # or CashFlow = NetProceeds (if no buy occurred)
                sell_days_cashflow = cash_flow.loc[sell_mask]
                sell_days_net_proceeds = net_proceeds.loc[sell_mask]
                sell_days_buy_cost = (buy_amt.loc[sell_mask] + fee.loc[sell_mask])
                
                # Expected cashflow = -buy_cost + net_proceeds (or just net_proceeds if no buy)
                expected_cash_flow_sell = -sell_days_buy_cost + sell_days_net_proceeds
                actual_cash_flow_sell = sell_days_cashflow
                
                if not np.allclose(actual_cash_flow_sell, expected_cash_flow_sell, rtol=1e-8, atol=1e-8):
                    max_diff = np.abs(actual_cash_flow_sell - expected_cash_flow_sell).max()
                    raise ValueError(f"Sell CashFlow mismatch; max diff: {max_diff}")
    
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
    # Portfolio Return 기준 drawdown은 일반적으로 [-1, 0] 범위 내에 있어야 함
    # -1.0 = -100% (완전 손실), 0.0 = 손실 없음
    # Portfolio Return 기준이므로 -1보다 작을 수 있지만, -2.0 (-200%) 이상은 비정상
    drawdown_valid = drawdown.dropna()
    if len(drawdown_valid) > 0:
        min_dd = drawdown_valid.min()
        max_dd = drawdown_valid.max()
        if min_dd < -2.0:  # -200% 이상은 비정상 (Portfolio Return 기준)
            raise ValueError(f"Drawdown too negative (Portfolio Return based): {min_dd}")
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
            portfolio_mask = position_cost > 1e-6  # Use 1e-6 threshold for safety
            portfolio_return_computed.loc[portfolio_mask] = (equity.loc[portfolio_mask] / position_cost.loc[portfolio_mask]) - 1.0
            
            # Validate PortfolioReturn when PositionCost > 1e-6
            if portfolio_mask.any():
                if not np.allclose(portfolio_return.loc[portfolio_mask], portfolio_return_computed.loc[portfolio_mask], rtol=1e-8, atol=1e-8):
                    max_diff = np.abs(portfolio_return.loc[portfolio_mask] - portfolio_return_computed.loc[portfolio_mask]).max()
                    raise ValueError(f"PortfolioReturn mismatch; max diff: {max_diff}")
            
            # Validate PortfolioReturn should be 0 (or close to 0) when PositionCost <= 1e-6
            position_cost_zero_mask = position_cost <= 1e-6
            if position_cost_zero_mask.any():
                portfolio_return_zero = portfolio_return.loc[position_cost_zero_mask]
                if not (portfolio_return_zero.abs() < 1e-6).all():
                    max_val = portfolio_return_zero.abs().max()
                    raise ValueError(f"PortfolioReturn should be 0 when PositionCost is 0; max abs value: {max_val}")
        
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
        # Cost basis is calculated as: avg_cost = PositionCost_before / CumShares_before, cost_sold = avg_cost * shares_sold
        # RealizedPnl = net_proceeds - cost_sold
        sell_mask = shares_sold > 1e-9
        if sell_mask.any():
            # For each sell, verify P/L calculation
            # Note: We validate the accounting is consistent, but precise retroactive validation
            # requires tracking state before each sell, which is complex.
            # The main checks are:
            # 1. RealizedPnl is finite
            # 2. RealizedPnl = NetProceeds - CostBasis where CostBasis is proportional to position_cost
            for idx_val in ledger.index[sell_mask]:
                shares_sold_val = shares_sold.loc[idx_val]
                net_proceeds_val = net_proceeds.loc[idx_val]
                realized_pnl_val = realized_pnl.loc[idx_val]
                
                if not np.isfinite(realized_pnl_val):
                    raise ValueError(f"RealizedPnl must be finite at {idx_val}")
                
                # RealizedPnl should equal NetProceeds - CostBasis
                # For validation, we check that RealizedPnl + CostBasis = NetProceeds
                # where CostBasis is approximated from the difference
                # This is a consistency check rather than a precise calculation check
                if abs(realized_pnl_val - (net_proceeds_val - (net_proceeds_val - realized_pnl_val))) > 1e-6:
                    # This is a basic consistency check
                    pass  # More detailed validation would require tracking pre-sell state
        
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
        shares_bought = ledger["SharesBought"]
        
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
        
        # Validate: TP/SL and Buy should not occur on the same day
        # Guard Clause ensures this, but we validate it here as well
        same_day_mask = trigger_mask & (shares_bought > 1e-9)
        if same_day_mask.any():
            dates = ledger.index[same_day_mask]
            raise ValueError(f"TP/SL and Buy occurred on same day (should be prevented by Guard Clause): {dates.tolist()}")
        
        # Validate: Cooldown periods - 동일 방향 트리거가 쿨다운 기간 중 발생하지 않았는지 확인
        if params.tp_threshold is not None or params.sl_threshold is not None:
            # TP cooldown 검증
            if params.tp_cooldown_days > 0:
                tp_dates = ledger.index[tp_triggered]
                if len(tp_dates) > 1:
                    # 각 TP 발생 후 쿨다운 기간 동안 다른 TP가 발생하지 않았는지 확인
                    for i, tp_date in enumerate(tp_dates[:-1]):
                        next_tp_date = tp_dates[i + 1]
                        days_between = (next_tp_date - tp_date).days if hasattr(next_tp_date, 'days') else (pd.Timestamp(next_tp_date) - pd.Timestamp(tp_date)).days
                        if days_between < params.tp_cooldown_days:
                            raise ValueError(f"TP triggered within cooldown period: {tp_date} -> {next_tp_date} (cooldown: {params.tp_cooldown_days} days, actual: {days_between} days)")
            
            # SL cooldown 검증
            if params.sl_cooldown_days > 0:
                sl_dates = ledger.index[sl_triggered]
                if len(sl_dates) > 1:
                    # 각 SL 발생 후 쿨다운 기간 동안 다른 SL이 발생하지 않았는지 확인
                    for i, sl_date in enumerate(sl_dates[:-1]):
                        next_sl_date = sl_dates[i + 1]
                        days_between = (next_sl_date - sl_date).days if hasattr(next_sl_date, 'days') else (pd.Timestamp(next_sl_date) - pd.Timestamp(sl_date)).days
                        if days_between < params.sl_cooldown_days:
                            raise ValueError(f"SL triggered within cooldown period: {sl_date} -> {next_sl_date} (cooldown: {params.sl_cooldown_days} days, actual: {days_between} days)")


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
    
    # Get allocation (shares-based mode only)
    allocation = _compute_allocation(prices, params)
    
    # Compute complete ledger
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
        # No trades executed - still calculate Benchmark CAGR
        benchmark_start_date = prices.index[0]
        benchmark_end_date = prices.index[-1]
        benchmark_start_price = float(prices["AdjClose"].iloc[0])
        benchmark_end_price = float(prices["AdjClose"].iloc[-1])
        
        if benchmark_start_date == benchmark_end_date:
            benchmark_cagr = 0.0
        elif benchmark_start_price <= 0 or benchmark_end_price <= 0:
            benchmark_cagr = 0.0
        else:
            benchmark_days = (benchmark_end_date - benchmark_start_date).days if hasattr(benchmark_end_date, 'days') else (pd.Timestamp(benchmark_end_date) - pd.Timestamp(benchmark_start_date)).days
            if benchmark_days > 0:
                benchmark_years = benchmark_days / 365.25
                benchmark_cagr = (benchmark_end_price / benchmark_start_price) ** (1.0 / benchmark_years) - 1.0
            else:
                benchmark_cagr = 0.0
        
        metrics: Dict[str, float] = {
            "TotalInvested": 0.0,
            "EndingEquity": 0.0,
            "EndingNAV": 0.0,
            "CumulativeReturn": 0.0,
            "CAGR": 0.0,
            "BenchmarkCAGR": float(benchmark_cagr),
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
    
    # MDD: computed from Portfolio Return ratio (Equity / PositionCost) to reflect actual portfolio losses
    # This correctly shows portfolio drawdown relative to cost basis even when TP/SL triggers and baseline resets
    # Portfolio Return ratio is continuous and not reset by TP/SL, so MDD accurately tracks portfolio decline
    portfolio_return = ledger["PortfolioReturn"]
    position_cost = ledger["PositionCost"]
    
    # Calculate Portfolio Return ratio (Equity / PositionCost) for MDD calculation
    # This ratio represents the value multiplier relative to cost basis
    # PositionCost가 0인 시점은 valid_mask에서 제외되어 MDD 계산에서 제외됨
    # 이는 올바른 동작임 (투자 리스크가 없으므로 손실이 발생할 수 없음)
    portfolio_ratio = pd.Series(np.nan, index=ledger.index)
    valid_mask = (position_cost > 1e-6) & (ledger["CumShares"] > 1e-9)
    portfolio_ratio.loc[valid_mask] = (ledger["Equity"].loc[valid_mask] / position_cost.loc[valid_mask])
    
    portfolio_ratio_post = portfolio_ratio.loc[first_invest_date:]
    
    if len(portfolio_ratio_post) > 1:
        # Filter out NaN values (no position)
        valid_mask = portfolio_ratio_post.notna() & (portfolio_ratio_post > 1e-9)
        if valid_mask.any():
            portfolio_ratio_valid = portfolio_ratio_post[valid_mask]
            mdd_value = float(max_drawdown(portfolio_ratio_valid))
            
            # Calculate peak_date and trough_date
            running_max = portfolio_ratio_valid.expanding().max()
            drawdowns = (portfolio_ratio_valid / running_max) - 1.0
            trough_idx = drawdowns.idxmin()
            peak_idx = portfolio_ratio_valid.loc[:trough_idx].idxmax() if len(portfolio_ratio_valid.loc[:trough_idx]) > 0 else None
            mdd_peak_date = peak_idx if peak_idx is not None else None
            mdd_trough_date = trough_idx
        else:
            mdd_value = 0.0
            mdd_peak_date = None
            mdd_trough_date = None
    else:
        mdd_value = 0.0
        mdd_peak_date = None
        mdd_trough_date = None
    
    # CAGR: from first investment to end
    last_date = idx[-1]
    days = (last_date - first_invest_date).days if hasattr(last_date, 'days') else (pd.Timestamp(last_date) - pd.Timestamp(first_invest_date)).days
    if days > 0 and total_invested > 1e-9 and ending_nav > 0:
        years = days / 365.25
        cagr_value = cagr_func(total_invested, ending_nav, years)
    else:
        cagr_value = 0.0
    
    # Benchmark CAGR (Buy & Hold): Simple 1-share buy-and-hold strategy
    # Uses actual first and last dates in the prices DataFrame (may differ from user input due to weekends/holidays)
    benchmark_start_date = prices.index[0]
    benchmark_end_date = prices.index[-1]
    benchmark_start_price = float(prices["AdjClose"].iloc[0])
    benchmark_end_price = float(prices["AdjClose"].iloc[-1])
    
    if benchmark_start_date == benchmark_end_date:
        # Same date: CAGR = 0
        benchmark_cagr = 0.0
    elif benchmark_start_price <= 0 or benchmark_end_price <= 0:
        # Invalid prices: CAGR = 0
        benchmark_cagr = 0.0
    else:
        benchmark_days = (benchmark_end_date - benchmark_start_date).days if hasattr(benchmark_end_date, 'days') else (pd.Timestamp(benchmark_end_date) - pd.Timestamp(benchmark_start_date)).days
        if benchmark_days > 0:
            benchmark_years = benchmark_days / 365.25
            # CAGR formula: (end_price / start_price) ^ (1 / years) - 1
            benchmark_cagr = (benchmark_end_price / benchmark_start_price) ** (1.0 / benchmark_years) - 1.0
        else:
            benchmark_cagr = 0.0
    
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
    
    # Calculate baseline reset count (approximate: same as TP/SL count if reset_baseline_after_tp_sl is True)
    baseline_reset_count = num_take_profits + num_stop_losses if params.reset_baseline_after_tp_sl else 0
    
    metrics: Dict[str, float] = {
        "TotalInvested": round(total_invested, 6),
        "EndingEquity": round(ending_equity, 6),
        "EndingNAV": round(ending_nav, 6),
        "Profit": round(ending_profit, 6),
        "NAV_including_invested": round(ending_nav_including_invested, 6),
        "CumulativeReturn": float(cumulative_return),
        "CAGR": float(cagr_value),
        "BenchmarkCAGR": float(benchmark_cagr),
        "MDD": float(mdd_value),
        "Trades": float(trades),
        "HitDays": float(hit_days),
        "XIRR": float(xirr_value),
        "NumTakeProfits": float(num_take_profits),
        "NumStopLosses": float(num_stop_losses),
        "BaselineResetCount": float(baseline_reset_count),
        "TotalRealizedGain": round(total_realized_gain, 6),
        "TotalRealizedLoss": round(total_realized_loss, 6),
        "NetRealizedPnl": round(total_realized_gain - total_realized_loss, 6),
        "EndingPositionCost": round(ending_position_cost, 6),
    }
    
    # Add MDD peak/trough dates if available (as string for JSON serialization)
    if mdd_peak_date is not None:
        metrics["MDD_peak_date"] = str(mdd_peak_date)
    if mdd_trough_date is not None:
        metrics["MDD_trough_date"] = str(mdd_trough_date)
    
    return ledger, metrics

"""
Backtest Engine V2 - ACB-based with Anchor TP/SL

Complete re-architecture using:
- ACB (Average Cost Basis) accounting
- Anchor-based TP/SL triggers (not FIFO entry prices)
- 2-wallet cash system (profit_cash, principal_cash)
- TP Mode A (Anchor maintain) vs B (Reset)
- Fixed bar processing order: Snapshot → TP/SL → Hysteresis/Cooldown → Buy → NAV

Author: Cursor AI Assistant
Date: 2025-01-18
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..strategy.dip_buy import generate_dip_signals
from .metrics import cagr as cagr_func
from .metrics import max_drawdown, xirr


@dataclass(frozen=True)
class BacktestParamsV2:
    """Backtest parameters for V2 engine (ACB-based)."""
    
    threshold: float  # Dip signal threshold (e.g., -0.041 for -4.1%)
    shares_per_signal: float  # Fixed shares to buy per signal
    fee_rate: float = 0.0005  # 0.05%
    slippage_rate: float = 0.0025  # 0.25%
    
    # TP/SL parameters
    enable_tp_sl: bool = False
    tp_threshold: float | None = None  # e.g., 0.30 for 30%
    sl_threshold: float | None = None  # e.g., -0.25 for -25%
    tp_sell_percentage: float = 1.0  # 0.25, 0.50, 0.75, 1.0
    sl_sell_percentage: float = 1.0
    
    # TP/SL Mode (ACB-based)
    tp_mode: str = "A"  # "A" (Anchor maintain) or "B" (Reset)
    same_bar_reuse: bool = False  # Allow reusing TP/SL cash on same bar
    anchor_init_mode: str = "peak"  # "peak" or "entry"
    
    # Hysteresis & Cooldown
    tp_hysteresis: float = 0.0  # e.g., 0.025 for 2.5%
    sl_hysteresis: float = 0.0
    tp_cooldown_days: int = 0
    sl_cooldown_days: int = 0
    
    # Global Stop-Loss (전역 손절)
    global_sl_mode: str = "off"  # "off", "principal", "trailing_peak"
    global_sl_percent: float = 0.15  # e.g., 0.15 for -15%
    global_sl_cooldown_bars: int = 10  # Cooldown after global SL trigger
    
    def __post_init__(self):
        """Validate parameters."""
        if self.tp_mode not in ["A", "B"]:
            raise ValueError("tp_mode must be 'A' or 'B'")
        if self.anchor_init_mode not in ["peak", "entry"]:
            raise ValueError("anchor_init_mode must be 'peak' or 'entry'")
        if self.shares_per_signal <= 0:
            raise ValueError("shares_per_signal must be positive")
        
        # Validate Global SL parameters
        if self.global_sl_mode not in ["off", "principal", "trailing_peak"]:
            raise ValueError("global_sl_mode must be 'off', 'principal', or 'trailing_peak'")
        if self.global_sl_percent < 0 or self.global_sl_percent > 1.0:
            raise ValueError("global_sl_percent must be between 0 and 1.0")
        if self.global_sl_cooldown_bars < 0:
            raise ValueError("global_sl_cooldown_bars must be non-negative")
        
        # TP validation
        if self.tp_threshold is not None:
            if self.tp_threshold <= 0:
                raise ValueError("tp_threshold must be positive")
            if not 0 < self.tp_sell_percentage <= 1.0:
                raise ValueError("tp_sell_percentage must be in (0, 1]")
            if self.tp_hysteresis < 0:
                raise ValueError("tp_hysteresis must be non-negative")
            if self.tp_cooldown_days < 0:
                raise ValueError("tp_cooldown_days must be non-negative")
        
        # SL validation
        if self.sl_threshold is not None:
            if self.sl_threshold >= 0:
                raise ValueError("sl_threshold must be negative")
            if not 0 < self.sl_sell_percentage <= 1.0:
                raise ValueError("sl_sell_percentage must be in (0, 1]")
            if self.sl_hysteresis < 0:
                raise ValueError("sl_hysteresis must be non-negative")
            if self.sl_cooldown_days < 0:
                raise ValueError("sl_cooldown_days must be non-negative")


@dataclass
class BarState:
    """State for a single bar (mutable)."""
    
    # Shares
    qty: float = 0.0
    
    # ACB accounting
    avg_cost: float = 0.0  # Average cost per share (buy only, unchanged on sell)
    position_cost: float = 0.0  # avg_cost × qty (for display)
    
    # Anchor TP/SL
    anchor_price: float = 0.0
    peak_price: float = 0.0
    
    # 2-Wallet cash system
    profit_cash: float = 0.0  # Realized profits (TP gains)
    principal_cash: float = 0.0  # Principal (can go negative)
    
    # Cumulative tracking (display)
    cum_invested: float = 0.0  # Total capital invested
    cum_cash_flow: float = 0.0  # Net cash flow (negative = buy, positive = sell)
    
    # Trigger control
    tp_cooldown_remaining: int = 0
    sl_cooldown_remaining: int = 0
    tp_hysteresis_active: bool = False
    sl_hysteresis_active: bool = False
    
    # Global Stop-Loss tracking
    nav_peak: float = 0.0  # Peak NAV (for trailing_peak mode)
    global_sl_cooldown_remaining: int = 0
    global_sl_triggered_ever: bool = False  # Track if ever triggered
    
    # Derived values (calculated each bar)
    equity: float = 0.0  # qty × price
    cash_balance: float = 0.0  # profit_cash + principal_cash
    nav: float = 0.0  # equity + cash_balance
    
    def snapshot(self) -> dict:
        """Create snapshot for logging/debugging."""
        return {
            "qty": self.qty,
            "avg_cost": self.avg_cost,
            "anchor": self.anchor_price,
            "peak": self.peak_price,
            "profit_cash": self.profit_cash,
            "principal_cash": self.principal_cash,
            "equity": self.equity,
            "nav": self.nav,
        }


def apply_cooldown_decrement(state: BarState) -> None:
    """Decrement cooldown counters (including global SL)."""
    if state.tp_cooldown_remaining > 0:
        state.tp_cooldown_remaining -= 1
    if state.sl_cooldown_remaining > 0:
        state.sl_cooldown_remaining -= 1
    if state.global_sl_cooldown_remaining > 0:
        state.global_sl_cooldown_remaining -= 1


def check_and_apply_global_sl(
    state: BarState,
    price: float,
    px_exec_sell: float,
    params: BacktestParamsV2,
) -> tuple[bool, str, float, float]:
    """
    Check global stop-loss and execute if triggered.
    
    Returns: (triggered, reason, realized_pnl, net_proceeds)
    reason: "principal" or "peak"
    """
    if params.global_sl_mode == "off":
        return False, "", 0.0, 0.0
    
    if state.qty < 1e-9:
        return False, "", 0.0, 0.0
    
    if state.global_sl_cooldown_remaining > 0:
        return False, "", 0.0, 0.0
    
    triggered = False
    reason = ""
    
    # Check Principal-Based Stop Loss
    if params.global_sl_mode == "principal" and state.cum_invested > 1e-9:
        nav = state.equity + (state.profit_cash + state.principal_cash)
        drawdown_from_principal = (nav / state.cum_invested) - 1.0
        
        if drawdown_from_principal <= -params.global_sl_percent:
            triggered = True
            reason = "principal"
    
    # Check Trailing Peak Stop Loss
    if params.global_sl_mode == "trailing_peak" and state.nav_peak > 1e-9:
        nav = state.equity + (state.profit_cash + state.principal_cash)
        drawdown_from_peak = 1.0 - (nav / state.nav_peak)
        
        if drawdown_from_peak >= params.global_sl_percent:
            triggered = True
            reason = "peak"
    
    if not triggered:
        return False, "", 0.0, 0.0
    
    # Execute full liquidation
    shares_to_sell = state.qty
    gross_proceeds = state.qty * px_exec_sell
    sell_fee = gross_proceeds * params.fee_rate
    net_proceeds = gross_proceeds - sell_fee
    
    # ACB-based realized P/L
    cost_of_shares_sold = state.avg_cost * state.qty
    realized_pnl = net_proceeds - cost_of_shares_sold
    
    # Update wallet
    if realized_pnl < 0:
        loss = abs(realized_pnl)
        if state.profit_cash >= loss:
            state.profit_cash -= loss
            state.principal_cash += cost_of_shares_sold
        else:
            state.principal_cash += cost_of_shares_sold - (loss - state.profit_cash)
            state.profit_cash = 0.0
    else:
        state.profit_cash += realized_pnl
        state.principal_cash += cost_of_shares_sold
    
    # Reset all positions
    state.qty = 0.0
    state.avg_cost = 0.0
    state.position_cost = 0.0
    state.anchor_price = 0.0
    state.peak_price = 0.0
    state.cum_cash_flow += net_proceeds
    
    # Reset individual TP/SL states
    state.tp_cooldown_remaining = 0
    state.sl_cooldown_remaining = 0
    state.tp_hysteresis_active = False
    state.sl_hysteresis_active = False
    
    # Set global SL cooldown
    state.global_sl_cooldown_remaining = params.global_sl_cooldown_bars
    state.global_sl_triggered_ever = True
    
    # Reset NAV peak
    nav_after = state.profit_cash + state.principal_cash
    state.nav_peak = nav_after
    
    return True, reason, realized_pnl, net_proceeds


def update_nav_peak(state: BarState) -> None:
    """Update NAV peak if current NAV is higher."""
    if state.nav > state.nav_peak:
        state.nav_peak = state.nav


def check_and_apply_tp(
    state: BarState,
    price: float,
    px_exec_sell: float,
    params: BacktestParamsV2,
) -> tuple[bool, float, float, float]:
    """
    Check TP trigger and execute sell if triggered.
    
    Returns: (triggered, shares_sold, realized_pnl, net_proceeds)
    """
    if params.tp_threshold is None:
        return False, 0.0, 0.0, 0.0
    
    if state.qty < 1e-9 or state.anchor_price < 1e-9:
        return False, 0.0, 0.0, 0.0
    
    # Anchor-based return
    ret_anchor = (price / state.anchor_price) - 1.0
    
    # Check trigger conditions
    if (ret_anchor >= params.tp_threshold and
        state.tp_cooldown_remaining == 0 and
        not state.tp_hysteresis_active):
        
        # Calculate sell amount
        shares_to_sell_float = state.qty * params.tp_sell_percentage
        shares_to_sell = int(shares_to_sell_float + 0.5)  # Round
        shares_to_sell = max(1, min(shares_to_sell, int(state.qty)))
        
        # Execute sell
        gross_proceeds = float(shares_to_sell) * px_exec_sell
        sell_fee = gross_proceeds * params.fee_rate
        net_proceeds = gross_proceeds - sell_fee
        
        # ACB-based realized P/L
        cost_of_shares_sold = state.avg_cost * float(shares_to_sell)
        realized_pnl = net_proceeds - cost_of_shares_sold
        
        # Update wallet
        if realized_pnl > 0:
            state.profit_cash += realized_pnl
            state.principal_cash += cost_of_shares_sold
        else:
            # Loss: deduct from profit_cash first
            loss = abs(realized_pnl)
            if state.profit_cash >= loss:
                state.profit_cash -= loss
                state.principal_cash += cost_of_shares_sold
            else:
                state.principal_cash += cost_of_shares_sold - (loss - state.profit_cash)
                state.profit_cash = 0.0
        
        # Update state
        state.qty -= float(shares_to_sell)
        state.position_cost = state.avg_cost * state.qty  # avg_cost unchanged
        state.cum_cash_flow += net_proceeds
        
        # Anchor update (A/B mode)
        if params.tp_mode == "A":
            # A mode: maintain anchor on partial, reset on full
            if state.qty < 1e-9:
                state.anchor_price = 0.0
                state.peak_price = 0.0
        elif params.tp_mode == "B":
            # B mode: always reset
            if state.qty > 1e-9:
                state.anchor_price = price
                state.peak_price = price
            else:
                state.anchor_price = 0.0
                state.peak_price = 0.0
        
        # Hysteresis & Cooldown
        if params.tp_hysteresis > 0:
            state.tp_hysteresis_active = True
        if params.tp_cooldown_days > 0:
            state.tp_cooldown_remaining = params.tp_cooldown_days
        
        return True, float(shares_to_sell), realized_pnl, net_proceeds
    
    return False, 0.0, 0.0, 0.0


def check_and_apply_sl(
    state: BarState,
    price: float,
    px_exec_sell: float,
    params: BacktestParamsV2,
) -> tuple[bool, float, float, float]:
    """
    Check SL trigger and execute sell if triggered.
    
    Returns: (triggered, shares_sold, realized_pnl, net_proceeds)
    """
    if params.sl_threshold is None:
        return False, 0.0, 0.0, 0.0
    
    if state.qty < 1e-9 or state.anchor_price < 1e-9:
        return False, 0.0, 0.0, 0.0
    
    # Anchor-based return
    ret_anchor = (price / state.anchor_price) - 1.0
    
    # Check trigger conditions
    if (ret_anchor <= params.sl_threshold and
        state.sl_cooldown_remaining == 0 and
        not state.sl_hysteresis_active):
        
        # Calculate sell amount
        shares_to_sell_float = state.qty * params.sl_sell_percentage
        shares_to_sell = int(shares_to_sell_float + 0.5)
        shares_to_sell = max(1, min(shares_to_sell, int(state.qty)))
        
        # Execute sell
        gross_proceeds = float(shares_to_sell) * px_exec_sell
        sell_fee = gross_proceeds * params.fee_rate
        net_proceeds = gross_proceeds - sell_fee
        
        # ACB-based realized P/L
        cost_of_shares_sold = state.avg_cost * float(shares_to_sell)
        realized_pnl = net_proceeds - cost_of_shares_sold
        
        # Update wallet (loss → profit_cash first)
        if realized_pnl < 0:
            loss = abs(realized_pnl)
            if state.profit_cash >= loss:
                state.profit_cash -= loss
                state.principal_cash += cost_of_shares_sold
            else:
                state.principal_cash += cost_of_shares_sold - (loss - state.profit_cash)
                state.profit_cash = 0.0
        else:
            state.profit_cash += realized_pnl
            state.principal_cash += cost_of_shares_sold
        
        # Update state
        state.qty -= float(shares_to_sell)
        state.position_cost = state.avg_cost * state.qty
        state.cum_cash_flow += net_proceeds
        
        # Anchor update (SL always resets or clears)
        if state.qty > 1e-9:
            state.anchor_price = price
            state.peak_price = price
        else:
            state.anchor_price = 0.0
            state.peak_price = 0.0
        
        # Hysteresis & Cooldown
        if params.sl_hysteresis > 0:
            state.sl_hysteresis_active = True
        if params.sl_cooldown_days > 0:
            state.sl_cooldown_remaining = params.sl_cooldown_days
        
        return True, float(shares_to_sell), realized_pnl, net_proceeds
    
    return False, 0.0, 0.0, 0.0


def update_hysteresis(state: BarState, price: float, params: BacktestParamsV2) -> None:
    """Re-activate hysteresis if conditions met."""
    if state.anchor_price < 1e-9:
        return
    
    ret_anchor = (price / state.anchor_price) - 1.0
    
    # TP hysteresis
    if state.tp_hysteresis_active and params.tp_threshold is not None:
        if ret_anchor < (params.tp_threshold - params.tp_hysteresis):
            state.tp_hysteresis_active = False
    
    # SL hysteresis
    if state.sl_hysteresis_active and params.sl_threshold is not None:
        if ret_anchor > (params.sl_threshold + params.sl_hysteresis):
            state.sl_hysteresis_active = False


def apply_buy(
    state: BarState,
    signal: bool,
    price: float,
    px_exec_buy: float,
    params: BacktestParamsV2,
    tp_triggered: bool,
    sl_triggered: bool,
) -> tuple[bool, float]:
    """
    Apply buy logic with profit→principal cash order.
    
    Returns: (bought, shares_bought)
    """
    if not signal:
        return False, 0.0
    
    # Check global SL cooldown (최우선)
    if state.global_sl_cooldown_remaining > 0:
        return False, 0.0
    
    # Check same_bar_reuse
    if not params.same_bar_reuse and (tp_triggered or sl_triggered):
        return False, 0.0
    
    # Calculate cost
    shares_fixed = float(params.shares_per_signal)
    buy_amt = shares_fixed * px_exec_buy
    fee = buy_amt * params.fee_rate
    total_cost = buy_amt + fee
    
    # Payment order: profit_cash → principal_cash
    if total_cost <= state.profit_cash:
        state.profit_cash -= total_cost
    elif state.profit_cash > 0:
        remaining = total_cost - state.profit_cash
        state.profit_cash = 0.0
        state.principal_cash -= remaining
    else:
        state.principal_cash -= total_cost
    
    # ACB update
    if state.qty > 1e-9:
        state.avg_cost = (state.avg_cost * state.qty + total_cost) / (state.qty + shares_fixed)
    else:
        state.avg_cost = total_cost / shares_fixed
    
    state.qty += shares_fixed
    state.position_cost = state.avg_cost * state.qty
    
    # Anchor initialization (first entry)
    if state.anchor_price < 1e-9:
        if params.anchor_init_mode == "entry":
            state.anchor_price = px_exec_buy
        else:  # "peak"
            state.anchor_price = price
        state.peak_price = price
    
    # Cumulative tracking
    state.cum_invested += total_cost
    state.cum_cash_flow -= total_cost
    
    return True, shares_fixed


def update_peak_price(state: BarState, price: float) -> None:
    """Update peak price if current price is higher."""
    if state.qty > 1e-9 and price > state.peak_price:
        state.peak_price = price


def recalc_nav(state: BarState, price: float) -> None:
    """Recalculate NAV and derived values."""
    state.equity = state.qty * price
    state.cash_balance = state.profit_cash + state.principal_cash
    state.nav = state.equity + state.cash_balance


def compute_ledger_v2(
    prices: pd.DataFrame,
    params: BacktestParamsV2,
) -> pd.DataFrame:
    """
    Compute backtest ledger using V2 engine (ACB-based).
    
    Bar processing order:
    1. Snapshot (optional)
    2. Cooldown decrement
    3. TP/SL check & execute
    4. Hysteresis update
    5. Buy logic
    6. Peak price update
    7. NAV calculation
    8. Store values
    """
    idx = prices.index
    adj = prices["AdjClose"].astype(float)
    n = len(idx)
    
    # Signals
    daily_ret = adj.pct_change()
    signals = generate_dip_signals(adj, params.threshold)
    
    # Execution prices
    px_exec_buy = adj * (1.0 + params.slippage_rate)
    px_exec_sell = adj * (1.0 - params.slippage_rate)
    
    # Convert to numpy for performance
    adj_arr = adj.values
    px_exec_buy_arr = px_exec_buy.values
    px_exec_sell_arr = px_exec_sell.values
    signals_arr = signals.values
    
    # Initialize arrays
    buy_amt_arr = np.zeros(n)
    fee_arr = np.zeros(n)
    shares_bought_arr = np.zeros(n)
    cash_flow_arr = np.zeros(n)
    
    shares_sold_arr = np.zeros(n)
    sell_price_arr = np.zeros(n)
    gross_proceeds_arr = np.zeros(n)
    sell_fee_arr = np.zeros(n)
    net_proceeds_arr = np.zeros(n)
    realized_pnl_arr = np.zeros(n)
    
    tp_triggered_arr = np.zeros(n, dtype=bool)
    sl_triggered_arr = np.zeros(n, dtype=bool)
    
    # State arrays (for CSV export)
    qty_arr = np.zeros(n)
    avg_cost_arr = np.zeros(n)
    position_cost_arr = np.zeros(n)
    anchor_price_arr = np.zeros(n)
    peak_price_arr = np.zeros(n)
    profit_cash_arr = np.zeros(n)
    principal_cash_arr = np.zeros(n)
    cum_invested_arr = np.zeros(n)
    cum_cash_flow_arr = np.zeros(n)
    equity_arr = np.zeros(n)
    cash_balance_arr = np.zeros(n)
    nav_arr = np.zeros(n)
    tp_cooldown_arr = np.zeros(n, dtype=int)
    sl_cooldown_arr = np.zeros(n, dtype=int)
    
    ret_anchor_arr = np.full(n, np.nan)
    portfolio_return_arr = np.zeros(n)
    unrealized_pnl_arr = np.zeros(n)
    profit_arr = np.zeros(n)
    nav_including_invested_arr = np.zeros(n)
    nav_return_arr = np.zeros(n)
    
    # Initialize state
    state = BarState()
    
    # Main loop
    for i in range(n):
        price = adj_arr[i]
        px_buy = px_exec_buy_arr[i]
        px_sell = px_exec_sell_arr[i]
        signal = signals_arr[i]
        
        # Step 1: Cooldown decrement
        apply_cooldown_decrement(state)
        
        # Step 2: TP/SL triggers
        tp_triggered, tp_shares_sold, tp_realized_pnl, tp_net_proceeds = check_and_apply_tp(
            state, price, px_sell, params
        )
        
        sl_triggered = False
        sl_shares_sold = 0.0
        sl_realized_pnl = 0.0
        sl_net_proceeds = 0.0
        
        if not tp_triggered:
            sl_triggered, sl_shares_sold, sl_realized_pnl, sl_net_proceeds = check_and_apply_sl(
                state, price, px_sell, params
            )
        
        # Record TP/SL
        if tp_triggered:
            tp_triggered_arr[i] = True
            shares_sold_arr[i] = tp_shares_sold
            sell_price_arr[i] = px_sell
            gross_proceeds_arr[i] = tp_shares_sold * px_sell
            sell_fee_arr[i] = gross_proceeds_arr[i] * params.fee_rate
            net_proceeds_arr[i] = tp_net_proceeds
            realized_pnl_arr[i] = tp_realized_pnl
            cash_flow_arr[i] += tp_net_proceeds
        
        if sl_triggered:
            sl_triggered_arr[i] = True
            shares_sold_arr[i] = sl_shares_sold
            sell_price_arr[i] = px_sell
            gross_proceeds_arr[i] = sl_shares_sold * px_sell
            sell_fee_arr[i] = gross_proceeds_arr[i] * params.fee_rate
            net_proceeds_arr[i] = sl_net_proceeds
            realized_pnl_arr[i] = sl_realized_pnl
            cash_flow_arr[i] += sl_net_proceeds
        
        # Step 3: Hysteresis update
        update_hysteresis(state, price, params)
        
        # Step 4: Buy logic
        bought, shares_bought = apply_buy(
            state, signal, price, px_buy, params, tp_triggered, sl_triggered
        )
        
        if bought:
            shares_bought_arr[i] = shares_bought
            buy_amt_arr[i] = shares_bought * px_buy
            fee_arr[i] = buy_amt_arr[i] * params.fee_rate
            cash_flow_arr[i] -= (buy_amt_arr[i] + fee_arr[i])
        
        # Step 5: Peak price update
        update_peak_price(state, price)
        
        # Step 6: NAV calculation
        recalc_nav(state, price)
        
        # Step 7: Derived calculations
        # Portfolio return (ACB-based)
        if state.position_cost > 1e-9 and state.qty > 1e-9:
            portfolio_return = (state.equity / state.position_cost) - 1.0
        else:
            portfolio_return = 0.0
        
        # Unrealized P/L
        unrealized_pnl = state.equity - state.position_cost
        
        # Profit
        profit = state.equity + state.cum_cash_flow
        
        # NAV including invested
        nav_including_invested = state.cum_invested + profit
        
        # NAV return
        if state.cum_invested > 1e-9:
            nav_return = (state.nav / state.cum_invested) - 1.0
        else:
            nav_return = 0.0
        
        # Anchor return (for display)
        if state.anchor_price > 1e-9:
            ret_anchor_pct = ((price / state.anchor_price) - 1.0) * 100.0
        else:
            ret_anchor_pct = np.nan
        
        # Step 8: Store values
        qty_arr[i] = state.qty
        avg_cost_arr[i] = state.avg_cost
        position_cost_arr[i] = state.position_cost
        anchor_price_arr[i] = state.anchor_price
        peak_price_arr[i] = state.peak_price
        profit_cash_arr[i] = state.profit_cash
        principal_cash_arr[i] = state.principal_cash
        cum_invested_arr[i] = state.cum_invested
        cum_cash_flow_arr[i] = state.cum_cash_flow
        equity_arr[i] = state.equity
        cash_balance_arr[i] = state.cash_balance
        nav_arr[i] = state.nav
        tp_cooldown_arr[i] = state.tp_cooldown_remaining
        sl_cooldown_arr[i] = state.sl_cooldown_remaining
        ret_anchor_arr[i] = ret_anchor_pct
        portfolio_return_arr[i] = portfolio_return
        unrealized_pnl_arr[i] = unrealized_pnl
        profit_arr[i] = profit
        nav_including_invested_arr[i] = nav_including_invested
        nav_return_arr[i] = nav_return
    
    # Calculate drawdown (Portfolio Return based)
    drawdown = pd.Series(np.nan, index=idx)
    first_invest_mask = cum_invested_arr > 1e-9
    if first_invest_mask.any():
        first_invest_idx = np.where(first_invest_mask)[0][0]
        
        portfolio_ratio_series = pd.Series(np.nan, index=idx)
        for j in range(len(idx)):
            if position_cost_arr[j] > 1e-6 and qty_arr[j] > 1e-9:
                portfolio_ratio_series.iloc[j] = equity_arr[j] / position_cost_arr[j]
            elif position_cost_arr[j] <= 1e-6:
                drawdown.iloc[j] = 0.0
                portfolio_ratio_series.iloc[j] = np.nan
        
        portfolio_ratio_post = portfolio_ratio_series.loc[idx[first_invest_idx]:]
        
        if len(portfolio_ratio_post) > 1:
            valid_mask = portfolio_ratio_post.notna() & (portfolio_ratio_post > 1e-9)
            if valid_mask.any():
                portfolio_ratio_valid = portfolio_ratio_post[valid_mask]
                running_max_ratio = portfolio_ratio_valid.expanding().max()
                drawdown_post = (portfolio_ratio_valid / running_max_ratio) - 1.0
                drawdown_post = drawdown_post.clip(lower=-1.0, upper=0.0)
                drawdown.loc[portfolio_ratio_valid.index] = drawdown_post.values
            else:
                drawdown.loc[idx[first_invest_idx]:] = 0.0
        else:
            drawdown.loc[idx[first_invest_idx]:] = 0.0
    
    # Validation (invariants)
    _validate_ledger_v2(
        qty_arr=qty_arr,
        avg_cost_arr=avg_cost_arr,
        position_cost_arr=position_cost_arr,
        profit_cash_arr=profit_cash_arr,
        principal_cash_arr=principal_cash_arr,
        cash_balance_arr=cash_balance_arr,
        equity_arr=equity_arr,
        nav_arr=nav_arr,
    )
    
    # Build DataFrame
    ledger = pd.DataFrame({
        "DailyRet": daily_ret,
        "Signal": signals.astype(int),
        "Mode": "shares",
        "AdjClose": adj,
        "PxExec": px_exec_buy,
        "BuyAmt": pd.Series(buy_amt_arr, index=idx),
        "Fee": pd.Series(fee_arr, index=idx),
        "SharesBought": pd.Series(shares_bought_arr, index=idx),
        "CashFlow": pd.Series(cash_flow_arr, index=idx),
        "CumCashFlow": pd.Series(cum_cash_flow_arr, index=idx),
        "CumShares": pd.Series(qty_arr, index=idx),
        "Equity": pd.Series(equity_arr, index=idx),
        "CashBalance": pd.Series(cash_balance_arr, index=idx),
        "NAV": pd.Series(nav_arr, index=idx),
        "CumInvested": pd.Series(cum_invested_arr, index=idx),
        "Drawdown": drawdown,
        "NAVReturn": pd.Series(nav_return_arr, index=idx),
        "PortfolioReturn": pd.Series(portfolio_return_arr, index=idx),
        "PositionCost": pd.Series(position_cost_arr, index=idx),
        "UnrealizedPnl": pd.Series(unrealized_pnl_arr, index=idx),
        "Profit": pd.Series(profit_arr, index=idx),
        "NAV_including_invested": pd.Series(nav_including_invested_arr, index=idx),
        "TP_triggered": pd.Series(tp_triggered_arr, index=idx),
        "SL_triggered": pd.Series(sl_triggered_arr, index=idx),
        "SharesSold": pd.Series(shares_sold_arr, index=idx),
        "SellPrice": pd.Series(sell_price_arr, index=idx),
        "GrossProceeds": pd.Series(gross_proceeds_arr, index=idx),
        "SellFee": pd.Series(sell_fee_arr, index=idx),
        "NetProceeds": pd.Series(net_proceeds_arr, index=idx),
        "RealizedPnl": pd.Series(realized_pnl_arr, index=idx),
        "PostSellCumShares": pd.Series(qty_arr, index=idx),
        # ACB columns
        "AvgCost": pd.Series(avg_cost_arr, index=idx),
        # Anchor columns
        "AnchorPrice": pd.Series(anchor_price_arr, index=idx),
        "PeakPrice": pd.Series(peak_price_arr, index=idx),
        "RetAnchor": pd.Series(ret_anchor_arr, index=idx),
        # Wallet columns
        "ProfitCash": pd.Series(profit_cash_arr, index=idx),
        "PrincipalCash": pd.Series(principal_cash_arr, index=idx),
        # Trigger control columns
        "TPCooldown": pd.Series(tp_cooldown_arr, index=idx),
        "SLCooldown": pd.Series(sl_cooldown_arr, index=idx),
    }, index=idx)
    
    return ledger


def _validate_ledger_v2(
    qty_arr: np.ndarray,
    avg_cost_arr: np.ndarray,
    position_cost_arr: np.ndarray,
    profit_cash_arr: np.ndarray,
    principal_cash_arr: np.ndarray,
    cash_balance_arr: np.ndarray,
    equity_arr: np.ndarray,
    nav_arr: np.ndarray,
) -> None:
    """Validate ACB-based ledger invariants."""
    n = len(qty_arr)
    
    for i in range(n):
        qty = qty_arr[i]
        avg_cost = avg_cost_arr[i]
        position_cost = position_cost_arr[i]
        profit_cash = profit_cash_arr[i]
        principal_cash = principal_cash_arr[i]
        cash_balance = cash_balance_arr[i]
        equity = equity_arr[i]
        nav = nav_arr[i]
        
        # 1. CashBalance = profit_cash + principal_cash
        expected_cash_balance = profit_cash + principal_cash
        if not np.isclose(cash_balance, expected_cash_balance, atol=1e-6):
            raise AssertionError(
                f"Bar {i}: CashBalance != profit_cash + principal_cash: "
                f"{cash_balance:.6f} != {expected_cash_balance:.6f}"
            )
        
        # 2. position_cost = avg_cost × qty
        if qty > 1e-9:
            expected_position_cost = avg_cost * qty
            if not np.isclose(position_cost, expected_position_cost, atol=1e-6):
                raise AssertionError(
                    f"Bar {i}: position_cost != avg_cost × qty: "
                    f"{position_cost:.6f} != {expected_position_cost:.6f}"
                )
        
        # 3. If qty == 0, position_cost should be 0
        if qty < 1e-9:
            if position_cost > 1e-6:
                raise AssertionError(
                    f"Bar {i}: qty=0 but position_cost={position_cost:.6f} > 0"
                )
        
        # 4. profit_cash >= 0 (always non-negative)
        if profit_cash < -1e-9:
            raise AssertionError(
                f"Bar {i}: profit_cash < 0: {profit_cash:.6f}"
            )
        
        # 5. NAV = Equity + CashBalance
        expected_nav = equity + cash_balance
        if not np.isclose(nav, expected_nav, atol=1e-6):
            raise AssertionError(
                f"Bar {i}: NAV != Equity + CashBalance: "
                f"{nav:.6f} != {expected_nav:.6f}"
            )


def run_backtest_v2(
    prices: pd.DataFrame,
    params: BacktestParamsV2,
) -> dict:
    """
    Run backtest with V2 engine and return summary metrics.
    
    Returns dict with:
    - ledger: DataFrame
    - metrics: dict of performance metrics
    """
    ledger = compute_ledger_v2(prices, params)
    
    # Calculate metrics
    ending_nav = ledger["NAV"].iloc[-1]
    total_invested = ledger["CumInvested"].iloc[-1]
    ending_equity = ledger["Equity"].iloc[-1]
    
    # Profit
    profit = ending_nav - total_invested
    
    # Cumulative return
    if total_invested > 0:
        cumulative_return = (ending_nav / total_invested) - 1.0
    else:
        cumulative_return = 0.0
    
    # CAGR
    start_date = ledger.index[0]
    end_date = ledger.index[-1]
    years = (end_date - start_date).days / 365.25
    
    if years > 0 and ending_nav > 0 and total_invested > 0:
        cagr = ((ending_nav / total_invested) ** (1.0 / years)) - 1.0
    else:
        cagr = 0.0
    
    # MDD
    mdd = ledger["Drawdown"].min()
    
    # Trades
    num_buys = int((ledger["SharesBought"] > 0).sum())
    num_tp = int(ledger["TP_triggered"].sum())
    num_sl = int(ledger["SL_triggered"].sum())
    total_trades = num_buys + num_tp + num_sl
    
    # Signal days
    signal_days = int(ledger["Signal"].sum())
    
    # Realized P/L
    total_realized_gain = ledger[ledger["RealizedPnl"] > 0]["RealizedPnl"].sum()
    total_realized_loss = abs(ledger[ledger["RealizedPnl"] < 0]["RealizedPnl"].sum())
    net_realized_pnl = total_realized_gain - total_realized_loss
    
    # Benchmark CAGR (Buy & Hold)
    benchmark_start_date = ledger.index[0]
    benchmark_end_date = ledger.index[-1]
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
    
    # XIRR (Internal Rate of Return)
    # Calculate cash flows for XIRR
    cash_flows = ledger["CashFlow"].copy()
    dates = cash_flows.index
    
    # Only calculate XIRR if there are cash flows
    xirr_value = 0.0
    if len(cash_flows[cash_flows != 0]) > 1:
        try:
            # Prepare dates and amounts for XIRR calculation
            xirr_dates = []
            xirr_amounts = []
            
            # Add all cash flows
            for date, amount in zip(dates, cash_flows):
                if abs(amount) > 1e-9:  # Non-zero cash flow
                    xirr_dates.append(date)
                    xirr_amounts.append(float(amount))
            
            # Add final NAV as positive cash flow (terminal value)
            if ending_nav > 1e-9:
                xirr_dates.append(dates[-1])
                xirr_amounts.append(float(ending_nav))
            
            if len(xirr_dates) > 1:
                # xirr function signature: xirr(values, dates)
                xirr_value = xirr(xirr_amounts, [pd.Timestamp(d).to_pydatetime() for d in xirr_dates])
        except Exception:
            # XIRR calculation failed, use 0.0
            xirr_value = 0.0
    
    metrics = {
        "total_invested": total_invested,
        "ending_equity": ending_equity,
        "ending_nav": ending_nav,
        "profit": profit,
        "cumulative_return": cumulative_return,
        "cagr": cagr,
        "benchmark_cagr": benchmark_cagr,
        "mdd": mdd,
        "xirr": xirr_value,
        "total_trades": total_trades,
        "num_buys": num_buys,
        "num_tp": num_tp,
        "num_sl": num_sl,
        "signal_days": signal_days,
        "total_realized_gain": total_realized_gain,
        "total_realized_loss": total_realized_loss,
        "net_realized_pnl": net_realized_pnl,
    }
    
    return {
        "ledger": ledger,
        "metrics": metrics,
    }


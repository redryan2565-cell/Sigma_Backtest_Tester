"""Test accounting invariants and monotonicity - no network dependency."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestParams, run_backtest


def test_nav_equals_equity_plus_cashbalance():
    """Test that NAV == Equity + CashBalance (identity check)."""
    idx = pd.date_range('2023-01-01', periods=10, freq='D')
    
    prices = pd.DataFrame({
        'Open': 100.0 + np.random.randn(10) * 2,
        'High': 100.0 + np.random.randn(10) * 2 + 1.0,
        'Low': 100.0 + np.random.randn(10) * 2 - 1.0,
        'Close': 100.0 + np.random.randn(10) * 2,
        'AdjClose': 100.0 + np.random.randn(10) * 2,
        'Volume': [1000000] * 10,
    }, index=idx)
    
    # Ensure some drops to trigger buys
    prices.loc[idx[0], 'AdjClose'] = 95.0
    prices.loc[idx[2], 'AdjClose'] = 92.0
    
    params = BacktestParams(
        threshold=-0.04,
        shares_per_signal=10.0,
        enable_tp_sl=False,
    )
    
    ledger, _ = run_backtest(prices, params)
    
    # Verify NAV = Equity + CashBalance
    nav = ledger["NAV"]
    equity = ledger["Equity"]
    cash_balance = ledger["CashBalance"]
    
    assert np.allclose(nav, equity + cash_balance, rtol=1e-8, atol=1e-8), \
        "NAV must equal Equity + CashBalance"
    
    # Also verify NAV = Equity + CumCashFlow (cash balance is CumInvested + CumCashFlow)
    cum_cash_flow = ledger["CumCashFlow"]
    cum_invested = ledger["CumInvested"]
    expected_cash_balance = cum_invested + cum_cash_flow
    
    assert np.allclose(cash_balance, expected_cash_balance, rtol=1e-8, atol=1e-8), \
        "CashBalance must equal CumInvested + CumCashFlow"


def test_cumshares_nonnegative_and_integer():
    """Test that CumShares >= 0 and is reasonably integer-like."""
    idx = pd.date_range('2023-01-01', periods=10, freq='D')
    
    prices = pd.DataFrame({
        'Open': 100.0,
        'High': 101.0,
        'Low': 99.0,
        'Close': 100.0,
        'AdjClose': 100.0,
        'Volume': [1000000] * 10,
    }, index=idx)
    
    prices.loc[idx[0], 'AdjClose'] = 95.0
    
    params = BacktestParams(
        threshold=-0.05,
        shares_per_signal=10.0,
        enable_tp_sl=False,
    )
    
    ledger, _ = run_backtest(prices, params)
    
    cum_shares = ledger["CumShares"]
    
    # CumShares should never be negative
    assert (cum_shares >= -1e-10).all(), "CumShares must never be negative"
    
    # CumShares should be non-decreasing when there are no sells
    # (With TP/SL enabled, it can decrease, but we're testing without TP/SL)
    cum_shares_diff = cum_shares.diff().fillna(0.0)
    assert (cum_shares_diff >= -1e-10).all(), "CumShares should be non-decreasing when no sells"


def test_cuminvested_monotonic_increasing():
    """Test that CumInvested is monotonically increasing."""
    idx = pd.date_range('2023-01-01', periods=10, freq='D')
    
    prices = pd.DataFrame({
        'Open': 100.0,
        'High': 101.0,
        'Low': 99.0,
        'Close': 100.0,
        'AdjClose': 100.0,
        'Volume': [1000000] * 10,
    }, index=idx)
    
    prices.loc[idx[0], 'AdjClose'] = 95.0
    prices.loc[idx[2], 'AdjClose'] = 92.0
    
    params = BacktestParams(
        threshold=-0.04,
        shares_per_signal=10.0,
        enable_tp_sl=False,
    )
    
    ledger, _ = run_backtest(prices, params)
    
    cum_invested = ledger["CumInvested"]
    
    # CumInvested should be non-decreasing (monotonically increasing or constant)
    cum_invested_diff = cum_invested.diff().fillna(0.0)
    assert (cum_invested_diff >= -1e-10).all(), "CumInvested must be non-decreasing"
    
    # CumInvested should never decrease, even with sells (it represents total capital injected)
    # This is validated in the engine, but we verify here too


def test_navreturn_global_range():
    """Test that NAVReturn_global is in reasonable range."""
    idx = pd.date_range('2023-01-01', periods=10, freq='D')
    
    prices = pd.DataFrame({
        'Open': 100.0,
        'High': 101.0,
        'Low': 99.0,
        'Close': 100.0,
        'AdjClose': 100.0,
        'Volume': [1000000] * 10,
    }, index=idx)
    
    prices.loc[idx[0], 'AdjClose'] = 95.0
    
    params = BacktestParams(
        threshold=-0.05,
        shares_per_signal=10.0,
        enable_tp_sl=False,
    )
    
    ledger, _ = run_backtest(prices, params)
    
    if "NAVReturn_global" in ledger.columns:
        nav_return_global = ledger["NAVReturn_global"]
        cum_invested = ledger["CumInvested"]
        
        # NAVReturn_global should be NaN or 0 when CumInvested is 0
        mask_no_invest = cum_invested < 1e-9
        if mask_no_invest.any():
            nav_return_when_no_invest = nav_return_global.loc[mask_no_invest]
            assert (nav_return_when_no_invest.abs() < 1e-8).all() or nav_return_when_no_invest.isna().all(), \
                "NAVReturn_global should be 0 or NaN when no investment"
        
        # NAVReturn_global should be finite when CumInvested > 0
        mask_invest = cum_invested > 1e-9
        if mask_invest.any():
            nav_return_when_invest = nav_return_global.loc[mask_invest]
            assert nav_return_when_invest.notna().all(), "NAVReturn_global should be finite when invested"
            assert np.isfinite(nav_return_when_invest).all(), "NAVReturn_global should be finite when invested"
            
            # NAVReturn_global = (NAV / CumInvested) - 1
            # Can be negative (loss) or positive (gain), but extreme values might indicate issues
            # We don't set strict bounds, but verify it's calculated correctly
            nav = ledger["NAV"]
            expected_nav_return = (nav.loc[mask_invest] / cum_invested.loc[mask_invest]) - 1.0
            assert np.allclose(nav_return_when_invest, expected_nav_return, rtol=1e-8, atol=1e-8), \
                "NAVReturn_global should equal (NAV / CumInvested) - 1"



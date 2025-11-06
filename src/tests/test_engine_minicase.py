"""Mini test case with 5-day toy time series - no network dependency."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestParams, run_backtest


def test_minicase_5day_series():
    """Test basic backtest with 5-day toy time series.
    
    Price series: AdjClose=[100, 95, 97, 92, 96]
    threshold=-0.04, fee=slip=0 → Buy signals on days 2 (95) and 4 (92)
    """
    idx = pd.date_range('2023-01-01', periods=5, freq='D')
    
    prices = pd.DataFrame({
        'Open': [100.0, 95.0, 97.0, 92.0, 96.0],
        'High': [101.0, 96.0, 98.0, 93.0, 97.0],
        'Low': [99.0, 94.0, 96.0, 91.0, 95.0],
        'Close': [100.0, 95.0, 97.0, 92.0, 96.0],
        'AdjClose': [100.0, 95.0, 97.0, 92.0, 96.0],
        'Volume': [1000000] * 5,
    }, index=idx)
    
    # threshold=-0.04 means buy on 5% drop (95/100-1=-0.05) and 8% drop (92/100-1=-0.08)
    # But since we look at daily returns, day 2: 95/100-1=-0.05, day 4: 92/97-1=-0.0515
    # Both should trigger with threshold=-0.04
    params = BacktestParams(
        threshold=-0.04,
        shares_per_signal=10.0,
        fee_rate=0.0,  # No fees for simplicity
        slippage_rate=0.0,  # No slippage for simplicity
        enable_tp_sl=False,
    )
    
    ledger, metrics = run_backtest(prices, params)
    
    # Expected signals: days 1 (95/100-1=-0.05) and 3 (92/97-1=-0.0515)
    signals = ledger["Signal"].astype(bool)
    assert signals.iloc[1] == True, "Day 1 (95) should trigger buy signal"
    assert signals.iloc[3] == True, "Day 3 (92) should trigger buy signal"
    assert signals.iloc[0] == False, "Day 0 (100) should not trigger"
    assert signals.iloc[2] == False, "Day 2 (97) should not trigger"
    assert signals.iloc[4] == False, "Day 4 (96) should not trigger"
    
    # Expected buys: 2 purchases, 10 shares each
    shares_bought = ledger["SharesBought"]
    assert shares_bought.iloc[1] == 10.0, "Day 1 should buy 10 shares"
    assert shares_bought.iloc[3] == 10.0, "Day 3 should buy 10 shares"
    assert shares_bought.iloc[0] == 0.0, "Day 0 should not buy"
    assert shares_bought.iloc[2] == 0.0, "Day 2 should not buy"
    assert shares_bought.iloc[4] == 0.0, "Day 4 should not buy"
    
    # CumShares validation
    cum_shares = ledger["CumShares"]
    assert cum_shares.iloc[0] == 0.0, "Day 0: no shares"
    assert cum_shares.iloc[1] == 10.0, "Day 1: 10 shares"
    assert cum_shares.iloc[2] == 10.0, "Day 2: still 10 shares"
    assert cum_shares.iloc[3] == 20.0, "Day 3: 20 shares"
    assert cum_shares.iloc[4] == 20.0, "Day 4: still 20 shares"
    
    # CumCashFlow validation (with fee=0, slip=0)
    # Day 1: buy 10 shares @ 95 = -950
    # Day 3: buy 10 shares @ 92 = -920
    cum_cash_flow = ledger["CumCashFlow"]
    assert abs(cum_cash_flow.iloc[0]) < 1e-6, "Day 0: no cash flow"
    assert abs(cum_cash_flow.iloc[1] - (-950.0)) < 1e-6, f"Day 1: CumCF should be -950, got {cum_cash_flow.iloc[1]}"
    assert abs(cum_cash_flow.iloc[2] - (-950.0)) < 1e-6, f"Day 2: CumCF should be -950, got {cum_cash_flow.iloc[2]}"
    assert abs(cum_cash_flow.iloc[3] - (-1870.0)) < 1e-6, f"Day 3: CumCF should be -1870, got {cum_cash_flow.iloc[3]}"
    assert abs(cum_cash_flow.iloc[4] - (-1870.0)) < 1e-6, f"Day 4: CumCF should be -1870, got {cum_cash_flow.iloc[4]}"
    
    # NAV validation: NAV = Equity + CashBalance
    # Day 4: Equity = 20 shares * 96 = 1920, CashBalance = CumInvested + CumCF = 1870 - 1870 = 0
    # NAV = 1920 + 0 = 1920
    # Actually, let's recalculate: CumInvested = 950 + 920 = 1870
    # CashBalance = CumInvested + CumCF = 1870 - 1870 = 0
    # Equity = 20 * 96 = 1920
    # NAV = 1920 + 0 = 1920
    nav = ledger["NAV"]
    equity = ledger["Equity"]
    cash_balance = ledger["CashBalance"]
    
    # Verify NAV = Equity + CashBalance
    assert np.allclose(nav, equity + cash_balance, rtol=1e-8, atol=1e-8), "NAV must equal Equity + CashBalance"
    
    # Verify CumInvested
    cum_invested = ledger["CumInvested"]
    assert abs(cum_invested.iloc[4] - 1870.0) < 1e-6, f"Day 4: CumInvested should be 1870, got {cum_invested.iloc[4]}"
    
    # Final day NAV check (approximate)
    # Day 4: Equity = 20 * 96 = 1920, CashBalance = 0, NAV = 1920
    assert abs(nav.iloc[4] - 1920.0) < 1e-6, f"Day 4: NAV should be ~1920, got {nav.iloc[4]}"
    
    # Verify metrics
    assert abs(metrics["TotalInvested"] - 1870.0) < 1e-6, f"TotalInvested should be 1870, got {metrics['TotalInvested']}"
    assert metrics["Trades"] == 2.0, f"Trades should be 2, got {metrics['Trades']}"
    assert metrics["HitDays"] == 2.0, f"HitDays should be 2, got {metrics['HitDays']}"



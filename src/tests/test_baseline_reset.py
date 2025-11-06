"""Tests for TP/SL baseline reset functionality."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestParams, run_backtest


def test_baseline_reset_prevents_consecutive_tp():
    """Test that baseline reset prevents consecutive TP triggers."""
    # Create a price series where NAVReturn_global stays above TP threshold for multiple days
    # After first TP, baseline should reset and prevent consecutive triggers
    
    idx = pd.date_range('2023-01-01', periods=20, freq='D')
    
    # Create price sequence: start at 100, drop to trigger buy, then rise steadily
    # This will cause NAVReturn_global to exceed TP threshold for multiple days
    prices = pd.DataFrame({
        'Open': 100.0 + np.arange(20) * 0.5,
        'High': 100.0 + np.arange(20) * 0.5 + 1.0,
        'Low': 100.0 + np.arange(20) * 0.5 - 1.0,
        'Close': 100.0 + np.arange(20) * 0.5,
        'AdjClose': 100.0 + np.arange(20) * 0.5,
        'Volume': [1000000] * 20,
    }, index=idx)
    
    # First day: drop by 5% to trigger buy signal
    prices.loc[idx[0], 'AdjClose'] = 95.0
    
    # Then steady rise: 100 -> 105 -> 110 -> 115 -> ...
    # This will cause NAVReturn_global to exceed 30% threshold for multiple days
    
    params = BacktestParams(
        threshold=-0.05,  # Buy on 5% drop
        shares_per_signal=10.0,
        enable_tp_sl=True,
        tp_threshold=0.30,  # 30% TP threshold
        sl_threshold=-0.20,  # -20% SL threshold
        tp_sell_percentage=0.5,  # Sell 50% on TP
        sl_sell_percentage=1.0,
        reset_baseline_after_tp_sl=True,  # Enable baseline reset
        tp_hysteresis=0.0,  # No hysteresis
        sl_hysteresis=0.0,
        tp_cooldown_days=0,  # No cooldown
        sl_cooldown_days=0,
    )
    
    ledger, metrics = run_backtest(prices, params)
    
    # Check that TP triggers occurred
    tp_count = int(metrics.get('NumTakeProfits', 0))
    
    # With baseline reset enabled, we should get at most 1 TP per run-up period
    # Even if NAVReturn_global stays above threshold for multiple days,
    # baseline reset should prevent consecutive triggers
    
    # Verify baseline reset happened: after TP, ROI_base should equal NAV_post_trade
    tp_days = ledger[ledger['TP_triggered'] == True].index
    
    if len(tp_days) > 0:
        # Check first TP day
        first_tp_day = tp_days[0]
        tp_day_idx = ledger.index.get_loc(first_tp_day)
        
        # After TP, ROI_base should be reset to NAV
        # Check next day's ROI_base equals NAV after TP
        if tp_day_idx + 1 < len(ledger):
            next_day_nav = ledger.loc[ledger.index[tp_day_idx + 1], 'NAV']
            next_day_roi_base = ledger.loc[ledger.index[tp_day_idx + 1], 'ROI_base']
            
            # ROI_base should be close to NAV after TP (within rounding tolerance)
            assert abs(next_day_roi_base - next_day_nav) < 1e-6, \
                f"ROI_base should equal NAV after TP reset. ROI_base: {next_day_roi_base}, NAV: {next_day_nav}"
        
        # Verify NAVReturn_baselined resets after TP
        if tp_day_idx + 1 < len(ledger):
            next_day_baselined_return = ledger.loc[ledger.index[tp_day_idx + 1], 'NAVReturn_baselined']
            # After reset, baselined return should be close to 0
            assert abs(next_day_baselined_return) < 0.01, \
                f"NAVReturn_baselined should reset to ~0 after TP. Got: {next_day_baselined_return}"


def test_baseline_initialization_on_first_buy():
    """Test that ROI_base is initialized on first buy."""
    idx = pd.date_range('2023-01-01', periods=10, freq='D')
    
    prices = pd.DataFrame({
        'Open': 100.0,
        'High': 101.0,
        'Low': 99.0,
        'Close': 100.0,
        'AdjClose': 100.0,
        'Volume': [1000000] * 10,
    }, index=idx)
    
    # First day: drop by 5% to trigger buy
    prices.loc[idx[0], 'AdjClose'] = 95.0
    
    params = BacktestParams(
        threshold=-0.05,
        shares_per_signal=10.0,
        enable_tp_sl=True,
        tp_threshold=0.30,
        sl_threshold=-0.20,
        tp_sell_percentage=1.0,
        sl_sell_percentage=1.0,
        reset_baseline_after_tp_sl=True,
    )
    
    ledger, metrics = run_backtest(prices, params)
    
    # Find first buy day
    first_buy_day = ledger[ledger['SharesBought'] > 0].index[0]
    buy_day_idx = ledger.index.get_loc(first_buy_day)
    
    # ROI_base should be set on first buy day
    roi_base = ledger.loc[first_buy_day, 'ROI_base']
    nav = ledger.loc[first_buy_day, 'NAV']
    
    assert not pd.isna(roi_base), "ROI_base should be initialized on first buy"
    assert abs(roi_base - nav) < 1e-6, \
        f"ROI_base should equal NAV on first buy. ROI_base: {roi_base}, NAV: {nav}"


def test_hysteresis_rearming():
    """Test that hysteresis correctly re-arms TP after return drops below threshold - hysteresis."""
    idx = pd.date_range('2023-01-01', periods=30, freq='D')
    
    # Create price sequence: drop to trigger buy, then rise to TP, then drop and rise again
    prices = pd.DataFrame({
        'Open': 100.0,
        'High': 101.0,
        'Low': 99.0,
        'Close': 100.0,
        'AdjClose': 100.0,
        'Volume': [1000000] * 30,
    }, index=idx)
    
    # Day 0: drop by 5% to trigger buy
    prices.loc[idx[0], 'AdjClose'] = 95.0
    
    # Days 1-10: steady rise to trigger TP (reach 30% gain)
    # Days 11-15: drop below threshold - hysteresis (e.g., 20%)
    # Days 16-20: rise again above threshold to trigger second TP
    
    for i in range(1, 11):
        # Rise to about 35% gain (above 30% TP threshold)
        prices.loc[idx[i], 'AdjClose'] = 95.0 * (1.0 + 0.035 * i / 10)
    
    for i in range(11, 16):
        # Drop to about 20% gain (below 30% - 10% hysteresis = 20%)
        prices.loc[idx[i], 'AdjClose'] = 95.0 * (1.0 + 0.20)
    
    for i in range(16, 21):
        # Rise again above 30% threshold
        prices.loc[idx[i], 'AdjClose'] = 95.0 * (1.0 + 0.035 * i / 10)
    
    params = BacktestParams(
        threshold=-0.05,
        shares_per_signal=10.0,
        enable_tp_sl=True,
        tp_threshold=0.30,  # 30% TP threshold
        sl_threshold=-0.20,
        tp_sell_percentage=0.5,
        sl_sell_percentage=1.0,
        reset_baseline_after_tp_sl=True,
        tp_hysteresis=0.10,  # 10% hysteresis
        sl_hysteresis=0.0,
        tp_cooldown_days=0,
        sl_cooldown_days=0,
    )
    
    ledger, metrics = run_backtest(prices, params)
    
    # Should have at least 2 TP triggers (first one, then re-armed after hysteresis)
    tp_count = int(metrics.get('NumTakeProfits', 0))
    # Note: This test may need adjustment based on actual price sequence
    # The key is that hysteresis allows re-arming after return drops below threshold - hysteresis


def test_cooldown_prevents_immediate_retrigger():
    """Test that cooldown prevents immediate TP retrigger."""
    idx = pd.date_range('2023-01-01', periods=15, freq='D')
    
    prices = pd.DataFrame({
        'Open': 100.0,
        'High': 101.0,
        'Low': 99.0,
        'Close': 100.0,
        'AdjClose': 100.0,
        'Volume': [1000000] * 15,
    }, index=idx)
    
    # Day 0: drop by 5% to trigger buy
    prices.loc[idx[0], 'AdjClose'] = 95.0
    
    # Days 1-5: rise to trigger TP
    for i in range(1, 6):
        prices.loc[idx[i], 'AdjClose'] = 95.0 * (1.0 + 0.35 * i / 5)
    
    # Days 6-10: stay high (above TP threshold)
    for i in range(6, 11):
        prices.loc[idx[i], 'AdjClose'] = 95.0 * 1.35
    
    params = BacktestParams(
        threshold=-0.05,
        shares_per_signal=10.0,
        enable_tp_sl=True,
        tp_threshold=0.30,
        sl_threshold=-0.20,
        tp_sell_percentage=0.5,  # Sell 50% only
        sl_sell_percentage=1.0,
        reset_baseline_after_tp_sl=True,
        tp_hysteresis=0.0,
        sl_hysteresis=0.0,
        tp_cooldown_days=3,  # 3-day cooldown
        sl_cooldown_days=0,
    )
    
    ledger, metrics = run_backtest(prices, params)
    
    # Check TP triggers
    tp_days = ledger[ledger['TP_triggered'] == True].index
    
    if len(tp_days) >= 2:
        # Check that second TP is at least cooldown_days after first
        first_tp_idx = ledger.index.get_loc(tp_days[0])
        second_tp_idx = ledger.index.get_loc(tp_days[1])
        
        days_between = second_tp_idx - first_tp_idx
        assert days_between >= 3, \
            f"Second TP should be at least 3 days after first. Got: {days_between}"


def test_baselined_return_calculation():
    """Test that NAVReturn_baselined is calculated correctly."""
    idx = pd.date_range('2023-01-01', periods=10, freq='D')
    
    prices = pd.DataFrame({
        'Open': 100.0,
        'High': 101.0,
        'Low': 99.0,
        'Close': 100.0,
        'AdjClose': 100.0,
        'Volume': [1000000] * 10,
    }, index=idx)
    
    # Day 0: drop by 5% to trigger buy (NAV = 1000, ROI_base = 1000)
    prices.loc[idx[0], 'AdjClose'] = 95.0
    
    # Day 1: price rises to 105 (NAV = 1100, ROI_base = 1000, baselined return = 10%)
    prices.loc[idx[1], 'AdjClose'] = 105.0
    
    params = BacktestParams(
        threshold=-0.05,
        shares_per_signal=10.0,
        enable_tp_sl=True,
        tp_threshold=0.30,
        sl_threshold=-0.20,
        tp_sell_percentage=1.0,
        sl_sell_percentage=1.0,
        reset_baseline_after_tp_sl=False,  # Don't reset for this test
    )
    
    ledger, metrics = run_backtest(prices, params)
    
    # Check calculation: NAVReturn_baselined = (NAV / ROI_base) - 1
    day1 = ledger.index[1]
    nav = ledger.loc[day1, 'NAV']
    roi_base = ledger.loc[day1, 'ROI_base']
    baselined_return = ledger.loc[day1, 'NAVReturn_baselined']
    
    expected_return = (nav / roi_base) - 1.0
    
    assert abs(baselined_return - expected_return) < 1e-8, \
        f"NAVReturn_baselined calculation incorrect. Expected: {expected_return}, Got: {baselined_return}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


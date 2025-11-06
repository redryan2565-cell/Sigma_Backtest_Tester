"""Test MDD (Maximum Drawdown) metric - no network dependency."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestParams, run_backtest
from src.backtest.metrics import max_drawdown


def test_mdd_range():
    """Test that MDD is in valid range: -1 <= MDD <= 0."""
    idx = pd.date_range('2023-01-01', periods=20, freq='D')
    
    # Create a price series with some volatility
    prices = pd.DataFrame({
        'Open': 100.0 + np.random.randn(20) * 5,
        'High': 100.0 + np.random.randn(20) * 5 + 1.0,
        'Low': 100.0 + np.random.randn(20) * 5 - 1.0,
        'Close': 100.0 + np.random.randn(20) * 5,
        'AdjClose': 100.0 + np.random.randn(20) * 5,
        'Volume': [1000000] * 20,
    }, index=idx)
    
    # Ensure some drops to trigger buys
    prices.loc[idx[0], 'AdjClose'] = 95.0
    prices.loc[idx[2], 'AdjClose'] = 92.0
    
    params = BacktestParams(
        threshold=-0.04,
        shares_per_signal=10.0,
        enable_tp_sl=False,
    )
    
    ledger, metrics = run_backtest(prices, params)
    
    # Verify MDD is in valid range
    mdd = metrics["MDD"]
    assert -1.0 <= mdd <= 0.0, f"MDD should be in range [-1, 0], got {mdd}"
    
    # Verify MDD calculation from NAV series
    nav = ledger["NAV"]
    cum_invested = ledger["CumInvested"]
    
    # Find first investment date
    first_invest_mask = cum_invested > 1e-9
    if first_invest_mask.any():
        first_invest_date = nav.index[first_invest_mask][0]
        nav_from_first_invest = nav.loc[first_invest_date:]
        
        if len(nav_from_first_invest) > 1:
            computed_mdd = max_drawdown(nav_from_first_invest)
            assert abs(computed_mdd - mdd) < 1e-6, \
                f"MDD from metrics should match computed MDD. Metrics: {mdd}, Computed: {computed_mdd}"


def test_mdd_edge_cases():
    """Test MDD with edge cases: empty series, single value, etc."""
    # Empty series
    empty_series = pd.Series([], dtype=float)
    mdd_empty = max_drawdown(empty_series)
    assert mdd_empty == 0.0, f"MDD for empty series should be 0.0, got {mdd_empty}"
    
    # Single value
    single_value = pd.Series([100.0])
    mdd_single = max_drawdown(single_value)
    assert mdd_single == 0.0, f"MDD for single value should be 0.0, got {mdd_single}"
    
    # Two values - no drawdown
    two_values_up = pd.Series([100.0, 110.0])
    mdd_two_up = max_drawdown(two_values_up)
    assert mdd_two_up == 0.0, f"MDD for increasing series should be 0.0, got {mdd_two_up}"
    
    # Two values - drawdown
    two_values_down = pd.Series([100.0, 90.0])
    mdd_two_down = max_drawdown(two_values_down)
    assert abs(mdd_two_down - (-0.10)) < 1e-6, \
        f"MDD for [100, 90] should be -0.10, got {mdd_two_down}"
    
    # Perfect drawdown (goes to zero)
    zero_drawdown = pd.Series([100.0, 50.0, 0.0])
    mdd_zero = max_drawdown(zero_drawdown)
    assert abs(mdd_zero - (-1.0)) < 1e-6, \
        f"MDD for series going to zero should be -1.0, got {mdd_zero}"
    
    # No drawdown (monotonically increasing)
    increasing = pd.Series([100.0, 110.0, 120.0, 130.0])
    mdd_inc = max_drawdown(increasing)
    assert mdd_inc == 0.0, f"MDD for increasing series should be 0.0, got {mdd_inc}"
    
    # Maximum drawdown in middle
    peak_then_drop = pd.Series([100.0, 150.0, 120.0, 140.0])
    mdd_peak = max_drawdown(peak_then_drop)
    # MDD from peak (150) to lowest after peak (120) = (120/150) - 1 = -0.20
    assert abs(mdd_peak - (-0.20)) < 1e-6, \
        f"MDD for peak-then-drop should be -0.20, got {mdd_peak}"


def test_mdd_with_negative_values():
    """Test MDD handles edge cases with zeros or negative values."""
    # Series with zeros
    with_zeros = pd.Series([100.0, 0.0, 50.0])
    mdd_zeros = max_drawdown(with_zeros)
    # Should filter zeros and compute from positive values
    assert -1.0 <= mdd_zeros <= 0.0, f"MDD should be in range [-1, 0], got {mdd_zeros}"
    
    # Series with all zeros (should return 0.0)
    all_zeros = pd.Series([0.0, 0.0, 0.0])
    mdd_all_zeros = max_drawdown(all_zeros)
    assert mdd_all_zeros == 0.0, f"MDD for all zeros should be 0.0, got {mdd_all_zeros}"


def test_mdd_in_backtest_results():
    """Test that MDD in backtest results is valid."""
    idx = pd.date_range('2023-01-01', periods=30, freq='D')
    
    # Create a realistic price series with volatility
    np.random.seed(42)  # For reproducibility
    returns = np.random.randn(30) * 0.02  # 2% daily volatility
    prices_array = 100.0 * np.exp(np.cumsum(returns))
    
    prices = pd.DataFrame({
        'Open': prices_array,
        'High': prices_array * 1.01,
        'Low': prices_array * 0.99,
        'Close': prices_array,
        'AdjClose': prices_array,
        'Volume': [1000000] * 30,
    }, index=idx)
    
    # Ensure some drops to trigger buys
    prices.loc[idx[0], 'AdjClose'] = prices_array[0] * 0.95
    
    params = BacktestParams(
        threshold=-0.04,
        shares_per_signal=10.0,
        enable_tp_sl=False,
    )
    
    ledger, metrics = run_backtest(prices, params)
    
    # Verify MDD is in valid range
    mdd = metrics["MDD"]
    assert -1.0 <= mdd <= 0.0, f"MDD should be in range [-1, 0], got {mdd}"
    
    # Verify MDD matches ledger Drawdown column
    drawdown = ledger["Drawdown"]
    drawdown_valid = drawdown.dropna()
    
    if len(drawdown_valid) > 0:
        # MDD should be the minimum drawdown
        min_drawdown = drawdown_valid.min()
        assert abs(mdd - min_drawdown) < 1e-6, \
            f"MDD from metrics should match minimum Drawdown. MDD: {mdd}, Min Drawdown: {min_drawdown}"



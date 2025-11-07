import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestParams, run_backtest


def _sample_prices() -> pd.DataFrame:
    idx = pd.to_datetime([
        "2023-01-02",
        "2023-01-03",
        "2023-01-04",
        "2023-01-05",
    ])
    values = [100.0, 95.0, 94.0, 97.0]
    data = {
        "Open": values,
        "High": [v * 1.01 for v in values],
        "Low": [v * 0.99 for v in values],
        "Close": values,
        "AdjClose": values,
        "Volume": [1_000_000] * len(values),
    }
    return pd.DataFrame(data, index=idx)


def test_nav_and_drawdown_remain_well_formed_for_share_mode() -> None:
    prices = _sample_prices()
    params = BacktestParams(
        threshold=-0.04,
        shares_per_signal=1.0,
        fee_rate=0.0005,
        slippage_rate=0.0005,
    )

    ledger, metrics = run_backtest(prices, params)

    signals = ledger["Signal"] > 0
    assert signals.any(), "expected at least one buy signal"

    nav = ledger["NAV"].astype(float)
    equity = ledger["Equity"].astype(float)

    assert np.allclose(nav.values, equity.values)

    drawdown = ledger["Drawdown"].dropna()
    if not drawdown.empty:
        assert (drawdown >= -1.0 - 1e-9).all()
        assert (drawdown <= 1e-9).all()

    assert metrics["EndingNAV"] == pytest.approx(nav.iloc[-1])
    assert metrics["EndingEquity"] == pytest.approx(equity.iloc[-1])

    total_invested = metrics["TotalInvested"]
    if total_invested > 0:
        expected_cum_return = metrics["EndingNAV"] / total_invested - 1.0
        assert metrics["CumulativeReturn"] == pytest.approx(expected_cum_return)

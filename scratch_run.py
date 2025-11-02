from __future__ import annotations

import json
from datetime import date

from src.backtest.engine import BacktestParams, run_backtest
from src.data.yfin import YFinanceFeed


def main() -> None:
    # 짧은 구간으로 먼저 확인 (네트워크 지연 최소화)
    start = date(2023, 1, 3)
    end = date(2023, 2, 15)
    ticker = "TQQQ"

    feed = YFinanceFeed()
    prices = feed.get_daily(ticker, start, end)

    params = BacktestParams(
        threshold=-0.041,
        weekly_budget=500.0,
        mode="split",
        carryover=True,
        fee_rate=0.0005,
        slippage_rate=0.0005,
    )

    daily, metrics = run_backtest(prices, params)

    print(json.dumps(metrics, indent=2, default=float))
    daily.to_csv("daily.csv", encoding="utf-8-sig")
    print("saved: daily.csv")


if __name__ == "__main__":
    main()



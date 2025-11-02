from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Protocol

import pandas as pd


class PriceFeed(ABC):
    """Abstract base class for price data providers.

    Implementations should return a DataFrame indexed by datetime (UTC-naive),
    containing columns: ['Open','High','Low','Close','AdjClose','Volume'].
    """

    @abstractmethod
    def get_daily(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Fetch daily OHLCV data.

        Args:
            ticker: Ticker symbol, e.g., 'TQQQ'.
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            DataFrame with columns ['Open','High','Low','Close','AdjClose','Volume'].
        """


class SupportsGetDaily(Protocol):
    def get_daily(self, ticker: str, start: date, end: date) -> pd.DataFrame:  # pragma: no cover - protocol
        ...



from __future__ import annotations

from datetime import date, datetime
from typing import Dict, Any

import pandas as pd
import requests

from ..config import get_settings
from .base import PriceFeed


class AlphaVantageFeed(PriceFeed):
    """Alpha Vantage TIME_SERIES_DAILY_ADJUSTED feed.

    Requires ALPHA_VANTAGE_KEY in environment or .env.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or get_settings().alpha_vantage_key
        if not self.api_key:
            raise RuntimeError("Alpha Vantage API key not configured (ALPHA_VANTAGE_KEY)")

    def get_daily(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        params: Dict[str, Any] = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": "full",
            "datatype": "json",
            "apikey": self.api_key,
        }
        r = requests.get(self.BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "Time Series (Daily)" not in data:
            raise ValueError(f"Unexpected Alpha Vantage response: {data.get('Note') or data}")
        ts = data["Time Series (Daily)"]
        records = []
        for ds, row in ts.items():
            records.append(
                {
                    "Date": pd.to_datetime(ds),
                    "Open": float(row["1. open"]),
                    "High": float(row["2. high"]),
                    "Low": float(row["3. low"]),
                    "Close": float(row["4. close"]),
                    "AdjClose": float(row["5. adjusted close"]),
                    "Volume": int(row["6. volume"]),
                }
            )
        df = pd.DataFrame.from_records(records).set_index("Date").sort_index()
        df = df.loc[(df.index.date >= start) & (df.index.date <= end)]
        if df.empty:
            raise ValueError("No data returned for the given range from Alpha Vantage")
        return df



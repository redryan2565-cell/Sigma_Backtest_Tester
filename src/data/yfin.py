from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from ..config import get_settings
from .base import PriceFeed
from .cache import DataCache, get_cache


class YFinanceFeed(PriceFeed):
    """PriceFeed implementation using yfinance.

    Returns OHLCV with 'AdjClose' column.
    """

    def __init__(self, session: Optional[object] = None, cache: Optional[DataCache] = None) -> None:
        self._session = session
        if cache is None:
            # Use settings to configure cache
            settings = get_settings()
            self._cache = DataCache(
                cache_dir=settings.cache_dir,
                ttl_hours=settings.cache_ttl_hours,
                enabled=settings.cache_enabled,
            )
        else:
            self._cache = cache

    def get_daily(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        # Validate inputs
        if not ticker or not isinstance(ticker, str) or len(ticker.strip()) == 0:
            raise ValueError("Ticker symbol must be a non-empty string")
        if start > end:
            raise ValueError(f"Start date ({start}) must be <= end date ({end})")
        
        # Check cache first
        cached_data = self._cache.get(ticker, start, end)
        if cached_data is not None:
            return cached_data
        
        try:
            import yfinance as yf
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("yfinance is required for YFinanceFeed") from exc

        # yfinance end is exclusive for download; add one day to include end
        start_dt = datetime.combine(start, datetime.min.time())
        end_dt = datetime.combine(end + timedelta(days=1), datetime.min.time())

        df = yf.download(
            ticker,
            start=start_dt,
            end=end_dt,
            auto_adjust=False,
            progress=False,
            session=self._session,
        )
        if df is None or df.empty:
            raise ValueError("No data returned from yfinance for the given range")
        # Handle MultiIndex columns (can happen even for single ticker in some versions)
        if isinstance(df.columns, pd.MultiIndex):
            # If top level looks like the ticker symbol, slice it; otherwise collapse to the last level
            top = df.columns.get_level_values(0)
            if str(ticker) in set(map(str, top)):
                try:
                    df = df.xs(str(ticker), axis=1, level=0)
                except Exception:
                    df.columns = df.columns.get_level_values(-1)
            else:
                df.columns = df.columns.get_level_values(-1)

        df = df.rename(columns={"Adj Close": "AdjClose"})
        required = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            # Fallback to Ticker().history API which is often more stable
            try:
                tkr = yf.Ticker(str(ticker))
                h = tkr.history(start=start_dt, end=end_dt, auto_adjust=False, actions=False)
                if isinstance(h.columns, pd.MultiIndex):
                    h.columns = h.columns.get_level_values(-1)
                h = h.rename(columns={"Adj Close": "AdjClose"})
                df = h
                missing = [c for c in required if c not in df.columns]
            except Exception:
                pass
            if missing:
                # provide context to caller
                raise ValueError(
                    f"Missing columns from yfinance data: {missing}; got columns={list(df.columns)}"
                )

        # ensure index is DatetimeIndex sorted ascending
        df = df.sort_index()
        df.index = pd.to_datetime(df.index)

        # keep only required columns and cast types
        out = df[required].copy()
        for c in ["Open", "High", "Low", "Close", "AdjClose"]:
            series = out[c]
            # If a stray dimension remains, squeeze to Series
            if isinstance(series, pd.DataFrame):
                if series.shape[1] == 1:
                    series = series.iloc[:, 0]
                else:
                    # collapse by taking first column defensively
                    series = series.iloc[:, 0]
            out[c] = pd.to_numeric(series, errors="coerce")
        out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").astype("Int64")
        out = out.dropna(subset=["AdjClose"])  # require AdjClose

        # trim to inclusive end (defensive)
        out = out.loc[(out.index.date >= start) & (out.index.date <= end)]
        if out.empty:
            raise ValueError("No data after trimming to requested date range")

        # Cache the result
        self._cache.set(ticker, start, end, out)

        return out



from __future__ import annotations

import re
from datetime import date, datetime, timedelta

import pandas as pd

from ...config import get_settings
from ...data.base import PriceFeed
from ...data.cache import DataCache


class YFinanceFeed(PriceFeed):
    """PriceFeed implementation using yfinance.

    Returns OHLCV with 'AdjClose' column.
    """

    def __init__(self, session: object | None = None, cache: DataCache | None = None) -> None:
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

        ticker = ticker.strip().upper()

        # Security: Validate ticker format to prevent injection attacks
        if len(ticker) > 15:
            raise ValueError("Ticker symbol is too long (max 15 characters)")
        if not re.match(r'^[A-Z0-9.\-]+$', ticker):
            raise ValueError("Ticker symbol contains invalid characters. Only alphanumeric characters, dots, and hyphens are allowed.")

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

    def validate_ticker(self, ticker: str) -> bool:
        """Validate if a ticker symbol exists and is supported.

        Args:
            ticker: Ticker symbol to validate (e.g., 'TQQQ', 'AAPL').

        Returns:
            True if ticker exists and is valid, False otherwise.
        """
        if not ticker or not isinstance(ticker, str) or len(ticker.strip()) == 0:
            return False

        ticker = ticker.strip().upper()

        # Security: Validate ticker format to prevent injection attacks
        # Ticker symbols typically contain only alphanumeric characters, dots, and hyphens
        # Length limit: 1-15 characters (most tickers are 1-5 chars, but some can be longer)
        if len(ticker) > 15:
            return False

        # Only allow alphanumeric, dots, and hyphens
        # This prevents SQL injection, command injection, path traversal, etc.
        if not re.match(r'^[A-Z0-9.\-]+$', ticker):
            return False

        try:
            import yfinance as yf
        except Exception:
            # If yfinance is not available, cannot validate
            return False

        try:
            import warnings
            # Suppress yfinance warnings for invalid tickers (404 errors are expected)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                tkr = yf.Ticker(ticker)
                # Try to get basic info - if ticker doesn't exist, this will fail or return empty
                info = tkr.info

            # Check if we got valid info (should have at least 'symbol' key)
            if not info or len(info) == 0:
                return False

            # Check if symbol matches (case-insensitive)
            symbol = info.get('symbol', '').upper()
            if symbol != ticker:
                # Sometimes Yahoo returns different symbol (e.g., for delisted stocks)
                # Try to get history as additional check
                try:
                    hist = tkr.history(period="1d")
                    if hist is None or hist.empty:
                        return False
                except Exception:
                    return False

            return True
        except Exception:
            # Any exception means ticker is likely invalid
            return False



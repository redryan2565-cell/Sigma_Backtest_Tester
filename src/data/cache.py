from __future__ import annotations

import hashlib
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


class DataCache:
    """Disk-based cache for price data with TTL support.
    
    Caches price data using pickle format. Cache keys are based on
    (ticker, start_date, end_date). Cache entries expire after TTL hours.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl_hours: int = 24,
        enabled: bool = True,
    ) -> None:
        """Initialize cache.
        
        Args:
            cache_dir: Directory for cache files. Defaults to .cache/ in project root.
            ttl_hours: Time to live in hours. Defaults to 24.
            enabled: Whether caching is enabled. Defaults to True.
        """
        self.enabled = enabled
        self.ttl_hours = ttl_hours
        
        if cache_dir is None:
            # Default to .cache/ in project root (look for src/ parent)
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / ".cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, ticker: str, start: date, end: date) -> str:
        """Generate cache key from parameters."""
        key_str = f"{ticker}_{start.isoformat()}_{end.isoformat()}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{cache_key}.pkl"

    def get(self, ticker: str, start: date, end: date) -> Optional[pd.DataFrame]:
        """Retrieve cached data if available and not expired.
        
        Args:
            ticker: Stock ticker symbol.
            start: Start date.
            end: End date.
            
        Returns:
            Cached DataFrame if found and valid, None otherwise.
        """
        if not self.enabled:
            return None
            
        cache_key = self._cache_key(ticker, start, end)
        cache_path = self._cache_path(cache_key)
        
        if not cache_path.exists():
            return None
            
        try:
            # Check expiration
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            age_hours = (datetime.now() - mtime).total_seconds() / 3600
            if age_hours > self.ttl_hours:
                # Expired, remove it
                cache_path.unlink()
                return None
            
            # Load cached data
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
            
            # Validate cached data structure
            if not isinstance(cached_data, dict):
                return None
            if "ticker" not in cached_data or "start" not in cached_data or "end" not in cached_data:
                return None
            if cached_data["ticker"] != ticker or cached_data["start"] != start or cached_data["end"] != end:
                return None
            if "data" not in cached_data or not isinstance(cached_data["data"], pd.DataFrame):
                return None
                
            return cached_data["data"]
        except Exception:
            # On any error, remove corrupted cache
            try:
                cache_path.unlink()
            except Exception:
                pass
            return None

    def set(self, ticker: str, start: date, end: date, data: pd.DataFrame) -> None:
        """Store data in cache.
        
        Args:
            ticker: Stock ticker symbol.
            start: Start date.
            end: End date.
            data: DataFrame to cache.
        """
        if not self.enabled:
            return
            
        cache_key = self._cache_key(ticker, start, end)
        cache_path = self._cache_path(cache_key)
        
        try:
            cached_data = {
                "ticker": ticker,
                "start": start,
                "end": end,
                "data": data,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cached_data, f)
        except Exception:
            # Silently fail on cache write errors
            pass

    def clear(self, ticker: Optional[str] = None) -> None:
        """Clear cache entries.
        
        Args:
            ticker: If provided, clear only entries for this ticker. Otherwise clear all.
        """
        if not self.cache_dir.exists():
            return
            
        if ticker is None:
            # Clear all
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception:
                    pass
        else:
            # Clear specific ticker (need to check contents)
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    with open(cache_file, "rb") as f:
                        cached_data = pickle.load(f)
                    if isinstance(cached_data, dict) and cached_data.get("ticker") == ticker:
                        cache_file.unlink()
                except Exception:
                    pass

    def invalidate_expired(self) -> int:
        """Remove expired cache entries.
        
        Returns:
            Number of entries removed.
        """
        if not self.cache_dir.exists():
            return 0
            
        removed = 0
        now = datetime.now()
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                age_hours = (now - mtime).total_seconds() / 3600
                if age_hours > self.ttl_hours:
                    cache_file.unlink()
                    removed += 1
            except Exception:
                pass
                
        return removed


# Global cache instance (singleton pattern)
_cache_instance: Optional[DataCache] = None


def get_cache() -> DataCache:
    """Get global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = DataCache()
    return _cache_instance


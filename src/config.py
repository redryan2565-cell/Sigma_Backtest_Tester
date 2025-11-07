from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment and optional .env file.

    Attributes:
        alpha_vantage_key: API key for Alpha Vantage (optional).
        cache_enabled: Whether data caching is enabled. Defaults to True.
        cache_ttl_hours: Cache TTL in hours. Defaults to 24.
        cache_dir: Cache directory path. Defaults to .cache/ in project root.
        debug_mode: Whether to show detailed error messages. Defaults to False (production mode).
        developer_mode: Whether to enable developer features. Defaults to False (production mode).
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to INFO in production, DEBUG if debug_mode is True.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    alpha_vantage_key: str | None = Field(default=None, alias="ALPHA_VANTAGE_KEY")
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_ttl_hours: int = Field(default=24, alias="CACHE_TTL_HOURS")
    cache_dir: Path | None = Field(default=None, alias="CACHE_DIR")
    debug_mode: bool = Field(default=False, alias="DEBUG_MODE")
    developer_mode: bool = Field(default=False, alias="DEVELOPER_MODE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


def get_settings() -> Settings:
    """Return singleton settings instance.

    Returns:
        Settings: Settings populated from environment.
    """
    return Settings()  # type: ignore[call-arg]


def setup_logging(settings: Settings | None = None) -> None:
    """Setup logging configuration based on settings.

    Args:
        settings: Settings instance. If None, will load from get_settings().
    """
    if settings is None:
        settings = get_settings()

    # Determine log level
    if settings.debug_mode:
        level = logging.DEBUG
    else:
        # Parse LOG_LEVEL environment variable
        level_str = settings.log_level.upper()
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level = level_map.get(level_str, logging.INFO)

    # Configure logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress noisy third-party loggers in production
    if not settings.debug_mode:
        logging.getLogger("yfinance").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)



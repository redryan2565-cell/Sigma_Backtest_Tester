from __future__ import annotations

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
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    alpha_vantage_key: str | None = Field(default=None, alias="ALPHA_VANTAGE_KEY")
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_ttl_hours: int = Field(default=24, alias="CACHE_TTL_HOURS")
    cache_dir: Path | None = Field(default=None, alias="CACHE_DIR")


def get_settings() -> Settings:
    """Return singleton settings instance.

    Returns:
        Settings: Settings populated from environment.
    """

    return Settings()  # type: ignore[call-arg]



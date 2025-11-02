from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment and optional .env file.

    Attributes:
        alpha_vantage_key: API key for Alpha Vantage (optional).
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    alpha_vantage_key: str | None = Field(default=None, alias="ALPHA_VANTAGE_KEY")


def get_settings() -> Settings:
    """Return singleton settings instance.

    Returns:
        Settings: Settings populated from environment.
    """

    return Settings()  # type: ignore[call-arg]



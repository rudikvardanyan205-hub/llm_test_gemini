"""Settings loaded from environment variables and an optional local .env file."""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MissingAPIKeyError(RuntimeError):
    """Raised before any request when the API key is absent."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str | None = None
    default_model: str = "gemini-2.5-flash"
    request_timeout: float = Field(default=30.0, gt=0)
    max_concurrent: int = Field(default=5, ge=1)

    def require_api_key(self) -> str:
        value = self.gemini_api_key.strip() if self.gemini_api_key else ""
        if not value or value == "replace-with-your-api-key":
            raise MissingAPIKeyError(
                "GEMINI_API_KEY is missing. Copy .env.example to .env and add a valid API key."
            )
        return value

"""Application configuration and shared constants."""

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from constants import DEFAULT_MAX_INPUT_LENGTH

_APP_DIR = Path(__file__).resolve().parent


class Intent(StrEnum):
    """Supported user message intents."""

    SAVE_MEMORY = "SAVE_MEMORY"
    QUERY_MEMORY = "QUERY_MEMORY"
    UPDATE_MEMORY = "UPDATE_MEMORY"
    DELETE_MEMORY = "DELETE_MEMORY"
    SHOPPING_ADD = "SHOPPING_ADD"
    SHOPPING_REMOVE = "SHOPPING_REMOVE"
    SHOPPING_SHOW = "SHOPPING_SHOW"
    GENERAL_CHAT = "GENERAL_CHAT"
    UNKNOWN = "UNKNOWN"


class MemoryType(StrEnum):
    """Categories of stored memories."""

    OBJECT_LOCATION = "object_location"
    CONTACT = "contact"
    PREFERENCE = "preference"
    FACT = "fact"
    LIST = "list"
    MISC = "misc"


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=(_APP_DIR / ".env", _APP_DIR.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    openai_api_key: str = Field(
        default="test-key",
        validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key"),
    )
    openai_model: str = Field(
        default="gpt-4o",
        validation_alias=AliasChoices("OPENAI_MODEL", "openai_model"),
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices(
            "EMBEDDING_MODEL",
            "OPENAI_EMBEDDING_MODEL",
            "embedding_model",
        ),
    )
    database_path: str = Field(
        default="hippo.db",
        validation_alias=AliasChoices("DATABASE_PATH", "SQLITE_DB_PATH", "database_path"),
    )
    user_id: str = "user_1"
    llm_timeout_seconds: float = Field(
        default=30.0,
        validation_alias=AliasChoices("LLM_TIMEOUT_SECONDS", "TIMEOUT_SECONDS"),
    )
    max_retrieval_results: int = 5
    semantic_search_top_k: int = 10
    rerank_confidence_threshold: float = 0.4
    max_input_length: int = DEFAULT_MAX_INPUT_LENGTH
    log_level: str = Field(
        default="WARNING",
        validation_alias=AliasChoices("LOG_LEVEL", "log_level"),
    )
    structured_logging: bool = Field(
        default=False,
        validation_alias=AliasChoices("STRUCTURED_LOGGING", "structured_logging"),
    )
    analytics_admin_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANALYTICS_ADMIN_KEY", "analytics_admin_key"),
    )
    cors_origins: str = Field(
        default="*",
        validation_alias=AliasChoices("HIPPO_CORS_ORIGINS", "CORS_ORIGINS", "cors_origins"),
    )
    hippo_env: str = Field(
        default="development",
        validation_alias=AliasChoices("HIPPO_ENV", "hippo_env"),
    )
    session_secret: str | None = Field(
        default=None,
        validation_alias=AliasChoices("HIPPO_SESSION_SECRET", "session_secret"),
    )
    chat_rate_limit_per_session: int = Field(
        default=30,
        validation_alias=AliasChoices(
            "CHAT_RATE_LIMIT_PER_SESSION",
            "chat_rate_limit_per_session",
        ),
    )

    @model_validator(mode="after")
    def validate_production_settings(self) -> Self:
        """Fail fast when production is misconfigured."""
        if self.hippo_env.lower() != "production":
            return self

        if not self.openai_api_key or self.openai_api_key == "test-key":
            raise ValueError("OPENAI_API_KEY must be set in production.")

        if self.cors_origins.strip() == "*":
            raise ValueError(
                "HIPPO_CORS_ORIGINS must list explicit origins in production."
            )

        if not self.analytics_admin_key:
            raise ValueError("ANALYTICS_ADMIN_KEY must be set in production.")

        if not self.session_secret:
            raise ValueError("HIPPO_SESSION_SECRET must be set in production.")

        return self


def get_settings() -> Settings:
    """Load and return application settings."""
    return Settings()


def get_database_path(settings: Settings | None = None) -> Path:
    """Resolve the SQLite database path relative to the app directory."""
    resolved_settings = settings or get_settings()
    path = Path(resolved_settings.database_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path

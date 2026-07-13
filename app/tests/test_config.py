"""Tests for settings validation."""

from __future__ import annotations

import pytest

from config import Settings, WhatsAppProvider


def test_whatsapp_requires_webhook_secret_when_enabled(
    temp_db_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WhatsApp credentials should require a webhook secret."""
    monkeypatch.delenv("WHATSAPP_WEBHOOK_SECRET", raising=False)
    with pytest.raises(ValueError, match="WHATSAPP_WEBHOOK_SECRET"):
        Settings(
            openai_api_key="test-key",
            database_path=str(temp_db_path),
            session_secret="secret",
            whatsapp_access_token="token",
            whatsapp_phone_number_id="123",
        )


def test_auth_disabled_only_allowed_in_development(temp_db_path) -> None:
    """Auth bypass must not be enabled outside development."""
    with pytest.raises(ValueError, match="HIPPO_AUTH_DISABLED"):
        Settings(
            openai_api_key="real-key",
            database_path=str(temp_db_path),
            hippo_env="production",
            session_secret="secret",
            analytics_admin_key="admin",
            cors_origins="https://example.com",
            auth_disabled=True,
        )


def test_session_secret_required_outside_auth_bypass(temp_db_path) -> None:
    """Non-production environments still require a session secret by default."""
    with pytest.raises(ValueError, match="HIPPO_SESSION_SECRET"):
        Settings(
            openai_api_key="test-key",
            database_path=str(temp_db_path),
            hippo_env="development",
            session_secret=None,
            auth_disabled=False,
        )


def test_production_requires_anthropic_key(temp_db_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Production should require an Anthropic API key for the agent loop."""
    monkeypatch.setenv("HIPPO_ENV", "production")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        Settings(
            openai_api_key="real-key",
            database_path=str(temp_db_path),
            hippo_env="production",
            session_secret="secret",
            analytics_admin_key="admin",
            cors_origins="https://example.com",
        )


def test_twilio_requires_account_sid(temp_db_path) -> None:
    """Twilio WhatsApp should require an account SID."""
    with pytest.raises(ValueError, match="TWILIO_ACCOUNT_SID"):
        Settings(
            openai_api_key="test-key",
            database_path=str(temp_db_path),
            session_secret="secret",
            whatsapp_provider=WhatsAppProvider.TWILIO,
            whatsapp_webhook_secret="secret",
            whatsapp_access_token="token",
            whatsapp_phone_number_id="123",
        )

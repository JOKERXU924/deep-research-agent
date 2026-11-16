"""Load only the explicitly selected .env file; never search parent projects."""

import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from deep_research.errors import ConfigurationError


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: SecretStr
    model: str = Field(default="deepseek-flash", min_length=1, max_length=100)
    base_url: str = "https://api.deepseek.com"
    timeout_seconds: float = Field(default=60, gt=0, le=300, allow_inf_nan=False)

    @field_validator("api_key")
    @classmethod
    def validate_key(cls, value: SecretStr) -> SecretStr:
        key = value.get_secret_value().strip()
        if not key or any(not 33 <= ord(character) <= 126 for character in key):
            raise ValueError("a nonempty printable ASCII API key is required")
        return SecretStr(key)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        value = value.strip()
        if not value or not all(c.isascii() and (c.isalnum() or c in "-_.") for c in value):
            raise ValueError("invalid model identifier")
        return value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        url = urlsplit(value)
        if (
            url.scheme != "https"
            or url.hostname != "api.deepseek.com"
            or url.port not in (None, 443)
            or url.username is not None
            or url.password is not None
            or url.path not in ("", "/v1")
            or url.query
            or url.fragment
        ):
            raise ValueError("M0 only supports the official DeepSeek HTTPS endpoint")
        return value


def load_settings(env_file: Path = Path(".env")) -> Settings:
    try:
        values = dict(dotenv_values(env_file, interpolate=False)) if env_file.is_file() else {}
    except (OSError, UnicodeError):
        raise ConfigurationError("configuration", "无法读取指定的 .env 文件。") from None

    def get(name: str, default: str = "") -> str:
        return os.environ.get(name, values.get(name) or default)

    try:
        return Settings(
            api_key=SecretStr(get("DEEPSEEK_API_KEY")),
            model=get("DEEPSEEK_MODEL", "deepseek-flash"),
            base_url=get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout_seconds=get("MODEL_TIMEOUT_SECONDS", "60"),
        )
    except ValidationError:
        # Pydantic errors may include rejected input; don't print them for secrets.
        raise ConfigurationError(
            "configuration",
            "模型配置无效：请检查 DEEPSEEK_API_KEY、模型名、官方地址及超时设置。",
        ) from None

"""Immutable, server-owned model choices; only the selected profile reads a key.

The TOML file is trusted local configuration, not an HTTP request payload.  Even so,
endpoints are tightly bounded so a spelling mistake cannot redirect a cloud key.
Public snapshots identify configuration without exporting its endpoint or key name.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tomllib
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from deep_research.errors import ConfigurationError
from deep_research.model_budget import ModelBudgetLimits

ProviderName = Literal["deepseek", "openai", "mimo", "vllm"]
MAX_PROFILE_BYTES = 64 * 1024
MAX_PROFILES = 32
PROFILE_ID_PATTERN = r"[a-z][a-z0-9_-]{0,47}"
MODEL_NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}"
ENV_NAME_PATTERN = r"[A-Z][A-Z0-9_]{0,63}"


def _configuration_error() -> ConfigurationError:
    # TOML/Pydantic errors may contain raw input; never chain those to public errors.
    return ConfigurationError(
        "configuration",
        "模型配置档案无效：请检查版本、唯一标识、官方地址、预算和密钥环境变量设置。",
    )


def _endpoint(provider: str, value: str) -> str:
    if not value or len(value) > 300 or any(ord(c) <= 32 or ord(c) >= 127 for c in value):
        raise ValueError("invalid model endpoint")
    if "?" in value or "#" in value or "\\" in value:
        raise ValueError("endpoint must not contain query, fragment or backslash")
    try:
        url = urlsplit(value)
        port = url.port
        if url.username is not None or url.password is not None:
            raise ValueError("endpoint must not contain credentials")
        path = url.path.rstrip("/")
        if provider == "vllm":
            if (
                url.scheme != "http"
                or url.hostname not in {"127.0.0.1", "::1"}
                or port is None
                or not 1 <= port <= 65535
                or path != "/v1"
            ):
                raise ValueError("vLLM requires a numeric loopback endpoint with port and /v1")
            host = "[::1]" if url.hostname == "::1" else "127.0.0.1"
            return f"http://{host}:{port}/v1"
        hosts = {
            "deepseek": ("api.deepseek.com", {"", "/v1"}),
            "openai": ("api.openai.com", {"/v1"}),
            "mimo": ("api.xiaomimimo.com", {"/v1"}),
        }
        host, paths = hosts.get(provider, ("", set()))
        if (
            not host
            or url.scheme != "https"
            or url.hostname != host
            or port not in {None, 443}
            or path not in paths
        ):
            raise ValueError("cloud providers require their official HTTPS endpoint")
        return f"https://{host}{path}"
    except ValueError:
        raise ValueError("unsupported model endpoint") from None


class _FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)


class ProviderSettings(_FrozenConfig):
    """Private adapter settings. Never include this model in HTTP/audit payloads."""

    provider: ProviderName
    model: str = Field(min_length=1, max_length=200)
    base_url: str = Field(min_length=1, max_length=300, repr=False)
    api_key: SecretStr | None = Field(default=None, repr=False)
    timeout_seconds: float = Field(default=60.0, gt=0, le=300, allow_inf_nan=False)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not re.fullmatch(MODEL_NAME_PATTERN, value):
            raise ValueError("invalid model identifier")
        return value

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str, info: ValidationInfo) -> str:
        return _endpoint(info.data.get("provider", ""), value)

    @field_validator("api_key")
    @classmethod
    def validate_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        key = value.get_secret_value().strip()
        if not 1 <= len(key) <= 1024 or any(not 33 <= ord(c) <= 126 for c in key):
            raise ValueError("a nonempty printable ASCII API key is required")
        return SecretStr(key)

    @model_validator(mode="after")
    def require_cloud_key(self) -> Self:
        if self.provider != "vllm" and self.api_key is None:
            raise ValueError("cloud profiles require an API key")
        return self


class ModelProfile(_FrozenConfig):
    id: str = Field(min_length=1, max_length=48)
    label: str = Field(min_length=1, max_length=80)
    provider: ProviderName
    model: str = Field(min_length=1, max_length=200)
    base_url: str = Field(min_length=1, max_length=300, repr=False)
    api_key_env: str | None = Field(default=None, repr=False)
    timeout_seconds: float = Field(default=60.0, gt=0, le=300, allow_inf_nan=False)
    budget: ModelBudgetLimits = Field(default_factory=ModelBudgetLimits)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not re.fullmatch(PROFILE_ID_PATTERN, value):
            raise ValueError("invalid profile identifier")
        return value

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError("invalid profile label")
        try:
            value.encode("utf-8")
        except UnicodeError:
            raise ValueError("invalid profile label encoding") from None
        return value

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        return ProviderSettings.validate_model(value)

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str, info: ValidationInfo) -> str:
        return _endpoint(info.data.get("provider", ""), value)

    @field_validator("api_key_env")
    @classmethod
    def validate_env_name(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(ENV_NAME_PATTERN, value):
            raise ValueError("invalid API key environment variable name")
        return value

    @model_validator(mode="after")
    def require_cloud_key_reference(self) -> Self:
        if self.provider != "vllm" and self.api_key_env is None:
            raise ValueError("cloud profiles require an API key environment variable")
        return self

    @property
    def fingerprint(self) -> str:
        configuration = {
            "version": 1,
            "id": self.id,
            "label": self.label,
            "provider": self.provider,
            "model": self.model,
            "endpoint_sha256": hashlib.sha256(self.base_url.encode("utf-8")).hexdigest(),
            "timeout_seconds": self.timeout_seconds,
            "budget": self.budget.model_dump(mode="json"),
        }
        canonical = json.dumps(
            configuration, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def public_snapshot(self) -> dict:
        """Return a detached public identity; changing it cannot mutate the profile."""
        return {
            "id": self.id,
            "label": self.label,
            "provider": self.provider,
            "model": self.model,
            "fingerprint": self.fingerprint,
        }

    def load_settings(self, env_file: Path = Path(".env")) -> ProviderSettings:
        """Read only this profile's key, with environment overriding explicit .env.

        A keyless loopback profile does not read any .env file. No model/URL override
        is taken from the environment: the selected immutable profile owns those.
        """
        key = None
        try:
            if self.api_key_env is not None:
                if self.api_key_env in os.environ:
                    raw_key = os.environ[self.api_key_env]
                elif env_file.is_file():
                    with env_file.open("rb") as handle:
                        content = handle.read(MAX_PROFILE_BYTES + 1)
                    if len(content) > MAX_PROFILE_BYTES:
                        raise ValueError("environment file exceeds limit")
                    values = dotenv_values(
                        stream=io.StringIO(content.decode("utf-8-sig")), interpolate=False
                    )
                    raw_key = values.get(self.api_key_env)
                else:
                    raw_key = None
                if raw_key is None:
                    raise ValueError("selected profile key is missing")
                key = SecretStr(raw_key)
            return ProviderSettings(
                provider=self.provider,
                model=self.model,
                base_url=self.base_url,
                api_key=key,
                timeout_seconds=self.timeout_seconds,
            )
        except (OSError, UnicodeError, ValueError):
            raise _configuration_error() from None


class ProfileRegistry(_FrozenConfig):
    version: Literal[1]
    default_profile: str
    profiles: tuple[ModelProfile, ...] = Field(min_length=1, max_length=MAX_PROFILES)

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("unsupported profile format version")
        return value

    @model_validator(mode="after")
    def validate_registry(self) -> Self:
        ids = [profile.id for profile in self.profiles]
        if len(ids) != len(set(ids)) or self.default_profile not in ids:
            raise ValueError("duplicate profile identifiers or missing default")
        return self

    def get(self, profile_id: str | None = None) -> ModelProfile:
        selected = self.default_profile if profile_id is None else profile_id
        for profile in self.profiles:
            if profile.id == selected:
                return profile
        raise ConfigurationError("configuration", "所选模型配置档案不存在。")

    def public_catalog(self) -> list[dict]:
        return [
            {**profile.public_snapshot(), "budget": profile.budget.model_dump(mode="json")}
            for profile in self.profiles
        ]


def load_profiles(path: Path) -> ProfileRegistry:
    """Load once at startup. Editing a file cannot switch an already running job."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_PROFILE_BYTES + 1)
        if len(raw) > MAX_PROFILE_BYTES:
            raise ValueError("profile file exceeds limit")
        document = tomllib.loads(raw.decode("utf-8"))
        # TOML arrays are lists, but the public registry keeps an immutable tuple.
        profiles = document.get("profiles")
        if not isinstance(profiles, list) or not 1 <= len(profiles) <= MAX_PROFILES:
            raise ValueError("invalid profile array")
        document["profiles"] = tuple(ModelProfile.model_validate(item) for item in profiles)
        return ProfileRegistry.model_validate(document)
    except (OSError, UnicodeError, ValueError, TypeError):
        raise _configuration_error() from None

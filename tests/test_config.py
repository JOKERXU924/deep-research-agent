import pytest
from pydantic import SecretStr, ValidationError

from deep_research.config import Settings, load_settings
from deep_research.errors import ConfigurationError


@pytest.fixture(autouse=True)
def clean_model_environment(monkeypatch):
    for name in (
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_BASE_URL",
        "MODEL_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_missing_key_fails_locally_without_echoing_inputs(tmp_path):
    with pytest.raises(ConfigurationError) as caught:
        load_settings(tmp_path / ".env")
    assert caught.value.code == "configuration"


def test_explicit_env_and_environment_precedence(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=file-test-key\nDEEPSEEK_MODEL=demo-model\nMODEL_TIMEOUT_SECONDS=15\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-test-key")
    settings = load_settings(env_file)
    assert settings.api_key.get_secret_value() == "environment-test-key"
    assert settings.model == "demo-model"
    assert settings.timeout_seconds == 15
    assert "environment-test-key" not in repr(settings)


def test_never_searches_for_parent_env(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=parent-test-key", encoding="utf-8")
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    with pytest.raises(ConfigurationError):
        load_settings()


@pytest.mark.parametrize(
    "url",
    [
        "http://api.deepseek.com",
        "https://example.com",
        "https://api.deepseek.com.evil.test",
        "https://api.deepseek.com:444",
        "https://user:password@api.deepseek.com",
        "https://api.deepseek.com?key=secret-marker",
        "https://api.deepseek.com/#frag",
        "https://api.deepseek.com/other",
    ],
)
def test_credentials_only_go_to_the_configured_official_endpoint(url):
    with pytest.raises(ValidationError):
        Settings(api_key=SecretStr("test-key"), base_url=url)


@pytest.mark.parametrize("url", ["https://api.deepseek.com/", "https://api.deepseek.com/v1/"])
def test_supported_endpoint_prefixes(url):
    settings = Settings(api_key=SecretStr("test-key"), base_url=url)
    assert settings.base_url == url.rstrip("/")


@pytest.mark.parametrize("timeout", ["bad", "0", "-1", "301", "nan", "inf"])
def test_invalid_settings_errors_do_not_expose_key(timeout, monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret-marker")
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", timeout)
    with pytest.raises(ConfigurationError) as caught:
        load_settings(tmp_path / ".env")
    assert "test-secret-marker" not in str(caught.value)


@pytest.mark.parametrize("key", ["", "  ", "demo-é", "demo\nkey", "demo\x7fkey"])
def test_invalid_header_characters_fail_during_configuration(key):
    with pytest.raises(ValidationError):
        Settings(api_key=SecretStr(key))

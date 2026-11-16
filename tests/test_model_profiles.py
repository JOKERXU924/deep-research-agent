import json
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from deep_research.errors import ConfigurationError
from deep_research.model_budget import ModelBudgetLimits
from deep_research.model_profiles import (
    MAX_PROFILE_BYTES,
    ModelProfile,
    ProfileRegistry,
    ProviderSettings,
    load_profiles,
)

BASE = {
    "id": "deepseek-main",
    "label": "DeepSeek 官方",
    "provider": "deepseek",
    "model": "deepseek-flash",
    "base_url": "https://api.deepseek.com",
    "api_key_env": "DEEPSEEK_API_KEY",
}
TOML = """version = 1
default_profile = "deepseek-main"
[[profiles]]
id = "deepseek-main"
label = "DeepSeek 官方"
provider = "deepseek"
model = "deepseek-flash"
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
"""


@pytest.fixture(autouse=True)
def clean_keys(monkeypatch):
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "MIMO_API_KEY", "VLLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def write_config(tmp_path, content=TOML):
    path = tmp_path / "profiles.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_example_catalog_loads_without_any_key():
    path = Path(__file__).resolve().parents[1] / "model-profiles.example.toml"
    registry = load_profiles(path)
    assert registry.default_profile == "deepseek-main"
    assert [entry["provider"] for entry in registry.public_catalog()] == [
        "deepseek",
        "openai",
        "mimo",
        "vllm",
    ]
    assert len(registry.profiles) == 4


def test_registry_is_immutable_and_public_values_are_detached(tmp_path):
    registry = load_profiles(write_config(tmp_path))
    assert isinstance(registry.profiles, tuple)
    profile = registry.get()
    assert profile == registry.get("deepseek-main")
    assert profile.budget == ModelBudgetLimits()
    for instance, attr, value in (
        (registry, "default_profile", "other"),
        (profile, "model", "other"),
        (profile.budget, "max_calls", 1),
    ):
        with pytest.raises(ValidationError):
            setattr(instance, attr, value)
    catalog = registry.public_catalog()
    catalog[0]["budget"]["max_calls"] = 1
    catalog[0]["id"] = "other"
    assert profile.budget.max_calls == 9
    assert profile.id == "deepseek-main"


def test_snapshot_and_catalog_exclude_all_private_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "private-key-marker")
    registry = load_profiles(write_config(tmp_path))
    snapshot = registry.get().public_snapshot()
    assert set(snapshot) == {"id", "label", "provider", "model", "fingerprint"}
    assert len(snapshot["fingerprint"]) == 64
    assert all(character in "0123456789abcdef" for character in snapshot["fingerprint"])
    serialized = json.dumps(registry.public_catalog())
    for private in (
        "api.deepseek.com",
        "base_url",
        "api_key_env",
        "DEEPSEEK_API_KEY",
        "private-key-marker",
    ):
        assert private not in serialized


def test_catalog_load_does_not_read_env_file_or_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "bad\nkey")
    (tmp_path / ".env").write_bytes(b"\xff\xfe")
    registry = load_profiles(write_config(tmp_path))
    assert registry.get().id == "deepseek-main"


@pytest.mark.parametrize(
    "content",
    [
        TOML.replace("version = 1", "version = 2"),
        TOML.replace("version = 1", "version = true"),
        TOML.replace("version = 1", 'version = "1"'),
        TOML.replace("version = 1", "version = 1\nextra = 1"),
        TOML.replace('default_profile = "deepseek-main"', 'default_profile = "missing"'),
        TOML.replace('api_key_env = "DEEPSEEK_API_KEY"', 'api_key = "private-key-marker"'),
        TOML.replace('api_key_env = "DEEPSEEK_API_KEY"', 'api_key_env = "lowercase"'),
        TOML.replace('api_key_env = "DEEPSEEK_API_KEY"', ""),
        TOML.replace('id = "deepseek-main"', 'id = "../other"'),
        TOML.replace('id = "deepseek-main"', 'id = "UPPERCASE"'),
        TOML.replace('model = "deepseek-flash"', 'model = "bad model"'),
        TOML.replace('label = "DeepSeek 官方"', 'label = "  "'),
        TOML + "timeout_seconds = true\n",
        TOML + "timeout_seconds = 0.0\n",
        TOML + "timeout_seconds = inf\n",
        TOML + "[profiles.budget]\nmax_calls = 0\n",
        TOML + "[profiles.budget]\nmax_calls = 10\n",
        TOML + "[profiles.budget]\nmax_calls = true\n",
        TOML + "[profiles.budget]\nmax_output_tokens = 36865\n",
        TOML + "[profiles.budget]\nmax_input_bytes = 589825\n",
        TOML + "[profiles.budget]\nextra = 1\n",
        TOML + TOML[TOML.index("[[profiles]]") :],
        "version = 1\ndefault_profile = 'main'\nprofiles = []\n",
        "version = 1\ndefault_profile = 'main'\nprofiles = {}\n",
        "invalid TOML private-key-marker",
    ],
)
def test_invalid_registry_fails_without_echoing_content(tmp_path, content):
    with pytest.raises(ConfigurationError) as caught:
        load_profiles(write_config(tmp_path, content))
    assert caught.value.code == "configuration"
    assert "private-key-marker" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_missing_version_is_rejected(tmp_path):
    with pytest.raises(ConfigurationError):
        load_profiles(write_config(tmp_path, TOML.replace("version = 1\n", "")))


def test_invalid_encoding_missing_and_oversized_file(tmp_path):
    path = tmp_path / "profiles.toml"
    with pytest.raises(ConfigurationError):
        load_profiles(path)
    path.write_bytes(b"\xff")
    with pytest.raises(ConfigurationError):
        load_profiles(path)
    path.write_bytes(TOML.encode("utf-8") + b" " * MAX_PROFILE_BYTES)
    with pytest.raises(ConfigurationError):
        load_profiles(path)


def test_profile_count_is_bounded(tmp_path):
    header = 'version = 1\ndefault_profile = "p0"\n'
    block = TOML[TOML.index("[[profiles]]") :]
    document = header + "\n".join(block.replace("deepseek-main", f"p{n}") for n in range(33))
    with pytest.raises(ConfigurationError):
        load_profiles(write_config(tmp_path, document))


@pytest.mark.parametrize(
    "provider,url",
    [
        ("deepseek", "http://api.deepseek.com"),
        ("deepseek", "https://api.deepseek.com.evil.test"),
        ("deepseek", "https://user:private-key-marker@api.deepseek.com"),
        ("deepseek", "https://api.deepseek.com?key=private-key-marker"),
        ("deepseek", "https://api.deepseek.com?"),
        ("deepseek", "https://api.deepseek.com#"),
        ("deepseek", "https://api.deepseek.com:444/v1"),
        ("deepseek", "https://api.deepseek.com:bad/v1"),
        ("deepseek", "https://api.deepseek.com/v1/chat/completions"),
        ("deepseek", " https://api.deepseek.com"),
        ("deepseek", "https://api.deepseek.com\n"),
        ("deepseek", "https://api.deepseek.com\\evil.test"),
        ("openai", "https://api.deepseek.com/v1"),
        ("openai", "https://api.openai.com"),
        ("openai", "https://api.openai.com/v1/../v1"),
        ("mimo", "https://api.xiaomimimo.com"),
        ("mimo", "http://api.xiaomimimo.com/v1"),
        ("vllm", "http://localhost:8000/v1"),
        ("vllm", "http://127.0.0.1/v1"),
        ("vllm", "http://127.0.0.1:8000"),
        ("vllm", "http://127.0.0.2:8000/v1"),
        ("vllm", "http://127.1:8000/v1"),
        ("vllm", "http://2130706433:8000/v1"),
        ("vllm", "http://192.168.1.2:8000/v1"),
        ("vllm", "http://[::ffff:127.0.0.1]:8000/v1"),
        ("vllm", "https://127.0.0.1:8000/v1"),
        ("vllm", "http://127.0.0.1:0/v1"),
        ("vllm", "http://127.0.0.1:65536/v1"),
        ("vllm", "file:///v1"),
        ("vllm", "http://user:pass@127.0.0.1:8000/v1"),
    ],
)
def test_strict_provider_endpoint_allowlists(provider, url):
    with pytest.raises(ValidationError) as caught:
        ModelProfile(**{**BASE, "provider": provider, "base_url": url})
    assert "private-key-marker" not in str(caught.value)


@pytest.mark.parametrize(
    "provider,url,canonical",
    [
        ("deepseek", "https://api.deepseek.com/", "https://api.deepseek.com"),
        ("deepseek", "https://api.deepseek.com/v1/", "https://api.deepseek.com/v1"),
        ("deepseek", "https://API.DEEPSEEK.COM:443", "https://api.deepseek.com"),
        ("openai", "https://api.openai.com/v1/", "https://api.openai.com/v1"),
        ("mimo", "https://api.xiaomimimo.com/v1", "https://api.xiaomimimo.com/v1"),
        ("vllm", "http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1"),
        ("vllm", "http://[::1]:8000/v1/", "http://[::1]:8000/v1"),
    ],
)
def test_endpoint_normalization(provider, url, canonical):
    profile = ModelProfile(**{**BASE, "provider": provider, "base_url": url})
    assert profile.base_url == canonical


def test_key_loading_is_lazy_explicit_and_environment_wins(tmp_path, monkeypatch):
    profile = ModelProfile(**BASE)
    env_file = tmp_path / "chosen.env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=file-test-key\nDEEPSEEK_MODEL=ignored\n", encoding="utf-8"
    )
    assert profile.load_settings(env_file).api_key.get_secret_value() == "file-test-key"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-test-key")
    settings = profile.load_settings(env_file)
    assert settings.api_key.get_secret_value() == "environment-test-key"
    assert settings.model == "deepseek-flash"
    assert "environment-test-key" not in repr(settings)
    assert "environment-test-key" not in settings.model_dump_json()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    with pytest.raises(ConfigurationError):
        profile.load_settings(env_file)


def test_env_does_not_interpolate_or_search_parent(tmp_path, monkeypatch):
    profile = ModelProfile(**BASE)
    (tmp_path / ".env").write_text('DEEPSEEK_API_KEY="${OTHER_SECRET}"\n', encoding="utf-8")
    monkeypatch.setenv("OTHER_SECRET", "must-not-expand")
    assert profile.load_settings(tmp_path / ".env").api_key.get_secret_value() == "${OTHER_SECRET}"
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    with pytest.raises(ConfigurationError):
        profile.load_settings()


def test_only_selected_profile_needs_its_key(tmp_path, monkeypatch):
    registry = load_profiles(Path(__file__).resolve().parents[1] / "model-profiles.example.toml")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    assert registry.get().load_settings(tmp_path / ".env").provider == "deepseek"
    with pytest.raises(ConfigurationError):
        registry.get("openai-main").load_settings(tmp_path / ".env")


def test_keyless_loopback_does_not_read_environment_file(tmp_path):
    profile = ModelProfile(
        **{**BASE, "provider": "vllm", "base_url": "http://127.0.0.1:8000/v1", "api_key_env": None}
    )
    path = tmp_path / ".env"
    path.write_bytes(b"\xff")
    assert profile.load_settings(path).api_key is None
    authenticated = ModelProfile(**{**profile.model_dump(), "api_key_env": "VLLM_API_KEY"})
    with pytest.raises(ConfigurationError):
        authenticated.load_settings(tmp_path / "missing.env")


@pytest.mark.parametrize(
    "key", ["", " ", "key\nvalue", "private-key-marker\x7f", "中文", "x" * 1025]
)
def test_bad_keys_do_not_leak_to_error(tmp_path, monkeypatch, key):
    monkeypatch.setenv("DEEPSEEK_API_KEY", key)
    with pytest.raises(ConfigurationError) as caught:
        ModelProfile(**BASE).load_settings(tmp_path / ".env")
    assert "private-key-marker" not in str(caught.value)


def test_explicit_env_size_and_encoding_are_bounded(tmp_path):
    path = tmp_path / ".env"
    profile = ModelProfile(**BASE)
    for content in (b"\xff", b"DEEPSEEK_API_KEY=" + b"a" * MAX_PROFILE_BYTES):
        path.write_bytes(content)
        with pytest.raises(ConfigurationError):
            profile.load_settings(path)


def test_settings_cannot_send_cloud_request_without_key():
    with pytest.raises(ValidationError):
        ProviderSettings(
            provider="deepseek", model="deepseek-flash", base_url="https://api.deepseek.com"
        )
    settings = ProviderSettings(
        provider="deepseek",
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key=SecretStr("test-key"),
    )
    with pytest.raises(ValidationError):
        settings.model = "other"


def test_fingerprint_is_stable_sensitive_to_config_but_not_key_rotation(tmp_path, monkeypatch):
    profile = ModelProfile(**BASE)
    assert profile.fingerprint == ModelProfile(**BASE).fingerprint
    assert (
        profile.fingerprint
        == ModelProfile(**{**BASE, "base_url": "https://api.deepseek.com:443/"}).fingerprint
    )
    assert (
        profile.fingerprint
        == ModelProfile(**{**BASE, "api_key_env": "ANOTHER_API_KEY"}).fingerprint
    )
    for change in (
        {"model": "another-model"},
        {"base_url": "https://api.deepseek.com/v1"},
        {"timeout_seconds": 30.0},
        {"budget": ModelBudgetLimits(max_calls=1)},
    ):
        assert profile.fingerprint != ModelProfile(**{**BASE, **change}).fingerprint
    monkeypatch.setenv("DEEPSEEK_API_KEY", "first-key")
    first = profile.load_settings(tmp_path / ".env")
    fingerprint = profile.fingerprint
    monkeypatch.setenv("DEEPSEEK_API_KEY", "rotated-key")
    second = profile.load_settings(tmp_path / ".env")
    assert first.api_key != second.api_key
    assert fingerprint == profile.fingerprint


def test_unknown_profile_has_stable_safe_error(tmp_path):
    registry = load_profiles(write_config(tmp_path))
    with pytest.raises(ConfigurationError) as caught:
        registry.get("private-key-marker")
    assert "private-key-marker" not in str(caught.value)


def test_loaded_registry_is_not_changed_by_editing_its_file(tmp_path):
    path = write_config(tmp_path)
    first = load_profiles(path)
    path.write_text(TOML.replace("deepseek-flash", "another-model"), encoding="utf-8")
    second = load_profiles(path)
    assert first.get().model == "deepseek-flash"
    assert first.get().fingerprint != second.get().fingerprint


def test_registry_constructor_rejects_lists_and_duplicate_identifiers():
    profile = ModelProfile(**BASE)
    with pytest.raises(ValidationError):
        ProfileRegistry(version=1, default_profile=profile.id, profiles=[profile])
    with pytest.raises(ValidationError):
        ProfileRegistry(version=1, default_profile=profile.id, profiles=(profile, profile))


def test_label_must_support_utf8_snapshot_serialization():
    with pytest.raises(ValidationError):
        ModelProfile(**{**BASE, "label": "broken\ud800label"})


def test_explicit_budget_fields_are_retained_in_catalog(tmp_path):
    path = write_config(
        tmp_path,
        TOML
        + "[profiles.budget]\nmax_calls = 2\nmax_input_bytes = 1000\nmax_output_tokens = 4096\n",
    )
    registry = load_profiles(path)
    assert registry.public_catalog()[0]["budget"] == {
        "max_calls": 2,
        "max_input_bytes": 1000,
        "max_output_tokens": 4096,
    }

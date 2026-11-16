"""CLI tests replace the live entry point and never load credentials or call a model."""

import json
import sys
from pathlib import Path

import pytest

from deep_research import cli
from deep_research.errors import ModelError
from deep_research.gateway import StructuredResult
from deep_research.schemas import ChatMessage, ModelRequest, ModelResponse, ResearchOutline


def test_main_writes_success_as_json_and_passes_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[Path, str]] = []
    expected = {"status": "ok", "outline": {"topic": "引用", "questions": ["为什么？"]}}

    async def fake_smoke(env_file: Path, prompt: str) -> dict[str, object]:
        calls.append((env_file, prompt))
        return expected

    monkeypatch.setattr(cli, "run_smoke", fake_smoke)
    monkeypatch.setattr(
        sys, "argv", ["research", "smoke", "--env-file", "test.env", "--prompt", "测试问题"]
    )

    assert cli.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == expected
    assert "引用" in captured.out
    assert captured.err == ""
    assert calls == [(Path("test.env"), "测试问题")]


def test_main_writes_public_error_to_stderr_and_returns_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fake_smoke(*_: object) -> dict[str, object]:
        raise ModelError("rate_limit", "模型服务限流。", status_code=429)

    monkeypatch.setattr(cli, "run_smoke", fake_smoke)
    monkeypatch.setattr(sys, "argv", ["research", "smoke"])

    assert cli.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "status": "error",
        "code": "rate_limit",
        "message": "模型服务限流。",
        "http_status": 429,
    }
    assert "Traceback" not in captured.err


def test_main_invalid_input_returns_two_without_printing_validation_payload(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fake_smoke(env_file: Path, prompt: str) -> dict[str, object]:
        ModelRequest(messages=(ChatMessage(role="user", content=prompt),))
        pytest.fail("Blank input must fail validation before this point")

    monkeypatch.setattr(cli, "run_smoke", fake_smoke)
    monkeypatch.setattr(sys, "argv", ["research", "smoke", "--prompt", "   "])

    assert cli.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "输入无效" in captured.err
    assert "Traceback" not in captured.err
    assert "input_value" not in captured.err


def test_main_keyboard_interrupt_returns_conventional_cancellation_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fake_smoke(*_: object) -> dict[str, object]:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_smoke", fake_smoke)
    monkeypatch.setattr(sys, "argv", ["research", "smoke"])

    assert cli.main() == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "已取消" in captured.err


@pytest.mark.parametrize("should_fail", [False, True])
async def test_smoke_assembles_gateway_and_closes_provider_on_success_or_failure(
    monkeypatch: pytest.MonkeyPatch, should_fail: bool
) -> None:
    configured = object()
    env_files: list[Path] = []
    events: list[str] = []
    outline = ResearchOutline(topic="引用", questions=["证据来自哪里？"])
    response = ModelResponse(
        provider="fake", model="offline", request_id="test-request", content="{}", latency_ms=1.234
    )

    def fake_settings(env_file: Path) -> object:
        env_files.append(env_file)
        return configured

    class FakeProvider:
        def __init__(self, settings: object) -> None:
            assert settings is configured

        async def __aenter__(self) -> "FakeProvider":
            events.append("opened")
            return self

        async def __aexit__(self, *_: object) -> None:
            events.append("closed")

    class FakeGateway:
        def __init__(self, provider: FakeProvider) -> None:
            assert isinstance(provider, FakeProvider)

        async def generate_structured(
            self, request: ModelRequest, schema: type[ResearchOutline]
        ) -> StructuredResult[ResearchOutline]:
            assert request.messages == (ChatMessage(role="user", content="写提纲"),)
            assert schema is ResearchOutline
            events.append("generated")
            if should_fail:
                raise ModelError("timeout", "Request timed out.")
            return StructuredResult(data=outline, response=response)

    monkeypatch.setattr(cli, "load_settings", fake_settings)
    monkeypatch.setattr(cli, "DeepSeekProvider", FakeProvider)
    monkeypatch.setattr(cli, "ModelGateway", FakeGateway)
    env_file = Path("never-read.env")

    if should_fail:
        with pytest.raises(ModelError, match="timed out"):
            await cli.run_smoke(env_file, "写提纲")
    else:
        assert await cli.run_smoke(env_file, "写提纲") == {
            "status": "ok",
            "provider": "fake",
            "model": "offline",
            "request_id": "test-request",
            "latency_ms": 1.23,
            "usage": None,
            "outline": outline.model_dump(),
        }
    assert env_files == [env_file]
    assert events == ["opened", "generated", "closed"]

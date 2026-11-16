"""Provider contracts use an in-memory transport: no network or real credentials."""

import asyncio
import json
from collections.abc import AsyncIterator
from copy import deepcopy

import httpx
import pytest
from pydantic import SecretStr

from deep_research.config import Settings
from deep_research.errors import ModelError
from deep_research.providers import deepseek
from deep_research.providers.deepseek import DeepSeekProvider
from deep_research.schemas import MAX_OUTPUT_BYTES, ChatMessage, ModelRequest


def settings(**overrides: object) -> Settings:
    return Settings(api_key=SecretStr("offline-test-key"), **overrides)


def request() -> ModelRequest:
    return ModelRequest(
        messages=(ChatMessage(role="user", content="Explain evidence tracking."),),
        max_output_tokens=128,
    )


def completion() -> dict:
    return {
        "id": "completion-test-1",
        "model": "actual-model-version",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Keep the source and quotation."},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }


class RecordingStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


async def test_success_sends_one_bounded_request_and_normalizes_response() -> None:
    captured: list[httpx.Request] = []
    body = json.dumps(completion()).encode()
    stream = RecordingStream(body[:20], body[20:])

    def handle(incoming: httpx.Request) -> httpx.Response:
        captured.append(incoming)
        return httpx.Response(200, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), trust_env=False) as client:
        async with DeepSeekProvider(settings(), client=client) as provider:
            result = await provider.complete(request())
        assert not client.is_closed  # Injected resources belong to the caller.

    assert stream.closed
    assert len(captured) == 1
    sent = captured[0]
    assert str(sent.url) == "https://api.deepseek.com/chat/completions"
    assert sent.method == "POST"
    assert sent.headers["authorization"] == "Bearer offline-test-key"
    assert json.loads(sent.content) == {
        "model": settings().model,
        "messages": [{"role": "user", "content": "Explain evidence tracking."}],
        "max_tokens": 128,
        "response_format": {"type": "text"},
        "thinking": {"type": "disabled"},
        "stream": False,
    }
    assert result.provider == "deepseek"
    assert result.model == "actual-model-version"
    assert result.request_id == "completion-test-1"
    assert result.content == "Keep the source and quotation."
    assert result.finish_reason == "stop"
    assert result.usage is not None
    assert result.usage.model_dump() == {
        "input_tokens": 10,
        "output_tokens": 8,
        "total_tokens": 18,
    }
    assert result.latency_ms >= 0


async def test_missing_usage_is_unknown_not_zero() -> None:
    body = completion()
    del body["usage"]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)), trust_env=False
    ) as client:
        result = await DeepSeekProvider(settings(), client=client).complete(request())
    assert result.usage is None


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, "authentication"),
        (403, "authentication"),
        (402, "billing"),
        (429, "rate_limit"),
        (500, "upstream"),
        (503, "upstream"),
        (400, "http_error"),
        (302, "http_error"),
    ],
)
async def test_http_failure_is_classified_sanitized_and_never_retried(
    status: int, code: str
) -> None:
    calls: list[httpx.Request] = []
    stream = RecordingStream(b"sensitive-upstream-body")

    def handle(incoming: httpx.Request) -> httpx.Response:
        calls.append(incoming)
        return httpx.Response(
            status,
            headers={"location": "https://unexpected.example/collect"},
            stream=stream,
        )

    # Even an injected client configured to follow redirects must not forward the key.
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), follow_redirects=True, trust_env=False
    ) as client:
        with pytest.raises(ModelError) as caught:
            await DeepSeekProvider(settings(), client=client).complete(request())
    assert caught.value.code == code
    assert caught.value.status_code == status
    assert "sensitive-upstream-body" not in str(caught.value)
    assert "offline-test-key" not in str(caught.value)
    assert len(calls) == 1
    assert stream.closed


def invalid_completions() -> list:
    result = [(b"not JSON", "protocol")]
    for field, value, code in [
        ("finish_reason", "length", "truncated"),
        ("finish_reason", "tool_calls", "incomplete"),
        ("finish_reason", "content_filter", "incomplete"),
        ("content", "  \n ", "empty_output"),
        ("content", None, "empty_output"),
        ("content", 42, "protocol"),
        ("tool_calls", [{"id": "call-1", "type": "function"}], "incomplete"),
    ]:
        body = completion()
        choice = body["choices"][0]
        target = choice if field == "finish_reason" else choice["message"]
        target[field] = value
        result.append((json.dumps(body).encode(), code))
    for mutation in [
        {"choices": []},
        {"choices": [deepcopy(completion()["choices"][0])] * 2},
        {"usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 99}},
        {"usage": {"prompt_tokens": -1, "completion_tokens": 8, "total_tokens": 7}},
    ]:
        result.append((json.dumps(completion() | mutation).encode(), "protocol"))
    return result


@pytest.mark.parametrize(("body", "code"), invalid_completions())
async def test_malformed_or_incomplete_completion_is_rejected(body: bytes, code: str) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body)), trust_env=False
    ) as client:
        with pytest.raises(ModelError) as caught:
            await DeepSeekProvider(settings(), client=client).complete(request())
    assert caught.value.code == code


@pytest.mark.parametrize("over_budget", [False, True])
async def test_raw_http_body_limit_includes_all_streamed_bytes(over_budget: bool) -> None:
    body = json.dumps(completion()).encode()
    padding = b" " * (deepseek.MAX_HTTP_RESPONSE_BYTES - len(body) + int(over_budget))
    stream = RecordingStream(body, padding)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream)), trust_env=False
    ) as client:
        provider = DeepSeekProvider(settings(), client=client)
        if over_budget:
            with pytest.raises(ModelError) as caught:
                await provider.complete(request())
            assert caught.value.code == "output_too_large"
        else:
            assert (await provider.complete(request())).content
    assert stream.closed


async def test_decoded_output_limit_counts_utf8_bytes() -> None:
    body = completion()
    content = "中" * (MAX_OUTPUT_BYTES // 3 + 1)
    assert len(content) < MAX_OUTPUT_BYTES < len(content.encode())
    body["choices"][0]["message"]["content"] = content
    wire = json.dumps(body, ensure_ascii=False).encode()
    assert len(wire) < deepseek.MAX_HTTP_RESPONSE_BYTES
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=wire)), trust_env=False
    ) as client:
        with pytest.raises(ModelError) as caught:
            await DeepSeekProvider(settings(), client=client).complete(request())
    assert caught.value.code == "output_too_large"


@pytest.mark.parametrize("failure", [httpx.ReadTimeout, httpx.ConnectError])
async def test_transport_failure_is_sanitized(failure: type[httpx.RequestError]) -> None:
    def handle(incoming: httpx.Request) -> httpx.Response:
        raise failure("sensitive transport details", request=incoming)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), trust_env=False) as client:
        with pytest.raises(ModelError) as caught:
            await DeepSeekProvider(settings(), client=client).complete(request())
    assert caught.value.code == ("timeout" if failure is httpx.ReadTimeout else "network")
    assert "sensitive transport details" not in str(caught.value)


async def test_wall_clock_deadline_covers_a_stalled_response_and_closes_it() -> None:
    class StalledStream(RecordingStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"{"
            await asyncio.Event().wait()

    stream = StalledStream()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream)), trust_env=False
    ) as client:
        with pytest.raises(ModelError) as caught:
            async with asyncio.timeout(2):
                await DeepSeekProvider(settings(timeout_seconds=0.02), client=client).complete(
                    request()
                )
    assert caught.value.code == "timeout"
    assert stream.closed


async def test_cancellation_propagates_and_closes_response() -> None:
    entered = asyncio.Event()

    class WaitingStream(RecordingStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            entered.set()
            await asyncio.Event().wait()
            yield b"unreachable"

    stream = WaitingStream()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream)), trust_env=False
    ) as client:
        provider = DeepSeekProvider(settings(), client=client)
        task = asyncio.create_task(provider.complete(request()))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert stream.closed
            assert not client.is_closed
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


async def test_owned_client_is_closed_once_and_closed_provider_rejects_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OwnedClient(httpx.AsyncClient):
        closes = 0

        async def aclose(self) -> None:
            self.closes += 1
            await super().aclose()

    client = OwnedClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200)), trust_env=False
    )
    monkeypatch.setattr(deepseek.httpx, "AsyncClient", lambda **_: client)
    provider = DeepSeekProvider(settings())
    await provider.aclose()
    await provider.aclose()
    assert client.is_closed
    assert client.closes == 1
    with pytest.raises(ModelError) as caught:
        await provider.complete(request())
    assert caught.value.code == "closed"

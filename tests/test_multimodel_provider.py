"""Provider dialects and failure boundaries; never use network or real keys."""

import asyncio
import json
from collections.abc import AsyncIterator
from copy import deepcopy

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from deep_research.errors import ModelError
from deep_research.model_profiles import ProviderSettings
from deep_research.providers import chat_compatible
from deep_research.providers.chat_compatible import ChatCompatibleProvider
from deep_research.schemas import MAX_OUTPUT_BYTES, ChatMessage, ModelRequest

PROVIDERS = ("deepseek", "openai", "mimo", "vllm")
URLS = {
    "deepseek": "https://api.deepseek.com",
    "openai": "https://api.openai.com/v1",
    "mimo": "https://api.xiaomimimo.com/v1",
    "vllm": "http://127.0.0.1:8000/v1",
}


def settings(provider: str = "openai", **overrides: object) -> ProviderSettings:
    return ProviderSettings.model_validate(
        {
            "provider": provider,
            "model": "example-model",
            "base_url": URLS[provider],
            "api_key": SecretStr("offline-placeholder-not-a-real-key"),
            **overrides,
        }
    )


def request(**overrides: object) -> ModelRequest:
    return ModelRequest.model_validate(
        {
            "messages": (ChatMessage(role="user", content="Return JSON with evidence."),),
            "max_output_tokens": 128,
            "response_format": "json_object",
            **overrides,
        }
    )


def completion() -> dict:
    return {
        "id": "response-test-1",
        "model": "actual-model-version",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"fact":"Evidence matters."}'},
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


@pytest.mark.parametrize("provider_name", PROVIDERS)
@pytest.mark.parametrize("format_name", ["text", "json_object"])
async def test_exact_vendor_dialect_and_response_normalization(
    provider_name: str, format_name: str
) -> None:
    calls = []
    body = json.dumps(completion()).encode()
    stream = RecordingStream(body[:17], body[17:])

    def handle(incoming: httpx.Request) -> httpx.Response:
        calls.append(incoming)
        return httpx.Response(200, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), trust_env=False) as client:
        async with ChatCompatibleProvider(settings(provider_name), client=client) as provider:
            result = await provider.complete(request(response_format=format_name))
        assert not client.is_closed
    assert stream.closed
    assert len(calls) == 1
    outgoing = calls[0]
    assert outgoing.method == "POST"
    assert str(outgoing.url) == URLS[provider_name] + "/chat/completions"
    if provider_name == "mimo":
        assert outgoing.headers["api-key"] == "offline-placeholder-not-a-real-key"
        assert "authorization" not in outgoing.headers
    else:
        assert outgoing.headers["authorization"] == "Bearer offline-placeholder-not-a-real-key"
        assert "api-key" not in outgoing.headers
    expected = {
        "model": "example-model",
        "messages": [{"role": "user", "content": "Return JSON with evidence."}],
        "response_format": {"type": format_name},
        "stream": False,
    }
    token_key = "max_completion_tokens" if provider_name in ("openai", "mimo") else "max_tokens"
    expected[token_key] = 128
    if provider_name in ("deepseek", "mimo"):
        expected["thinking"] = {"type": "disabled"}
    assert json.loads(outgoing.content) == expected
    assert result.provider == provider_name
    assert result.model == "actual-model-version"
    assert result.request_id == "response-test-1"
    assert result.usage is not None and result.usage.total_tokens == 18
    assert result.latency_ms >= 0


async def test_local_vllm_can_omit_authentication_instead_of_inventing_a_key() -> None:
    calls = []

    def handle(incoming: httpx.Request) -> httpx.Response:
        calls.append(incoming)
        return httpx.Response(200, json=completion())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), trust_env=False) as client:
        result = await ChatCompatibleProvider(
            settings("vllm", api_key=None), client=client
        ).complete(request())
    assert result.provider == "vllm"
    assert "authorization" not in calls[0].headers
    assert "api-key" not in calls[0].headers


@pytest.mark.parametrize("provider_name", PROVIDERS)
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
        (307, "http_error"),
    ],
)
async def test_http_errors_do_not_retry_redirect_fallback_or_leak(
    provider_name: str, status: int, code: str
) -> None:
    calls = []
    stream = RecordingStream(b"secret upstream body")

    def handle(incoming: httpx.Request) -> httpx.Response:
        calls.append(incoming)
        return httpx.Response(
            status, headers={"Location": "https://unexpected.example/secret"}, stream=stream
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), follow_redirects=True, trust_env=False
    ) as client:
        with pytest.raises(ModelError) as caught:
            await ChatCompatibleProvider(settings(provider_name), client=client).complete(request())
    assert caught.value.code == code
    assert caught.value.status_code == status
    assert caught.value.response_usage is None
    assert len(calls) == 1
    assert "secret" not in str(caught.value)
    assert "placeholder" not in str(caught.value)
    assert stream.closed


def malformed_completions() -> list[tuple[bytes, str, bool]]:
    result = []
    for field, value, code in [
        ("finish_reason", "length", "truncated"),
        ("finish_reason", "tool_calls", "incomplete"),
        ("finish_reason", "content_filter", "incomplete"),
        ("finish_reason", None, "protocol"),
        ("content", " \n ", "empty_output"),
        ("content", None, "empty_output"),
        ("content", 42, "protocol"),
        ("content", "\ud800", "protocol"),
        ("tool_calls", [{"id": "call-1"}], "incomplete"),
        ("function_call", {}, "incomplete"),
        ("refusal", "Cannot comply.", "refusal"),
        ("refusal", "", "refusal"),
        ("role", "user", "protocol"),
    ]:
        body = completion()
        choice = body["choices"][0]
        target = choice if field == "finish_reason" else choice["message"]
        target[field] = value
        result.append((json.dumps(body).encode(), code, True))
    for mutation in [
        {"choices": []},
        {"choices": [deepcopy(completion()["choices"][0])] * 2},
        {"model": "\ud800"},
        {"id": " "},
        {"id": "response\nwith-control"},
        {"model": "model\u007fwith-control"},
        {"id": "x" * 257},
    ]:
        result.append((json.dumps(completion() | mutation).encode(), "protocol", True))
    for usage in [
        {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 99},
        {"prompt_tokens": -1, "completion_tokens": 8, "total_tokens": 7},
        {"prompt_tokens": True, "completion_tokens": 8, "total_tokens": 9},
        {"prompt_tokens": "10", "completion_tokens": 8, "total_tokens": 18},
        {"prompt_tokens": 10.0, "completion_tokens": 8, "total_tokens": 18},
        {},
        [],
    ]:
        result.append((json.dumps(completion() | {"usage": usage}).encode(), "protocol", False))
    for raw in [
        b"not JSON",
        b"null",
        b"[]",
        b"\xff",
        b'{"choices":NaN}',
        b'{"choices":Infinity}',
        b'{"choices":-Infinity}',
        b'{"extra":1e999}',
        b'{"choices": [], "choices": []}',
        b'{"nested": {"key": 1, "key": 2}}',
        b"[" * 2000 + b"]" * 2000,
    ]:
        result.append((raw, "protocol", False))
    return result


@pytest.mark.parametrize(("body", "code", "known_usage"), malformed_completions())
async def test_reject_bad_output_but_preserve_valid_independent_usage(
    body: bytes, code: str, known_usage: bool
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body)), trust_env=False
    ) as client:
        with pytest.raises(ModelError) as caught:
            await ChatCompatibleProvider(settings(), client=client).complete(request())
    assert caught.value.code == code
    if known_usage:
        assert caught.value.response_usage is not None
        assert caught.value.response_usage.total_tokens == 18
    else:
        assert caught.value.response_usage is None


@pytest.mark.parametrize("unknown", [None, "missing"])
async def test_missing_usage_stays_unknown(unknown: str | None) -> None:
    body = completion()
    if unknown == "missing":
        del body["usage"]
    else:
        body["usage"] = None
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)), trust_env=False
    ) as client:
        result = await ChatCompatibleProvider(settings(), client=client).complete(request())
    assert result.usage is None


async def test_reasoning_trace_is_not_exported_as_answer() -> None:
    body = completion()
    body["choices"][0]["message"]["reasoning_content"] = "private reasoning metadata"
    body["usage"]["completion_tokens_details"] = {"reasoning_tokens": 5}
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)), trust_env=False
    ) as client:
        result = await ChatCompatibleProvider(settings(), client=client).complete(request())
    assert "private reasoning" not in result.model_dump_json()
    assert result.usage is not None and result.usage.output_tokens == 8


@pytest.mark.parametrize("extra", [0, 1])
async def test_http_byte_limit_and_stream_cleanup(extra: int) -> None:
    body = json.dumps(completion()).encode()
    padding = b" " * (chat_compatible.MAX_HTTP_RESPONSE_BYTES - len(body) + extra)
    stream = RecordingStream(body, padding)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream)), trust_env=False
    ) as client:
        provider = ChatCompatibleProvider(settings(), client=client)
        if extra:
            with pytest.raises(ModelError) as caught:
                await provider.complete(request())
            assert caught.value.code == "output_too_large"
            assert caught.value.response_usage is None
        else:
            assert (await provider.complete(request())).content
    assert stream.closed


@pytest.mark.parametrize("extra", [0, 1])
async def test_output_limit_counts_utf8_bytes_before_trimming(extra: int) -> None:
    body = completion()
    body["choices"][0]["message"]["content"] = "中" * (MAX_OUTPUT_BYTES // 3) + "a" * (1 + extra)
    wire = json.dumps(body, ensure_ascii=False).encode()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=wire)), trust_env=False
    ) as client:
        provider = ChatCompatibleProvider(settings(), client=client)
        if extra:
            with pytest.raises(ModelError) as caught:
                await provider.complete(request())
            assert caught.value.code == "output_too_large"
            assert caught.value.response_usage is not None
        else:
            assert len((await provider.complete(request())).content.encode()) == MAX_OUTPUT_BYTES


@pytest.mark.parametrize("failure", [httpx.ReadTimeout, httpx.ConnectError, httpx.ReadError])
async def test_transport_errors_are_sanitized_without_retry(
    failure: type[httpx.RequestError],
) -> None:
    calls = []

    def handle(incoming: httpx.Request) -> httpx.Response:
        calls.append(incoming)
        raise failure("secret transport details", request=incoming)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), trust_env=False) as client:
        with pytest.raises(ModelError) as caught:
            await ChatCompatibleProvider(settings(), client=client).complete(request())
    assert caught.value.code == ("timeout" if failure is httpx.ReadTimeout else "network")
    assert "secret" not in str(caught.value)
    assert len(calls) == 1


async def test_whole_call_deadline_stops_a_stalled_stream() -> None:
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
                await ChatCompatibleProvider(
                    settings(timeout_seconds=0.02), client=client
                ).complete(request())
    assert caught.value.code == "timeout"
    assert stream.closed


async def test_cancel_propagates_and_closes_only_owned_response() -> None:
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
        task = asyncio.create_task(
            ChatCompatibleProvider(settings(), client=client).complete(request())
        )
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


async def test_owned_client_no_ambient_proxy_and_closed_once(
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
    options = []

    def make_client(**kwargs: object) -> httpx.AsyncClient:
        options.append(kwargs)
        return client

    monkeypatch.setattr(chat_compatible.httpx, "AsyncClient", make_client)
    provider = ChatCompatibleProvider(settings())
    assert options == [{"follow_redirects": False, "trust_env": False}]
    await provider.aclose()
    await provider.aclose()
    assert client.is_closed and client.closes == 1
    with pytest.raises(ModelError) as caught:
        await provider.complete(request())
    assert caught.value.code == "closed"
    with pytest.raises(ModelError, match="已关闭"):
        await provider.__aenter__()


async def test_invalid_model_construct_request_cannot_reach_transport() -> None:
    calls = []

    def handle(incoming: httpx.Request) -> httpx.Response:
        calls.append(incoming)
        return httpx.Response(200, json=completion())

    invalid = request().model_copy(update={"max_output_tokens": 0})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), trust_env=False) as client:
        with pytest.raises(ModelError) as caught:
            await ChatCompatibleProvider(settings(), client=client).complete(invalid)
    assert caught.value.code == "invalid_request"
    assert calls == []


def test_settings_snapshot_rejects_bypassed_endpoint_validation() -> None:
    invalid = settings().model_copy(update={"base_url": "https://unexpected.example/v1"})
    with pytest.raises(ValidationError):
        ChatCompatibleProvider(invalid)

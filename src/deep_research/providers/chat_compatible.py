"""Bounded Chat Completions transport; compatibility is not protocol identity.

There are no SDK retries, redirects, implicit model switches, tool calls, or
reasoning traces in the application contract. All tests use an offline transport.
"""

import asyncio
import json
import math
from time import perf_counter
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from deep_research.errors import ModelError
from deep_research.model_profiles import ProviderSettings
from deep_research.schemas import MAX_OUTPUT_BYTES, ModelRequest, ModelResponse, TokenUsage

MAX_HTTP_RESPONSE_BYTES = 1024 * 1024


class _WireModel(BaseModel):
    # New vendor metadata is tolerated, but fields we consume remain strict.
    model_config = ConfigDict(extra="ignore", strict=True)


class _Message(_WireModel):
    role: Literal["assistant"]
    content: str | None
    tool_calls: list[dict[str, Any]] | None = None
    function_call: dict[str, Any] | None = None
    refusal: str | None = None


class _Choice(_WireModel):
    index: int = Field(ge=0, le=0)
    finish_reason: str
    message: _Message


class _Completion(_WireModel):
    id: str = Field(min_length=1, max_length=256)
    model: str = Field(min_length=1, max_length=256)
    choices: list[_Choice] = Field(min_length=1, max_length=1)


class _Usage(_WireModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(_: str) -> None:
    raise ValueError("non-finite JSON value")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _http_error(status: int) -> ModelError:
    if status in (401, 403):
        code, message = "authentication", "模型服务拒绝认证或当前账号没有权限。"
    elif status == 429:
        code, message = "rate_limit", "模型服务限流；本次未自动重试或切换模型。"
    elif status == 402:
        code, message = "billing", "模型服务提示余额不足。"
    elif status >= 500:
        code, message = "upstream", "模型服务暂时不可用。"
    else:
        code, message = "http_error", "模型服务拒绝请求，请检查模型和接口配置。"
    return ModelError(code, message, status_code=status)


def _parse_response(body: bytes, provider: str, elapsed_ms: float) -> ModelResponse:
    usage = None
    try:
        # JSON duplicates and NaN must not silently change the consumed values.
        raw = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
            parse_float=_finite_float,
        )
        if not isinstance(raw, dict):
            raise ValueError("completion must be an object")
        if raw.get("usage") is not None:
            wire_usage = _Usage.model_validate(raw["usage"])
            usage = TokenUsage(
                input_tokens=wire_usage.prompt_tokens,
                output_tokens=wire_usage.completion_tokens,
                total_tokens=wire_usage.total_tokens,
            )
        completion = _Completion.model_validate(raw)
        # Invalid Unicode must never escape as a late report/database exception.
        completion.id.encode("utf-8")
        completion.model.encode("utf-8")
        if any(
            not value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value)
            for value in (completion.id, completion.model)
        ):
            raise ValueError("invalid response metadata")
    except (ValueError, UnicodeError, RecursionError):
        raise ModelError(
            "protocol", "模型服务返回了无法识别的响应结构。", response_usage=usage
        ) from None

    choice = completion.choices[0]
    if choice.finish_reason == "length":
        raise ModelError(
            "truncated", "模型输出被截断，不能作为完整结果使用。", response_usage=usage
        )
    if (
        choice.finish_reason != "stop"
        or choice.message.tool_calls
        or choice.message.function_call is not None
    ):
        raise ModelError("incomplete", "模型未正常完成文本回复。", response_usage=usage)
    if choice.message.refusal is not None:
        raise ModelError("refusal", "模型服务拒绝生成此次回复。", response_usage=usage)
    content = choice.message.content
    if content is None or not content.strip():
        raise ModelError("empty_output", "模型返回了空文本。", response_usage=usage)
    try:
        output_bytes = len(content.encode("utf-8"))
    except UnicodeError:
        raise ModelError(
            "protocol", "模型服务返回了无效的文本编码。", response_usage=usage
        ) from None
    if output_bytes > MAX_OUTPUT_BYTES:
        raise ModelError("output_too_large", "模型输出超过应用大小限制。", response_usage=usage)
    return ModelResponse(
        provider=provider,
        model=completion.model,
        request_id=completion.id,
        content=content,
        usage=usage,
        latency_ms=elapsed_ms,
    )


class ChatCompatibleProvider:
    def __init__(
        self, settings: ProviderSettings, *, client: httpx.AsyncClient | None = None
    ) -> None:
        # Snapshot and revalidate even if a caller used model_construct/model_copy.
        self.settings = ProviderSettings.model_validate(settings.model_dump())
        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else httpx.AsyncClient(follow_redirects=False, trust_env=False)
        )
        self._closed = False

    async def __aenter__(self) -> "ChatCompatibleProvider":
        if self._closed:
            raise ModelError("closed", "模型连接已关闭。")
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            if self._owns_client:
                await self._client.aclose()

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [message.model_dump() for message in request.messages],
            "response_format": {"type": request.response_format},
            "stream": False,
        }
        # OpenAI and MiMo document max_completion_tokens (including reasoning).
        token_key = (
            "max_completion_tokens"
            if self.settings.provider in ("openai", "mimo")
            else "max_tokens"
        )
        payload[token_key] = request.max_output_tokens
        if self.settings.provider in ("deepseek", "mimo"):
            payload["thinking"] = {"type": "disabled"}
        return payload

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self._closed:
            raise ModelError("closed", "模型连接已关闭。")
        try:
            request = ModelRequest.model_validate(request.model_dump())
        except (ValidationError, ValueError, UnicodeError):
            raise ModelError("invalid_request", "模型请求不符合应用契约。") from None
        payload = self._payload(request)
        headers = {"Accept": "application/json", "Accept-Encoding": "identity"}
        if self.settings.api_key is not None:
            key = self.settings.api_key.get_secret_value()
            if self.settings.provider == "mimo":
                headers["api-key"] = key
            else:
                headers["Authorization"] = "Bearer " + key
        started = perf_counter()
        try:
            # Read timeouts alone do not bound a response that keeps trickling.
            async with asyncio.timeout(self.settings.timeout_seconds):
                async with self._client.stream(
                    "POST",
                    self.settings.base_url + "/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.settings.timeout_seconds,
                    follow_redirects=False,
                ) as response:
                    if response.status_code != 200:
                        raise _http_error(response.status_code)
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(body) + len(chunk) > MAX_HTTP_RESPONSE_BYTES:
                            raise ModelError("output_too_large", "模型响应超过应用大小限制。")
                        body.extend(chunk)
        except (TimeoutError, httpx.TimeoutException):
            raise ModelError("timeout", "模型请求超时，本次未自动重试。") from None
        except httpx.RequestError:
            raise ModelError("network", "无法连接模型服务，请检查网络。") from None
        # CancelledError propagates; the response context still closes its stream.
        return _parse_response(
            bytes(body), self.settings.provider, (perf_counter() - started) * 1000
        )

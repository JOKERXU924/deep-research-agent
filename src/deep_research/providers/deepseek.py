"""One bounded, non-streaming DeepSeek call. No hidden retries."""

import asyncio
from time import perf_counter
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from deep_research.config import Settings
from deep_research.errors import ModelError
from deep_research.schemas import MAX_OUTPUT_BYTES, ModelRequest, ModelResponse, TokenUsage

MAX_HTTP_RESPONSE_BYTES = 1024 * 1024


class _WireModel(BaseModel):
    # Providers may add metadata; fields we depend on are still strictly checked.
    model_config = ConfigDict(extra="ignore", strict=True)


class _Message(_WireModel):
    role: Literal["assistant"]
    content: str | None
    tool_calls: list[dict[str, Any]] | None = None


class _Choice(_WireModel):
    index: Literal[0]
    finish_reason: str
    message: _Message


class _Usage(_WireModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class _Completion(_WireModel):
    id: str = Field(min_length=1, max_length=256)
    model: str = Field(min_length=1, max_length=100)
    choices: list[_Choice] = Field(min_length=1, max_length=1)
    usage: _Usage | None = None


def _http_error(status: int) -> ModelError:
    if status in (401, 403):
        code, message = "authentication", "模型服务拒绝认证或当前账号没有权限。"
    elif status == 429:
        code, message = "rate_limit", "模型服务限流，请稍后再试。"
    elif status == 402:
        code, message = "billing", "模型服务提示余额不足。"
    elif status >= 500:
        code, message = "upstream", "模型服务暂时不可用。"
    else:
        code, message = "http_error", "模型服务拒绝请求，请检查模型和接口配置。"
    return ModelError(code, message, status_code=status)


class DeepSeekProvider:
    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(follow_redirects=False)
        self._closed = False

    async def __aenter__(self) -> "DeepSeekProvider":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            if self._owns_client:
                await self._client.aclose()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self._closed:
            raise ModelError("closed", "模型连接已关闭。")
        payload = {
            "model": self.settings.model,
            "messages": [message.model_dump() for message in request.messages],
            "max_tokens": request.max_output_tokens,
            "response_format": {"type": request.response_format},
            "thinking": {"type": "disabled"},
            "stream": False,
        }
        started = perf_counter()
        try:
            # A wall-clock deadline covers the whole call, not just each read.
            async with asyncio.timeout(self.settings.timeout_seconds):
                async with self._client.stream(
                    "POST",
                    self.settings.base_url + "/chat/completions",
                    headers={"Authorization": "Bearer " + self.settings.api_key.get_secret_value()},
                    json=payload,
                    timeout=self.settings.timeout_seconds,
                    follow_redirects=False,
                ) as response:
                    if response.status_code != 200:
                        raise _http_error(response.status_code)
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_HTTP_RESPONSE_BYTES:
                            raise ModelError("output_too_large", "模型响应超过应用大小限制。")
        except (TimeoutError, httpx.TimeoutException):
            raise ModelError("timeout", "模型请求超时，本次未自动重试。") from None
        except httpx.RequestError:
            raise ModelError("network", "无法连接模型服务，请检查网络。") from None
        # asyncio.CancelledError deliberately propagates to the caller.
        try:
            completion = _Completion.model_validate_json(bytes(body))
        except ValidationError:
            raise ModelError("protocol", "模型服务返回了无法识别的响应结构。") from None
        choice = completion.choices[0]
        if choice.finish_reason == "length":
            raise ModelError("truncated", "模型输出被截断，不能作为完整结果使用。")
        if choice.finish_reason != "stop" or choice.message.tool_calls:
            raise ModelError("incomplete", "模型未正常完成文本回复。")
        content = choice.message.content
        if content is None or not content.strip():
            raise ModelError("empty_output", "模型返回了空文本。")
        if len(content.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise ModelError("output_too_large", "模型输出超过应用大小限制。")
        try:
            usage = completion.usage
            return ModelResponse(
                provider="deepseek",
                model=completion.model,
                request_id=completion.id,
                content=content,
                usage=None
                if usage is None
                else TokenUsage(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                ),
                latency_ms=(perf_counter() - started) * 1000,
            )
        except ValidationError:
            raise ModelError("protocol", "模型服务返回了不一致的响应字段。") from None

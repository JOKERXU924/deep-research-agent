"""Gateway tests verify vendor-independent requests and schema enforcement."""

import json

import pytest
from pydantic import ValidationError

from deep_research.errors import ModelError, StructuredOutputError
from deep_research.gateway import ModelGateway
from deep_research.schemas import (
    MAX_INPUT_BYTES,
    ChatMessage,
    ModelRequest,
    ModelResponse,
    ResearchOutline,
)


class FakeProvider:
    def __init__(self, content: str = '{"topic":"Evidence","questions":["What is a source?"]}'):
        self.calls: list[ModelRequest] = []
        self.response = ModelResponse(
            provider="fake",
            model="offline",
            request_id="offline-request",
            content=content,
            latency_ms=1.0,
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        return self.response


def request(content: str = "Make a research outline.") -> ModelRequest:
    return ModelRequest(messages=(ChatMessage(role="user", content=content),), max_output_tokens=99)


async def test_plain_generation_passes_the_contract_through_unchanged() -> None:
    provider = FakeProvider("Plain reply")
    incoming = request()
    result = await ModelGateway(provider).generate(incoming)
    assert provider.calls == [incoming]
    assert provider.calls[0] is incoming
    assert result is provider.response


async def test_structured_generation_adds_schema_without_mutating_original_request() -> None:
    provider = FakeProvider()
    gateway = ModelGateway(provider)
    incoming = request()
    before = incoming.model_dump_json()

    result = await gateway.generate_structured(incoming, ResearchOutline)

    assert result.data == ResearchOutline(topic="Evidence", questions=["What is a source?"])
    assert result.response is provider.response
    assert len(provider.calls) == 1
    sent = provider.calls[0]
    assert sent.response_format == "json_object"
    assert sent.max_output_tokens == 99
    assert sent.messages[1:] == incoming.messages
    assert sent.messages[0].role == "system"
    instruction, schema_json = sent.messages[0].content.split("\n", maxsplit=1)
    assert "JSON" in instruction
    assert json.loads(schema_json) == ResearchOutline.model_json_schema()
    assert incoming.model_dump_json() == before
    assert incoming.response_format == "text"


async def test_added_schema_must_fit_the_input_budget_before_calling_provider() -> None:
    overhead = len(request("x").model_dump_json().encode()) - 1
    incoming = request("x" * (MAX_INPUT_BYTES - overhead))
    assert len(incoming.model_dump_json().encode()) == MAX_INPUT_BYTES
    provider = FakeProvider()

    with pytest.raises(ValidationError, match="application limit"):
        await ModelGateway(provider).generate_structured(incoming, ResearchOutline)

    assert provider.calls == []


@pytest.mark.parametrize(
    ("content", "code"),
    [
        ('{"topic":"Evidence",', "invalid_json"),
        ('{"topic":"Evidence","questions":[]}', "schema_mismatch"),
        ('{"topic":"Evidence","questions":[42]}', "schema_mismatch"),
        ('{"topic":"Evidence","questions":["Why?"],"invented":true}', "schema_mismatch"),
    ],
)
async def test_json_mode_does_not_bypass_local_validation(content: str, code: str) -> None:
    provider = FakeProvider(content)
    with pytest.raises(StructuredOutputError) as caught:
        await ModelGateway(provider).generate_structured(request(), ResearchOutline)
    assert caught.value.code == code
    assert len(provider.calls) == 1  # M0 does not silently repair or retry with paid calls.


async def test_provider_error_propagates_without_retry_or_fabricated_output() -> None:
    class FailedProvider(FakeProvider):
        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.calls.append(request)
            raise ModelError("rate_limit", "Try later.", status_code=429)

    provider = FailedProvider()
    with pytest.raises(ModelError) as caught:
        await ModelGateway(provider).generate_structured(request(), ResearchOutline)
    assert caught.value.code == "rate_limit"
    assert caught.value.status_code == 429
    assert len(provider.calls) == 1

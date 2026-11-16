"""Future agents depend on this gateway, not a vendor SDK or response shape."""

import json
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from deep_research.schemas import ChatMessage, ModelRequest, ModelResponse
from deep_research.structured import parse_structured


class ModelProvider(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse: ...


@dataclass(frozen=True)
class StructuredResult[T: BaseModel]:
    data: T
    response: ModelResponse
    parse_mode: str = "strict"


def structured_request(request: ModelRequest, schema: type[BaseModel]) -> ModelRequest:
    """The exact outgoing request, including the schema's application byte cost."""
    instructions = ChatMessage(
        role="system",
        content=(
            "Return exactly one JSON object matching this JSON Schema. "
            "Do not include markdown or explanations.\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        ),
    )
    return ModelRequest(
        messages=(instructions, *request.messages),
        max_output_tokens=request.max_output_tokens,
        response_format="json_object",
    )


class ModelGateway:
    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    async def generate(self, request: ModelRequest) -> ModelResponse:
        return await self.provider.complete(request)

    async def generate_structured[T: BaseModel](
        self, request: ModelRequest, schema: type[T]
    ) -> StructuredResult[T]:
        # Reconstruct to revalidate the budget after adding the schema prompt.
        response = await self.generate(structured_request(request, schema))
        return StructuredResult(data=parse_structured(response.content, schema), response=response)

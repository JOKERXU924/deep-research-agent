"""The application contract is independent of a model vendor's JSON envelope."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

MAX_INPUT_BYTES = 64 * 1024
MAX_OUTPUT_BYTES = 256 * 1024

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: NonEmptyText


class ModelRequest(StrictModel):
    messages: tuple[ChatMessage, ...] = Field(min_length=1, max_length=100)
    max_output_tokens: int = Field(default=512, ge=1, le=4096)
    response_format: Literal["text", "json_object"] = "text"

    @model_validator(mode="after")
    def check_input_size(self) -> Self:
        # An application byte limit, not a claim about the model's token window.
        if len(self.model_dump_json().encode("utf-8")) > MAX_INPUT_BYTES:
            raise ValueError("model input exceeds the 64 KiB application limit")
        return self


class TokenUsage(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def check_total(self) -> Self:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("token usage total is inconsistent")
        return self


class ModelResponse(StrictModel):
    provider: NonEmptyText
    model: NonEmptyText
    request_id: NonEmptyText
    content: NonEmptyText
    finish_reason: Literal["stop"] = "stop"
    usage: TokenUsage | None = None
    latency_ms: float = Field(ge=0, allow_inf_nan=False)


class ResearchOutline(StrictModel):
    """M0's demonstration output; this is not a researched factual report."""

    topic: NonEmptyText
    questions: list[NonEmptyText] = Field(min_length=1, max_length=5)

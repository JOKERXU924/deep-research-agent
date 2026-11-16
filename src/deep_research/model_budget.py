"""M7 application reservations, not a tokenizer or a monetary billing guarantee."""

import re
from typing import Self

from pydantic import Field, ValidationError, model_validator

from deep_research.errors import ResearchError
from deep_research.schemas import ModelRequest, StrictModel, TokenUsage


class ModelBudgetLimits(StrictModel):
    max_calls: int = Field(default=9, ge=1, le=9)
    max_input_bytes: int = Field(default=9 * 65536, ge=1, le=9 * 65536)
    max_output_tokens: int = Field(default=9 * 4096, ge=1, le=9 * 4096)


def validate_profile_snapshot(profile: dict) -> dict:
    """Copy a deliberately tiny public profile; never persist endpoints or keys."""
    valid = isinstance(profile, dict) and set(profile) == {
        "id",
        "label",
        "provider",
        "model",
        "fingerprint",
    }
    if valid:
        valid = all(
            isinstance(value, str)
            and value.strip() == value
            and 1 <= len(value) <= 256
            and not any(ord(char) < 32 or ord(char) == 127 for char in value)
            for value in profile.values()
        )
    if valid:
        valid = (
            re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", profile["id"]) is not None
            and len(profile["label"]) <= 80
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}", profile["model"]) is not None
            and re.fullmatch(r"[a-f0-9]{64}", profile["fingerprint"]) is not None
            and profile["provider"] in {"deepseek", "openai", "mimo", "vllm", "offline_demo"}
        )
    if not valid:
        raise ResearchError("model_profile_invalid", "模型配置公开快照无效。")
    try:
        for value in profile.values():
            value.encode("utf-8")
    except UnicodeError:
        raise ResearchError("model_profile_invalid", "模型配置公开快照无效。") from None
    return dict(profile)


class _Reserved(StrictModel):
    calls: int = Field(ge=0, le=9)
    input_bytes: int = Field(ge=0, le=9 * 65536)
    output_tokens: int = Field(ge=0, le=9 * 4096)


class _BudgetSummary(StrictModel):
    limits: ModelBudgetLimits
    effective_max_calls: int = Field(ge=1, le=9)
    reserved: _Reserved
    known_usage: TokenUsage
    usage_known_calls: int = Field(ge=0, le=9)
    usage_unknown_calls: int = Field(ge=0, le=9)
    usage_complete: bool

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if (
            self.effective_max_calls > self.limits.max_calls
            or self.reserved.calls > self.effective_max_calls
            or self.reserved.input_bytes > self.limits.max_input_bytes
            or self.reserved.output_tokens > self.limits.max_output_tokens
            or self.usage_known_calls + self.usage_unknown_calls != self.reserved.calls
            or self.usage_complete != (self.usage_unknown_calls == 0)
            or (self.usage_known_calls == 0 and self.known_usage.total_tokens != 0)
            or (
                self.reserved.calls == 0
                and (self.reserved.input_bytes or self.reserved.output_tokens)
            )
        ):
            raise ValueError("inconsistent model budget summary")
        return self


def validate_budget_summary(value: dict) -> dict:
    """Strict public allowlist used by Web summaries and audit responses."""
    try:
        return _BudgetSummary.model_validate(value).model_dump()
    except ValidationError:
        raise ResearchError("model_budget_invalid", "模型预算摘要无效。") from None


class ModelBudget:
    """Single-event-loop ledger. Accepted reservations are never refunded.

    prepare() is pure: the caller durably saves its returned reservation before
    commit() and dispatch. All bytes include the structured-output schema.
    """

    def __init__(self, limits: ModelBudgetLimits, *, architecture_limit: int = 9) -> None:
        if not isinstance(limits, ModelBudgetLimits):
            raise ResearchError("model_budget_invalid", "模型预算配置无效。")
        self.limits = ModelBudgetLimits.model_validate(limits.model_dump())
        if type(architecture_limit) is not int or not 1 <= architecture_limit <= 9:
            raise ResearchError("model_budget_invalid", "模型调用架构上限无效。")
        self.effective_max_calls = min(self.limits.max_calls, architecture_limit)
        self.reserved = {"calls": 0, "input_bytes": 0, "output_tokens": 0}
        self.known_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self.usage_known_calls = 0

    def prepare(self, request: ModelRequest) -> dict:
        reservation = {
            "input_bytes": len(request.model_dump_json().encode("utf-8")),
            "output_tokens": request.max_output_tokens,
        }
        if (
            self.reserved["calls"] >= self.effective_max_calls
            or self.reserved["input_bytes"] + reservation["input_bytes"]
            > self.limits.max_input_bytes
            or self.reserved["output_tokens"] + reservation["output_tokens"]
            > self.limits.max_output_tokens
        ):
            raise ResearchError(
                "model_budget_exceeded", "本次研究的模型调用预算已耗尽，未发送新请求。"
            )
        return reservation

    def commit(self, reservation: dict) -> None:
        self.reserved["calls"] += 1
        self.reserved["input_bytes"] += reservation["input_bytes"]
        self.reserved["output_tokens"] += reservation["output_tokens"]

    def observe(self, usage: TokenUsage | None) -> None:
        if usage is not None:
            checked = TokenUsage.model_validate(usage.model_dump())
            self.usage_known_calls += 1
            for key, value in checked.model_dump().items():
                self.known_usage[key] += value

    def summary(self) -> dict:
        unknown = self.reserved["calls"] - self.usage_known_calls
        return {
            "limits": self.limits.model_dump(),
            "effective_max_calls": self.effective_max_calls,
            "reserved": dict(self.reserved),
            "known_usage": dict(self.known_usage),
            "usage_known_calls": self.usage_known_calls,
            "usage_unknown_calls": unknown,
            "usage_complete": unknown == 0,
        }


def budget_from_calls(limits: dict, architecture_limit: int, calls: list[dict]) -> dict:
    """Rebuild the conservative view after interruption or missing final summary."""
    budget = ModelBudget(
        ModelBudgetLimits.model_validate(limits), architecture_limit=architecture_limit
    )
    for call in calls:
        details = call["details"]
        reservation = details.get("reservation")
        if reservation is None:
            raise ResearchError("model_budget_invalid", "模型调用缺少预算预留记录。")
        budget.commit(reservation)
        if details.get("usage") is not None:
            budget.observe(TokenUsage.model_validate(details["usage"]))
    return validate_budget_summary(budget.summary())

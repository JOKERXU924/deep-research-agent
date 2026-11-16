"""M7 budgets reserve application quantities and never claim monetary accounting."""

import pytest
from pydantic import ValidationError

from deep_research.errors import ModelError, ResearchError
from deep_research.gateway import structured_request
from deep_research.model_budget import (
    ModelBudget,
    ModelBudgetLimits,
    budget_from_calls,
    validate_budget_summary,
    validate_profile_snapshot,
)
from deep_research.research_models import ResearchPlan
from deep_research.schemas import ChatMessage, ModelRequest, TokenUsage


def outgoing():
    return structured_request(
        ModelRequest(
            messages=(ChatMessage(role="user", content="中文 budget"),), max_output_tokens=50
        ),
        ResearchPlan,
    )


@pytest.mark.parametrize(
    "key,maximum", [("max_calls", 9), ("max_input_bytes", 589824), ("max_output_tokens", 36864)]
)
@pytest.mark.parametrize("bad", [0, -1, True, "2", 1.5, None])
def test_limits_strict(key, maximum, bad):
    with pytest.raises(ValidationError):
        ModelBudgetLimits(**{key: bad})


@pytest.mark.parametrize(
    "key,maximum", [("max_calls", 9), ("max_input_bytes", 589824), ("max_output_tokens", 36864)]
)
def test_limits_exact_bounds_and_frozen(key, maximum):
    assert getattr(ModelBudgetLimits(**{key: 1}), key) == 1
    limits = ModelBudgetLimits(**{key: maximum})
    with pytest.raises(ValidationError):
        ModelBudgetLimits(**{key: maximum + 1})
    with pytest.raises(ValidationError):
        setattr(limits, key, 1)


def test_limits_forbid_extra_and_revalidate_constructed():
    with pytest.raises(ValidationError):
        ModelBudgetLimits(dollars=2)
    with pytest.raises(ValidationError):
        ModelBudget(ModelBudgetLimits.model_construct(max_calls=100))


def test_exact_final_schema_request_bytes_and_pure_prepare():
    request = outgoing()
    raw_bytes = len(request.model_dump_json().encode("utf-8"))
    budget = ModelBudget(
        ModelBudgetLimits(max_calls=1, max_input_bytes=raw_bytes, max_output_tokens=50)
    )
    reservation = budget.prepare(request)
    assert reservation == {"input_bytes": raw_bytes, "output_tokens": 50}
    assert budget.summary()["reserved"]["calls"] == 0
    assert raw_bytes > len(request.messages[-1].content.encode("utf-8"))
    budget.commit(reservation)
    assert budget.summary()["reserved"] == {"calls": 1, **reservation}
    with pytest.raises(ResearchError, match="预算") as raised:
        budget.prepare(request)
    assert raised.value.code == "model_budget_exceeded"
    assert budget.summary()["reserved"]["calls"] == 1


@pytest.mark.parametrize("field", ["max_calls", "max_input_bytes", "max_output_tokens"])
def test_multiple_calls_reserve_aggregate_and_reject_without_mutation(field):
    request = outgoing()
    values = {
        "max_calls": 2,
        "max_input_bytes": len(request.model_dump_json().encode("utf-8")) * 2,
        "max_output_tokens": 100,
    }
    values[field] -= 1
    budget = ModelBudget(ModelBudgetLimits(**values))
    budget.commit(budget.prepare(request))
    before = budget.summary()
    with pytest.raises(ResearchError) as raised:
        budget.prepare(request)
    assert raised.value.code == "model_budget_exceeded"
    assert budget.summary() == before


def test_effective_architecture_limit_and_usage_is_not_reservation():
    budget = ModelBudget(ModelBudgetLimits(), architecture_limit=3)
    for _ in range(3):
        budget.commit(budget.prepare(outgoing()))
    budget.observe(TokenUsage(input_tokens=100, output_tokens=3, total_tokens=103))
    budget.observe(None)
    value = budget.summary()
    assert value["effective_max_calls"] == 3
    assert value["reserved"]["output_tokens"] == 150
    assert value["known_usage"]["output_tokens"] == 3
    assert value["usage_unknown_calls"] == 2
    assert value["usage_complete"] is False
    assert validate_budget_summary(value) == value
    with pytest.raises(ResearchError):
        budget.prepare(outgoing())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda v: v.update(secret="hidden"),
        lambda v: v["reserved"].update(calls=True),
        lambda v: v["limits"].update(max_calls=100),
        lambda v: v.update(usage_complete=True),
        lambda v: v.update(usage_unknown_calls=0),
        lambda v: v["known_usage"].update(total_tokens=2),
        lambda v: v.update(effective_max_calls=10),
    ],
)
def test_public_budget_summary_strict_and_consistent(mutation):
    budget = ModelBudget(ModelBudgetLimits())
    budget.commit(budget.prepare(outgoing()))
    value = budget.summary()
    mutation(value)
    with pytest.raises(ResearchError) as raised:
        validate_budget_summary(value)
    assert raised.value.code == "model_budget_invalid"


def profile():
    return {
        "id": "demo",
        "label": "离线演示",
        "provider": "offline_demo",
        "model": "fixture/model",
        "fingerprint": "a" * 64,
    }


def test_profile_snapshot_is_defensive():
    value = profile()
    copied = validate_profile_snapshot(value)
    value["model"] = "other"
    assert copied["model"] == "fixture/model"


@pytest.mark.parametrize(
    "field,value",
    [
        ("api_key", "secret"),
        ("base_url", "http://secret"),
        ("id", "../path"),
        ("label", "bad\nlabel"),
        ("label", "x" * 81),
        ("model", "http://host?token=secret"),
        ("provider", "unknown"),
        ("fingerprint", "A" * 64),
        ("label", "\ud800"),
    ],
)
def test_profile_public_snapshot_rejects_secrets_and_invalid_values(field, value):
    item = profile()
    item[field] = value
    with pytest.raises(ResearchError) as raised:
        validate_profile_snapshot(item)
    assert raised.value.code == "model_profile_invalid"


def test_reconstruct_in_flight_and_failed_usage_without_final_summary():
    result = budget_from_calls(
        ModelBudgetLimits().model_dump(),
        3,
        [
            {
                "status": "in_flight",
                "details": {"reservation": {"input_bytes": 10, "output_tokens": 20}},
            },
            {
                "status": "failed",
                "details": {
                    "reservation": {"input_bytes": 30, "output_tokens": 40},
                    "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                },
            },
        ],
    )
    assert result["reserved"] == {"calls": 2, "input_bytes": 40, "output_tokens": 60}
    assert result["usage_known_calls"] == result["usage_unknown_calls"] == 1
    assert result["known_usage"]["total_tokens"] == 5


def test_model_error_preserves_only_valid_usage():
    error = ModelError(
        "model_truncated",
        "Truncated",
        response_usage=TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5),
    )
    assert error.response_usage.total_tokens == 5
    assert ModelError("unknown", "Unknown").response_usage is None
    with pytest.raises(TypeError):
        ModelError("wrong", "Wrong", response_usage={"payload": "secret"})
    with pytest.raises(ValidationError):
        ModelError(
            "wrong",
            "Wrong",
            response_usage=TokenUsage.model_construct(
                input_tokens=1, output_tokens=2, total_tokens=5
            ),
        )

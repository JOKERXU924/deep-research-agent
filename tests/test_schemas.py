import pytest
from pydantic import ValidationError

from deep_research.schemas import MAX_INPUT_BYTES, ChatMessage, ModelRequest, TokenUsage


def test_message_and_request_reject_invalid_types():
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="   ")
    with pytest.raises(ValidationError):
        ModelRequest(messages=())
    with pytest.raises(ValidationError):
        ModelRequest(messages=(ChatMessage(role="user", content="hello"),), max_output_tokens=True)


def test_input_budget_is_serialized_utf8_not_character_count():
    with pytest.raises(ValidationError):
        ModelRequest(messages=(ChatMessage(role="user", content="中" * (MAX_INPUT_BYTES // 3)),))


def test_usage_is_validated_instead_of_silently_invented():
    with pytest.raises(ValidationError):
        TokenUsage(input_tokens=10, output_tokens=2, total_tokens=100)
    with pytest.raises(ValidationError):
        TokenUsage(input_tokens=-1, output_tokens=2, total_tokens=1)

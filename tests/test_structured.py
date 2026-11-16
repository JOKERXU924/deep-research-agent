import pytest
from pydantic import BaseModel, Field

from deep_research.errors import StructuredOutputError
from deep_research.schemas import MAX_OUTPUT_BYTES, StrictModel
from deep_research.structured import parse_structured


class Result(StrictModel):
    answer: str
    count: int = Field(ge=0)


@pytest.mark.parametrize(
    "text",
    [
        '{"answer":"中文与 emoji 😀", "count":2}',
        ' \n{"answer":"中文与 emoji 😀", "count":2}\n ',
        '```json\n{"answer":"中文与 emoji 😀", "count":2}\n```',
        '```\r\n{"answer":"中文与 emoji 😀", "count":2}\r\n```',
    ],
)
def test_complete_json_and_whole_fence_are_accepted(text):
    result = parse_structured(text, Result)
    assert result.answer == "中文与 emoji 😀"
    assert result.count == 2


@pytest.mark.parametrize(
    "text",
    [
        "",
        '{"answer":"unfinished',
        'Here is JSON: {"answer":"x", "count":2}',
        '```json\n{"answer":"x", "count":2}\n```\nexplanation',
        '{"answer":"x", "count":2}{"answer":"y", "count":3}',
        '{"answer":"x", "count":1, "count":2}',
        '{"answer":"x", "count":NaN}',
        '{"answer":"x", "count":Infinity}',
        '{"answer":"x", "count":-Infinity}',
        '{"answer":"x", "count":1e999}',
    ],
)
def test_malformed_or_ambiguous_json_is_rejected(text):
    with pytest.raises(StructuredOutputError) as caught:
        parse_structured(text, Result)
    assert caught.value.code == "invalid_json"


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "null",
        '"text"',
        '{"answer":"x"}',
        '{"answer":"x", "count":"2"}',
        '{"answer":"x", "count":true}',
        '{"answer":"x", "count":-1}',
        '{"answer":"x", "count":1, "unknown":"secret-marker"}',
    ],
)
def test_schema_requires_correct_types_fields_and_constraints(text):
    with pytest.raises(StructuredOutputError) as caught:
        parse_structured(text, Result)
    assert caught.value.code == "schema_mismatch"
    assert "secret-marker" not in str(caught.value)


def test_size_limit_counts_utf8_bytes():
    with pytest.raises(StructuredOutputError) as caught:
        parse_structured("中" * (MAX_OUTPUT_BYTES // 3 + 1), Result)
    assert caught.value.code == "output_too_large"


def test_nested_duplicate_key_is_not_silently_overwritten():
    with pytest.raises(StructuredOutputError) as caught:
        parse_structured('{"outer":{"same":1,"same":2}}', Result)
    assert caught.value.code == "invalid_json"


def test_extra_fields_rejected_even_for_a_plain_pydantic_model():
    class PlainModel(BaseModel):
        answer: str

    with pytest.raises(StructuredOutputError):
        parse_structured('{"answer":"ok","extra":1}', PlainModel)

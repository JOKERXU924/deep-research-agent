"""Parse complete JSON, then validate its shape; never execute or guess content."""

import json
import math
import re

from pydantic import BaseModel, ValidationError

from deep_research.errors import StructuredOutputError
from deep_research.schemas import MAX_OUTPUT_BYTES
from deep_research.structured_diagnostics import validation_diagnostic

_FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```\Z", re.DOTALL)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError("nonstandard JSON number")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("JSON number exceeds the supported finite range")
    return result


def parse_structured[T: BaseModel](content: str, schema: type[T]) -> T:
    if len(content.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise StructuredOutputError("output_too_large", "模型输出超过应用大小限制。")
    text = content.strip()
    if match := _FENCE.fullmatch(text):
        text = match.group("body").strip()
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (ValueError, RecursionError):
        raise StructuredOutputError(
            "invalid_json", "模型未返回完整合法的 JSON；不接受解释文字、重复键或非标准数字。"
        ) from None
    if not isinstance(value, dict):
        raise StructuredOutputError(
            "schema_mismatch",
            "结构化输出必须是 JSON 对象。",
            validation=validation_diagnostic(schema.__name__, [{"type": "object_required"}]),
        )
    try:
        return schema.model_validate(value, strict=True, extra="forbid")
    except ValidationError as error:
        diagnostic = validation_diagnostic(
            schema.__name__,
            error.errors(include_input=False, include_context=False, include_url=False),
        )
        kinds = sorted({item["kind"] for item in diagnostic["issues"]})
        raise StructuredOutputError(
            "schema_mismatch",
            "模型 JSON 未通过字段校验：" + ", ".join(kinds),
            validation=diagnostic,
        ) from None

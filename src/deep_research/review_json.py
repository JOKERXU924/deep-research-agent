"""M5-only bounded JSON envelope recovery; never repair syntax or execute text."""

from pydantic import BaseModel

from deep_research.errors import StructuredOutputError
from deep_research.schemas import MAX_OUTPUT_BYTES
from deep_research.structured import parse_structured
from deep_research.structured_diagnostics import validate_diagnostic


def _checked_parse(content, schema):
    value = parse_structured(content, schema)
    if schema.__name__ == "BlueRepairs":
        targets = [o.target_id for o in value.operations if o.target_id is not None]
        if len(targets) != len(set(targets)):
            raise StructuredOutputError(
                "schema_mismatch",
                "修复操作重复引用同一目标。",
                validation=validate_diagnostic(
                    dict(
                        schema="BlueRepairs",
                        truncated=False,
                        issues=[dict(path="operations.*.target_id", kind="duplicate_reference")],
                    )
                ),
            )
    return value


def parse_review_output[T: BaseModel](content: str, schema: type[T]) -> tuple[T, str]:
    try:
        if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise StructuredOutputError("output_too_large", "审查输出超过应用大小限制。")
    except UnicodeError:
        raise StructuredOutputError("invalid_json", "审查输出不是合法UTF-8文本。") from None
    try:
        return _checked_parse(content, schema), "strict"
    except StructuredOutputError as error:
        if error.code != "invalid_json":
            raise
    # Recover exactly one complete top-level object surrounded by short prose.
    # Braces in JSON strings are tracked, not matched with a greedy regex.
    spans = []
    depth = 0
    quoted = False
    escaped = False
    start = 0
    for index, char in enumerate(content):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                break
            if depth == 0:
                spans.append((start, index + 1))
    if depth != 0 or quoted or len(spans) != 1:
        raise StructuredOutputError("invalid_json", "审查输出不含唯一完整JSON对象。")
    start, end = spans[0]
    wrappers = content[:start] + content[end:]
    if len(wrappers.encode("utf-8")) > 1024 or any(char in wrappers for char in "[]{}"):
        raise StructuredOutputError("invalid_json", "审查输出的JSON包裹文本不明确或过长。")
    return _checked_parse(content[start:end], schema), "envelope"

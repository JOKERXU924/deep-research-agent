"""Bounded local dataset loading; no download and no gold in task payloads."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from urllib.parse import quote

from pydantic import ValidationError

from deep_research.benchmark_models import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkDocument,
    BenchmarkGold,
    SupportingFact,
)
from deep_research.errors import ResearchError

MAX_DATASET_BYTES = 16 * 1024 * 1024
MAX_HOTPOT_BYTES = 64 * 1024 * 1024
MAX_HOTPOT_ROWS = 10000


def _error() -> ResearchError:
    # Never include a filesystem path, raw row, or Pydantic's input values.
    return ResearchError(
        "benchmark_data",
        "评测数据无效：请检查格式、唯一编号、证据引用和本地文件大小限制。",
    )


def _pairs(items: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError("nonfinite JSON constant")


def _decode(raw: bytes) -> object:
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)


def _bounded_read(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if not raw or len(raw) > limit:
        raise ValueError("empty or oversized local dataset")
    return raw


def _validated(dataset: BenchmarkDataset) -> BenchmarkDataset:
    # Revalidate even objects made with model_construct/model_copy(update=...).
    return BenchmarkDataset.model_validate_json(dataset.model_dump_json())


def load_benchmark(path: Path | None = None) -> BenchmarkDataset:
    try:
        if path is None:
            resource = files("deep_research").joinpath("benchmarks", "researchbench_v1.json")
            with resource.open("rb") as stream:
                raw = stream.read(MAX_DATASET_BYTES + 1)
            if not raw or len(raw) > MAX_DATASET_BYTES:
                raise ValueError("empty or oversized package dataset")
        else:
            raw = _bounded_read(path, MAX_DATASET_BYTES)
        # Decode once to reject duplicate keys; JSON-mode validation preserves strict tuples.
        decoded = _decode(raw)
        return BenchmarkDataset.model_validate_json(
            json.dumps(decoded, ensure_ascii=False, allow_nan=False)
        )
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError, ValidationError):
        raise _error() from None


def dataset_fingerprint(dataset: BenchmarkDataset) -> str:
    try:
        validated = _validated(dataset)
        canonical = json.dumps(
            validated.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
    except (UnicodeError, ValueError, TypeError, RecursionError, ValidationError):
        raise _error() from None


def public_case(case: BenchmarkCase) -> dict:
    """Projection, not a blacklist: evaluator labels cannot reach model prompts."""
    validated = BenchmarkCase.model_validate_json(case.model_dump_json())
    return {
        "id": validated.id,
        "question": validated.question,
        "documents": [doc.model_dump(mode="json") for doc in validated.documents],
    }


def _hotpot_case(row: object) -> BenchmarkCase:
    required = {"_id", "question", "answer", "supporting_facts", "context"}
    allowed = required | {"type", "level"}
    if not isinstance(row, dict) or not required <= row.keys() or not row.keys() <= allowed:
        raise ValueError("invalid HotpotQA row fields")
    if "type" in row and row["type"] not in {"bridge", "comparison"}:
        raise ValueError("invalid HotpotQA question type")
    if "level" in row and row["level"] not in {"easy", "medium", "hard"}:
        raise ValueError("invalid HotpotQA difficulty")
    context = row["context"]
    if not isinstance(context, list) or not 2 <= len(context) <= 10:
        raise ValueError("HotpotQA requires 2 to 10 context documents")
    documents = []
    for entry in context:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError("invalid HotpotQA context entry")
        title, sentences = entry
        if (
            not isinstance(title, str)
            or not isinstance(sentences, list)
            or not 1 <= len(sentences) <= 1000
            or any(not isinstance(sentence, str) for sentence in sentences)
        ):
            raise ValueError("invalid HotpotQA sentence list")
        documents.append(
            BenchmarkDocument(
                id="hp-" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:20],
                title=title,
                text="\n".join(sentences),
                url="https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe=""),
                sentences=tuple(sentences),
            )
        )
    raw_facts = row["supporting_facts"]
    if not isinstance(raw_facts, list) or not 1 <= len(raw_facts) <= 100:
        raise ValueError("invalid HotpotQA supporting facts")
    facts = []
    for fact in raw_facts:
        if not isinstance(fact, list) or len(fact) != 2:
            raise ValueError("invalid HotpotQA supporting reference")
        facts.append(SupportingFact(title=fact[0], sentence_index=fact[1]))
    return BenchmarkCase(
        id=row["_id"],
        domain="hotpotqa",
        question=row["question"],
        kind="hotpotqa",
        documents=tuple(documents),
        gold=BenchmarkGold(answer_aliases=(row["answer"],), supporting_facts=tuple(facts)),
    )


def import_hotpot(path: Path, limit: int = 35) -> BenchmarkDataset:
    """Import a user-supplied dev-style file, retaining every selected row's context.

    All rows are schema-checked (including duplicate IDs) before selecting the first
    ``limit``. This is a deterministic local subset, not the official benchmark score.
    """
    try:
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("HotpotQA subset limit must be between 1 and 50")
        raw = _bounded_read(path, MAX_HOTPOT_BYTES)
        rows = _decode(raw)
        if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_HOTPOT_ROWS:
            raise ValueError("HotpotQA expects a bounded nonempty row list")
        cases = []
        ids = set()
        for index, row in enumerate(rows):
            case = _hotpot_case(row)
            if case.id in ids:
                raise ValueError("duplicate HotpotQA row IDs")
            ids.add(case.id)
            if index < limit:
                cases.append(case)
        digest = hashlib.sha256(raw).hexdigest()
        return BenchmarkDataset(
            id=f"hotpotqa-{digest[:12]}-{len(cases)}",
            description=(
                f"Local HotpotQA-format subset: first {len(cases)} of {len(rows)} rows; "
                "original answers, context sentences and supporting references preserved."
            ),
            provenance=(
                "User-supplied local HotpotQA dev-style JSON. File origin and split are not "
                "authenticated; no download, no official full-benchmark score, no synthetic "
                "replacement for original supporting facts."
            ),
            cases=tuple(cases),
            source_sha256=digest,
            split="local-unverified-subset",
        )
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError, ValidationError):
        raise _error() from None

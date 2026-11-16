"""M8 dataset tests use authored fixtures only, never download official data."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import deep_research.benchmark_data as data_module
from deep_research.benchmark_data import (
    dataset_fingerprint,
    import_hotpot,
    load_benchmark,
    public_case,
)
from deep_research.benchmark_models import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkDocument,
    BenchmarkGold,
    GoldFact,
    SupportingFact,
)
from deep_research.errors import ResearchError


def _dataset() -> dict:
    data = load_benchmark().model_dump(mode="json")
    data["cases"] = data["cases"][:1]
    return data


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _row(identifier: str = "test-1") -> dict:
    return {
        "_id": identifier,
        "question": "  Which color is assigned to the room used by Team Test?  ",
        "answer": "violet",
        "supporting_facts": [["Team Test", 0], ["Room Test", 1]],
        "context": [
            ["Team Test", ["Team Test uses Room Test.", "This is an authored parser fixture."]],
            ["Room Test", ["Room Test is upstairs.\nIt has one door.", "Its color is violet."]],
            ["Distractor Test", ["This unrelated room is green."]],
        ],
        "type": "bridge",
        "level": "hard",
    }


def test_builtin_has_35_cases_11_domains_and_explicit_limitations() -> None:
    dataset = load_benchmark()
    assert len(dataset.cases) == 35
    assert len({case.domain for case in dataset.cases}) == 11
    assert "fictional" in dataset.description
    assert "not human calibrated" in dataset.description
    assert "correlated" in dataset.provenance
    assert all(len(case.documents) >= 2 for case in dataset.cases)
    assert all(len(case.gold.facts) >= 2 for case in dataset.cases)
    assert all(case.kind == "research" for case in dataset.cases)
    for case in dataset.cases:
        assert {source for fact in case.gold.facts for source in fact.source_ids} == {
            doc.id for doc in case.documents
        }
        assert all(doc.url.startswith("https://benchmark.invalid/") for doc in case.documents)


def test_public_case_is_a_whitelist_projection() -> None:
    case = load_benchmark().cases[0]
    payload = public_case(case)
    assert set(payload) == {"id", "question", "documents"}
    assert "gold" not in payload
    assert "answer_aliases" not in json.dumps(payload)
    assert "source_ids" not in json.dumps(payload)
    assert payload["documents"][0]["text"] == case.documents[0].text
    payload["documents"][0]["text"] = "changed"
    assert case.documents[0].text != "changed"


def test_all_nested_contracts_are_frozen() -> None:
    case = load_benchmark().cases[0]
    for model, field, value in [
        (load_benchmark(), "id", "changed"),
        (case, "question", "changed"),
        (case.gold, "answer_aliases", ("changed",)),
        (case.documents[0], "text", "changed"),
        (case.gold.facts[0], "statement", "changed"),
        (SupportingFact(title="Title", sentence_index=0), "sentence_index", 1),
    ]:
        with pytest.raises(ValidationError, match="frozen"):
            setattr(model, field, value)
    assert isinstance(case.documents, tuple)
    assert isinstance(case.gold.facts, tuple)


def test_round_trip_and_fingerprint_ignore_json_key_order(tmp_path: Path) -> None:
    data = _dataset()
    a = load_benchmark(_write(tmp_path / "a.json", data))
    (tmp_path / "b.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=7, sort_keys=True), encoding="utf-8"
    )
    b = load_benchmark(tmp_path / "b.json")
    assert a == b
    assert dataset_fingerprint(a) == dataset_fingerprint(b)
    assert len(dataset_fingerprint(a)) == 64
    data["cases"][0]["gold"]["answer_aliases"] = ["new answer"]
    changed = load_benchmark(_write(tmp_path / "c.json", data))
    assert dataset_fingerprint(a) != dataset_fingerprint(changed)


def test_fingerprint_revalidates_constructed_model() -> None:
    invalid = load_benchmark().model_copy(update={"cases": ()})
    with pytest.raises(ResearchError) as caught:
        dataset_fingerprint(invalid)
    assert caught.value.code == "benchmark_data"


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not JSON",
        b"[]",
        b"null",
        b'{"version":1,"version":1}',
        b'{"a":{"nested":1,"nested":2}}',
        b'{"version":NaN}',
        b'{"version":Infinity}',
        b'{"version":1e999}',
        b"\xff",
        b"\xef\xbb\xbf{}",
        b"[" * 1200 + b"]" * 1200,
    ],
)
def test_invalid_json_is_safe_error(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "private-location.json"
    path.write_bytes(raw)
    with pytest.raises(ResearchError) as caught:
        load_benchmark(path)
    assert caught.value.code == "benchmark_data"
    assert "private-location" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_missing_and_bounded_files_fail_safely(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ResearchError):
        load_benchmark(tmp_path / "missing.json")
    monkeypatch.setattr(data_module, "MAX_DATASET_BYTES", 20)
    with pytest.raises(ResearchError):
        load_benchmark(_write(tmp_path / "large.json", _dataset()))


@pytest.mark.parametrize("version", [True, False, "1", 1.0, 0, 2, None])
def test_dataset_version_is_strict(tmp_path: Path, version: object) -> None:
    data = _dataset()
    data["version"] = version
    with pytest.raises(ResearchError):
        load_benchmark(_write(tmp_path / "bad.json", data))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(extra="secret"),
        lambda d: d.update(cases=[]),
        lambda d: d["cases"].append(copy.deepcopy(d["cases"][0])),
        lambda d: d["cases"][0].update(domain="Unsafe domain"),
        lambda d: d["cases"][0].update(question=" " * 10),
        lambda d: d["cases"][0].update(question="x" * 1001),
        lambda d: d["cases"][0].update(question="x\x00y"),
        lambda d: d["cases"][0].update(question="\ud800"),
        lambda d: d["cases"][0].update(kind="unknown"),
        lambda d: d["cases"][0].update(documents=d["cases"][0]["documents"][:1]),
        lambda d: d["cases"][0]["documents"][1].update(id="assignment"),
        lambda d: d["cases"][0]["documents"][1].update(
            title=d["cases"][0]["documents"][0]["title"]
        ),
        lambda d: d["cases"][0]["documents"][1].update(url=d["cases"][0]["documents"][0]["url"]),
        lambda d: d["cases"][0]["gold"]["facts"][0].update(source_ids=["missing"]),
        lambda d: d["cases"][0]["gold"]["facts"][0].update(source_ids=["assignment", "assignment"]),
        lambda d: d["cases"][0]["gold"]["facts"][0].update(aliases=["same", "same"]),
        lambda d: d["cases"][0]["gold"]["facts"][0].update(aliases=[" "]),
        lambda d: d["cases"][0]["gold"]["facts"][1].update(id="bridge"),
        lambda d: d["cases"][0]["gold"].update(answer_aliases=["same", "same"]),
        lambda d: d["cases"][0]["gold"].update(facts=[]),
        lambda d: d.update(source_sha256="not-a-hash"),
        lambda d: d.update(split="\x00"),
    ],
)
def test_dataset_invariants(tmp_path: Path, mutation) -> None:
    data = _dataset()
    mutation(data)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
    with pytest.raises(ResearchError):
        load_benchmark(path)


@pytest.mark.parametrize(
    "url",
    [
        "file:///private.txt",
        "javascript:alert(1)",
        "https://user:password@example.com/a",
        "https://example.com\\@evil.test/a",
        "https://example.com/space here",
        "https://example.com:invalid/a",
        "https:///missing-host",
        "https://example.com/\x7f",
    ],
)
def test_document_url_validation(url: str) -> None:
    with pytest.raises(ValidationError):
        BenchmarkDocument(id="doc", title="title", text="text", url=url)


def test_document_utf8_byte_limit_not_character_limit() -> None:
    kwargs = {"id": "doc", "title": "title", "url": "https://benchmark.invalid/doc"}
    assert BenchmarkDocument(text="a" * 65536, **kwargs).text
    with pytest.raises(ValidationError):
        BenchmarkDocument(text="a" * 65537, **kwargs)
    with pytest.raises(ValidationError):
        BenchmarkDocument(text="中" * 22000, **kwargs)


def test_sentence_text_must_preserve_original_sequence() -> None:
    with pytest.raises(ValidationError):
        BenchmarkDocument(
            id="doc",
            title="title",
            url="https://benchmark.invalid/doc",
            text="first second",
            sentences=("first", "second"),
        )


def test_strict_python_contract_does_not_coerce_lists_or_numbers() -> None:
    with pytest.raises(ValidationError):
        GoldFact(id="fact", statement="statement", aliases=["alias"], source_ids=("doc",))
    with pytest.raises(ValidationError):
        SupportingFact(title="title", sentence_index=True)
    with pytest.raises(ValidationError):
        SupportingFact(title="title", sentence_index="1")
    with pytest.raises(ValidationError):
        BenchmarkGold()
    with pytest.raises(ValidationError):
        BenchmarkDataset.model_validate(_dataset())


def test_hotpot_preserves_original_text_gold_and_all_context(tmp_path: Path) -> None:
    row = _row()
    path = _write(tmp_path / "authored-parser-fixture.json", [row])
    dataset = import_hotpot(path)
    case = dataset.cases[0]
    assert case.question == row["question"]
    assert case.gold.answer_aliases == ("violet",)
    assert case.gold.facts == ()
    assert len(case.documents) == 3  # The distractor must not be discarded using gold.
    assert case.documents[1].sentences == tuple(row["context"][1][1])
    assert case.documents[1].text == "\n".join(row["context"][1][1])
    assert case.gold.supporting_facts[1] == SupportingFact(title="Room Test", sentence_index=1)
    assert dataset.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert dataset.split == "local-unverified-subset"
    assert "not authenticated" in dataset.provenance
    assert "supporting_facts" not in json.dumps(public_case(case))


def test_hotpot_deterministic_prefix_subset_validates_remaining_rows(tmp_path: Path) -> None:
    rows = [_row("one"), _row("two"), _row("three")]
    path = _write(tmp_path / "rows.json", rows)
    selected = import_hotpot(path, limit=2)
    assert [case.id for case in selected.cases] == ["one", "two"]
    assert "first 2 of 3" in selected.description
    assert dataset_fingerprint(selected) == dataset_fingerprint(import_hotpot(path, limit=2))
    rows[-1]["supporting_facts"] = [["missing", 0]]
    with pytest.raises(ResearchError):
        import_hotpot(_write(path, rows), limit=2)


@pytest.mark.parametrize("limit", [0, 51, -1, True, "2", 1.0, None])
def test_hotpot_limit_is_strict(tmp_path: Path, limit: object) -> None:
    with pytest.raises(ResearchError):
        import_hotpot(_write(tmp_path / "data.json", [_row()]), limit=limit)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(extra="secret"),
        lambda r: r.pop("answer"),
        lambda r: r.update(answer=3),
        lambda r: r.update(answer=""),
        lambda r: r.update(question="x" * 1001),
        lambda r: r.update(_id=True),
        lambda r: r.update(type="other"),
        lambda r: r.update(level="unknown"),
        lambda r: r.update(context=[]),
        lambda r: r.update(context=r["context"] * 4),
        lambda r: r["context"].append(r["context"][0]),
        lambda r: r["context"][0].append("extra"),
        lambda r: r["context"][0].__setitem__(1, "not-a-list"),
        lambda r: r["context"][0].__setitem__(1, [1]),
        lambda r: r["context"][0].__setitem__(1, ["中" * 22000]),
        lambda r: r.update(supporting_facts=[]),
        lambda r: r.update(supporting_facts=[["missing", 0]]),
        lambda r: r.update(supporting_facts=[["Team Test", 2]]),
        lambda r: r.update(supporting_facts=[["Team Test", -1]]),
        lambda r: r.update(supporting_facts=[["Team Test", True]]),
        lambda r: r.update(supporting_facts=[["Team Test", "0"]]),
        lambda r: r.update(supporting_facts=[["Team Test", 0, "extra"]]),
        lambda r: r.update(supporting_facts=[["Team Test", 0], ["Team Test", 0]]),
    ],
)
def test_hotpot_invalid_rows_are_rejected(tmp_path: Path, mutation) -> None:
    row = _row()
    mutation(row)
    with pytest.raises(ResearchError) as caught:
        import_hotpot(_write(tmp_path / "authored-bad-fixture.json", [row]))
    assert caught.value.code == "benchmark_data"
    assert "secret" not in str(caught.value)


def test_hotpot_duplicate_ids_after_selected_prefix_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ResearchError):
        import_hotpot(_write(tmp_path / "duplicate.json", [_row("same"), _row("same")]), limit=1)


@pytest.mark.parametrize("value", [[], {}, None, "wrong", [None]])
def test_hotpot_requires_row_array(tmp_path: Path, value: object) -> None:
    with pytest.raises(ResearchError):
        import_hotpot(_write(tmp_path / "bad.json", value))


def test_hotpot_bounded_file_and_row_count(tmp_path: Path, monkeypatch) -> None:
    path = _write(tmp_path / "fixture.json", [_row("one"), _row("two")])
    monkeypatch.setattr(data_module, "MAX_HOTPOT_ROWS", 1)
    with pytest.raises(ResearchError):
        import_hotpot(path)
    monkeypatch.setattr(data_module, "MAX_HOTPOT_ROWS", 10000)
    monkeypatch.setattr(data_module, "MAX_HOTPOT_BYTES", 10)
    with pytest.raises(ResearchError):
        import_hotpot(path)


def test_hotpot_duplicate_json_keys_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-keys.json"
    path.write_text('[{"_id":"a","_id":"b"}]', encoding="utf-8")
    with pytest.raises(ResearchError):
        import_hotpot(path)


def test_hotpot_dataset_round_trips_and_no_network_is_needed(tmp_path: Path) -> None:
    dataset = import_hotpot(_write(tmp_path / "input.json", [_row()]))
    exported = tmp_path / "export.json"
    exported.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    restored = load_benchmark(exported)
    assert restored == dataset
    assert dataset_fingerprint(restored) == dataset_fingerprint(dataset)


def test_supporting_facts_need_original_sentence_metadata() -> None:
    case = load_benchmark().cases[0].model_dump(mode="json")
    case["gold"]["supporting_facts"] = [
        {"title": case["documents"][0]["title"], "sentence_index": 0}
    ]
    with pytest.raises(ValidationError):
        BenchmarkCase.model_validate_json(json.dumps(case))

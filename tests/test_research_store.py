"""Offline transactional persistence checks; no model or network access."""

import hashlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from deep_research.research_store import ResearchStore, ResearchStoreError


def source(source_id: str = "S1", text: str = "前言。研究需要证据。🧪结束。") -> dict:
    return {
        "source_id": source_id,
        "url": "https://example.com/article",
        "title": "研究资料",
        "text": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "retrieved_at": "2026-10-15T12:00:00+00:00",
        "truncated": False,
    }


def evidence(evidence_id: str = "E1", source_id: str = "S1") -> dict:
    return {
        "evidence_id": evidence_id,
        "source_id": source_id,
        "start": 3,
        "end": 10,
        "quote": "研究需要证据。",
    }


@pytest.fixture
def store(tmp_path: Path):
    with ResearchStore(tmp_path / "nested" / "research.sqlite3") as instance:
        yield instance


def test_create_and_read_run(store: ResearchStore) -> None:
    run_id = store.create_run("为什么需要来源引用？")
    assert str(UUID(run_id)) == run_id
    run = store.get_run(run_id)
    assert run is not None
    assert run["question"] == "为什么需要来源引用？"
    assert run["status"] == "running"
    assert datetime.fromisoformat(run["created_at"]).utcoffset() == timedelta(0)
    assert run["finished_at"] is None
    assert run["error_code"] is None
    assert run["usage"] == []
    assert store.get_run("missing") is None
    assert store.get_report(run_id) is None


def test_sources_and_unicode_character_offsets_round_trip(store: ResearchStore) -> None:
    run_id = store.create_run("研究证据")
    item = source()
    quote = evidence()
    emoji = {"evidence_id": "E2", "source_id": "S1", "start": 10, "end": 11, "quote": "🧪"}
    store.save_sources(run_id, [item], [quote, emoji])
    assert store.get_sources(run_id) == [item]
    assert store.get_evidence(run_id) == [quote, emoji]


def test_reopen_preserves_sources_report_and_usage(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "research.sqlite3"
    report = {"title": "结论", "claims": [{"text": "需要引用", "evidence_ids": ["E1"]}]}
    usage = [{"model": "test", "input_tokens": 10, "output_tokens": 5, "total_tokens": 15}]
    with ResearchStore(path) as store:
        run_id = store.create_run("为什么需要引用？")
        store.save_sources(run_id, [source()], [evidence()])
        store.complete_run(run_id, report, usage, "output/run-1")
    with ResearchStore(path) as reopened:
        assert reopened.get_sources(run_id) == [source()]
        assert reopened.get_evidence(run_id) == [evidence()]
        assert reopened.get_report(run_id) == report
        run = reopened.get_run(run_id)
        assert run is not None and run["status"] == "completed"
        assert run["usage"] == usage
        assert run["output_dir"] == "output/run-1"
        assert datetime.fromisoformat(run["finished_at"]).utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    "mutation",
    [
        {"sha256": "0" * 64},
        {"truncated": 1},
        {"retrieved_at": "2026-10-15T12:00:00"},
        {"text": "different content"},
    ],
)
def test_invalid_source_rolls_back_entire_batch(store: ResearchStore, mutation: dict) -> None:
    run_id = store.create_run("test")
    with pytest.raises(ResearchStoreError, match="校验"):
        store.save_sources(run_id, [source("S1"), source("S2") | mutation], [])
    assert store.get_sources(run_id) == []


@pytest.mark.parametrize(
    "mutation",
    [
        {"start": -1},
        {"end": 100},
        {"start": True},
        {"end": 3},
        {"quote": "不是原文"},
        {"source_id": "missing"},
        {"start": 9, "end": 30},  # UTF-8 byte positions must not be used as character offsets.
    ],
)
def test_invalid_evidence_rolls_back_sources_and_evidence(
    store: ResearchStore, mutation: dict
) -> None:
    run_id = store.create_run("test")
    with pytest.raises(ResearchStoreError):
        store.save_sources(run_id, [source()], [evidence(), evidence("E2") | mutation])
    assert store.get_sources(run_id) == []
    assert store.get_evidence(run_id) == []


def test_duplicate_source_does_not_overwrite_or_save_other_rows(store: ResearchStore) -> None:
    run_id = store.create_run("test")
    store.save_sources(run_id, [source()], [])
    with pytest.raises(ResearchStoreError) as error:
        store.save_sources(run_id, [source("S2"), source("S1", "替换的正文")], [])
    assert error.value.code == "store_conflict"
    assert store.get_sources(run_id) == [source()]


def test_duplicate_evidence_rolls_back_new_source(store: ResearchStore) -> None:
    run_id = store.create_run("test")
    store.save_sources(run_id, [source()], [evidence()])
    with pytest.raises(ResearchStoreError) as error:
        store.save_sources(run_id, [source("S2")], [evidence("E1", "S2")])
    assert error.value.code == "store_conflict"
    assert store.get_sources(run_id) == [source()]
    assert store.get_evidence(run_id) == [evidence()]


def test_evidence_can_reference_previously_saved_source_in_same_run(store: ResearchStore) -> None:
    run_id = store.create_run("test")
    store.save_sources(run_id, [source()], [])
    store.save_sources(run_id, [], [evidence()])
    assert store.get_evidence(run_id) == [evidence()]


def test_runs_are_isolated_and_cannot_reference_each_others_sources(store: ResearchStore) -> None:
    first, second = store.create_run("first"), store.create_run("second")
    store.save_sources(first, [source()], [evidence()])
    with pytest.raises(ResearchStoreError):
        store.save_sources(second, [], [evidence()])
    assert store.get_sources(second) == []
    assert store.get_evidence(second) == []
    # A source ID is local to a run, not a global identifier shared between runs.
    store.save_sources(second, [source()], [evidence()])
    assert store.get_sources(first) == store.get_sources(second) == [source()]


@pytest.mark.parametrize("terminal_status", ["completed", "failed", "cancelled"])
def test_terminal_runs_cannot_be_changed(store: ResearchStore, terminal_status: str) -> None:
    run_id = store.create_run("test")
    store.save_sources(run_id, [source()], [evidence()])
    if terminal_status == "completed":
        store.complete_run(run_id, {"title": "original"}, [], "output/run")
    else:
        store.fail_run(run_id, "initial_error", terminal_status)
    original_run, original_report = store.get_run(run_id), store.get_report(run_id)
    with pytest.raises(ResearchStoreError) as error:
        store.save_sources(run_id, [source("S2")], [])
    assert error.value.code == "store_terminal"
    with pytest.raises(ResearchStoreError):
        store.complete_run(run_id, {"title": "overwrite"}, [], "new-output")
    store.fail_run(run_id, "later_error", "failed")
    assert store.get_run(run_id) == original_run
    assert store.get_report(run_id) == original_report
    assert store.get_sources(run_id) == [source()]
    assert store.get_evidence(run_id) == [evidence()]


def test_bad_report_serialization_does_not_complete_run(store: ResearchStore) -> None:
    run_id = store.create_run("test")
    with pytest.raises(ResearchStoreError):
        store.complete_run(run_id, {"score": float("nan")}, [], "output/run")
    assert store.get_run(run_id)["status"] == "running"
    assert store.get_report(run_id) is None


def test_failure_stores_only_code_and_keeps_evidence(store: ResearchStore) -> None:
    run_id = store.create_run("test")
    store.save_sources(run_id, [source()], [evidence()])
    for code, status in [("raw error body with secret", "failed"), ("timeout", "running")]:
        with pytest.raises(ResearchStoreError):
            store.fail_run(run_id, code, status)
    store.fail_run(run_id, "timeout")
    assert store.get_run(run_id)["error_code"] == "timeout"
    assert store.get_sources(run_id) == [source()]
    assert store.get_evidence(run_id) == [evidence()]


def test_missing_run_cannot_receive_writes(store: ResearchStore) -> None:
    with pytest.raises(ResearchStoreError) as error:
        store.save_sources("missing", [source()], [])
    assert error.value.code == "store_not_found"
    with pytest.raises(ResearchStoreError):
        store.complete_run("missing", {}, [], "output/run")
    with pytest.raises(ResearchStoreError):
        store.fail_run("missing", "timeout")


def test_composite_foreign_key_is_enabled_on_store_connection(store: ResearchStore) -> None:
    first, second = store.create_run("first"), store.create_run("second")
    store.save_sources(first, [source()], [])
    connection = store._db()
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO research_evidence VALUES (?, ?, ?, ?, ?, ?)",
            (second, "E1", "S1", 0, 1, "x"),
        )


def test_parameterized_sql_preserves_question_as_data(store: ResearchStore) -> None:
    question = "'); DROP TABLE research_runs; --"
    run_id = store.create_run(question)
    assert store.get_run(run_id)["question"] == question
    assert store.get_run("' OR 1=1 --") is None
    assert store.get_run(store.create_run("still works"))["status"] == "running"


def test_close_is_idempotent_and_future_operations_fail(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path / "research.sqlite3")
    store.close()
    store.close()
    with pytest.raises(ResearchStoreError) as error:
        store.create_run("test")
    assert error.value.code == "store_closed"

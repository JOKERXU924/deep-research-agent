"""Bounded, restart-safe retrieval events with fail-closed damaged-data handling."""

import sqlite3

import pytest
from pydantic import ValidationError

from deep_research.dag_journal import DagJournal, DagJournalError
from deep_research.research_store import ResearchStore
from deep_research.retrieval_quality import RetrievalQualityOptions

EVENT = {
    "phase": "gap",
    "kind": "search",
    "status": "started",
    "request_id": "gap.search.1",
    "query": "光反应",
}


@pytest.fixture
def audit(tmp_path):
    path = tmp_path / "quality.sqlite"
    with ResearchStore(path) as store, DagJournal(path) as journal:
        run_id = store.create_run("光合作用与碳固定")
        journal.create_run(run_id, {})
        yield path, journal, run_id


def test_old_snapshot_unchanged_without_new_events(audit):
    _, journal, run_id = audit
    assert "retrieval_events" not in journal.snapshot(run_id)


def test_retrieval_events_survive_restart_with_order_and_request_id(audit):
    path, journal, run_id = audit
    journal.record_retrieval_event(run_id, EVENT)
    journal.record_retrieval_event(run_id, {**EVENT, "status": "completed"})
    before = journal.snapshot(run_id)["retrieval_events"]
    with DagJournal(path) as reopened:
        assert reopened.snapshot(run_id)["retrieval_events"] == before
    assert [item["seq"] for item in before] == [1, 2]
    assert before[0]["event"]["query"] == "光反应"
    assert all(item["created_at"] for item in before)


@pytest.mark.parametrize(
    "update",
    [
        {"password": "must not enter public audit"},
        {"status": "whatever"},
        {"score": float("nan")},
        {"query": "x" * 121},
        {"evidence_ids": ["E0"]},
        {"topic_terms": True},
        {"url": "x" * 2049},
        {"request_id": "unsafe\ncode"},
    ],
)
def test_invalid_event_never_partially_appends(audit, update):
    _, journal, run_id = audit
    with pytest.raises(DagJournalError):
        journal.record_retrieval_event(run_id, EVENT | update)
    assert "retrieval_events" not in journal.snapshot(run_id)


def test_unknown_run_and_sixty_fifth_event_rejected(audit):
    _, journal, run_id = audit
    with pytest.raises(DagJournalError):
        journal.record_retrieval_event("missing", EVENT)
    for _ in range(64):
        journal.record_retrieval_event(run_id, EVENT)
    with pytest.raises(DagJournalError):
        journal.record_retrieval_event(run_id, EVENT)
    assert len(journal.snapshot(run_id)["retrieval_events"]) == 64


@pytest.mark.parametrize(
    "sql,params",
    [
        ("UPDATE retrieval_events SET event_json=?", ['{"phase":"unknown"}']),
        ("UPDATE retrieval_events SET event_json=?", ["x" * 70000]),
        ("UPDATE retrieval_events SET created_at=?", ["invalid time"]),
        ("UPDATE retrieval_events SET seq=2", []),
    ],
)
def test_modified_database_is_revalidated_on_read(audit, sql, params):
    path, journal, run_id = audit
    journal.record_retrieval_event(run_id, EVENT)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute(sql, params)
    with pytest.raises(DagJournalError):
        journal.snapshot(run_id)


@pytest.mark.parametrize(
    "field,value",
    [
        ("gap_searches", 3),
        ("gap_fetches", 3),
        ("gap_timeout_seconds", 31),
        ("gap_timeout_seconds", 0),
        ("gap_timeout_seconds", float("inf")),
        ("filter_irrelevant", 1),
    ],
)
def test_configuration_cannot_expand_hard_limits(field, value):
    with pytest.raises(ValidationError):
        RetrievalQualityOptions(**{field: value})

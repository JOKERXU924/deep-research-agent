"""Offline persistence and append-only audit checks for M2."""

import hashlib
import sqlite3
from pathlib import Path

import pytest

import deep_research.dag_journal as module
from deep_research.dag_journal import DagJournal, DagJournalError
from deep_research.research_store import ResearchStore


def task(task_id="search", dependencies=None, max_attempts=2):
    return {
        "id": task_id,
        "dependencies": dependencies or [],
        "timeout_seconds": 5.0,
        "max_attempts": max_attempts,
    }


def event(seq, *, task_id="search", state="pending", attempt=0, error_code=None):
    return {
        "seq": seq,
        "task_id": task_id,
        "state": state,
        "attempt": attempt,
        "error_code": error_code,
        "timestamp": "2026-10-15T12:00:00+00:00",
    }


@pytest.fixture
def audit(tmp_path: Path):
    path = tmp_path / "nested" / "research.sqlite3"
    with ResearchStore(path) as store, DagJournal(path) as journal:
        run_id = store.create_run("Research the cited evidence.")
        journal.create_run(run_id, {"max_concurrency": 2})
        yield journal, store, run_id


def test_create_references_existing_m1_run_and_rejects_duplicates(audit):
    journal, _, run_id = audit
    with pytest.raises(DagJournalError) as caught:
        journal.create_run("missing-run", {})
    assert caught.value.code == "dag_audit_conflict"
    with pytest.raises(DagJournalError):
        journal.create_run(run_id, {"overwrite": True})
    snapshot = journal.snapshot(run_id)
    assert snapshot["config"] == {"max_concurrency": 2}
    assert snapshot["batches"] == []
    assert snapshot["summary"] is None
    assert journal.snapshot("missing-run") is None


def test_batch_metadata_and_continuous_events_update_latest_state(audit):
    journal, _, run_id = audit
    journal.begin_batch(run_id, 0, [task(), task("fetch", ["search"])])
    events = [
        event(1),
        event(2, state="ready"),
        event(3, state="running", attempt=1),
        event(4, state="retry_wait", attempt=1, error_code="retrieval_timeout"),
        event(5, state="running", attempt=2),
        event(6, state="succeeded", attempt=2),
    ]
    for item in events:
        journal.record_event(run_id, 0, item)
    batch = journal.snapshot(run_id)["batches"][0]
    assert batch["events"] == events
    first, second = batch["tasks"]
    assert first["id"] == "search"
    assert first["state"] == "succeeded" and first["attempt"] == 2
    assert first["error_code"] is None
    assert second["dependencies"] == ["search"]
    assert second["state"] == "pending" and second["attempt"] == 0


@pytest.mark.parametrize("seq", [0, 2, True])
def test_first_event_must_start_at_one(audit, seq):
    journal, _, run_id = audit
    journal.begin_batch(run_id, 0, [task()])
    with pytest.raises(DagJournalError):
        journal.record_event(run_id, 0, event(seq))
    snapshot = journal.snapshot(run_id)["batches"][0]
    assert snapshot["events"] == []
    assert snapshot["tasks"][0]["state"] == "pending"


def test_duplicate_and_gap_events_do_not_change_latest_state(audit):
    journal, _, run_id = audit
    journal.begin_batch(run_id, 0, [task()])
    journal.record_event(run_id, 0, event(1, state="running", attempt=1))
    for seq in (1, 3):
        with pytest.raises(DagJournalError) as caught:
            journal.record_event(run_id, 0, event(seq, state="succeeded", attempt=1))
        assert caught.value.code == "dag_audit_conflict"
    snapshot = journal.snapshot(run_id)["batches"][0]
    assert len(snapshot["events"]) == 1
    assert snapshot["tasks"][0]["state"] == "running"


@pytest.mark.parametrize(
    "changes",
    [
        {"task_id": "unregistered"},
        {"state": "invented"},
        {"attempt": 3},
        {"attempt": True},
        {"timestamp": "not-a-date"},
        {"timestamp": "2026-10-15T12:00:00"},
        {"error_code": "secret from raw error body"},
        {"extra": "raw model text"},
    ],
)
def test_bad_event_is_atomic_and_does_not_consume_sequence_number(audit, changes):
    journal, _, run_id = audit
    journal.begin_batch(run_id, 0, [task()])
    with pytest.raises(DagJournalError):
        journal.record_event(run_id, 0, event(1) | changes)
    journal.record_event(run_id, 0, event(1))
    assert journal.snapshot(run_id)["batches"][0]["events"] == [event(1)]


@pytest.mark.parametrize("batch", [-1, 2, True])
def test_only_two_batches_are_supported(audit, batch):
    journal, _, run_id = audit
    with pytest.raises(DagJournalError):
        journal.begin_batch(run_id, batch, [task()])


@pytest.mark.parametrize(
    "tasks",
    [
        [],
        [task(), task()],
        [task("fetch", ["unknown"])],
        [task("self", ["self"])],
        [task() | {"timeout_seconds": float("inf")}],
        [task() | {"max_attempts": True}],
        [task() | {"timeout_seconds": 10**1000}],
        [task() | {"dependencies": ["fetch", "fetch"]}, task("fetch")],
        [task(f"task-{index}") for index in range(129)],
    ],
)
def test_bad_batches_do_not_register_partial_tasks(audit, tasks):
    journal, _, run_id = audit
    with pytest.raises(DagJournalError):
        journal.begin_batch(run_id, 0, tasks)
    assert journal.snapshot(run_id)["batches"] == []
    journal.begin_batch(run_id, 0, [task()])


def test_batch_ids_and_event_sequences_are_isolated(audit):
    journal, store, first_run = audit
    second_run = store.create_run("second")
    journal.create_run(second_run, {})
    for run_id, batch in [(first_run, 0), (first_run, 1), (second_run, 0)]:
        journal.begin_batch(run_id, batch, [task()])
        journal.record_event(run_id, batch, event(1))
    with pytest.raises(DagJournalError):
        journal.begin_batch(first_run, 0, [task("different")])
    assert [item["batch"] for item in journal.snapshot(first_run)["batches"]] == [0, 1]
    assert len(journal.snapshot(second_run)["batches"]) == 1


def test_event_limit_applies_across_batches(audit, monkeypatch):
    journal, _, run_id = audit
    monkeypatch.setattr(module, "MAX_EVENTS", 2)
    journal.begin_batch(run_id, 0, [task()])
    journal.begin_batch(run_id, 1, [task()])
    journal.record_event(run_id, 0, event(1))
    journal.record_event(run_id, 0, event(2, state="ready"))
    with pytest.raises(DagJournalError) as caught:
        journal.record_event(run_id, 1, event(1))
    assert caught.value.code == "dag_audit_conflict"
    assert journal.snapshot(run_id)["batches"][1]["events"] == []


def test_final_summary_is_idempotent_but_cannot_be_overwritten(audit):
    journal, _, run_id = audit
    summary = {"degraded": True, "replan_count": 1, "reason": "deadline"}
    journal.save_summary(run_id, summary)
    journal.save_summary(run_id, dict(reversed(list(summary.items()))))
    with pytest.raises(DagJournalError):
        journal.save_summary(run_id, {"degraded": False})
    assert journal.snapshot(run_id)["summary"] == summary


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), object(), {1, 2}])
def test_non_json_config_and_summary_are_rejected(audit, bad_value):
    journal, store, run_id = audit
    new_run = store.create_run("new")
    with pytest.raises(DagJournalError) as caught:
        journal.create_run(new_run, {"value": bad_value})
    assert caught.value.code == "dag_audit_validation"
    assert journal.snapshot(new_run) is None
    with pytest.raises(DagJournalError):
        journal.save_summary(run_id, {"value": bad_value})
    assert journal.snapshot(run_id)["summary"] is None


def test_model_call_ledger_records_success_failure_and_cancellation(audit):
    journal, _, run_id = audit
    completed = journal.start_model_call(run_id, "plan")
    details = {
        "provider": "fake",
        "model": "offline",
        "request_id": "request-1",
        "latency_ms": 12.5,
        "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
    }
    journal.finish_model_call(run_id, completed, "completed", details)
    failed = journal.start_model_call(run_id, "replan")
    journal.finish_model_call(run_id, failed, "failed", {"error_code": "rate_limit"})
    cancelled = journal.start_model_call(run_id, "report")
    journal.finish_model_call(run_id, cancelled, "cancelled", {"error_code": "cancelled"})
    calls = journal.snapshot(run_id)["model_calls"]
    assert [item["call_id"] for item in calls] == [1, 2, 3]
    assert [item["status"] for item in calls] == ["completed", "failed", "cancelled"]
    assert calls[0]["details"] == details
    assert calls[1]["details"] == {"error_code": "rate_limit"}
    assert all(item["finished_at"] is not None for item in calls)
    with pytest.raises(DagJournalError) as caught:
        journal.start_model_call(run_id, "another")
    assert caught.value.code == "dag_audit_conflict"


def test_model_call_finish_is_exactly_once(audit):
    journal, _, run_id = audit
    call_id = journal.start_model_call(run_id, "plan")
    journal.finish_model_call(run_id, call_id, "failed", {"error_code": "timeout"})
    with pytest.raises(DagJournalError) as caught:
        journal.finish_model_call(run_id, call_id, "completed", {})
    assert caught.value.code == "dag_audit_conflict"
    with pytest.raises(DagJournalError):
        journal.finish_model_call(run_id, 2, "completed", {})
    assert journal.snapshot(run_id)["model_calls"][0]["status"] == "failed"


@pytest.mark.parametrize(
    "details",
    [
        {"content": "raw model output"},
        {"prompt": "raw prompt"},
        {"no_change": "true"},
        {"no_change": 1},
        {"error_code": "raw upstream error with secret"},
        {"latency_ms": float("nan")},
        {"latency_ms": True},
        {"provider": "provider\nraw"},
        {"latency_ms": 10**1000},
        {"usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 4}},
        {"usage": {"input_tokens": True, "output_tokens": 2, "total_tokens": 3}},
        {"usage": {"raw": "upstream payload"}},
    ],
)
def test_bad_model_details_do_not_finish_or_expose_raw_payload(audit, details):
    journal, _, run_id = audit
    call_id = journal.start_model_call(run_id, "plan")
    with pytest.raises(DagJournalError) as caught:
        journal.finish_model_call(run_id, call_id, "completed", details)
    assert caught.value.code == "dag_audit_validation"
    assert "raw" not in str(caught.value) and "secret" not in str(caught.value)
    call = journal.snapshot(run_id)["model_calls"][0]
    assert call["status"] == "in_flight" and call["details"] == {}


def test_same_database_preserves_m1_report_and_reopening_preserves_unknown_usage(tmp_path: Path):
    path = tmp_path / "research.sqlite3"
    text = "Original saved evidence."
    source = {
        "source_id": "S1",
        "url": "https://example.org/",
        "title": "Original",
        "text": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "retrieved_at": "2026-10-15T00:00:00+00:00",
        "truncated": False,
    }
    with ResearchStore(path) as store:
        run_id = store.create_run("question")
        store.save_sources(run_id, [source], [])
        store.complete_run(run_id, {"title": "M1 report"}, [], "outputs/original")
        original = store.get_run(run_id)
    with DagJournal(path) as journal:
        journal.create_run(run_id, {"run_timeout_seconds": 90})
        journal.begin_batch(run_id, 0, [task()])
        journal.record_event(run_id, 0, event(1, state="running", attempt=1))
        journal.start_model_call(run_id, "plan")
    with DagJournal(path) as journal, ResearchStore(path) as store:
        snapshot = journal.snapshot(run_id)
        call = snapshot["model_calls"][0]
        assert call["status"] == "in_flight"
        assert call["finished_at"] is None and call["details"] == {}
        assert snapshot["batches"][0]["tasks"][0]["state"] == "running"
        assert store.get_run(run_id) == original
        assert store.get_sources(run_id) == [source]
        assert store.get_report(run_id) == {"title": "M1 report"}


def test_composite_foreign_keys_and_atomic_event_state_write(audit):
    journal, _, run_id = audit
    journal.begin_batch(run_id, 0, [task()])
    connection = journal._db()
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO dag_events VALUES (?, 0, 1, 'missing', 'running', 1, NULL, ?)",
            (run_id, "2026-10-15T00:00:00+00:00"),
        )
    connection.execute(
        "CREATE TRIGGER block_task_state BEFORE UPDATE ON dag_tasks "
        "BEGIN SELECT RAISE(ABORT, 'synthetic secret db body'); END"
    )
    with pytest.raises(DagJournalError) as caught:
        journal.record_event(run_id, 0, event(1, state="running", attempt=1))
    assert "secret" not in str(caught.value)
    snapshot = journal.snapshot(run_id)["batches"][0]
    assert snapshot["events"] == []
    assert snapshot["tasks"][0]["state"] == "pending"


def test_closed_connection_reports_sanitized_error(audit):
    journal, _, run_id = audit
    journal.close()
    journal.close()
    with pytest.raises(DagJournalError) as caught:
        journal.snapshot(run_id)
    assert caught.value.code == "dag_audit_io"

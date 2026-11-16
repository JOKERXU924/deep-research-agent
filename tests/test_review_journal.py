"""M5 append-only review storage: no models, network, or real research data."""

import json
import sqlite3
from datetime import datetime, timedelta

import pytest

from deep_research.research_store import ResearchStore
from deep_research.review_journal import (
    MAX_EVENTS,
    MAX_JSON_BYTES,
    ReviewJournal,
    ReviewJournalError,
)


@pytest.fixture
def storage(tmp_path):
    path = tmp_path / "nested" / "review.sqlite"
    with ResearchStore(path) as store, ReviewJournal(path) as journal:
        run_id = store.create_run("Review the original evidence.")
        yield store, journal, run_id, path


def initial(**changes):
    return {"kind": "initial", "round": 0, "report_hash": "a" * 64, **changes}


def test_new_and_missing_runs_are_distinct_and_foreign_keys_are_enforced(storage):
    _, journal, run_id, _ = storage
    assert journal.snapshot(run_id) == []
    for operation in (
        lambda: journal.snapshot("missing"),
        lambda: journal.append("missing", initial()),
    ):
        with pytest.raises(ReviewJournalError) as caught:
            operation()
        assert caught.value.code == "review_not_found"
    assert journal._db().execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        journal._db().execute(
            "INSERT INTO review_events VALUES (?, ?, ?, ?)",
            ("missing", 1, '{"kind":"stop"}', "2026-10-16T00:00:00+00:00"),
        )
    assert journal.snapshot(run_id) == []


def test_events_are_ordered_detached_and_survive_reopening(storage):
    _, journal, run_id, path = storage
    events = [
        initial(report={"title": "原文😀", "nested": [True, None, 0, 0.5]}),
        {"kind": "red", "round": 1, "findings": []},
        {"kind": "blue", "round": 1, "patches": [{"operation": "VERIFY"}]},
        {"kind": "stop", "round": 2, "reason": "round_limit", "report_hash": "F" * 64},
    ]
    expected = json.loads(json.dumps(events, ensure_ascii=False))
    for event in events:
        assert journal.append(run_id, event) is None
    events[0]["report"]["nested"].append("caller mutation")
    snapshot = journal.snapshot(run_id)
    assert [item["seq"] for item in snapshot] == [1, 2, 3, 4]
    assert [item["event"] for item in snapshot] == expected
    assert all(
        datetime.fromisoformat(item["created_at"]).utcoffset() == timedelta(0) for item in snapshot
    )
    snapshot[0]["event"]["report"]["title"] = "changed returned snapshot"
    with ReviewJournal(path) as reopened:
        assert [item["event"] for item in reopened.snapshot(run_id)] == expected


def test_duplicate_payloads_do_not_overwrite_and_two_connections_share_sequence(storage):
    store, journal, first, path = storage
    second = store.create_run("A separate run.")
    payload = initial(seq=999)  # User payload metadata cannot replace the journal sequence.
    journal.append(first, payload)
    with ReviewJournal(path) as another:
        another.append(first, payload)
        another.append(second, {"kind": "stop"})
    assert [item["seq"] for item in journal.snapshot(first)] == [1, 2]
    assert [item["event"] for item in journal.snapshot(first)] == [payload, payload]
    assert [item["seq"] for item in journal.snapshot(second)] == [1]
    with pytest.raises(sqlite3.IntegrityError):
        journal._db().execute(
            "INSERT INTO review_events VALUES (?, ?, ?, ?)",
            (first, 1, '{"kind":"stop"}', "2026-10-16T00:00:00+00:00"),
        )
    assert len(journal.snapshot(first)) == 2


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_terminal_runs_are_readable_but_no_longer_writable(storage, status):
    store, journal, run_id, _ = storage
    journal.append(run_id, initial())
    before = journal.snapshot(run_id)
    if status == "completed":
        store.complete_run(run_id, {"fixture": True}, [], "fixture/output")
    else:
        store.fail_run(run_id, "fixture_stop", status=status)
    with pytest.raises(ReviewJournalError) as caught:
        journal.append(run_id, {"kind": "stop", "round": 1})
    assert caught.value.code == "review_audit_conflict"
    assert journal.snapshot(run_id) == before


@pytest.mark.parametrize(
    "event",
    [
        None,
        [],
        {},
        {"kind": []},
        {"kind": "unknown"},
        initial(round=True),
        initial(round=-1),
        initial(round=3),
        initial(round=1.0),
        initial(report_hash="a" * 63),
        initial(report_hash="g" * 64),
        initial(report_hash=None),
        initial(payload=float("nan")),
        initial(payload=float("inf")),
        initial(payload={1: "coercion"}),
        initial(payload=("not", "a", "list")),
        initial(payload={"set"}),
        initial(payload=object()),
        initial(payload="\ud800"),
    ],
)
def test_bad_events_are_sanitized_and_do_not_consume_sequence(storage, event):
    _, journal, run_id, _ = storage
    with pytest.raises(ReviewJournalError) as caught:
        journal.append(run_id, event)
    assert caught.value.code == "review_audit_validation"
    assert journal.snapshot(run_id) == []
    journal.append(run_id, initial())
    assert journal.snapshot(run_id)[0]["seq"] == 1


def test_cycles_and_overly_deep_json_are_rejected_without_recursion_failure(storage):
    _, journal, run_id, _ = storage
    cyclic = []
    cyclic.append(cyclic)
    deep = []
    for _ in range(35):
        deep = [deep]
    for value in (cyclic, deep):
        with pytest.raises(ReviewJournalError) as caught:
            journal.append(run_id, initial(payload=value))
        assert caught.value.code == "review_audit_validation"
    assert journal.snapshot(run_id) == []


def test_exact_utf8_json_byte_boundary_is_allowed_but_one_more_byte_is_atomic(storage):
    _, journal, run_id, _ = storage
    base = {"kind": "stop", "message": ""}
    available = MAX_JSON_BYTES - len(json.dumps(base, separators=(",", ":")).encode("utf-8"))
    event = base | {"message": "汉" * (available // 3) + "a" * (available % 3)}
    assert (
        len(json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode()) == MAX_JSON_BYTES
    )
    journal.append(run_id, event)
    with pytest.raises(ReviewJournalError) as caught:
        journal.append(run_id, event | {"message": event["message"] + "a"})
    assert caught.value.code == "review_audit_validation"
    assert [item["event"] for item in journal.snapshot(run_id)] == [event]


def test_event_cap_is_per_run_and_never_discards_earlier_audit(storage):
    store, journal, run_id, _ = storage
    for _ in range(MAX_EVENTS):
        journal.append(run_id, initial())
    with pytest.raises(ReviewJournalError) as caught:
        journal.append(run_id, {"kind": "stop"})
    assert caught.value.code == "review_audit_conflict"
    assert [item["seq"] for item in journal.snapshot(run_id)] == list(range(1, MAX_EVENTS + 1))
    another = store.create_run("Independent capacity.")
    journal.append(another, {"kind": "stop"})
    assert len(journal.snapshot(another)) == 1


def test_database_failure_after_insert_rolls_back_and_hides_native_error(storage):
    _, journal, run_id, _ = storage
    journal._db().executescript("""
        CREATE TRIGGER reject_review_insert AFTER INSERT ON review_events
        BEGIN SELECT RAISE(FAIL, 'private-upstream-payload'); END;
    """)
    with pytest.raises(ReviewJournalError) as caught:
        journal.append(run_id, initial())
    assert caught.value.code == "review_audit_conflict"
    assert "private-upstream-payload" not in str(caught.value)
    assert journal.snapshot(run_id) == []
    journal._db().execute("DROP TRIGGER reject_review_insert")
    journal.append(run_id, initial())
    assert journal.snapshot(run_id)[0]["seq"] == 1


@pytest.mark.parametrize(
    "raw",
    [
        "{not JSON",
        "null",
        "[]",
        '{"kind":"unknown"}',
        '{"kind":"red","round":true}',
        '{"kind":"stop","value":NaN}',
        '{"kind":"stop","value":1e999}',
        '{"kind":"initial","kind":"stop"}',
        '{"kind":"stop","nested":{"x":1,"x":2}}',
        '{"kind":"stop","report_hash":"wrong"}',
    ],
)
def test_corrupt_json_is_not_trusted_on_read_or_subsequent_append(storage, raw):
    _, journal, run_id, _ = storage
    journal.append(run_id, initial())
    journal._db().execute("UPDATE review_events SET event_json = ? WHERE run_id = ?", (raw, run_id))
    for operation in (lambda: journal.snapshot(run_id), lambda: journal.append(run_id, initial())):
        with pytest.raises(ReviewJournalError) as caught:
            operation()
        assert caught.value.code == "review_audit_validation"
        assert raw not in str(caught.value)
    assert (
        journal._db()
        .execute("SELECT COUNT(*) FROM review_events WHERE run_id = ?", (run_id,))
        .fetchone()[0]
        == 1
    )


@pytest.mark.parametrize(
    "change",
    [
        {"seq": 2},
        {"created_at": "not-a-date"},
        {"created_at": "2026-10-16T12:00:00"},
        {"created_at": "2026-10-16T12:00:00+08:00"},
    ],
)
def test_corrupt_sequence_and_timestamp_are_rejected(storage, change):
    _, journal, run_id, _ = storage
    journal.append(run_id, initial())
    if "seq" in change:
        journal._db().execute(
            "UPDATE review_events SET seq = ? WHERE run_id = ?", (change["seq"], run_id)
        )
    else:
        journal._db().execute(
            "UPDATE review_events SET created_at = ? WHERE run_id = ?",
            (change["created_at"], run_id),
        )
    with pytest.raises(ReviewJournalError) as caught:
        journal.snapshot(run_id)
    assert caught.value.code == "review_audit_validation"


def test_oversized_corrupt_database_value_is_rejected_before_json_parsing(storage):
    _, journal, run_id, _ = storage
    journal.append(run_id, initial())
    journal._db().execute("PRAGMA ignore_check_constraints = ON")
    journal._db().execute(
        "UPDATE review_events SET event_json = ? WHERE run_id = ?",
        ('{"kind":"stop","body":"' + "a" * MAX_JSON_BYTES + '"}', run_id),
    )
    with pytest.raises(ReviewJournalError) as caught:
        journal.snapshot(run_id)
    assert caught.value.code == "review_audit_validation"


def test_closed_journal_rejects_reads_writes_and_reentry_but_close_is_idempotent(storage):
    _, journal, run_id, _ = storage
    journal.append(run_id, initial())
    journal.close()
    journal.close()
    for operation in (
        lambda: journal.append(run_id, initial()),
        lambda: journal.snapshot(run_id),
        journal.__enter__,
    ):
        with pytest.raises(ReviewJournalError) as caught:
            operation()
        assert caught.value.code == "review_audit_closed"


def test_sql_and_open_failures_are_sanitized(storage, tmp_path):
    _, journal, run_id, _ = storage
    journal._db().execute("DROP TABLE review_events")
    for operation in (lambda: journal.append(run_id, initial()), lambda: journal.snapshot(run_id)):
        with pytest.raises(ReviewJournalError) as caught:
            operation()
        assert caught.value.code == "review_audit_io"
        assert "review_events" not in str(caught.value)
    with pytest.raises(ReviewJournalError) as caught:
        ReviewJournal(tmp_path)
    assert caught.value.code == "review_audit_io"
    assert str(tmp_path) not in str(caught.value)

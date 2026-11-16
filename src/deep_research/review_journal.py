"""Bounded append-only review audit in the research database, with strict reads."""

import json
import math
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from deep_research.errors import ResearchError

MAX_EVENTS = 32
MAX_JSON_BYTES = 64 * 1024
MAX_JSON_DEPTH = 32
_KINDS = frozenset({"initial", "red", "blue", "evidence_gap", "stop"})
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_HASH = re.compile(r"[0-9a-fA-F]{64}\Z")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_events (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL CHECK (seq BETWEEN 1 AND 32),
    event_json TEXT NOT NULL CHECK (
        typeof(event_json) = 'text' AND length(CAST(event_json AS BLOB)) <= 65536
    ),
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, seq),
    FOREIGN KEY (run_id) REFERENCES research_runs(id)
);
"""


class ReviewJournalError(ResearchError):
    """Stable errors that never expose database paths, SQL, or review payloads."""


def _invalid() -> ReviewJournalError:
    return ReviewJournalError("review_audit_validation", "审查审计数据校验失败，未保存本次修改。")


def _conflict() -> ReviewJournalError:
    return ReviewJournalError("review_audit_conflict", "审查任务已结束、记录冲突或超过审计上限。")


def _identifier(value: object) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise _invalid()
    return value


def _json_values(value: object) -> None:
    """Reject coercions, cycles, non-finite numbers and excessive nesting before encoding."""
    pending = [(value, 0)]
    visited = 0
    while pending:
        item, depth = pending.pop()
        visited += 1
        if depth > MAX_JSON_DEPTH or visited > MAX_JSON_BYTES:
            raise _invalid()
        if item is None or type(item) in (bool, int):
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise _invalid()
        elif type(item) is str:
            if len(item.encode("utf-8")) > MAX_JSON_BYTES:
                raise _invalid()
        elif type(item) is list:
            if len(item) > MAX_JSON_BYTES:
                raise _invalid()
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is dict:
            if len(item) > MAX_JSON_BYTES or any(type(key) is not str for key in item):
                raise _invalid()
            pending.extend((key, depth + 1) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        else:
            raise _invalid()


def _encode_event(event: object) -> str:
    if (
        not isinstance(event, dict)
        or not isinstance(event.get("kind"), str)
        or event["kind"] not in _KINDS
    ):
        raise _invalid()
    if "round" in event and (type(event["round"]) is not int or not 0 <= event["round"] <= 2):
        raise _invalid()
    if "report_hash" in event and (
        not isinstance(event["report_hash"], str) or _HASH.fullmatch(event["report_hash"]) is None
    ):
        raise _invalid()
    try:
        _json_values(event)
        encoded = json.dumps(
            event, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
            raise _invalid()
        return encoded
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise _invalid() from None


def _decode_event(raw: object) -> dict:
    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        raise ValueError

    if not isinstance(raw, str):
        raise _invalid()
    try:
        if len(raw.encode("utf-8")) > MAX_JSON_BYTES:
            raise _invalid()
        event = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
        _encode_event(event)
        return event
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise _invalid() from None


def _timestamp(value: object) -> str:
    if not isinstance(value, str) or len(value) > 64:
        raise _invalid()
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        raise _invalid() from None
    if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
        raise _invalid()
    return value


class ReviewJournal:
    """One owning-thread connection; audit reads never resume a review.

    Each run has at most 32 events of 64 KiB, so stored event JSON is at most
    2 MiB per run. Ordering semantics belong to the engine; identical payloads
    are separate append attempts, never an overwrite or an idempotent replay.
    """

    def __init__(self, path: Path) -> None:
        self._connection: sqlite3.Connection | None = None
        self._closed = False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(path, isolation_level=None)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.executescript(_SCHEMA)
        except (OSError, sqlite3.Error):
            if self._connection is not None:
                self._connection.close()
            self._closed = True
            raise ReviewJournalError("review_audit_io", "无法打开审查审计数据库。") from None

    def __enter__(self) -> "ReviewJournal":
        self._db()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            if self._connection is not None:
                self._connection.close()
            self._closed = True

    def _db(self) -> sqlite3.Connection:
        if self._closed or self._connection is None:
            raise ReviewJournalError("review_audit_closed", "审查审计数据库已关闭。")
        return self._connection

    @contextmanager
    def _transaction(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self._db()
        try:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield connection
                connection.commit()
            except BaseException:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass  # Cleanup must not replace the original exception or cancellation.
                raise
        except sqlite3.IntegrityError:
            raise _conflict() from None
        except sqlite3.Error:
            raise ReviewJournalError("review_audit_io", "审查审计数据库操作失败。") from None

    @staticmethod
    def _require_run(connection: sqlite3.Connection, run_id: str, *, running: bool = False) -> None:
        row = connection.execute(
            "SELECT status FROM research_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise ReviewJournalError("review_not_found", "研究任务不存在。")
        if row["status"] not in ("running", "completed", "failed", "cancelled"):
            raise _invalid()
        if running and row["status"] != "running":
            raise _conflict()

    @staticmethod
    def _events(connection: sqlite3.Connection, run_id: str) -> list[dict]:
        # Do not materialize an arbitrarily large value if a database was damaged
        # outside this API. One extra row detects violations of the total cap.
        rows = connection.execute(
            "SELECT seq, created_at, CASE WHEN length(CAST(event_json AS BLOB)) <= ? "
            "THEN event_json ELSE NULL END AS event_json FROM review_events "
            "WHERE run_id = ? ORDER BY seq LIMIT ?",
            (MAX_JSON_BYTES, run_id, MAX_EVENTS + 1),
        ).fetchall()
        if len(rows) > MAX_EVENTS:
            raise _invalid()
        result = []
        for expected, row in enumerate(rows, start=1):
            if type(row["seq"]) is not int or row["seq"] != expected:
                raise _invalid()
            result.append(
                {
                    "seq": expected,
                    "event": _decode_event(row["event_json"]),
                    "created_at": _timestamp(row["created_at"]),
                }
            )
        return result

    def append(self, run_id: str, event: dict) -> None:
        run_id = _identifier(run_id)
        encoded = _encode_event(event)
        with self._transaction(write=True) as connection:
            self._require_run(connection, run_id, running=True)
            previous = self._events(connection, run_id)
            if len(previous) >= MAX_EVENTS:
                raise _conflict()
            connection.execute(
                "INSERT INTO review_events (run_id, seq, event_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (run_id, len(previous) + 1, encoded, datetime.now(UTC).isoformat()),
            )

    def snapshot(self, run_id: str) -> list[dict]:
        run_id = _identifier(run_id)
        with self._transaction() as connection:
            self._require_run(connection, run_id)
            return self._events(connection, run_id)

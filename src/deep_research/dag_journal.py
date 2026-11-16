"""Persistent scheduling and model-call audit, independent of M1 source storage."""

import json
import math
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deep_research.errors import ResearchError
from deep_research.retrieval_quality import validate_retrieval_event
from deep_research.structured_diagnostics import validate_diagnostic

MAX_TASKS = 128
MAX_EVENTS = 4096
MAX_MODEL_CALLS = 3
MAX_REVIEW_MODEL_CALLS = 9
MAX_JSON_BYTES = 64 * 1024
TASK_STATES = frozenset(
    {
        "pending",
        "ready",
        "running",
        "retry_wait",
        "succeeded",
        "failed",
        "timed_out",
        "skipped",
        "cancelled",
    }
)
_CODE = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_DETAIL_KEYS = {
    "provider",
    "model",
    "request_id",
    "latency_ms",
    "usage",
    "error_code",
    "reservation",
    "validation",
    "no_change",
}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS dag_runs (
    run_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    summary_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES research_runs(id)
);
CREATE TABLE IF NOT EXISTS dag_batches (
    run_id TEXT NOT NULL,
    batch INTEGER NOT NULL CHECK (batch IN (0, 1)),
    PRIMARY KEY (run_id, batch),
    FOREIGN KEY (run_id) REFERENCES dag_runs(run_id)
);
CREATE TABLE IF NOT EXISTS dag_tasks (
    run_id TEXT NOT NULL,
    batch INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    dependencies_json TEXT NOT NULL,
    timeout_seconds REAL NOT NULL,
    max_attempts INTEGER NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    attempt INTEGER NOT NULL DEFAULT 0,
    error_code TEXT,
    updated_at TEXT,
    PRIMARY KEY (run_id, batch, task_id),
    FOREIGN KEY (run_id, batch) REFERENCES dag_batches(run_id, batch)
);
CREATE TABLE IF NOT EXISTS dag_events (
    run_id TEXT NOT NULL,
    batch INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    state TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    error_code TEXT,
    timestamp TEXT NOT NULL,
    PRIMARY KEY (run_id, batch, seq),
    FOREIGN KEY (run_id, batch, task_id) REFERENCES dag_tasks(run_id, batch, task_id)
);
CREATE TABLE IF NOT EXISTS dag_model_calls (
    run_id TEXT NOT NULL,
    call_id INTEGER NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('in_flight', 'completed', 'failed', 'cancelled')),
    details_json TEXT NOT NULL DEFAULT '{}',
    started_at TEXT NOT NULL,
    finished_at TEXT,
    PRIMARY KEY (run_id, call_id),
    FOREIGN KEY (run_id) REFERENCES dag_runs(run_id)
);
CREATE TABLE IF NOT EXISTS retrieval_events (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL CHECK (seq BETWEEN 1 AND 64),
    event_json TEXT NOT NULL CHECK (length(CAST(event_json AS BLOB)) <= 65536),
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, seq),
    FOREIGN KEY (run_id) REFERENCES dag_runs(run_id)
);
"""


class DagJournalError(ResearchError):
    """Public audit error; raw database exceptions and payloads are never exposed."""


def _invalid() -> DagJournalError:
    return DagJournalError("dag_audit_validation", "调度审计数据校验失败，未保存本次修改。")


def _conflict() -> DagJournalError:
    return DagJournalError("dag_audit_conflict", "调度审计记录重复、顺序冲突或超过限制。")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _identifier(value: object) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise _invalid()
    return value


def _code(value: object) -> str | None:
    if value is not None and (not isinstance(value, str) or _CODE.fullmatch(value) is None):
        raise _invalid()
    return value


def _batch(value: object) -> int:
    if type(value) is not int or value not in (0, 1):
        raise _invalid()
    return value


def _json(value: object) -> str:
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
            raise ValueError
        return encoded
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise _invalid() from None


def _model_call_limit(config: dict) -> int:
    limit = config.get("model_call_limit", MAX_MODEL_CALLS)
    if type(limit) is not int or not MAX_MODEL_CALLS <= limit <= MAX_REVIEW_MODEL_CALLS:
        raise _invalid()
    return limit


def _timestamp(value: object) -> str:
    if not isinstance(value, str) or len(value) > 64:
        raise _invalid()
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        raise _invalid() from None
    if timestamp.tzinfo is None:
        raise _invalid()
    return value


def _finite_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _details(details: object) -> str:
    if not isinstance(details, dict) or not set(details) <= _DETAIL_KEYS:
        raise _invalid()
    if "no_change" in details and type(details["no_change"]) is not bool:
        raise _invalid()
    for field in ("provider", "model", "request_id"):
        if field in details:
            value = details[field]
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 256
                or any(ord(char) < 32 or ord(char) == 127 for char in value)
            ):
                raise _invalid()
    if "latency_ms" in details:
        latency = details["latency_ms"]
        if not _finite_number(latency) or latency < 0:
            raise _invalid()
    if "error_code" in details:
        _code(details["error_code"])
    if "validation" in details:
        try:
            if details.get("error_code") != "schema_mismatch":
                raise ValueError
            validate_diagnostic(details["validation"])
        except ValueError:
            raise _invalid() from None
    if "reservation" in details:
        reservation = details["reservation"]
        if (
            not isinstance(reservation, dict)
            or set(reservation) != {"input_bytes", "output_tokens"}
            or type(reservation["input_bytes"]) is not int
            or not 1 <= reservation["input_bytes"] <= 65536
            or type(reservation["output_tokens"]) is not int
            or not 1 <= reservation["output_tokens"] <= 4096
        ):
            raise _invalid()
    usage = details.get("usage")
    if usage is not None:
        if not isinstance(usage, dict) or set(usage) != {
            "input_tokens",
            "output_tokens",
            "total_tokens",
        }:
            raise _invalid()
        if any(type(value) is not int or value < 0 for value in usage.values()):
            raise _invalid()
        if usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"]:
            raise _invalid()
    return _json(details)


class DagJournal:
    """Append-only event/call audit on its own connection in the M1 database.

    Scheduler state-transition rules belong to the scheduler; this layer enforces
    registered tasks, continuous per-batch sequence numbers and bounded storage.
    Reopening never resumes work or invents usage for in-flight model calls.
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
            raise DagJournalError("dag_audit_io", "无法打开调度审计数据库。") from None

    def __enter__(self) -> "DagJournal":
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
            raise DagJournalError("dag_audit_io", "调度审计数据库已关闭。")
        return self._connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._db()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        except sqlite3.IntegrityError:
            raise _conflict() from None
        except sqlite3.Error:
            raise DagJournalError("dag_audit_io", "调度审计保存失败。") from None

    @staticmethod
    def _require_run(connection: sqlite3.Connection, run_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM dag_runs WHERE run_id = ?", (_identifier(run_id),)
        ).fetchone()
        if row is None:
            raise _invalid()
        return row

    def create_run(self, run_id: str, config: dict) -> None:
        run_id = _identifier(run_id)
        if not isinstance(config, dict):
            raise _invalid()
        config = dict(config)
        _model_call_limit(config)
        if "model_budget" in config or "model_profile" in config:
            from deep_research.model_budget import ModelBudgetLimits, validate_profile_snapshot

            try:
                if "model_budget" in config:
                    config["model_budget"] = ModelBudgetLimits.model_validate(
                        config["model_budget"]
                    ).model_dump()
                if "model_profile" in config:
                    config["model_profile"] = validate_profile_snapshot(config["model_profile"])
            except (ValueError, ResearchError):
                raise _invalid() from None
        config_json = _json(config)
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO dag_runs (run_id, config_json, created_at) VALUES (?, ?, ?)",
                (run_id, config_json, _now()),
            )

    def begin_batch(self, run_id: str, batch: int, tasks: list[dict]) -> None:
        batch = _batch(batch)
        if not isinstance(tasks, list) or not 1 <= len(tasks) <= MAX_TASKS:
            raise _invalid()
        ids: set[str] = set()
        rows: list[tuple] = []
        for ordinal, task in enumerate(tasks):
            if not isinstance(task, dict) or set(task) != {
                "id",
                "dependencies",
                "timeout_seconds",
                "max_attempts",
            }:
                raise _invalid()
            task_id = _identifier(task["id"])
            dependencies = task["dependencies"]
            if not isinstance(dependencies, list) or len(dependencies) > MAX_TASKS:
                raise _invalid()
            dependency_ids = [_identifier(value) for value in dependencies]
            if len(set(dependency_ids)) != len(dependency_ids) or task_id in dependency_ids:
                raise _invalid()
            if task_id in ids:
                raise _conflict()
            ids.add(task_id)
            timeout, attempts = task["timeout_seconds"], task["max_attempts"]
            if not _finite_number(timeout) or timeout <= 0:
                raise _invalid()
            if type(attempts) is not int or not 1 <= attempts <= MAX_EVENTS:
                raise _invalid()
            rows.append(
                (run_id, batch, task_id, ordinal, _json(dependency_ids), float(timeout), attempts)
            )
        if any(dependency not in ids for task in tasks for dependency in task["dependencies"]):
            raise _invalid()
        with self._transaction() as connection:
            self._require_run(connection, run_id)
            connection.execute("INSERT INTO dag_batches VALUES (?, ?)", (run_id, batch))
            connection.executemany(
                "INSERT INTO dag_tasks "
                "(run_id, batch, task_id, ordinal, dependencies_json, "
                "timeout_seconds, max_attempts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def record_event(self, run_id: str, batch: int, event: dict) -> None:
        batch = _batch(batch)
        if not isinstance(event, dict) or set(event) != {
            "seq",
            "task_id",
            "state",
            "attempt",
            "error_code",
            "timestamp",
        }:
            raise _invalid()
        task_id = _identifier(event["task_id"])
        state, seq, attempt = event["state"], event["seq"], event["attempt"]
        if not isinstance(state, str) or state not in TASK_STATES:
            raise _invalid()
        if type(seq) is not int or not 1 <= seq <= MAX_EVENTS:
            raise _invalid()
        if type(attempt) is not int or attempt < 0:
            raise _invalid()
        error_code, timestamp = _code(event["error_code"]), _timestamp(event["timestamp"])
        with self._transaction() as connection:
            self._require_run(connection, run_id)
            task = connection.execute(
                "SELECT max_attempts FROM dag_tasks WHERE run_id = ? AND batch = ? AND task_id = ?",
                (run_id, batch, task_id),
            ).fetchone()
            if task is None or attempt > task["max_attempts"]:
                raise _invalid()
            last = connection.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM dag_events WHERE run_id = ? AND batch = ?",
                (run_id, batch),
            ).fetchone()[0]
            count = connection.execute(
                "SELECT COUNT(*) FROM dag_events WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            if seq != last + 1 or count >= MAX_EVENTS:
                raise _conflict()
            connection.execute(
                "INSERT INTO dag_events "
                "(run_id, batch, seq, task_id, state, attempt, error_code, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, batch, seq, task_id, state, attempt, error_code, timestamp),
            )
            connection.execute(
                "UPDATE dag_tasks SET state = ?, attempt = ?, error_code = ?, updated_at = ? "
                "WHERE run_id = ? AND batch = ? AND task_id = ?",
                (state, attempt, error_code, timestamp, run_id, batch, task_id),
            )

    def save_summary(self, run_id: str, summary: dict) -> None:
        if not isinstance(summary, dict):
            raise _invalid()
        summary_json = _json(summary)
        with self._transaction() as connection:
            run = self._require_run(connection, run_id)
            if run["summary_json"] is not None:
                if run["summary_json"] != summary_json:
                    raise _conflict()
                return
            connection.execute(
                "UPDATE dag_runs SET summary_json = ? WHERE run_id = ?", (summary_json, run_id)
            )

    def start_model_call(self, run_id: str, stage: str, details: dict | None = None) -> int:
        if _code(stage) is None:
            raise _invalid()
        if details is not None and (
            not isinstance(details, dict) or set(details) != {"reservation"}
        ):
            raise _invalid()
        details_json = _details(details if details is not None else {})
        with self._transaction() as connection:
            run = self._require_run(connection, run_id)
            try:
                config = json.loads(run["config_json"])
                limit = _model_call_limit(config)
            except (ValueError, TypeError, AttributeError):
                raise _invalid() from None
            count = connection.execute(
                "SELECT COUNT(*) FROM dag_model_calls WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            if "model_budget" in config:
                configured = config["model_budget"]
                reservation = (details or {}).get("reservation")
                if reservation is None:
                    raise _invalid()
                previous = [
                    json.loads(row[0])["reservation"]
                    for row in connection.execute(
                        "SELECT details_json FROM dag_model_calls WHERE run_id = ?", (run_id,)
                    ).fetchall()
                ]
                if (
                    count >= min(limit, configured["max_calls"])
                    or sum(item["input_bytes"] for item in previous) + reservation["input_bytes"]
                    > configured["max_input_bytes"]
                    or sum(item["output_tokens"] for item in previous)
                    + reservation["output_tokens"]
                    > configured["max_output_tokens"]
                ):
                    raise ResearchError(
                        "model_budget_exceeded", "模型调用预算已耗尽，未预留新请求。"
                    )
            if count >= limit:
                raise _conflict()
            call_id = count + 1
            connection.execute(
                "INSERT INTO dag_model_calls "
                "(run_id, call_id, stage, status, details_json, started_at) "
                "VALUES (?, ?, ?, 'in_flight', ?, ?)",
                (run_id, call_id, stage, details_json, _now()),
            )
            return call_id

    def finish_model_call(self, run_id: str, call_id: int, status: str, details: dict) -> None:
        if type(call_id) is not int or not 1 <= call_id <= MAX_REVIEW_MODEL_CALLS:
            raise _invalid()
        if status not in ("completed", "failed", "cancelled"):
            raise _invalid()
        _details(details)
        with self._transaction() as connection:
            run = self._require_run(connection, run_id)
            try:
                limit = _model_call_limit(json.loads(run["config_json"]))
            except (ValueError, TypeError, AttributeError):
                raise _invalid() from None
            if call_id > limit:
                raise _invalid()
            row = connection.execute(
                "SELECT status, details_json FROM dag_model_calls WHERE run_id = ? AND call_id = ?",
                (run_id, call_id),
            ).fetchone()
            if row is None:
                raise _invalid()
            if row["status"] != "in_flight":
                raise _conflict()
            previous_details = json.loads(row["details_json"])
            if "reservation" in details and details["reservation"] != previous_details.get(
                "reservation"
            ):
                raise _conflict()
            details_json = _details({**previous_details, **details})
            connection.execute(
                "UPDATE dag_model_calls SET status = ?, details_json = ?, finished_at = ? "
                "WHERE run_id = ? AND call_id = ?",
                (status, details_json, _now(), run_id, call_id),
            )

    def record_retrieval_event(self, run_id: str, event: dict) -> None:
        try:
            encoded = _json(validate_retrieval_event(event))
        except ValueError:
            raise _invalid() from None
        with self._transaction() as connection:
            self._require_run(connection, run_id)
            seq = connection.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM retrieval_events WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            if seq > 64:
                raise _conflict()
            connection.execute(
                "INSERT INTO retrieval_events VALUES (?, ?, ?, ?)",
                (run_id, seq, encoded, _now()),
            )

    def snapshot(self, run_id: str) -> dict[str, Any] | None:
        run_id = _identifier(run_id)
        connection = self._db()
        try:
            # A read transaction keeps metadata and event lists mutually consistent.
            connection.execute("BEGIN")
            try:
                run = connection.execute(
                    "SELECT * FROM dag_runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                if run is None:
                    return None
                batches = []
                for batch_row in connection.execute(
                    "SELECT batch FROM dag_batches WHERE run_id = ? ORDER BY batch", (run_id,)
                ).fetchall():
                    batch = batch_row["batch"]
                    tasks = []
                    for row in connection.execute(
                        "SELECT task_id, dependencies_json, timeout_seconds, max_attempts, state, "
                        "attempt, error_code, updated_at FROM dag_tasks "
                        "WHERE run_id = ? AND batch = ? ORDER BY ordinal",
                        (run_id, batch),
                    ).fetchall():
                        task = dict(row)
                        task["id"] = task.pop("task_id")
                        task["dependencies"] = json.loads(task.pop("dependencies_json"))
                        tasks.append(task)
                    events = [
                        dict(row)
                        for row in connection.execute(
                            "SELECT seq, task_id, state, attempt, error_code, timestamp "
                            "FROM dag_events "
                            "WHERE run_id = ? AND batch = ? ORDER BY seq",
                            (run_id, batch),
                        ).fetchall()
                    ]
                    batches.append({"batch": batch, "tasks": tasks, "events": events})
                calls = []
                for row in connection.execute(
                    "SELECT call_id, stage, status, details_json, started_at, finished_at "
                    "FROM dag_model_calls WHERE run_id = ? ORDER BY call_id",
                    (run_id,),
                ).fetchall():
                    call = dict(row)
                    call["details"] = json.loads(call.pop("details_json"))
                    # A local database may be damaged or edited outside this API.
                    # Revalidate the public allowlist when reading, not just writing.
                    _details(call["details"])
                    if (
                        type(call["call_id"]) is not int
                        or not 1 <= call["call_id"] <= MAX_REVIEW_MODEL_CALLS
                        or _code(call["stage"]) is None
                        or call["status"] not in ("in_flight", "completed", "failed", "cancelled")
                    ):
                        raise _invalid()
                    _timestamp(call["started_at"])
                    if call["finished_at"] is not None:
                        _timestamp(call["finished_at"])
                    calls.append(call)
                result = {
                    "run_id": run_id,
                    "config": json.loads(run["config_json"]),
                    "batches": batches,
                    "summary": json.loads(run["summary_json"]) if run["summary_json"] else None,
                    "model_calls": calls,
                }
                retrieval_events = []
                for row in connection.execute(
                    "SELECT seq, substr(event_json, 1, 65537) AS event_json, created_at, "
                    "length(CAST(event_json AS BLOB)) AS size FROM retrieval_events "
                    "WHERE run_id=? ORDER BY seq LIMIT 65",
                    (run_id,),
                ).fetchall():
                    if row["size"] > MAX_JSON_BYTES:
                        raise _invalid()
                    retrieval_events.append(
                        {
                            "seq": row["seq"],
                            "event": validate_retrieval_event(json.loads(row["event_json"])),
                            "created_at": _timestamp(row["created_at"]),
                        }
                    )
                if retrieval_events:
                    if len(retrieval_events) > 64 or (
                        [item["seq"] for item in retrieval_events]
                        != list(range(1, len(retrieval_events) + 1))
                    ):
                        raise _invalid()
                    result["retrieval_events"] = retrieval_events
                if "model_budget" in result["config"]:
                    from deep_research.model_budget import budget_from_calls

                    result["model_budget"] = budget_from_calls(
                        result["config"]["model_budget"],
                        _model_call_limit(result["config"]),
                        calls,
                    )
                return result
            finally:
                connection.rollback()
        except (sqlite3.Error, ValueError, RecursionError):
            raise DagJournalError("dag_audit_io", "无法读取调度审计记录。") from None

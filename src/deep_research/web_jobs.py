"""Durable single-worker Web queue with bounded audits and fail-closed cancellation.

This is deliberately not a distributed scheduler. A sidecar OS advisory lock is
held for the manager lifetime. All job transitions and their events commit in
one SQLite transaction; prior CLI research runs are never implicitly resumed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

from pydantic import ValidationError

from deep_research.errors import ResearchError
from deep_research.model_budget import validate_budget_summary, validate_profile_snapshot
from deep_research.web_models import WebRequest

MAX_EVENTS = 512
MAX_PROGRESS_EVENTS = MAX_EVENTS - 8
TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})
Runner = Callable[[WebRequest, Callable[[str], None], Callable[[str], None]], Awaitable[dict]]
_CODE = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS web_jobs (
 id TEXT PRIMARY KEY,
 idempotency_key TEXT NOT NULL UNIQUE,
 request_hash TEXT NOT NULL,
 request_json TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN
  ('queued','running','cancelling','completed','failed','cancelled','interrupted')),
 run_id TEXT UNIQUE,
 result_json TEXT,
 error_code TEXT,
 cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)),
 dropped_events INTEGER NOT NULL DEFAULT 0 CHECK(dropped_events >= 0),
 created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL,
 started_at TEXT,
 finished_at TEXT
);
CREATE TABLE IF NOT EXISTS web_events (
 job_id TEXT NOT NULL REFERENCES web_jobs(id),
 seq INTEGER NOT NULL CHECK(seq BETWEEN 1 AND 512),
 kind TEXT NOT NULL,
 data_json TEXT NOT NULL CHECK(length(CAST(data_json AS BLOB)) <= 8192),
 created_at TEXT NOT NULL,
 PRIMARY KEY(job_id,seq)
);
CREATE INDEX IF NOT EXISTS web_jobs_pending ON web_jobs(status,created_at);
"""


def _error(code: str) -> ResearchError:
    codes = {
        "web_not_found": (404, "未找到此 Web 任务。"),
        "web_conflict": (409, "任务提交或状态发生冲突。"),
        "web_capacity": (429, "等待队列已满，请稍后再试。"),
        "web_validation": (422, "Web 任务参数无效。"),
        "web_io": (503, "任务记录暂时不可用，已停止接受新任务。"),
        "web_closed": (503, "任务记录已关闭。"),
        "web_unavailable": (503, "任务服务尚未就绪或已停止。"),
        "web_locked": (409, "此数据库已有 Web 服务，请勿同时启动多个实例。"),
    }
    status, message = codes[code]
    return ResearchError(code, message, status_code=status)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _uuid(value: object) -> str:
    if not isinstance(value, str):
        raise _error("web_validation")
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError):
        raise _error("web_validation") from None
    if str(parsed) != value.lower():
        raise _error("web_validation")
    return str(parsed)


def _encode(value: object) -> str:
    try:
        result = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
        if len(result.encode("utf-8")) > 8192:
            raise ValueError
        return result
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise _error("web_validation") from None


def _safe_message(message: object) -> str:
    # Pipeline progress is trusted code, but its future implementations must not
    # accidentally forward provider payloads or machine-local paths to browsers.
    if not isinstance(message, str):
        raise _error("web_validation")
    if re.search(
        r"(?i)(?:sk-[\w-]+|bearer\s+\S+|api[_-]?key|token\s*[=:]|"
        r"[a-z]:[\\/]|\\\\|(?:^|\s)/(?:[^\s]+/)[^\s]*)",
        message,
    ):
        return "研究进度已更新（敏感细节已隐藏）。"
    return message.encode("utf-8")[:1000].decode("utf-8", errors="ignore")


def _safe_result(result: object, run_id: str | None) -> dict:
    if not isinstance(result, dict) or run_id is None or result.get("run_id") != run_id:
        raise _error("web_validation")
    clean: dict = {"run_id": run_id}
    if not isinstance(result.get("status"), str) or result["status"] not in {
        "completed",
        "degraded",
    }:
        raise _error("web_validation")
    clean["status"] = result["status"]
    if type(result.get("degraded", False)) is not bool:
        raise _error("web_validation")
    clean["degraded"] = result.get("degraded", False)
    for name, validate in (
        ("model_profile", validate_profile_snapshot),
        ("model_budget", validate_budget_summary),
    ):
        if name in result:
            try:
                clean[name] = validate(result[name])
            except ResearchError:
                raise _error("web_validation") from None
    for name in ("source_count", "evidence_count", "model_call_count", "replan_count"):
        if name in result:
            if type(result[name]) is not int or not 0 <= result[name] <= 1_000_000:
                raise _error("web_validation")
            clean[name] = result[name]
    if "engine" in result:
        if result["engine"] != "dag":
            raise _error("web_validation")
        clean["engine"] = "dag"
    if "report_kind" in result:
        if not isinstance(result["report_kind"], str) or result["report_kind"] not in {
            "model_report",
            "evidence_digest",
        }:
            raise _error("web_validation")
        clean["report_kind"] = result["report_kind"]
    if "reasons" in result:
        reasons = result["reasons"]
        if (
            not isinstance(reasons, list)
            or len(reasons) > 20
            or any(not isinstance(value, str) or not _CODE.fullmatch(value) for value in reasons)
        ):
            raise _error("web_validation")
        clean["reasons"] = list(reasons)
    if "counters" in result:
        counters = result["counters"]
        if not isinstance(counters, dict) or set(counters) != {"model", "search", "fetch"}:
            raise _error("web_validation")
        if any(
            type(value) is not int or not 0 <= value <= 1_000_000 for value in counters.values()
        ):
            raise _error("web_validation")
        clean["counters"] = dict(counters)
    if "memory_write" in result:
        memory = result["memory_write"]
        if (
            not isinstance(memory, dict)
            or not isinstance(memory.get("status"), str)
            or memory["status"]
            not in {
                "saved",
                "skipped",
                "failed",
            }
        ):
            raise _error("web_validation")
        public_memory = {"status": memory["status"]}
        for name in ("code", "reason"):
            if name in memory:
                value = memory[name]
                public_memory[name] = (
                    value
                    if isinstance(value, str) and _CODE.fullmatch(value)
                    else "memory_unexpected"
                )
        for name in (
            "added_items",
            "added_provenance",
            "deduplicated",
            "skipped_recalled",
            "created_conflicts",
            "created_candidates",
        ):
            if name in memory:
                value = memory[name]
                if type(value) is not int or not 0 <= value <= 1_000_000:
                    raise _error("web_validation")
                public_memory[name] = value
        clean["memory_write"] = public_memory
    if "compression" in result:
        compression = result["compression"]
        if (
            not isinstance(compression, dict)
            or not isinstance(compression.get("status"), str)
            or compression["status"]
            not in {
                "completed",
                "skipped_deadline",
                "interrupted",
                "pending",
                "skipped",
            }
        ):
            raise _error("web_validation")
        public_compression = {"status": compression["status"]}
        if "encoder_kind" in compression:
            if not isinstance(compression["encoder_kind"], str) or compression[
                "encoder_kind"
            ] not in {
                "lexical",
                "local-e5",
                "semantic",
                "sentence-transformers",
            }:
                raise _error("web_validation")
            public_compression["encoder_kind"] = compression["encoder_kind"]
        if "report_evidence_bytes" in compression:
            value = compression["report_evidence_bytes"]
            if type(value) is not int or not 0 <= value <= 1_000_000:
                raise _error("web_validation")
            public_compression["report_evidence_bytes"] = value
        clean["compression"] = public_compression
    review = result.get("review")
    if review is not None:
        if not isinstance(review, dict):
            raise _error("web_validation")
        # Explicit allowlist: neither draft text nor arbitrary nested errors are
        # stored in the Web summary. Full research artifacts remain separate.
        clean_review: dict = {}
        if "status" in review:
            if not isinstance(review["status"], str) or review["status"] not in {
                "passed",
                "unresolved",
                "interrupted",
                "skipped",
            }:
                raise _error("web_validation")
            clean_review["status"] = review["status"]
        if "stop_reason" in review:
            if not isinstance(review["stop_reason"], str) or not _CODE.fullmatch(
                review["stop_reason"]
            ):
                raise _error("web_validation")
            clean_review["stop_reason"] = review["stop_reason"]
        for key in ("rounds", "calls", "format_repairs"):
            if key in review:
                if type(review[key]) is not int or not 0 <= review[key] <= 100:
                    raise _error("web_validation")
                clean_review[key] = review[key]
        scores = review.get("final_scores")
        if scores is not None:
            if not isinstance(scores, dict) or set(scores) != {
                "evidence_support",
                "logical_consistency",
                "citation_quality",
            }:
                raise _error("web_validation")
            if any(type(score) is not int or not 0 <= score <= 100 for score in scores.values()):
                raise _error("web_validation")
            clean_review["final_scores"] = dict(scores)
        clean["review"] = clean_review
    _encode(clean)
    return clean


class WebJobStore:
    """Synchronous, owning-thread SQLite access; callers own its lifetime."""

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self._connection: sqlite3.Connection | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.path, isolation_level=None)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.executescript(_SCHEMA)
        except (OSError, sqlite3.Error):
            self.close()
            raise _error("web_io") from None

    def __enter__(self) -> WebJobStore:
        self._db()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            raise _error("web_closed")
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
        except sqlite3.Error:
            raise _error("web_io") from None

    def _rows(self, query: str, args: tuple = ()) -> list[sqlite3.Row]:
        try:
            return self._db().execute(query, args).fetchall()
        except sqlite3.Error:
            raise _error("web_io") from None

    @staticmethod
    def _job(row: sqlite3.Row) -> dict:
        try:
            request = WebRequest.model_validate_json(row["request_json"])
            result = json.loads(row["result_json"]) if row["result_json"] is not None else None
            if result is not None:
                result = _safe_result(result, row["run_id"])
            return {
                key: row[key]
                for key in (
                    "id",
                    "status",
                    "run_id",
                    "error_code",
                    "dropped_events",
                    "created_at",
                    "updated_at",
                    "started_at",
                    "finished_at",
                )
            } | {
                "request": request.model_dump(),
                "result": result,
                "cancel_requested": bool(row["cancel_requested"]),
            }
        except (ValueError, TypeError, ValidationError, ResearchError):
            raise _error("web_io") from None

    def get(self, job_id: str) -> dict:
        # URLs with an arbitrary unknown identifier are a 404, not a validation
        # oracle. Idempotency keys, by contrast, are strictly UUID validated.
        if not isinstance(job_id, str) or not 1 <= len(job_id) <= 128:
            raise _error("web_not_found")
        rows = self._rows("SELECT * FROM web_jobs WHERE id=?", (job_id,))
        if not rows:
            raise _error("web_not_found")
        return self._job(rows[0])

    def list(self, limit: int = 30) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise _error("web_validation")
        return [
            self._job(row)
            for row in self._rows(
                "SELECT * FROM web_jobs ORDER BY created_at DESC,id DESC LIMIT ?", (limit,)
            )
        ]

    def events(self, job_id: str, after: int = 0) -> list[dict]:
        self.get(job_id)
        if type(after) is not int or not 0 <= after <= MAX_EVENTS:
            raise _error("web_validation")
        rows = self._rows(
            "SELECT * FROM web_events WHERE job_id=? AND seq>? ORDER BY seq", (job_id, after)
        )
        try:
            return [
                {
                    "seq": row["seq"],
                    "kind": row["kind"],
                    "data": json.loads(row["data_json"]),
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
        except (ValueError, TypeError):
            raise _error("web_io") from None

    @staticmethod
    def _event(db: sqlite3.Connection, job_id: str, kind: str, data: dict, now: str) -> None:
        seq = db.execute(
            "SELECT COALESCE(MAX(seq),0)+1 FROM web_events WHERE job_id=?", (job_id,)
        ).fetchone()[0]
        if seq > MAX_EVENTS:
            raise _error("web_io")
        db.execute(
            "INSERT INTO web_events VALUES(?,?,?,?,?)", (job_id, seq, kind, _encode(data), now)
        )

    def submit(self, request: WebRequest, key: str, max_queued: int) -> tuple[dict, bool]:
        key = _uuid(key)
        if (
            not isinstance(request, WebRequest)
            or type(max_queued) is not int
            or not 1 <= max_queued <= 100
        ):
            raise _error("web_validation")
        # Revalidate frozen instances as model_construct can bypass validation.
        try:
            request = WebRequest.model_validate(request.model_dump())
        except ValidationError:
            raise _error("web_validation") from None
        payload = _encode(request.model_dump())
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        with self._transaction() as db:
            row = db.execute("SELECT * FROM web_jobs WHERE idempotency_key=?", (key,)).fetchone()
            if row is not None:
                # M6 persisted requests before profile_id existed. Verify
                # the old bytes against their checksum before adding schema
                # defaults, then compare intent rather than a versioned shape.
                # This is corruption detection, not protection from a writer
                # who can alter both the payload and its checksum in the DB.
                try:
                    previous_payload = row["request_json"]
                    if (
                        not isinstance(previous_payload, str)
                        or hashlib.sha256(previous_payload.encode("utf-8")).hexdigest()
                        != row["request_hash"]
                    ):
                        raise _error("web_io")
                    previous = WebRequest.model_validate_json(previous_payload)
                    normalized_previous = _encode(previous.model_dump())
                except (ValueError, TypeError, UnicodeError, ValidationError):
                    raise _error("web_io") from None
                if normalized_previous != payload:
                    raise _error("web_conflict")
                return self._job(row), False
            count = db.execute("SELECT COUNT(*) FROM web_jobs WHERE status='queued'").fetchone()[0]
            if count >= max_queued:
                raise _error("web_capacity")
            job_id, now = str(uuid4()), _now()
            db.execute(
                "INSERT INTO web_jobs(id,idempotency_key,request_hash,request_json,status,"
                "created_at,updated_at) VALUES(?,?,?,?,'queued',?,?)",
                (job_id, key, digest, payload, now, now),
            )
            self._event(db, job_id, "queued", {"status": "queued"}, now)
        return self.get(job_id), True

    def begin_next(self) -> dict | None:
        with self._transaction() as db:
            row = db.execute(
                # Timestamp resolution can tie on Windows. UUID ordering is
                # random; rowid preserves submission order for that timestamp.
                "SELECT id FROM web_jobs WHERE status='queued' ORDER BY created_at,rowid LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            job_id, now = row["id"], _now()
            db.execute(
                "UPDATE web_jobs SET status='running',started_at=?,updated_at=? WHERE id=?",
                (now, now, job_id),
            )
            self._event(db, job_id, "running", {"status": "running"}, now)
        return self.get(job_id)

    def pending_ids(self) -> list[str]:
        return [
            row["id"]
            for row in self._rows(
                "SELECT id FROM web_jobs WHERE status IN ('queued','running','cancelling')"
            )
        ]

    def bind_run(self, job_id: str, run_id: str) -> None:
        run_id = _uuid(run_id)
        with self._transaction() as db:
            job = self.get(job_id)
            if job["status"] != "running" or job["run_id"] is not None:
                raise _error("web_conflict")
            now = _now()
            db.execute(
                "UPDATE web_jobs SET run_id=?,updated_at=? WHERE id=?", (run_id, now, job_id)
            )
            self._event(db, job_id, "run_created", {"run_id": run_id}, now)

    def progress(self, job_id: str, message: str) -> None:
        message = _safe_message(message)
        with self._transaction() as db:
            job = self.get(job_id)
            if job["status"] not in {"running", "cancelling"}:
                raise _error("web_conflict")
            count = db.execute(
                "SELECT COUNT(*) FROM web_events WHERE job_id=? AND kind='progress'", (job_id,)
            ).fetchone()[0]
            now = _now()
            if count >= MAX_PROGRESS_EVENTS:
                db.execute(
                    "UPDATE web_jobs SET dropped_events=dropped_events+1,updated_at=? WHERE id=?",
                    (now, job_id),
                )
            else:
                self._event(db, job_id, "progress", {"message": message}, now)
                db.execute("UPDATE web_jobs SET updated_at=? WHERE id=?", (now, job_id))

    def cancel(self, job_id: str) -> dict:
        with self._transaction() as db:
            job = self.get(job_id)
            if job["status"] in TERMINAL or job["status"] == "cancelling":
                return job
            if job["status"] == "running" and not job["cancel_requested"] and job["run_id"]:
                has_research = (
                    db.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_runs'"
                    ).fetchone()
                    is not None
                )
                if has_research:
                    committed = db.execute(
                        "SELECT status FROM research_runs WHERE id=?", (job["run_id"],)
                    ).fetchone()
                    if committed is not None and committed["status"] == "completed":
                        # Research committed before this cancellation request.
                        # The adapter may still be closing provider connections;
                        # do not interrupt that cleanup or hide its valid result.
                        return job
            status = "cancelled" if job["status"] == "queued" else "cancelling"
            now = _now()
            db.execute(
                "UPDATE web_jobs SET status=?,cancel_requested=1,updated_at=?,finished_at=? "
                "WHERE id=?",
                (status, now, now if status == "cancelled" else None, job_id),
            )
            self._event(db, job_id, status, {"status": status}, now)
        return self.get(job_id)

    def finish(
        self, job_id: str, status: str, *, result: dict | None = None, error_code: str | None = None
    ) -> dict:
        if not isinstance(status, str) or status not in {"completed", "failed", "cancelled"}:
            raise _error("web_validation")
        if error_code is not None and (
            not isinstance(error_code, str) or not _CODE.fullmatch(error_code)
        ):
            raise _error("web_validation")
        with self._transaction() as db:
            job = self.get(job_id)
            if job["status"] not in {"running", "cancelling"}:
                raise _error("web_conflict")
            if job["cancel_requested"]:
                status, result, error_code = "cancelled", None, None
            clean = _safe_result(result, job["run_id"]) if status == "completed" else None
            now = _now()
            db.execute(
                "UPDATE web_jobs SET status=?,result_json=?,error_code=?,updated_at=?,"
                "finished_at=? WHERE id=?",
                (
                    status,
                    _encode(clean) if clean is not None else None,
                    error_code,
                    now,
                    now,
                    job_id,
                ),
            )
            self._event(db, job_id, status, {"status": status, "error_code": error_code}, now)
        return self.get(job_id)

    def recover(self) -> int:
        with self._transaction() as db:
            rows = db.execute(
                "SELECT id,run_id FROM web_jobs WHERE status IN ('queued','running','cancelling')"
            ).fetchall()
            has_research = (
                db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_runs'"
                ).fetchone()
                is not None
            )
            for row in rows:
                now = _now()
                db.execute(
                    "UPDATE web_jobs SET status='interrupted',error_code='web_interrupted',"
                    "updated_at=?,finished_at=? WHERE id=?",
                    (now, now, row["id"]),
                )
                self._event(db, row["id"], "interrupted", {"status": "interrupted"}, now)
                if has_research and row["run_id"] is not None:
                    db.execute(
                        "UPDATE research_runs SET status='failed',error_code='web_interrupted',"
                        "finished_at=? WHERE id=? AND status='running'",
                        (now, row["run_id"]),
                    )
            return len(rows)


class _InstanceLock:
    def __init__(self, path: Path):
        self.path = path.with_suffix(path.suffix + ".web.lock")
        self.handle: BinaryIO | None = None

    def acquire(self) -> None:
        try:
            handle = self.path.open("a+b")
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.handle = handle
        except OSError:
            if "handle" in locals():
                handle.close()
            raise _error("web_locked") from None

    def release(self) -> None:
        if self.handle is not None:
            # Closing the descriptor releases either platform's advisory lock.
            self.handle.close()
            self.handle = None


class WebManager:
    def __init__(self, store: WebJobStore, runner: Runner, max_queued: int = 8):
        if type(max_queued) is not int or not 1 <= max_queued <= 100:
            raise _error("web_validation")
        self.store, self.runner, self.max_queued = store, runner, max_queued
        self._lock = _InstanceLock(store.path)
        self._started = False
        self._closing = False
        self._healthy = True
        self._wake = asyncio.Event()
        self._worker: asyncio.Task | None = None
        self._active: asyncio.Task | None = None
        self._active_id: str | None = None
        self._cancel_sent = False

    async def start(self) -> None:
        if self._started or self._closing:
            raise _error("web_unavailable")
        self._lock.acquire()
        try:
            self.store.recover()
        except BaseException:
            self._lock.release()
            self._healthy = False
            raise
        self._started = True
        self._worker = asyncio.create_task(self._work(), name="research-web-worker")

    def _ready(self) -> None:
        if not self._started or self._closing or not self._healthy:
            raise _error("web_unavailable")

    def _broken(self, *, cancel_active: bool = True) -> None:
        self._healthy = False
        self._wake.set()
        if cancel_active:
            self._cancel_active()

    def _cancel_active(self) -> None:
        if self._active is not None and not self._active.done() and not self._cancel_sent:
            self._cancel_sent = True
            self._active.cancel()

    def _call(self, function: Callable, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ResearchError as exc:
            if exc.code in {"web_io", "web_closed"}:
                self._broken()
            raise

    def submit(self, request: WebRequest, idempotency_key: str) -> tuple[dict, bool]:
        self._ready()
        result = self._call(self.store.submit, request, idempotency_key, self.max_queued)
        self._wake.set()
        return result

    def get(self, job_id: str) -> dict:
        return self._call(self.store.get, job_id)

    def list(self, limit: int = 30) -> list[dict]:
        return self._call(self.store.list, limit)

    def events(self, job_id: str, after: int = 0) -> list[dict]:
        return self._call(self.store.events, job_id, after)

    def cancel(self, job_id: str) -> dict:
        job = self._call(self.store.cancel, job_id)
        if job["status"] == "cancelling" and self._active_id == job_id:
            self._cancel_active()
        self._wake.set()
        return job

    def _progress(self, job_id: str, message: str) -> None:
        try:
            if not self._healthy:
                raise _error("web_io")
            self.store.progress(job_id, message)
        except BaseException:
            if asyncio.current_task() is self._active:
                # CLI notification handlers intentionally catch Exception.
                # Raise cancellation directly (without a second pending cancel)
                # so an immediately-returning provider cannot run another call.
                self._broken(cancel_active=False)
                self._cancel_sent = True
                raise asyncio.CancelledError from None
            self._broken()
            raise

    def _bind(self, job_id: str, run_id: str) -> None:
        try:
            if not self._healthy:
                raise _error("web_io")
            self._call(self.store.bind_run, job_id, run_id)
        except BaseException:
            self._broken()
            raise

    async def _invoke_runner(self, job: dict) -> dict:
        # Even a wrongly implemented synchronous adapter is contained inside
        # this task, rather than killing the persistent worker before its try.
        return await self.runner(
            WebRequest.model_validate(job["request"]),
            partial(self._progress, job["id"]),
            partial(self._bind, job["id"]),
        )

    async def _work(self) -> None:
        while not self._closing and self._healthy:
            self._wake.clear()
            try:
                job = self._call(self.store.begin_next)
            except ResearchError:
                self._broken()
                return
            if job is None:
                await self._wake.wait()
                continue
            job_id = job["id"]
            self._active_id = job_id
            self._cancel_sent = False
            self._active = asyncio.create_task(
                self._invoke_runner(job),
                name=f"research-web-{job_id}",
            )
            result, status, code = None, "completed", None
            try:
                result = await self._active
            except asyncio.CancelledError:
                status = "cancelled" if self._healthy else "failed"
                code = None if self._healthy else "web_io"
            except Exception as exc:
                status = "failed"
                code = (
                    exc.code
                    if isinstance(exc, ResearchError)
                    and isinstance(exc.code, str)
                    and _CODE.fullmatch(exc.code)
                    else "web_execution"
                )
            try:
                self._call(self.store.finish, job_id, status, result=result, error_code=code)
            except ResearchError as exc:
                if exc.code == "web_validation":
                    try:
                        self._call(self.store.finish, job_id, "failed", error_code="web_execution")
                    except ResearchError:
                        self._broken()
                else:
                    self._broken()
            finally:
                self._active = None
                self._active_id = None

    async def close(self) -> None:
        if not self._started:
            self._lock.release()
            return
        self._closing = True
        try:
            for job_id in self.store.pending_ids():
                self.cancel(job_id)
        except ResearchError:
            self._broken()
        self._wake.set()
        try:
            if self._worker is not None:
                await self._worker
        finally:
            self._lock.release()
            self._started = False

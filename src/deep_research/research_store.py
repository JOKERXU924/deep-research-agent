"""Persist a research run and its exact source text in an isolated SQLite database."""

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from deep_research.errors import ResearchError

_ERROR_CODE = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_runs (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    created_at TEXT NOT NULL,
    finished_at TEXT,
    error_code TEXT,
    output_dir TEXT,
    report_json TEXT,
    usage_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS research_sources (
    run_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    truncated INTEGER NOT NULL CHECK (truncated IN (0, 1)),
    PRIMARY KEY (run_id, source_id),
    FOREIGN KEY (run_id) REFERENCES research_runs(id)
);
CREATE TABLE IF NOT EXISTS research_evidence (
    run_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    start_char INTEGER NOT NULL CHECK (start_char >= 0),
    end_char INTEGER NOT NULL CHECK (end_char > start_char),
    quote TEXT NOT NULL,
    PRIMARY KEY (run_id, evidence_id),
    FOREIGN KEY (run_id, source_id) REFERENCES research_sources(run_id, source_id)
);
"""


class ResearchStoreError(ResearchError):
    """A stable public storage error without SQL, payloads, or upstream error bodies."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _invalid() -> ResearchStoreError:
    return ResearchStoreError("store_validation", "研究记录校验失败，未保存本批数据。")


def _text(value: object, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise _invalid()
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise _invalid() from None
    return value


def _json(value: object) -> str:
    try:
        result = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        result.encode("utf-8")
        return result
    except (TypeError, ValueError, RecursionError, UnicodeError):
        raise _invalid() from None


class ResearchStore:
    """One synchronous connection; use from the owning thread and close when finished.

    Offsets are Python Unicode character indexes into the exact persisted text,
    not UTF-8 byte offsets. All writes reject terminal runs. A failed batch is
    rolled back in full, including source rows inserted before invalid evidence.
    """

    def __init__(self, path: Path) -> None:
        self._closed = False
        self._connection: sqlite3.Connection | None = None
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
            raise ResearchStoreError("store_io", "无法打开研究数据库。") from None

    def __enter__(self) -> "ResearchStore":
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
            raise ResearchStoreError("store_closed", "研究数据库已关闭。")
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
            raise ResearchStoreError(
                "store_conflict", "研究记录重复或引用无效，未保存本批数据。"
            ) from None
        except sqlite3.Error:
            raise ResearchStoreError("store_io", "研究数据库操作失败。") from None

    def _read(self, query: str, run_id: str) -> list[sqlite3.Row]:
        try:
            return self._db().execute(query, (_text(run_id),)).fetchall()
        except sqlite3.Error:
            raise ResearchStoreError("store_io", "无法读取研究记录。") from None

    @staticmethod
    def _require_running(connection: sqlite3.Connection, run_id: str) -> None:
        row = connection.execute(
            "SELECT status FROM research_runs WHERE id = ?", (_text(run_id),)
        ).fetchone()
        if row is None:
            raise ResearchStoreError("store_not_found", "研究任务不存在。")
        if row["status"] != "running":
            raise ResearchStoreError("store_terminal", "研究任务已结束，不能修改结果。")

    def create_run(self, question: str) -> str:
        question = _text(question)
        run_id = str(uuid4())
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO research_runs (id, question, status, created_at) "
                "VALUES (?, ?, 'running', ?)",
                (run_id, question, _now()),
            )
        return run_id

    def save_sources(self, run_id: str, sources: list[dict], evidence: list[dict]) -> None:
        if not isinstance(sources, list) or not isinstance(evidence, list):
            raise _invalid()
        with self._transaction() as connection:
            self._require_running(connection, run_id)
            for source in sources:
                self._save_source(connection, run_id, source)
            for item in evidence:
                self._save_evidence(connection, run_id, item)

    @staticmethod
    def _save_source(connection: sqlite3.Connection, run_id: str, source: dict) -> None:
        required = {"source_id", "url", "title", "text", "sha256", "retrieved_at", "truncated"}
        if not isinstance(source, dict) or set(source) != required:
            raise _invalid()
        source_id = _text(source["source_id"])
        url = _text(source["url"])
        title = _text(source["title"], allow_empty=True)
        text = _text(source["text"])
        digest = _text(source["sha256"])
        retrieved_at = _text(source["retrieved_at"])
        if type(source["truncated"]) is not bool:
            raise _invalid()
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != digest:
            raise _invalid()
        try:
            retrieved = datetime.fromisoformat(retrieved_at)
        except ValueError:
            raise _invalid() from None
        if retrieved.tzinfo is None or retrieved.utcoffset() != timedelta(0):
            raise _invalid()
        connection.execute(
            "INSERT INTO research_sources "
            "(run_id, source_id, url, title, text, sha256, retrieved_at, truncated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, source_id, url, title, text, digest, retrieved_at, source["truncated"]),
        )

    @staticmethod
    def _save_evidence(connection: sqlite3.Connection, run_id: str, item: dict) -> None:
        required = {"evidence_id", "source_id", "start", "end", "quote"}
        if not isinstance(item, dict) or set(item) != required:
            raise _invalid()
        evidence_id = _text(item["evidence_id"])
        source_id = _text(item["source_id"])
        quote = _text(item["quote"])
        start, end = item["start"], item["end"]
        if type(start) is not int or type(end) is not int:
            raise _invalid()
        source = connection.execute(
            "SELECT text FROM research_sources WHERE run_id = ? AND source_id = ?",
            (run_id, source_id),
        ).fetchone()
        if source is None or not 0 <= start < end <= len(source["text"]):
            raise _invalid()
        if source["text"][start:end] != quote:
            raise _invalid()
        connection.execute(
            "INSERT INTO research_evidence "
            "(run_id, evidence_id, source_id, start_char, end_char, quote) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, evidence_id, source_id, start, end, quote),
        )

    def complete_run(self, run_id: str, report: dict, usage: list[dict], output_dir: str) -> None:
        if not isinstance(report, dict) or not isinstance(usage, list):
            raise _invalid()
        if not all(isinstance(item, dict) for item in usage):
            raise _invalid()
        from deep_research.research_map import validate_research_map

        try:
            validate_research_map(report)
        except (ValueError, KeyError, TypeError):
            raise _invalid() from None
        report_json, usage_json = _json(report), _json(usage)
        output_dir = _text(output_dir)
        with self._transaction() as connection:
            self._require_running(connection, run_id)
            connection.execute(
                "UPDATE research_runs SET status = 'completed', finished_at = ?, "
                "report_json = ?, usage_json = ?, output_dir = ? WHERE id = ?",
                (_now(), report_json, usage_json, output_dir, run_id),
            )

    def fail_run(self, run_id: str, code: str, status: str = "failed") -> None:
        if status not in ("failed", "cancelled"):
            raise _invalid()
        if not isinstance(code, str) or _ERROR_CODE.fullmatch(code) is None:
            raise _invalid()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT status FROM research_runs WHERE id = ?", (_text(run_id),)
            ).fetchone()
            if row is None:
                raise ResearchStoreError("store_not_found", "研究任务不存在。")
            if row["status"] != "running":
                return
            connection.execute(
                "UPDATE research_runs SET status = ?, finished_at = ?, error_code = ? WHERE id = ?",
                (status, _now(), code, run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self._read(
            "SELECT id, question, status, created_at, finished_at, error_code, output_dir, "
            "usage_json FROM research_runs WHERE id = ?",
            run_id,
        )
        if not rows:
            return None
        result = dict(rows[0])
        try:
            result["usage"] = json.loads(result.pop("usage_json"))
        except (ValueError, RecursionError):
            raise ResearchStoreError("store_io", "研究记录中的 JSON 无法读取。") from None
        return result

    def get_sources(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._read(
            "SELECT source_id, url, title, text, sha256, retrieved_at, truncated "
            "FROM research_sources WHERE run_id = ? ORDER BY rowid",
            run_id,
        )
        return [dict(row) | {"truncated": bool(row["truncated"])} for row in rows]

    def get_evidence(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._read(
            "SELECT evidence_id, source_id, start_char, end_char, quote "
            "FROM research_evidence WHERE run_id = ? ORDER BY rowid",
            run_id,
        )
        return [
            {
                "evidence_id": row["evidence_id"],
                "source_id": row["source_id"],
                "start": row["start_char"],
                "end": row["end_char"],
                "quote": row["quote"],
            }
            for row in rows
        ]

    def get_report(self, run_id: str) -> dict[str, Any] | None:
        rows = self._read("SELECT report_json FROM research_runs WHERE id = ?", run_id)
        if not rows or rows[0]["report_json"] is None:
            return None
        try:
            report = json.loads(rows[0]["report_json"])
            from deep_research.research_map import validate_research_map

            validate_research_map(report)
            return report
        except (ValueError, RecursionError, KeyError, TypeError):
            raise ResearchStoreError("store_io", "研究报告中的 JSON 无法读取。") from None

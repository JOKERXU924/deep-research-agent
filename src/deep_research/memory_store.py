"""Source-backed, namespace-isolated memory; never a store of model-written facts.

SQLite is authoritative. The lexical vectors are a bounded, rebuildable index,
not semantic embeddings or factual verification. Historical research rows are
read-only here; only memory_* tables are written.
"""

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np
from pydantic import ValidationError

from deep_research.errors import ResearchError
from deep_research.memory_vector import (
    LexicalEncoder,
    conflict_reason,
    lexical_overlap,
    normalize_text,
    vector_scores,
)
from deep_research.research_models import ResearchReport

_SCOPE = re.compile(r"[a-zA-Z0-9_-]{1,64}\Z")
_MODES = {"wikipedia", "local_corpus", "offline_demo"}
_MAX_ITEMS = 1000
_MAX_SOURCES = 64
_MAX_EVIDENCE = 128
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    source_mode TEXT NOT NULL,
    text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    encoder_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector BLOB NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('active', 'superseded')),
    created_at TEXT NOT NULL,
    UNIQUE (scope, source_mode, content_hash),
    UNIQUE (id, scope, source_mode)
);
CREATE TABLE IF NOT EXISTS memory_provenance (
    memory_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    source_mode TEXT NOT NULL,
    origin_run_id TEXT NOT NULL,
    origin_evidence_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    PRIMARY KEY (memory_id, origin_run_id, origin_evidence_id),
    UNIQUE (scope, source_mode, origin_run_id, origin_evidence_id),
    FOREIGN KEY (memory_id, scope, source_mode) REFERENCES memory_items(id, scope, source_mode),
    FOREIGN KEY (origin_run_id, origin_evidence_id)
        REFERENCES research_evidence(run_id, evidence_id),
    FOREIGN KEY (origin_run_id, source_id) REFERENCES research_sources(run_id, source_id)
);
CREATE TABLE IF NOT EXISTS memory_conflicts (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    source_mode TEXT NOT NULL,
    left_id TEXT NOT NULL,
    right_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('open', 'resolved')),
    resolution_json TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    CHECK (left_id != right_id),
    UNIQUE (scope, source_mode, left_id, right_id, reason),
    FOREIGN KEY (left_id, scope, source_mode) REFERENCES memory_items(id, scope, source_mode),
    FOREIGN KEY (right_id, scope, source_mode) REFERENCES memory_items(id, scope, source_mode)
);
CREATE TABLE IF NOT EXISTS memory_candidates (
    scope TEXT NOT NULL,
    source_mode TEXT NOT NULL,
    left_id TEXT NOT NULL,
    right_id TEXT NOT NULL,
    score REAL NOT NULL,
    overlap REAL NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (scope, source_mode, left_id, right_id),
    CHECK (left_id != right_id),
    FOREIGN KEY (left_id, scope, source_mode) REFERENCES memory_items(id, scope, source_mode),
    FOREIGN KEY (right_id, scope, source_mode) REFERENCES memory_items(id, scope, source_mode)
);
"""


class MemoryError(ResearchError):
    """Stable memory_* errors without SQL, source bodies, or model payloads."""


def _invalid() -> MemoryError:
    return MemoryError("memory_validation", "记忆数据或参数校验失败，未保存本批数据。")


def _text(value: object, *, maximum: int = 4000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise _invalid()
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise _invalid() from None
    return value


def _namespace(scope: str, source_mode: str) -> None:
    if not isinstance(scope, str) or _SCOPE.fullmatch(scope) is None:
        raise _invalid()
    if not isinstance(source_mode, str) or source_mode not in _MODES:
        raise _invalid()


def _decode(value: str) -> object:
    def reject_constant(_: str) -> None:
        raise ValueError

    def finite_float(text: str) -> float:
        result = float(text)
        if not math.isfinite(result):
            raise ValueError
        return result

    try:
        return json.loads(value, parse_constant=reject_constant, parse_float=finite_float)
    except (ValueError, TypeError, RecursionError):
        raise _invalid() from None


def _timestamp(value: object, now: datetime) -> datetime:
    try:
        result = datetime.fromisoformat(_text(value, maximum=64))
    except (ValueError, TypeError):
        raise _invalid() from None
    if result.tzinfo is None or result.utcoffset() != timedelta(0) or result > now:
        raise _invalid()
    return result


def _source(row: sqlite3.Row) -> dict:
    return {
        name: bool(row[name]) if name == "truncated" else row[name]
        for name in ("source_id", "url", "title", "text", "sha256", "retrieved_at", "truncated")
    }


def _evidence(row: sqlite3.Row) -> dict:
    return {
        "evidence_id": row["evidence_id"],
        "source_id": row["source_id"],
        "start": row["start_char"],
        "end": row["end_char"],
        "quote": row["quote"],
    }


def _verify_source(source: dict, now: datetime) -> None:
    try:
        digest = hashlib.sha256(source["text"].encode("utf-8")).hexdigest()
        if digest != source["sha256"] or not source["text"].strip():
            raise _invalid()
        _timestamp(source["retrieved_at"], now)
    except (TypeError, AttributeError, UnicodeError, KeyError):
        raise _invalid() from None


def _verify_evidence(source: dict, evidence: dict) -> None:
    start, end = evidence["start"], evidence["end"]
    if (
        evidence["source_id"] != source["source_id"]
        or type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(source["text"])
        or source["text"][start:end] != evidence["quote"]
        or not evidence["quote"].strip()
    ):
        raise _invalid()


class MemoryStore:
    """A single owning-thread connection. Import and conflict decisions are atomic.

    ``now`` is injectable for deterministic age tests, not a persisted freshness
    override. Freshness always comes from the original source retrieval time.
    """

    def __init__(self, path: Path, *, now: Callable[[], datetime] | None = None) -> None:
        self._connection: sqlite3.Connection | None = None
        self._closed = False
        self._clock = now or (lambda: datetime.now(UTC))
        self._encoder = LexicalEncoder()
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
            raise MemoryError("memory_io", "无法打开记忆数据库。") from None

    def __enter__(self) -> "MemoryStore":
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
            raise MemoryError("memory_closed", "记忆数据库已关闭。")
        return self._connection

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise _invalid()
        return value.astimezone(UTC)

    @contextmanager
    def _transaction(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self._db()
        try:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        except sqlite3.IntegrityError:
            raise MemoryError("memory_conflict", "记忆记录重复或引用无效。") from None
        except sqlite3.Error:
            raise MemoryError("memory_io", "记忆数据库操作失败。") from None

    def _vector(self, row: sqlite3.Row) -> np.ndarray:
        if (
            row["encoder_id"] != self._encoder.encoder_id
            or row["dimensions"] != self._encoder.dimensions
            or not isinstance(row["vector"], bytes)
            or len(row["vector"]) != self._encoder.dimensions * 4
        ):
            raise MemoryError("memory_vector", "记忆索引损坏或版本不兼容，需要重建。")
        vector = np.frombuffer(row["vector"], dtype="<f4")
        if not np.isfinite(vector).all():
            raise MemoryError("memory_vector", "记忆索引损坏或版本不兼容，需要重建。")
        norm = float(np.linalg.norm(vector))
        if norm != 0 and not abs(norm - 1.0) < 0.0001:
            raise MemoryError("memory_vector", "记忆索引损坏或版本不兼容，需要重建。")
        return vector

    def _load_run(
        self, connection: sqlite3.Connection, run_id: str, now: datetime
    ) -> tuple[str, list[dict], list[dict], set[str]]:
        # Local import keeps future orchestration integrations free of import cycles.
        from deep_research.research import validate_citations

        row = connection.execute(
            "SELECT status, report_json FROM research_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise MemoryError("memory_not_found", "研究任务不存在。")
        if row["status"] != "completed" or row["report_json"] is None:
            raise MemoryError("memory_ineligible", "只能导入已完成且有有效报告的研究。")
        bundle = _decode(row["report_json"])
        if not isinstance(bundle, dict) or bundle.get("run_id") != run_id:
            raise _invalid()
        mode = bundle.get("retrieval_mode")
        _namespace("validation", mode)
        warnings = bundle.get("warnings")
        execution = bundle.get("execution", {})
        if not isinstance(warnings, list) or not isinstance(execution, dict):
            raise _invalid()
        if warnings or execution.get("degraded", False) is not False:
            raise MemoryError("memory_ineligible", "有警告或降级的研究不进入共享记忆。")
        sources = [
            _source(item)
            for item in connection.execute(
                "SELECT * FROM research_sources WHERE run_id = ? ORDER BY rowid", (run_id,)
            )
        ]
        evidence = [
            _evidence(item)
            for item in connection.execute(
                "SELECT * FROM research_evidence WHERE run_id = ? ORDER BY rowid", (run_id,)
            )
        ]
        if len(sources) > _MAX_SOURCES or len(evidence) > _MAX_EVIDENCE:
            raise MemoryError("memory_capacity", "研究来源或证据数量超过记忆导入上限。")
        for source in sources:
            _verify_source(source, now)
        try:
            report = ResearchReport.model_validate(bundle["report"])
            validate_citations(report, sources, evidence)
        except (ResearchError, ValidationError, KeyError, TypeError, ValueError, UnicodeError):
            raise _invalid() from None
        source_index = {source["source_id"]: source for source in sources}
        for item in evidence:
            _verify_evidence(source_index[item["source_id"]], item)
            _text(item["quote"])
        memory = bundle.get("memory", {})
        if not isinstance(memory, dict) or not isinstance(memory.get("recalled", []), list):
            raise _invalid()
        recalled: set[str] = set()
        evidence_ids = {item["evidence_id"] for item in evidence}
        for item in memory.get("recalled", []):
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("evidence_id"), str)
                or item["evidence_id"] not in evidence_ids
            ):
                raise _invalid()
            recalled.add(item["evidence_id"])
        compression = bundle.get("compression")
        if bundle.get("schema_version") == 4 and compression is None:
            raise _invalid()
        if bundle.get("schema_version") == 5:
            features = bundle.get("features")
            if (
                not isinstance(features, dict)
                or features.get("review") is not True
                or type(features.get("compression")) is not bool
                or features["compression"] != (compression is not None)
                or not isinstance(bundle.get("review"), dict)
            ):
                raise _invalid()
        if compression is not None:
            if not isinstance(compression, dict) or compression.get("status") != "completed":
                raise _invalid()
            selected = compression.get("selected_evidence_ids")
            if (
                not isinstance(selected, list)
                or not selected
                or not all(isinstance(key, str) for key in selected)
                or len(set(selected)) != len(selected)
                or not set(selected) <= evidence_ids
                or not recalled <= set(selected)
            ):
                raise _invalid()
            # M4 keeps the baseline spans for deadline recovery and audit, but
            # only actual report-context evidence is eligible for publication.
            evidence = [item for item in evidence if item["evidence_id"] in set(selected)]
            try:
                validate_citations(report, sources, evidence)
            except ResearchError:
                raise _invalid() from None
        review = bundle.get("review")
        if review is not None:
            from deep_research.review_models import ReviewOptions, ReviewScores, report_fingerprint

            if not isinstance(review, dict) or review.get("status") != "passed":
                raise MemoryError("memory_ineligible", "未通过审查的研究不进入共享记忆。")
            try:
                options = ReviewOptions.model_validate(review["options"])
                scores = ReviewScores.model_validate(review["final_scores"])
            except (KeyError, TypeError, ValidationError):
                raise _invalid() from None
            selected = review.get("evidence_ids")
            current_ids = {item["evidence_id"] for item in evidence}
            if (
                review.get("best_hash") != report_fingerprint(report)
                or review.get("finding_count") != 0
                or min(scores.model_dump().values()) < options.score_threshold
                or not isinstance(selected, list)
                or not selected
                or not all(isinstance(key, str) for key in selected)
                or len(set(selected)) != len(selected)
                or set(selected) != current_ids
                or not recalled <= set(selected)
            ):
                raise _invalid()
            # The set must match the M4-filtered context (or all M2 evidence),
            # never broaden it to unused archived evidence after review.
            try:
                validate_citations(report, sources, evidence)
            except ResearchError:
                raise _invalid() from None
        return mode, sources, evidence, recalled

    def ingest_run(self, run_id: str, *, scope: str) -> dict:
        _namespace(scope, "wikipedia")
        run_id = _text(run_id, maximum=128)
        now = self._now()
        with self._transaction(write=True) as connection:
            mode, sources, evidence, recalled = self._load_run(connection, run_id, now)
            source_index = {source["source_id"]: source for source in sources}
            result = {
                "scope": scope,
                "source_mode": mode,
                "run_id": run_id,
                "added_items": 0,
                "added_provenance": 0,
                "deduplicated": 0,
                "skipped_recalled": len(recalled),
                "created_conflicts": 0,
                "created_candidates": 0,
            }
            for item in evidence:
                if item["evidence_id"] in recalled:
                    continue
                normalized = normalize_text(item["quote"])
                if not normalized:
                    raise _invalid()
                digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                existing = connection.execute(
                    "SELECT * FROM memory_items "
                    "WHERE scope = ? AND source_mode = ? AND content_hash = ?",
                    (scope, mode, digest),
                ).fetchone()
                if existing is None:
                    count = connection.execute(
                        "SELECT COUNT(*) FROM memory_items WHERE scope = ? AND source_mode = ?",
                        (scope, mode),
                    ).fetchone()[0]
                    if count >= _MAX_ITEMS:
                        raise MemoryError("memory_capacity", "该命名空间的记忆条目已达到上限。")
                    vector = self._encoder.encode([item["quote"]])[0]
                    memory_id = str(uuid4())
                    connection.execute(
                        "INSERT INTO memory_items "
                        "(id, scope, source_mode, text, normalized_text, content_hash, "
                        "encoder_id, dimensions, vector, state, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)",
                        (
                            memory_id,
                            scope,
                            mode,
                            item["quote"],
                            normalized,
                            digest,
                            self._encoder.encoder_id,
                            self._encoder.dimensions,
                            vector.astype("<f4").tobytes(),
                            now.isoformat(),
                        ),
                    )
                    conflicts, candidates = self._compare_new(
                        connection,
                        memory_id,
                        scope,
                        mode,
                        item["quote"],
                        source_index[item["source_id"]],
                        vector,
                        now,
                    )
                    result["added_items"] += 1
                    result["created_conflicts"] += conflicts
                    result["created_candidates"] += candidates
                else:
                    if existing["normalized_text"] != normalized:
                        raise _invalid()
                    self._vector(existing)
                    memory_id = existing["id"]
                    result["deduplicated"] += 1
                added = connection.execute(
                    "INSERT INTO memory_provenance "
                    "(memory_id, scope, source_mode, origin_run_id, origin_evidence_id, "
                    "source_id, source_sha256) VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (memory_id, origin_run_id, origin_evidence_id) DO NOTHING",
                    (
                        memory_id,
                        scope,
                        mode,
                        run_id,
                        item["evidence_id"],
                        item["source_id"],
                        source_index[item["source_id"]]["sha256"],
                    ),
                ).rowcount
                result["added_provenance"] += added
                if added and existing is not None and existing["state"] == "active":
                    # A new origin can introduce a changed-URL relation even when
                    # the quote itself already existed under a different URL.
                    conflicts, candidates = self._compare_new(
                        connection,
                        memory_id,
                        scope,
                        mode,
                        item["quote"],
                        source_index[item["source_id"]],
                        self._vector(existing),
                        now,
                    )
                    result["created_conflicts"] += conflicts
                    result["created_candidates"] += candidates
        return result

    def _compare_new(
        self,
        connection: sqlite3.Connection,
        memory_id: str,
        scope: str,
        mode: str,
        text: str,
        source: dict,
        vector: np.ndarray,
        now: datetime,
    ) -> tuple[int, int]:
        conflicts = candidates = 0
        incoming_order = connection.execute(
            "SELECT rowid FROM memory_items WHERE id = ? AND scope = ? AND source_mode = ?",
            (memory_id, scope, mode),
        ).fetchone()[0]
        others = connection.execute(
            "SELECT rowid AS item_order, * FROM memory_items "
            "WHERE scope = ? AND source_mode = ? AND state = 'active' AND id != ? ORDER BY rowid",
            (scope, mode, memory_id),
        ).fetchall()
        for other in others:
            # Stable orientation even when a later provenance import discovers
            # a relationship involving an older item (not a newly created item).
            left_id, right_id = (
                (other["id"], memory_id)
                if other["item_order"] < incoming_order
                else (memory_id, other["id"])
            )
            other_vector = self._vector(other)
            changed = connection.execute(
                "SELECT 1 FROM memory_provenance p JOIN research_sources s "
                "ON s.run_id = p.origin_run_id AND s.source_id = p.source_id "
                "WHERE p.memory_id = ? AND p.scope = ? AND p.source_mode = ? "
                "AND s.url = ? AND s.sha256 != ? LIMIT 1",
                (other["id"], scope, mode, source["url"], source["sha256"]),
            ).fetchone()
            reasons = []
            if changed:
                reasons.append("source_changed")
            possible = conflict_reason(other["text"], text)
            if possible:
                reasons.append(possible)
            if reasons:
                for reason in reasons:
                    prior = connection.execute(
                        "SELECT 1 FROM memory_conflicts WHERE scope = ? AND source_mode = ? "
                        "AND reason = ? AND ((left_id = ? AND right_id = ?) "
                        "OR (left_id = ? AND right_id = ?))",
                        (scope, mode, reason, other["id"], memory_id, memory_id, other["id"]),
                    ).fetchone()
                    if prior is not None:
                        continue
                    connection.execute(
                        "INSERT INTO memory_conflicts "
                        "(id, scope, source_mode, left_id, right_id, reason, status, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, 'open', ?)",
                        (
                            str(uuid4()),
                            scope,
                            mode,
                            left_id,
                            right_id,
                            reason,
                            now.isoformat(),
                        ),
                    )
                    conflicts += 1
            else:
                score = vector_scores(vector, other_vector.reshape(1, -1))[0]
                overlap = lexical_overlap(other["text"], text)
                if score >= 0.92 and overlap >= 0.8:
                    prior = connection.execute(
                        "SELECT 1 FROM memory_candidates WHERE scope = ? AND source_mode = ? "
                        "AND ((left_id = ? AND right_id = ?) OR (left_id = ? AND right_id = ?))",
                        (scope, mode, other["id"], memory_id, memory_id, other["id"]),
                    ).fetchone()
                    if prior is not None:
                        continue
                    connection.execute(
                        "INSERT INTO memory_candidates "
                        "(scope, source_mode, left_id, right_id, score, overlap, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (scope, mode, left_id, right_id, score, overlap, now.isoformat()),
                    )
                    candidates += 1
        return conflicts, candidates

    @staticmethod
    def _conflict(row: sqlite3.Row) -> dict:
        result = dict(row)
        encoded = result.pop("resolution_json")
        result["resolution"] = None if encoded is None else _decode(encoded)
        return result

    def list_conflicts(self, *, scope: str, source_mode: str, status: str = "open") -> list[dict]:
        _namespace(scope, source_mode)
        if status not in ("open", "resolved", "all"):
            raise _invalid()
        with self._transaction() as connection:
            return [
                self._conflict(row)
                for row in connection.execute(
                    "SELECT * FROM memory_conflicts "
                    "WHERE scope = ? AND source_mode = ? "
                    "AND (? = 'all' OR status = ?) ORDER BY rowid",
                    (scope, source_mode, status, status),
                )
            ]

    def resolve_conflict(
        self, conflict_id: str, *, scope: str, source_mode: str, policy: str, reason: str
    ) -> dict:
        _namespace(scope, source_mode)
        conflict_id = _text(conflict_id, maximum=128)
        reason = _text(reason, maximum=1000)
        if policy not in ("prefer_existing", "prefer_incoming", "keep_both"):
            raise _invalid()
        with self._transaction(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM memory_conflicts WHERE id = ? AND scope = ? AND source_mode = ?",
                (conflict_id, scope, source_mode),
            ).fetchone()
            if row is None:
                raise MemoryError("memory_not_found", "当前命名空间中不存在该冲突记录。")
            if row["status"] != "open":
                raise MemoryError("memory_conflict", "冲突已处理，不能重复提交决定。")
            original_states = {
                item["id"]: item["state"]
                for item in connection.execute(
                    "SELECT id, state FROM memory_items "
                    "WHERE scope = ? AND source_mode = ? AND id IN (?, ?)",
                    (scope, source_mode, row["left_id"], row["right_id"]),
                )
            }
            # Resolving one relationship must not reverse another human decision.
            # An explicit future restoration API would need its own audit/policy.
            states = dict(original_states)
            if policy == "prefer_existing":
                if states[row["left_id"]] != "active":
                    raise MemoryError("memory_conflict", "所选记忆已被其他决定停用，不能隐式恢复。")
                states[row["right_id"]] = "superseded"
            elif policy == "prefer_incoming":
                if states[row["right_id"]] != "active":
                    raise MemoryError("memory_conflict", "所选记忆已被其他决定停用，不能隐式恢复。")
                states[row["left_id"]] = "superseded"
            for item_id, state in states.items():
                connection.execute(
                    "UPDATE memory_items SET state = ? "
                    "WHERE id = ? AND scope = ? AND source_mode = ?",
                    (state, item_id, scope, source_mode),
                )
            resolution = {
                "policy": policy,
                "reason": reason,
                "original_states": original_states,
                "resulting_states": states,
            }
            connection.execute(
                "UPDATE memory_conflicts SET status = 'resolved', resolution_json = ?, "
                "resolved_at = ? WHERE id = ? AND scope = ? AND source_mode = ?",
                (
                    json.dumps(resolution, ensure_ascii=False),
                    self._now().isoformat(),
                    conflict_id,
                    scope,
                    source_mode,
                ),
            )
            resolved = connection.execute(
                "SELECT * FROM memory_conflicts WHERE id = ?", (conflict_id,)
            ).fetchone()
            return self._conflict(resolved)

    def _provenance(
        self, connection: sqlite3.Connection, item: sqlite3.Row, now: datetime, cutoff: datetime
    ) -> dict | None:
        rows = connection.execute(
            "SELECT p.*, r.status AS run_status, r.report_json, s.url, s.title, s.text, "
            "s.sha256, s.retrieved_at, s.truncated, e.evidence_id, "
            "e.source_id AS evidence_source_id, e.start_char, e.end_char, e.quote "
            "FROM memory_provenance p "
            "JOIN research_runs r ON r.id = p.origin_run_id "
            "JOIN research_sources s ON s.run_id = p.origin_run_id AND s.source_id = p.source_id "
            "JOIN research_evidence e "
            "ON e.run_id = p.origin_run_id AND e.evidence_id = p.origin_evidence_id "
            "WHERE p.memory_id = ? AND p.scope = ? AND p.source_mode = ? ORDER BY p.rowid",
            (item["id"], item["scope"], item["source_mode"]),
        ).fetchall()
        eligible: list[tuple[datetime, dict]] = []
        normalized = normalize_text(item["text"])
        if (
            normalized != item["normalized_text"]
            or hashlib.sha256(normalized.encode("utf-8")).hexdigest() != item["content_hash"]
        ):
            raise _invalid()
        for row in rows:
            if row["run_status"] != "completed":
                continue
            bundle = _decode(row["report_json"])
            if not isinstance(bundle, dict) or bundle.get("retrieval_mode") != item["source_mode"]:
                raise _invalid()
            source, evidence = _source(row), _evidence(row)
            if row["source_sha256"] != source["sha256"]:
                raise _invalid()
            evidence["source_id"] = row["evidence_source_id"]
            _verify_source(source, now)
            _verify_evidence(source, evidence)
            if normalize_text(evidence["quote"]) != normalized:
                raise _invalid()
            retrieved = _timestamp(source["retrieved_at"], now)
            if retrieved >= cutoff:
                eligible.append(
                    (
                        retrieved,
                        {
                            "origin_run_id": row["origin_run_id"],
                            "origin_evidence_id": row["origin_evidence_id"],
                            "source": source,
                            "evidence": evidence,
                        },
                    )
                )
        if not eligible:
            return None
        return max(eligible, key=lambda pair: pair[0])[1]

    def search(
        self,
        query: str,
        *,
        scope: str,
        source_mode: str = "wikipedia",
        limit: int = 4,
        max_age_days: int = 30,
    ) -> dict:
        _namespace(scope, source_mode)
        query = _text(query)
        if type(limit) is not int or not 1 <= limit <= 10:
            raise _invalid()
        if type(max_age_days) is not int or not 1 <= max_age_days <= 3650:
            raise _invalid()
        now = self._now()
        try:
            cutoff = now - timedelta(days=max_age_days)
        except OverflowError:
            raise _invalid() from None
        result = {
            "scope": scope,
            "source_mode": source_mode,
            "encoder_id": self._encoder.encoder_id,
            "hits": [],
            "excluded_conflicts": 0,
            "excluded_stale": 0,
        }
        query_vector = self._encoder.encode([query])[0]
        with self._transaction() as connection:
            items = connection.execute(
                "SELECT * FROM memory_items "
                "WHERE scope = ? AND source_mode = ? AND state = 'active' ORDER BY rowid",
                (scope, source_mode),
            ).fetchall()
            conflicts = connection.execute(
                "SELECT * FROM memory_conflicts WHERE scope = ? AND source_mode = ? ORDER BY rowid",
                (scope, source_mode),
            ).fetchall()
            open_ids = {
                row[key]
                for row in conflicts
                if row["status"] == "open"
                for key in ("left_id", "right_id")
            }
            for item in items:
                vector = self._vector(item)
                if item["id"] in open_ids:
                    result["excluded_conflicts"] += 1
                    continue
                provenance = self._provenance(connection, item, now, cutoff)
                if provenance is None:
                    result["excluded_stale"] += 1
                    continue
                score = vector_scores(query_vector, vector.reshape(1, -1))[0]
                if score < 0.15 or lexical_overlap(query, item["text"]) <= 0:
                    continue
                notes = [
                    self._conflict(row)
                    for row in conflicts
                    if row["status"] == "resolved"
                    and item["id"] in (row["left_id"], row["right_id"])
                ]
                result["hits"].append(
                    {
                        "memory_id": item["id"],
                        "score": float(score),
                        **provenance,
                        "conflict_notes": notes,
                    }
                )
            result["hits"].sort(key=lambda hit: (-hit["score"], hit["memory_id"]))
            result["hits"] = result["hits"][:limit]
        return result

    def stats(self, *, scope: str, source_mode: str) -> dict:
        _namespace(scope, source_mode)
        with self._transaction() as connection:
            counts = {"active": 0, "superseded": 0}
            for row in connection.execute(
                "SELECT state, COUNT(*) AS n FROM memory_items "
                "WHERE scope = ? AND source_mode = ? GROUP BY state",
                (scope, source_mode),
            ):
                counts[row["state"]] = row["n"]
            result = {
                "scope": scope,
                "source_mode": source_mode,
                "encoder_id": self._encoder.encoder_id,
                "items": counts["active"] + counts["superseded"],
                **counts,
            }
            for key, query in (
                (
                    "provenance",
                    "SELECT COUNT(*) FROM memory_provenance WHERE scope = ? AND source_mode = ?",
                ),
                (
                    "open_conflicts",
                    "SELECT COUNT(*) FROM memory_conflicts "
                    "WHERE scope = ? AND source_mode = ? AND status = 'open'",
                ),
                (
                    "resolved_conflicts",
                    "SELECT COUNT(*) FROM memory_conflicts "
                    "WHERE scope = ? AND source_mode = ? AND status = 'resolved'",
                ),
                (
                    "duplicate_candidates",
                    "SELECT COUNT(*) FROM memory_candidates WHERE scope = ? AND source_mode = ?",
                ),
            ):
                result[key] = connection.execute(query, (scope, source_mode)).fetchone()[0]
            return result

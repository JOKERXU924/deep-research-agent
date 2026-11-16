"""Offline memory tests use exact source snapshots, never a model or the network."""

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from deep_research.memory_store import MemoryError, MemoryStore
from deep_research.research_store import ResearchStore

NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)
SCOPE = "project-a"
MODE = "wikipedia"
TEXT = "Plants use sunlight to produce sugar and oxygen."


def make_run(
    store: ResearchStore,
    text: str = TEXT,
    *,
    mode: str = MODE,
    url: str = "https://example.org/plants",
    retrieved_at: datetime = NOW,
    quotes: list[str] | None = None,
    warnings: list | None = None,
    degraded: bool = False,
    recalled: list[str] | None = None,
    status: str = "completed",
) -> tuple[str, dict, list[dict]]:
    run_id = store.create_run("How do plants use sunlight?")
    source = {
        "source_id": "S1",
        "url": url,
        "title": "Plant source",
        "text": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "retrieved_at": retrieved_at.isoformat(),
        "truncated": False,
    }
    evidence = [
        {
            "evidence_id": f"E{index + 1}",
            "source_id": "S1",
            "start": text.index(quote),
            "end": text.index(quote) + len(quote),
            "quote": quote,
        }
        for index, quote in enumerate(quotes or [text])
    ]
    store.save_sources(run_id, [source], evidence)
    if status == "completed":
        report = {
            "title": "Model title is not a memory",
            "sections": [
                {
                    "heading": "Result",
                    "claims": [{"text": "UNICORN_MODEL_CLAIM_NOT_SOURCE", "citations": ["E1"]}],
                }
            ],
            "limitations": ["Quotes establish provenance, not truth."],
        }
        bundle = {
            "run_id": run_id,
            "retrieval_mode": mode,
            "warnings": warnings or [],
            "execution": {"degraded": degraded},
            "report": report,
            "sources": [source],
            "evidence": evidence,
            "memory": {"recalled": [{"evidence_id": item} for item in recalled or []]},
        }
        store.complete_run(run_id, bundle, [], "output/fixture")
    elif status != "running":
        store.fail_run(run_id, "fixture_failure", status)
    return run_id, source, evidence


@pytest.fixture
def stores(tmp_path: Path):
    path = tmp_path / "nested" / "research.sqlite3"
    with ResearchStore(path) as research, MemoryStore(path, now=lambda: NOW) as memory:
        yield research, memory, path


def ingest(memory: MemoryStore, run: str, scope: str = SCOPE) -> dict:
    return memory.ingest_run(run, scope=scope)


def search(memory: MemoryStore, query: str = TEXT, **kwargs) -> dict:
    return memory.search(query, scope=kwargs.pop("scope", SCOPE), **kwargs)


def stats(memory: MemoryStore, **kwargs) -> dict:
    return memory.stats(
        scope=kwargs.pop("scope", SCOPE), source_mode=kwargs.pop("source_mode", MODE)
    )


def conflicts(memory: MemoryStore, **kwargs) -> list[dict]:
    return memory.list_conflicts(scope=SCOPE, source_mode=MODE, **kwargs)


def update_bundle(store: ResearchStore, run: str, update) -> None:
    bundle = store.get_report(run)
    update(bundle)
    store._db().execute(
        "UPDATE research_runs SET report_json = ? WHERE id = ?",
        (json.dumps(bundle), run),
    )


def test_ingestion_indexes_exact_evidence_and_preserves_research(stores) -> None:
    research, memory, _ = stores
    run, source, evidence = make_run(research)
    original = (research.get_run(run), research.get_report(run), research.get_sources(run))
    result = ingest(memory, run)
    assert result["added_items"] == result["added_provenance"] == 1
    found = search(memory)
    assert found["encoder_id"] == "lexical-hash-v1:1024"
    assert len(found["hits"]) == 1
    hit = found["hits"][0]
    assert hit["origin_run_id"] == run
    assert hit["origin_evidence_id"] == "E1"
    assert hit["source"] == source
    assert hit["evidence"] == evidence[0]
    assert hit["conflict_notes"] == []
    assert hit["score"] == pytest.approx(1.0)
    assert original == (research.get_run(run), research.get_report(run), research.get_sources(run))
    assert memory._db().execute("SELECT text FROM memory_items").fetchone()[0] == TEXT
    assert search(memory, "UNICORN_MODEL_CLAIM_NOT_SOURCE")["hits"] == []


def test_unicode_offsets_are_original_character_offsets(stores) -> None:
    research, memory, _ = stores
    text = "🧪前言。植物利用阳光制造糖。结束。"
    run, source, evidence = make_run(research, text, quotes=["植物利用阳光制造糖。"])
    ingest(memory, run)
    hit = search(memory, "植物利用阳光")["hits"][0]
    assert hit["source"] == source
    assert hit["evidence"] == evidence[0]
    assert hit["evidence"]["start"] == 4


@pytest.mark.parametrize("status", ["running", "failed", "cancelled"])
def test_only_completed_runs_can_be_imported(stores, status: str) -> None:
    research, memory, _ = stores
    run, _, _ = make_run(research, status=status)
    with pytest.raises(MemoryError) as error:
        ingest(memory, run)
    assert error.value.code == "memory_ineligible"
    assert stats(memory)["items"] == 0


@pytest.mark.parametrize("kwargs", [{"warnings": ["retrieval failed"]}, {"degraded": True}])
def test_warned_and_degraded_runs_are_ineligible(stores, kwargs: dict) -> None:
    research, memory, _ = stores
    run, _, _ = make_run(research, **kwargs)
    with pytest.raises(MemoryError) as error:
        ingest(memory, run)
    assert error.value.code == "memory_ineligible"
    assert stats(memory)["items"] == 0


def test_missing_run_and_missing_report(stores) -> None:
    research, memory, _ = stores
    with pytest.raises(MemoryError) as error:
        ingest(memory, "missing")
    assert error.value.code == "memory_not_found"
    run, _, _ = make_run(research)
    research._db().execute("UPDATE research_runs SET report_json = NULL WHERE id = ?", (run,))
    with pytest.raises(MemoryError) as error:
        ingest(memory, run)
    assert error.value.code == "memory_ineligible"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda bundle: bundle.update(run_id="other-run"),
        lambda bundle: bundle.update(retrieval_mode="unknown"),
        lambda bundle: bundle.update(warnings="not-a-list"),
        lambda bundle: bundle.update(execution="not-a-dict"),
        lambda bundle: bundle.update(report={"title": "not-a-report"}),
        lambda bundle: bundle["report"]["sections"][0]["claims"][0].update(citations=["E99"]),
        lambda bundle: bundle.update(memory={"recalled": [{"evidence_id": "E99"}]}),
        lambda bundle: bundle.update(memory={"recalled": [{"evidence_id": []}]}),
        lambda bundle: bundle.update(memory={"recalled": "bad"}),
    ],
)
def test_invalid_bundle_is_rejected_without_writes(stores, mutation) -> None:
    research, memory, _ = stores
    run, _, _ = make_run(research)
    update_bundle(research, run, mutation)
    with pytest.raises(MemoryError) as error:
        ingest(memory, run)
    assert error.value.code == "memory_validation"
    assert stats(memory)["items"] == 0


@pytest.mark.parametrize(
    "statement,args",
    [
        ("UPDATE research_sources SET sha256 = ? WHERE run_id = ?", ("0" * 64,)),
        ("UPDATE research_sources SET text = ? WHERE run_id = ?", ("tampered text",)),
        ("UPDATE research_evidence SET quote = ? WHERE run_id = ?", ("invented quote",)),
        ("UPDATE research_evidence SET end_char = ? WHERE run_id = ?", (10000,)),
    ],
)
def test_original_database_rows_are_validated_not_bundle_copies(stores, statement, args) -> None:
    research, memory, _ = stores
    run, _, _ = make_run(research)
    research._db().execute(statement, (*args, run))
    with pytest.raises(MemoryError) as error:
        ingest(memory, run)
    assert error.value.code == "memory_validation"
    assert stats(memory)["items"] == 0


def test_recalled_evidence_is_not_imported_again(stores) -> None:
    research, memory, _ = stores
    first, second = "Plants use sunlight.", "Plants produce oxygen."
    run, _, _ = make_run(research, first + " " + second, quotes=[first, second], recalled=["E1"])
    result = ingest(memory, run)
    assert result["skipped_recalled"] == result["added_items"] == 1
    assert memory._db().execute("SELECT text FROM memory_items").fetchone()[0] == second
    assert stats(memory)["provenance"] == 1


def test_all_recalled_is_a_successful_noop(stores) -> None:
    research, memory, _ = stores
    run, _, _ = make_run(research, recalled=["E1"])
    result = ingest(memory, run)
    assert result["skipped_recalled"] == 1
    assert result["added_items"] == 0


def test_normalized_dedup_preserves_distinct_origins_and_import_is_idempotent(stores) -> None:
    research, memory, _ = stores
    first, _, _ = make_run(research)
    second, _, _ = make_run(research, TEXT.upper(), url="https://example.org/other")
    ingest(memory, first)
    result = ingest(memory, second)
    assert result["added_items"] == 0
    assert result["deduplicated"] == result["added_provenance"] == 1
    again = ingest(memory, second)
    assert again["added_items"] == again["added_provenance"] == 0
    assert stats(memory)["items"] == 1
    assert stats(memory)["provenance"] == 2
    rows = memory._db().execute("SELECT origin_run_id FROM memory_provenance").fetchall()
    assert {row[0] for row in rows} == {first, second}


def test_namespace_and_source_mode_are_strictly_isolated(stores) -> None:
    research, memory, _ = stores
    for mode in ("wikipedia", "local_corpus", "offline_demo"):
        run, _, _ = make_run(research, mode=mode)
        ingest(memory, run)
        assert stats(memory, source_mode=mode)["items"] == 1
        assert search(memory, source_mode=mode)["hits"][0]["origin_run_id"] == run
    assert search(memory, scope="project-b")["hits"] == []
    assert stats(memory, scope="project-b")["items"] == 0
    assert memory.list_conflicts(scope="project-b", source_mode=MODE) == []


@pytest.mark.parametrize("bad_scope", ["", "a" * 65, "含中文", "a b", "../other", "a;DROP"])
def test_scope_validation(stores, bad_scope: str) -> None:
    research, memory, _ = stores
    run, _, _ = make_run(research)
    with pytest.raises(MemoryError):
        ingest(memory, run, bad_scope)
    with pytest.raises(MemoryError):
        search(memory, scope=bad_scope)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_mode": "all"},
        {"limit": 0},
        {"limit": 11},
        {"limit": True},
        {"max_age_days": 0},
        {"max_age_days": 3651},
        {"max_age_days": True},
    ],
)
def test_search_parameter_validation(stores, kwargs: dict) -> None:
    _, memory, _ = stores
    with pytest.raises(MemoryError):
        search(memory, **kwargs)


def test_age_uses_original_retrieval_time_and_reimport_cannot_refresh(stores) -> None:
    research, memory, _ = stores
    run, _, _ = make_run(research, retrieved_at=NOW - timedelta(days=31))
    ingest(memory, run)
    assert search(memory)["hits"] == []
    assert search(memory)["excluded_stale"] == 1
    ingest(memory, run)
    assert search(memory)["excluded_stale"] == 1
    assert search(memory, max_age_days=32)["hits"][0]["origin_run_id"] == run


def test_fresh_distinct_provenance_can_supply_same_quote(stores) -> None:
    research, memory, _ = stores
    old, _, _ = make_run(research, retrieved_at=NOW - timedelta(days=31))
    fresh, _, _ = make_run(research)
    ingest(memory, old)
    ingest(memory, fresh)
    hit = search(memory)["hits"][0]
    assert hit["origin_run_id"] == fresh
    assert stats(memory)["items"] == 1
    assert stats(memory)["provenance"] == 2


def test_future_timestamp_is_rejected(stores) -> None:
    research, memory, _ = stores
    run, _, _ = make_run(research, retrieved_at=NOW + timedelta(seconds=1))
    with pytest.raises(MemoryError) as error:
        ingest(memory, run)
    assert error.value.code == "memory_validation"


def test_age_boundary_and_clock_injection(stores) -> None:
    research, memory, path = stores
    run, _, _ = make_run(research, retrieved_at=NOW - timedelta(days=30))
    ingest(memory, run)
    assert len(search(memory)["hits"]) == 1
    with MemoryStore(path, now=lambda: NOW + timedelta(microseconds=1)) as later:
        assert search(later)["hits"] == []


def changed_sources(research, memory) -> tuple[str, str]:
    first, _, _ = make_run(research, "Plants use sunlight to produce sugar.")
    second, _, _ = make_run(research, "Plants use sunlight to produce sugar and oxygen.")
    ingest(memory, first)
    ingest(memory, second)
    return first, second


def test_changed_source_snapshot_creates_open_conflict_and_excludes_both(stores) -> None:
    research, memory, _ = stores
    changed_sources(research, memory)
    rows = conflicts(memory)
    assert len(rows) == 1
    assert rows[0]["reason"] == "source_changed"
    assert rows[0]["resolution"] is None
    assert search(memory)["hits"] == []
    assert search(memory)["excluded_conflicts"] == 2


def test_same_snapshot_different_chunks_do_not_trigger_source_changed(stores) -> None:
    research, memory, _ = stores
    first, second = "Plants use sunlight.", "Oceans contain salt water."
    run, _, _ = make_run(research, first + " " + second, quotes=[first, second])
    ingest(memory, run)
    assert conflicts(memory) == []
    assert stats(memory)["items"] == 2


@pytest.mark.parametrize(
    "policy,kept",
    [
        ("prefer_existing", 0),
        ("prefer_incoming", 1),
        ("keep_both", None),
    ],
)
def test_resolution_keeps_audit_and_never_deletes_or_reimports_superseded(
    stores, policy, kept
) -> None:
    research, memory, _ = stores
    runs = changed_sources(research, memory)
    row = conflicts(memory)[0]
    resolved = memory.resolve_conflict(
        row["id"],
        scope=SCOPE,
        source_mode=MODE,
        policy=policy,
        reason="Reviewed source dates; this is a retrieval policy, not proof of truth.",
    )
    assert resolved["status"] == "resolved"
    assert resolved["resolution"]["policy"] == policy
    assert set(resolved["resolution"]["original_states"].values()) == {"active"}
    assert resolved["resolved_at"] == NOW.isoformat()
    assert conflicts(memory) == []
    hits = search(memory, "Plants use sunlight")["hits"]
    assert len(hits) == (2 if kept is None else 1)
    if kept is not None:
        assert hits[0]["origin_run_id"] == runs[kept]
        ingest(memory, runs[1 - kept])
        assert stats(memory)["superseded"] == 1
    assert all(hit["conflict_notes"] for hit in hits)
    assert stats(memory)["items"] == 2
    assert stats(memory)["provenance"] == 2
    with pytest.raises(MemoryError) as error:
        memory.resolve_conflict(
            row["id"], scope=SCOPE, source_mode=MODE, policy=policy, reason="again"
        )
    assert error.value.code == "memory_conflict"


def test_resolution_cannot_cross_namespace_and_requires_reason(stores) -> None:
    research, memory, _ = stores
    changed_sources(research, memory)
    row = conflicts(memory)[0]
    with pytest.raises(MemoryError) as error:
        memory.resolve_conflict(
            row["id"], scope="other", source_mode=MODE, policy="keep_both", reason="reviewed"
        )
    assert error.value.code == "memory_not_found"
    with pytest.raises(MemoryError):
        memory.resolve_conflict(
            row["id"],
            scope=SCOPE,
            source_mode="offline_demo",
            policy="keep_both",
            reason="reviewed",
        )
    with pytest.raises(MemoryError):
        memory.resolve_conflict(
            row["id"], scope=SCOPE, source_mode=MODE, policy="keep_both", reason=" "
        )
    assert len(conflicts(memory)) == 1


def test_other_open_conflicts_still_exclude_item_after_one_resolution(stores) -> None:
    research, memory, _ = stores
    changed_sources(research, memory)
    third, _, _ = make_run(research, "Plants use sunlight to produce sugar, oxygen, and energy.")
    ingest(memory, third)
    rows = conflicts(memory)
    assert len(rows) >= 3
    memory.resolve_conflict(
        rows[0]["id"], scope=SCOPE, source_mode=MODE, policy="keep_both", reason="reviewed"
    )
    assert search(memory)["hits"] == []
    assert search(memory)["excluded_conflicts"] == 3


def test_close_reopen_preserves_vectors_origins_and_conflicts(stores) -> None:
    research, memory, path = stores
    changed_sources(research, memory)
    original = conflicts(memory)
    memory.close()
    memory.close()
    with MemoryStore(path, now=lambda: NOW) as reopened:
        assert stats(reopened)["items"] == 2
        assert conflicts(reopened) == original
        assert search(reopened)["excluded_conflicts"] == 2
    with pytest.raises(MemoryError) as error:
        search(memory)
    assert error.value.code == "memory_closed"


@pytest.mark.parametrize(
    "column,value",
    [
        ("encoder_id", "another-encoder"),
        ("dimensions", 64),
        ("vector", b"short"),
        ("vector", np.full(1024, np.nan, dtype="<f4").tobytes()),
        ("vector", np.full(1024, np.inf, dtype="<f4").tobytes()),
        ("vector", np.ones(1024, dtype="<f4").tobytes()),
    ],
)
def test_corrupt_or_incompatible_vectors_fail_closed(stores, column, value) -> None:
    research, memory, _ = stores
    run, _, _ = make_run(research)
    ingest(memory, run)
    statements = {
        "encoder_id": "UPDATE memory_items SET encoder_id = ?",
        "dimensions": "UPDATE memory_items SET dimensions = ?",
        "vector": "UPDATE memory_items SET vector = ?",
    }
    memory._db().execute(statements[column], (value,))
    with pytest.raises(MemoryError) as error:
        search(memory)
    assert error.value.code == "memory_vector"


def test_search_rechecks_original_provenance(stores) -> None:
    research, memory, _ = stores
    run, _, _ = make_run(research)
    ingest(memory, run)
    research._db().execute(
        "UPDATE research_evidence SET quote = 'tampered' WHERE run_id = ?", (run,)
    )
    with pytest.raises(MemoryError) as error:
        search(memory)
    assert error.value.code == "memory_validation"


def test_search_requires_origin_still_completed(stores) -> None:
    research, memory, _ = stores
    run, _, _ = make_run(research)
    ingest(memory, run)
    research._db().execute("UPDATE research_runs SET status = 'failed' WHERE id = ?", (run,))
    assert search(memory)["hits"] == []


def test_database_failure_rolls_back_entire_import_and_is_sanitized(stores) -> None:
    research, memory, _ = stores
    first, second = "Plants use sunlight.", "Oceans contain salt water."
    run, _, _ = make_run(research, first + " " + second, quotes=[first, second])
    memory._db().execute(
        "CREATE TRIGGER reject_second BEFORE INSERT ON memory_provenance "
        "WHEN NEW.origin_evidence_id = 'E2' BEGIN "
        "SELECT RAISE(ABORT, 'secret upstream SQL error'); END"
    )
    with pytest.raises(MemoryError) as error:
        ingest(memory, run)
    assert "secret" not in str(error.value)
    assert "SQL" not in str(error.value)
    assert stats(memory)["items"] == stats(memory)["provenance"] == 0
    assert research.get_run(run)["status"] == "completed"
    memory._db().execute("DROP TRIGGER reject_second")
    assert ingest(memory, run)["added_items"] == 2


def test_conflict_resolution_is_atomic(stores) -> None:
    research, memory, _ = stores
    changed_sources(research, memory)
    row = conflicts(memory)[0]
    memory._db().execute(
        "CREATE TRIGGER reject_resolution BEFORE UPDATE ON memory_conflicts BEGIN "
        "SELECT RAISE(ABORT, 'secret'); END"
    )
    with pytest.raises(MemoryError):
        memory.resolve_conflict(
            row["id"], scope=SCOPE, source_mode=MODE, policy="prefer_incoming", reason="reviewed"
        )
    assert stats(memory)["active"] == 2
    assert stats(memory)["superseded"] == 0
    assert conflicts(memory)[0]["status"] == "open"


def test_capacity_rejects_entire_batch_not_silent_truncation(stores, monkeypatch) -> None:
    research, memory, _ = stores
    monkeypatch.setattr("deep_research.memory_store._MAX_ITEMS", 1)
    first, second = "Plants use sunlight.", "Oceans contain salt water."
    run, _, _ = make_run(research, first + " " + second, quotes=[first, second])
    with pytest.raises(MemoryError) as error:
        ingest(memory, run)
    assert error.value.code == "memory_capacity"
    assert stats(memory)["items"] == 0


def test_per_run_limits_reject_before_import(stores, monkeypatch) -> None:
    research, memory, _ = stores
    monkeypatch.setattr("deep_research.memory_store._MAX_EVIDENCE", 1)
    first, second = "Plants use sunlight.", "Oceans contain salt water."
    run, _, _ = make_run(research, first + " " + second, quotes=[first, second])
    with pytest.raises(MemoryError) as error:
        ingest(memory, run)
    assert error.value.code == "memory_capacity"
    assert stats(memory)["items"] == 0


def test_foreign_keys_cover_origin_evidence_and_namespace(stores) -> None:
    research, memory, _ = stores
    run, _, _ = make_run(research)
    ingest(memory, run)
    connection = memory._db()
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute("UPDATE memory_provenance SET origin_evidence_id = 'missing'")
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute("UPDATE memory_provenance SET scope = 'another'")


def test_lexical_overlap_guard_blocks_hash_collision(stores, monkeypatch) -> None:
    research, memory, _ = stores
    run, _, _ = make_run(research)
    ingest(memory, run)
    monkeypatch.setattr("deep_research.memory_store.vector_scores", lambda *args: [1.0])
    assert search(memory, "zzzzz_unrelated_unique")["hits"] == []


def test_top_k_is_bounded_and_sorted(stores) -> None:
    research, memory, _ = stores
    for index, text in enumerate(
        [
            "Plants use sunlight.",
            "Plants use sunlight for energy.",
            "Oceans and plants have complex ecosystems.",
        ]
    ):
        run, _, _ = make_run(research, text, url=f"https://example.org/{index}")
        ingest(memory, run)
    hits = search(memory, "Plants use sunlight", limit=2)["hits"]
    assert len(hits) == 2
    assert hits[0]["score"] >= hits[1]["score"]


def test_dedup_new_provenance_can_introduce_source_changed_conflict(stores) -> None:
    research, memory, _ = stores
    first, _, _ = make_run(research, "Plants use sunlight.", url="https://example.org/a")
    second, _, _ = make_run(research, "Oceans contain salt water.", url="https://example.org/b")
    duplicate, _, _ = make_run(research, "Plants use sunlight.", url="https://example.org/b")
    ingest(memory, first)
    ingest(memory, second)
    assert conflicts(memory) == []
    result = ingest(memory, duplicate)
    assert result["added_items"] == 0
    assert result["created_conflicts"] == 1
    assert conflicts(memory)[0]["reason"] == "source_changed"
    assert search(memory)["excluded_conflicts"] == 2
    assert ingest(memory, duplicate)["created_conflicts"] == 0
    # Reverse-direction detection must not create the same pair/reason twice.
    reverse, _, _ = make_run(research, "Oceans contain salt water.", url="https://example.org/a")
    assert ingest(memory, reverse)["created_conflicts"] == 0
    assert len(conflicts(memory)) == 1


def test_multiple_conflict_resolutions_cannot_resurrect_superseded_items(stores) -> None:
    research, memory, _ = stores
    first, _, _ = make_run(research, "系统支持异步任务")
    second, _, _ = make_run(research, "系统不支持异步任务")
    ingest(memory, first)
    ingest(memory, second)
    rows = conflicts(memory)
    assert {row["reason"] for row in rows} == {"source_changed", "possible_negation"}
    memory.resolve_conflict(
        rows[0]["id"], scope=SCOPE, source_mode=MODE, policy="prefer_incoming", reason="reviewed"
    )
    assert search(memory, "系统异步任务")["hits"] == []  # Other relation remains open.
    with pytest.raises(MemoryError) as error:
        memory.resolve_conflict(
            rows[1]["id"], scope=SCOPE, source_mode=MODE, policy="prefer_existing", reason="reverse"
        )
    assert error.value.code == "memory_conflict"
    memory.resolve_conflict(
        rows[1]["id"], scope=SCOPE, source_mode=MODE, policy="keep_both", reason="leave prior state"
    )
    assert stats(memory)["superseded"] == 1
    assert stats(memory)["active"] == 1
    assert len(conflicts(memory, status="all")) == 2
    assert search(memory, "系统异步任务")["hits"][0]["origin_run_id"] == second


def test_approximate_duplicates_are_candidates_not_merged(stores) -> None:
    research, memory, _ = stores
    first, _, _ = make_run(research, "Plants use sunlight to produce sugar and oxygen.")
    second, _, _ = make_run(
        research,
        "Plants use sunlight to produce sugar and oxygen!",
        url="https://example.org/other",
    )
    ingest(memory, first)
    result = ingest(memory, second)
    assert result["added_items"] == 1
    assert result["created_candidates"] == 1
    assert stats(memory)["items"] == 2
    assert stats(memory)["duplicate_candidates"] == 1
    assert conflicts(memory) == []


def test_excessive_quote_length_is_rejected_without_truncation(stores) -> None:
    research, memory, _ = stores
    run, _, _ = make_run(research, "x" * 4001)
    with pytest.raises(MemoryError) as error:
        ingest(memory, run)
    assert error.value.code == "memory_validation"
    assert stats(memory)["items"] == 0


@pytest.mark.parametrize("encoded", ["{", '{"extra":NaN}', '{"extra":1e999}'])
def test_invalid_persisted_json_is_sanitized(stores, encoded) -> None:
    research, memory, _ = stores
    run, _, _ = make_run(research)
    research._db().execute("UPDATE research_runs SET report_json = ? WHERE id = ?", (encoded, run))
    with pytest.raises(MemoryError) as error:
        ingest(memory, run)
    assert error.value.code == "memory_validation"


def test_future_timestamp_added_after_import_fails_closed(stores) -> None:
    research, memory, _ = stores
    run, _, _ = make_run(research)
    ingest(memory, run)
    research._db().execute(
        "UPDATE research_sources SET retrieved_at = ? WHERE run_id = ?",
        ((NOW + timedelta(days=1)).isoformat(), run),
    )
    with pytest.raises(MemoryError) as error:
        search(memory)
    assert error.value.code == "memory_validation"

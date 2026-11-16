"""Cross-run memory integration with exact stored provenance and no network calls."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_dag_research import FakeGateway, FakeRetriever, TrackingStore

from deep_research.dag_journal import DagJournal
from deep_research.dag_research import DagOptions, run_dag_pipeline
from deep_research.errors import ResearchError
from deep_research.memory_store import MemoryError, MemoryStore
from deep_research.research import ResearchRunError, validate_citations
from deep_research.research_models import ResearchReport
from deep_research.retrieval import SourceDocument
from deep_research.schemas import MAX_INPUT_BYTES

QUESTION = "evidence source tracking"
SCOPE = "study"
MODE = "local_corpus"
OLD = SourceDocument(
    "https://example.org/history",
    "Source tracking notes",
    "Evidence source tracking preserves quoted original text.",
)
FRESH = SourceDocument(
    "https://example.org/current",
    "Current evidence notes",
    "New evidence source tracking records retrieval details.",
)


@pytest.fixture
def storage(tmp_path):
    database = tmp_path / "memory.sqlite"
    with TrackingStore(database) as store, DagJournal(database) as journal:
        with MemoryStore(database) as memory:
            yield store, journal, memory


async def execute(tmp_path, storage, *, gateway=None, retriever=None, enable_memory=True, **kwargs):
    store, journal, memory = storage
    memory_options = {"memory": memory, "memory_scope": SCOPE} if enable_memory else {}
    return await run_dag_pipeline(
        QUESTION,
        gateway=gateway or FakeGateway(),
        retriever=retriever or FakeRetriever({"initial": [FRESH]}),
        store=store,
        journal=journal,
        output_dir=tmp_path / "reports",
        **({"source_mode": MODE} | memory_options | kwargs),
    )


async def seed(tmp_path, storage):
    result = await execute(
        tmp_path, storage, retriever=FakeRetriever({"initial": [OLD]}), enable_memory=False
    )
    imported = storage[2].ingest_run(result["run_id"], scope=SCOPE)
    assert imported["added_items"] == 1
    return result


async def test_recalled_original_is_copied_to_current_run_with_new_citations_and_old_timestamp(
    tmp_path, storage
):
    seed_result = await seed(tmp_path, storage)
    store, _, memory = storage
    original = store.get_sources(seed_result["run_id"])[0]
    gateway = FakeGateway()
    retriever = FakeRetriever({"initial": [FRESH]})
    result = await execute(tmp_path, storage, gateway=gateway, retriever=retriever)

    assert result["status"] == "completed"
    assert result["memory"]["recalled_count"] == 1
    assert result["memory_write"]["status"] == "saved"
    assert retriever.fetch_calls == [FRESH.url]
    bundle = store.get_report(result["run_id"])
    assert bundle["schema_version"] == 3
    assert [item["url"] for item in bundle["sources"]] == [FRESH.url, OLD.url]
    reused_source = bundle["sources"][1]
    assert reused_source == original | {"source_id": "S2"}
    reused_evidence = bundle["evidence"][1]
    assert reused_evidence == {
        "evidence_id": "E2",
        "source_id": "S2",
        "start": 0,
        "end": len(OLD.text),
        "quote": OLD.text,
    }
    recall = bundle["memory"]["recalled"][0]
    assert recall["evidence_id"] == "E2"
    assert recall["source_id"] == "S2"
    assert recall["origin_run_id"] == seed_result["run_id"]
    assert recall["origin_evidence_id"] == "E1"
    assert recall["retrieved_at"] == original["retrieved_at"]
    assert recall["memory_id"]
    assert recall["score"] > 0
    assert recall["conflict_notes"] == []
    markdown = await asyncio.to_thread(Path(result["report_path"]).read_text, encoding="utf-8")
    assert "## 历史证据复用" in markdown
    assert "不代表本次重新联网核验" in markdown
    assert "E2：原任务 " + seed_result["run_id"] + " / E1" in markdown
    assert "原获取时间 " + original["retrieved_at"] in markdown
    assert "记忆条目 " + recall["memory_id"] in markdown
    context = json.loads(gateway.calls[-1][1].messages[-1].content)
    assert context["memory_reuse"] == bundle["memory"]
    assert "历史快照" in gateway.calls[-1][1].messages[0].content
    assert context["evidence"] == bundle["evidence"]
    validate_citations(
        ResearchReport.model_validate(bundle["report"]), bundle["sources"], bundle["evidence"]
    )

    # Republishing the report must not turn a recalled quotation into another observation.
    assert result["memory_write"]["skipped_recalled"] == 1
    assert result["memory_write"]["added_items"] == 1
    assert result["memory_write"]["added_provenance"] == 1
    assert memory.stats(scope=SCOPE, source_mode=MODE)["provenance"] == 2
    hits = memory.search(OLD.text, scope=SCOPE, source_mode=MODE)["hits"]
    old_hit = next(item for item in hits if item["memory_id"] == recall["memory_id"])
    assert old_hit["origin_run_id"] == seed_result["run_id"]
    assert store.get_sources(seed_result["run_id"])[0] == original


@pytest.mark.parametrize("matching", ["url", "content_hash"])
async def test_fresh_source_takes_precedence_over_matching_cached_material(
    tmp_path, storage, matching
):
    seed_result = await seed(tmp_path, storage)
    original = storage[0].get_sources(seed_result["run_id"])[0]
    fresh = (
        SourceDocument(OLD.url, "New retrieval", FRESH.text)
        if matching == "url"
        else SourceDocument(FRESH.url, "New origin for identical text", OLD.text)
    )
    result = await execute(tmp_path, storage, retriever=FakeRetriever({"initial": [fresh]}))
    assert result["memory"]["selected_count"] == 1
    assert result["memory"]["skipped_fresh"] == 1
    assert result["memory"]["recalled"] == []
    sources = storage[0].get_sources(result["run_id"])
    assert len(sources) == 1
    assert sources[0]["url"] == fresh.url
    assert sources[0]["text"] == fresh.text
    assert storage[0].get_sources(seed_result["run_id"])[0] == original


async def test_memory_disabled_preserves_m2_shape_and_performs_no_memory_io(
    tmp_path, storage, monkeypatch
):
    memory = storage[2]
    monkeypatch.setattr(memory, "search", lambda *a, **kw: pytest.fail("Unexpected recall"))
    monkeypatch.setattr(
        memory, "ingest_run", lambda *a, **kw: pytest.fail("Unexpected publication")
    )
    gateway = FakeGateway()
    result = await execute(tmp_path, storage, enable_memory=False, gateway=gateway)
    assert result["degraded"] is False
    assert "memory" not in result and "memory_write" not in result
    bundle = storage[0].get_report(result["run_id"])
    assert bundle["schema_version"] == 2
    assert "memory" not in bundle
    assert "memory_reuse" not in json.loads(gateway.calls[-1][1].messages[-1].content)


async def test_memory_read_failure_stops_before_any_model_or_retrieval_call(
    tmp_path, storage, monkeypatch
):
    def broken_search(*args, **kwargs):
        raise MemoryError("memory_io", "Memory could not be read.")

    monkeypatch.setattr(storage[2], "search", broken_search)
    gateway, retriever = FakeGateway(), FakeRetriever({"initial": [FRESH]})
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, storage, gateway=gateway, retriever=retriever)
    assert caught.value.code == "memory_io"
    assert gateway.calls == []
    assert retriever.search_calls == []
    assert storage[0].get_run(caught.value.run_id)["status"] == "failed"
    assert storage[0].get_report(caught.value.run_id) is None
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize(
    "error", [MemoryError("memory_io", "Unavailable."), RuntimeError("secret")]
)
async def test_memory_publication_failure_does_not_erase_a_completed_report(
    tmp_path, storage, monkeypatch, error
):
    calls, progress = [], []

    def broken_ingest(run_id, *, scope):
        calls.append((run_id, scope))
        assert storage[0].get_run(run_id)["status"] == "completed"
        raise error

    monkeypatch.setattr(storage[2], "ingest_run", broken_ingest)
    result = await execute(tmp_path, storage, on_progress=progress.append)
    assert result["status"] == "completed"
    expected = "memory_io" if isinstance(error, ResearchError) else "memory_unexpected"
    assert result["memory_write"] == {"status": "failed", "code": expected}
    assert calls == [(result["run_id"], SCOPE)]
    assert storage[0].get_run(result["run_id"])["status"] == "completed"
    bundle = storage[0].get_report(result["run_id"])
    assert bundle["report"]["sections"]
    saved = await asyncio.to_thread(Path(result["evidence_path"]).read_text, encoding="utf-8")
    assert json.loads(saved) == bundle
    assert any("记忆" in item and "失败" in item for item in progress)
    assert all("secret" not in item for item in progress)


async def test_cache_only_result_is_explicitly_degraded_and_never_republished(
    tmp_path, storage, monkeypatch
):
    seed_result = await seed(tmp_path, storage)
    before = storage[2].stats(scope=SCOPE, source_mode=MODE)
    monkeypatch.setattr(
        storage[2], "ingest_run", lambda *a, **kw: pytest.fail("Degraded report must not publish")
    )
    gateway = FakeGateway()
    retriever = FakeRetriever({"initial": [], "revised": []})
    result = await execute(tmp_path, storage, gateway=gateway, retriever=retriever)
    assert result["status"] == "completed"
    assert result["degraded"] is True
    assert "memory_only" in result["reasons"]
    assert result["memory_write"] == {"status": "skipped", "reason": "degraded_run"}
    assert result["replan_count"] == 1
    assert [stage for stage, _ in gateway.calls] == ["plan", "replan", "report"]
    assert len(retriever.search_calls) == 2
    assert retriever.fetch_calls == []
    assert storage[2].stats(scope=SCOPE, source_mode=MODE) == before
    assert (
        storage[0].get_sources(result["run_id"])[0]["retrieved_at"]
        == (storage[0].get_sources(seed_result["run_id"])[0]["retrieved_at"])
    )


@pytest.mark.parametrize("different", [{"memory_scope": "another"}, {"source_mode": "wikipedia"}])
async def test_recall_never_crosses_scope_or_source_mode(tmp_path, storage, different):
    await seed(tmp_path, storage)
    gateway = FakeGateway()
    with pytest.raises(ResearchRunError) as caught:
        await execute(
            tmp_path,
            storage,
            gateway=gateway,
            retriever=FakeRetriever({"initial": [], "revised": []}),
            **different,
        )
    assert caught.value.code == "no_evidence"
    assert [stage for stage, _ in gateway.calls] == ["plan", "replan"]
    assert storage[0].get_sources(caught.value.run_id) == []


async def test_stale_history_is_excluded_instead_of_refreshed_by_recall(
    tmp_path, storage, monkeypatch
):
    await seed(tmp_path, storage)
    monkeypatch.setattr(storage[2], "_now", lambda: datetime.now(UTC) + timedelta(days=31))
    assert storage[2].search(QUESTION, scope=SCOPE, source_mode=MODE)["excluded_stale"] == 1
    with pytest.raises(ResearchRunError) as caught:
        await execute(
            tmp_path,
            storage,
            retriever=FakeRetriever({"initial": [], "revised": []}),
            memory_max_age_days=30,
        )
    assert caught.value.code == "no_evidence"
    assert storage[0].get_sources(caught.value.run_id) == []


async def test_global_deadline_can_use_previously_recalled_memory_after_model_cleanup(
    tmp_path, storage, monkeypatch
):
    await seed(tmp_path, storage)
    gateway = FakeGateway(stall_stage="plan")
    before = storage[2].stats(scope=SCOPE, source_mode=MODE)
    monkeypatch.setattr(
        storage[2], "ingest_run", lambda *a, **kw: pytest.fail("Deadline digest must not publish")
    )
    result = await execute(
        tmp_path, storage, gateway=gateway, options=DagOptions(total_timeout_seconds=0.2)
    )
    assert gateway.entered.is_set() and gateway.drained.is_set()
    assert result["status"] == "completed"
    assert result["report_kind"] == "evidence_digest"
    assert "global_timeout" in result["reasons"]
    assert result["memory"]["recalled_count"] == 1
    assert result["memory_write"]["status"] == "skipped"
    assert len(gateway.calls) == 1
    bundle = storage[0].get_report(result["run_id"])
    assert bundle["report"]["sections"][0]["claims"][0]["text"] == "原文摘录：" + OLD.text
    assert storage[2].stats(scope=SCOPE, source_mode=MODE) == before


async def test_user_cancel_does_not_export_or_publish_a_cache_only_result(
    tmp_path, storage, monkeypatch
):
    await seed(tmp_path, storage)
    before = storage[2].stats(scope=SCOPE, source_mode=MODE)
    gateway = FakeGateway(stall_stage="plan")
    monkeypatch.setattr(
        storage[2], "ingest_run", lambda *a, **kw: pytest.fail("Cancelled run must not publish")
    )
    task = asyncio.create_task(execute(tmp_path, storage, gateway=gateway))
    try:
        await asyncio.wait_for(gateway.entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert gateway.drained.is_set()
        run_id = storage[0].last_run_id
        assert storage[0].get_run(run_id)["status"] == "cancelled"
        assert storage[0].get_report(run_id) is None
        assert storage[0].get_sources(run_id) == []
        assert not (tmp_path / "reports" / run_id).exists()
        assert storage[2].stats(scope=SCOPE, source_mode=MODE) == before
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_verbose_resolved_conflicts_are_bounded_in_prompt_but_preserved_in_audit(
    tmp_path, storage
):
    store, _, memory = storage
    for number in range(1, 8):
        historical = SourceDocument(
            f"https://example.org/history/{number}",
            "Evidence retention rule",
            f"Evidence source tracking retains exactly {number} observations.",
        )
        completed = await execute(
            tmp_path,
            storage,
            enable_memory=False,
            retriever=FakeRetriever({"initial": [historical]}),
        )
        memory.ingest_run(completed["run_id"], scope=SCOPE)

    conflicts = memory.list_conflicts(scope=SCOPE, source_mode=MODE)
    assert len(conflicts) == 21  # Seven different numerical claims, one conflict per pair.
    long_reason = "人工判断理由" + "证据" * 477
    assert len(long_reason) == 960
    for conflict in conflicts:
        memory.resolve_conflict(
            conflict["id"],
            scope=SCOPE,
            source_mode=MODE,
            policy="keep_both",
            reason=long_reason,
        )

    gateway = FakeGateway()
    result = await execute(tmp_path, storage, gateway=gateway)
    assert result["status"] == "completed"
    assert result["degraded"] is True
    assert "memory_conflict_retained" in result["reasons"]
    assert result["memory_write"]["status"] == "skipped"
    assert result["memory"]["recalled_count"] == 4
    assert [stage for stage, _ in gateway.calls] == ["plan", "report"]
    assert all(
        len(request.model_dump_json().encode()) <= MAX_INPUT_BYTES for _, request in gateway.calls
    )

    report_request = gateway.calls[-1][1]
    assert long_reason not in report_request.model_dump_json()
    prompt = json.loads(report_request.messages[-1].content)["memory_reuse"]
    assert len(prompt["recalled"]) == 4
    for item in prompt["recalled"]:
        assert len(item["conflict_notes"]) == 4
        assert item["conflict_count"] == 6
        assert item["conflict_details_truncated"] is True
        for note in item["conflict_notes"]:
            assert set(note) == {"id", "reason", "policy"}
            assert note["policy"] == "keep_both"
            assert note["reason"] != long_reason

    bundle = store.get_report(result["run_id"])
    # The previous implementation inserted this entire oversized metadata into the prompt.
    assert len(json.dumps(bundle["memory"], ensure_ascii=False).encode()) > MAX_INPUT_BYTES
    for item in bundle["memory"]["recalled"]:
        assert len(item["conflict_notes"]) == 6
        assert all(note["resolution"]["reason"] == long_reason for note in item["conflict_notes"])
    saved = await asyncio.to_thread(Path(result["evidence_path"]).read_text, encoding="utf-8")
    assert json.loads(saved)["memory"] == bundle["memory"]

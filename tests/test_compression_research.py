"""M4 integration: archived originals, bounded prompts, and unchanged memory provenance."""

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_dag_research import FakeGateway, FakeRetriever, TrackingStore

from deep_research import dag_research
from deep_research.compression import CompressionOptions, HierarchicalCompressor
from deep_research.dag_journal import DagJournal
from deep_research.dag_research import DagOptions, run_dag_pipeline
from deep_research.embeddings import LexicalEmbedding
from deep_research.errors import ResearchError
from deep_research.gateway import ModelGateway, StructuredResult
from deep_research.memory_store import MemoryError, MemoryStore
from deep_research.research import ResearchRunError
from deep_research.research_models import CitedClaim, ReportSection, ResearchReport
from deep_research.research_store import ResearchStoreError
from deep_research.retrieval import SourceDocument
from deep_research.schemas import MAX_INPUT_BYTES, ModelResponse

QUESTION = "evidence source tracking"
LINES = [
    "Evidence source tracking preserves the first baseline observation.",
    "Evidence source tracking preserves the second baseline observation.",
    "Evidence source tracking 保留中文😀原句，不能改写数值 42。",
]
DOCUMENT = SourceDocument(
    "https://example.org/fresh", "Original multilingual observations", "\n".join(LINES)
)


class SpanCompressor:
    """Pick a known original span so tests distinguish selected and archived evidence."""

    def __init__(self, *, line=2, budget=8192, stall=False, failure=None):
        self.options = SimpleNamespace(max_evidence_bytes=budget)
        self.line = line
        self.stall = stall
        self.failure = failure
        self.calls = []
        self.entered = asyncio.Event()
        self.drained = asyncio.Event()

    async def compress(self, question, sources):
        self.calls.append((question, sources))
        self.entered.set()
        if self.stall:
            try:
                await asyncio.Event().wait()
            finally:
                self.drained.set()
        if self.failure is not None:
            raise self.failure
        spans = []
        for source in sources:
            quote = source["text"].splitlines()[self.line]
            start = source["text"].index(quote)
            spans.append(
                {
                    "source_id": source["source_id"],
                    "start": start,
                    "end": start + len(quote),
                    "quote": quote,
                }
            )
        return SimpleNamespace(
            evidence=spans,
            audit={
                "schema_version": 1,
                "method": "test_span_selection",
                "encoder": {"encoder_id": "offline-test", "kind": "lexical", "dimensions": 1},
                "selected_count": len(spans),
            },
        )


@pytest.fixture
def storage(tmp_path):
    database = tmp_path / "compression.sqlite"
    with TrackingStore(database) as store, DagJournal(database) as journal:
        with MemoryStore(database) as memory:
            yield store, journal, memory


async def execute(tmp_path, storage, *, gateway=None, retriever=None, **kwargs):
    store, journal, _ = storage
    return await run_dag_pipeline(
        QUESTION,
        gateway=gateway or FakeGateway(),
        retriever=retriever or FakeRetriever({"initial": [DOCUMENT]}),
        store=store,
        journal=journal,
        output_dir=tmp_path / "reports",
        source_mode="local_corpus",
        **kwargs,
    )


async def test_selected_span_gets_new_id_and_archive_preserves_original_evidence(tmp_path, storage):
    gateway, compressor = FakeGateway(), SpanCompressor()
    result = await execute(tmp_path, storage, gateway=gateway, compressor=compressor)
    store, _, _ = storage
    assert result["status"] == "completed"
    assert [stage for stage, _ in gateway.calls] == ["plan", "report"]
    assert compressor.calls[0][0] == QUESTION
    assert compressor.calls[0][1][0]["text"] == DOCUMENT.text
    bundle = store.get_report(result["run_id"])
    assert bundle["schema_version"] == 4
    assert bundle["sources"][0]["text"] == DOCUMENT.text
    archived = store.get_evidence(result["run_id"])
    assert [item["evidence_id"] for item in archived] == ["E1", "E2", "E3"]
    assert [item["quote"] for item in archived] == LINES
    assert archived == bundle["evidence"]
    selected = archived[-1]
    assert DOCUMENT.text[selected["start"] : selected["end"]] == LINES[2]
    assert selected["start"] == DOCUMENT.text.index(LINES[2])
    assert "😀" in selected["quote"]
    context = json.loads(gateway.calls[-1][1].messages[-1].content)
    assert context["evidence"] == [selected]
    assert bundle["compression"]["selected_evidence_ids"] == ["E3"]
    assert bundle["compression"]["archived_evidence_count"] == 3
    assert bundle["compression"]["status"] == "completed"
    assert bundle["compression"]["report_evidence_bytes"] <= 8192
    saved = await asyncio.to_thread(Path(result["evidence_path"]).read_text, encoding="utf-8")
    assert json.loads(saved) == bundle


async def test_selecting_an_existing_span_reuses_id_without_duplicate_storage(tmp_path, storage):
    gateway = FakeGateway()
    result = await execute(tmp_path, storage, gateway=gateway, compressor=SpanCompressor(line=0))
    archived = storage[0].get_evidence(result["run_id"])
    assert [item["evidence_id"] for item in archived] == ["E1", "E2"]
    assert result["compression"]["selected_evidence_ids"] == ["E1"]
    assert json.loads(gateway.calls[-1][1].messages[-1].content)["evidence"] == [archived[0]]


async def test_real_lexical_compressor_keeps_exact_unicode_quotes_and_auditable_ranking(
    tmp_path, storage
):
    document = SourceDocument(
        "https://example.org/unicode",
        "原文证据 😀",
        "证据出处😀必须追踪到原文。\n原文证据要保存获取时间。\n压缩可能遗漏限定条件。",
    )
    gateway = FakeGateway()
    compressor = HierarchicalCompressor(
        LexicalEmbedding(), CompressionOptions(max_evidence=2, max_evidence_bytes=512)
    )
    result = await run_dag_pipeline(
        "如何保留原文证据出处😀？",
        gateway=gateway,
        retriever=FakeRetriever({"initial": [document]}),
        store=storage[0],
        journal=storage[1],
        output_dir=tmp_path / "reports",
        source_mode="local_corpus",
        compressor=compressor,
    )
    bundle = storage[0].get_report(result["run_id"])
    context = json.loads(gateway.calls[-1][1].messages[-1].content)
    assert 1 <= len(context["evidence"]) <= 2
    for item in context["evidence"]:
        assert document.text[item["start"] : item["end"]] == item["quote"]
        assert item in bundle["evidence"]
    audit = bundle["compression"]
    assert audit["encoder"]["kind"] == "lexical"
    assert audit["encoder"]["encoder_id"] == compressor.encoder.encoder_id
    assert audit["textrank"]["iterations"] >= 1
    assert audit["report_evidence_bytes"] <= 512
    assert audit["report_request_bytes"] <= MAX_INPUT_BYTES
    assert bundle["sources"][0]["text"] == document.text


async def test_report_cannot_cite_archived_evidence_that_was_not_sent_to_model(tmp_path, storage):
    class WrongCitationGateway(FakeGateway):
        async def generate_structured(self, request, schema):
            result = await super().generate_structured(request, schema)
            if schema is not ResearchReport:
                return result
            return StructuredResult(
                data=ResearchReport(
                    title="Unseen citation",
                    sections=[
                        ReportSection(
                            heading="Claim",
                            claims=[CitedClaim(text="An unsupported claim.", citations=["E1"])],
                        )
                    ],
                    limitations=["Test fixture."],
                ),
                response=result.response,
            )

    gateway = WrongCitationGateway()
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, storage, gateway=gateway, compressor=SpanCompressor())
    assert caught.value.code == "invalid_citation"
    assert [item["evidence_id"] for item in storage[0].get_evidence(caught.value.run_id)] == [
        "E1",
        "E2",
        "E3",
    ]
    assert storage[0].get_report(caught.value.run_id) is None
    assert storage[0].get_run(caught.value.run_id)["status"] == "failed"
    assert not (tmp_path / "reports").exists()


async def test_disabled_compression_preserves_m2_prompt_and_bundle(tmp_path, storage):
    gateway = FakeGateway()
    result = await execute(tmp_path, storage, gateway=gateway)
    assert "compression" not in result
    bundle = storage[0].get_report(result["run_id"])
    assert bundle["schema_version"] == 2
    assert "compression" not in bundle
    assert json.loads(gateway.calls[-1][1].messages[-1].content)["evidence"] == bundle["evidence"]


async def test_full_model_request_including_json_schema_stays_within_input_limit(tmp_path, storage):
    class RecordingProvider:
        def __init__(self):
            self.calls = []

        async def complete(self, request):
            self.calls.append(request)
            if len(self.calls) == 1:
                data = {"queries": ["initial"]}
            else:
                context = json.loads(request.messages[-1].content)
                item = context["evidence"][0]
                data = {
                    "title": "Compact report",
                    "sections": [
                        {
                            "heading": "Evidence",
                            "claims": [{"text": item["quote"], "citations": [item["evidence_id"]]}],
                        }
                    ],
                    "limitations": ["Offline fixture."],
                }
            return ModelResponse(
                provider="fake",
                model="offline",
                request_id=f"call-{len(self.calls)}",
                content=json.dumps(data, ensure_ascii=False),
                latency_ms=0.0,
            )

    provider = RecordingProvider()
    result = await execute(
        tmp_path, storage, gateway=ModelGateway(provider), compressor=SpanCompressor(budget=512)
    )
    assert result["status"] == "completed"
    assert len(provider.calls) == 2
    for request in provider.calls:
        assert request.response_format == "json_object"
        assert "JSON Schema" in request.messages[0].content
        assert len(request.model_dump_json().encode()) <= MAX_INPUT_BYTES
    evidence = json.loads(provider.calls[-1].messages[-1].content)["evidence"]
    assert len(json.dumps(evidence, ensure_ascii=False, separators=(",", ":")).encode()) <= 512


async def test_compressor_failure_is_explicit_and_does_not_silently_use_baseline(tmp_path, storage):
    gateway = FakeGateway()
    compressor = SpanCompressor(failure=ResearchError("compression_invalid", "Compression failed."))
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, storage, gateway=gateway, compressor=compressor)
    assert caught.value.code == "compression_invalid"
    assert [stage for stage, _ in gateway.calls] == ["plan"]
    assert len(storage[0].get_evidence(caught.value.run_id)) == 2
    assert storage[0].get_report(caught.value.run_id) is None
    assert not (tmp_path / "reports").exists()


async def test_compression_span_storage_failure_does_not_export_success(
    tmp_path, storage, monkeypatch
):
    store = storage[0]
    original_save = store.save_sources

    def fail_additional_span(run_id, sources, evidence):
        if not sources and any(item["evidence_id"] == "E3" for item in evidence):
            raise ResearchStoreError("store_io", "Selected span could not be stored.")
        return original_save(run_id, sources, evidence)

    monkeypatch.setattr(store, "save_sources", fail_additional_span)
    gateway = FakeGateway()
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, storage, gateway=gateway, compressor=SpanCompressor())
    assert caught.value.code == "store_io"
    assert [stage for stage, _ in gateway.calls] == ["plan"]
    assert store.get_run(caught.value.run_id)["status"] == "failed"
    assert len(store.get_evidence(caught.value.run_id)) == 2
    assert store.get_report(caught.value.run_id) is None


async def test_global_deadline_drains_compression_then_uses_baseline_without_model_call(
    tmp_path, storage, monkeypatch
):
    compressor, gateway = SpanCompressor(stall=True), FakeGateway()
    original_export = dag_research._export_bundle

    def export_after_cleanup(bundle, output_dir):
        assert compressor.drained.is_set()
        return original_export(bundle, output_dir)

    monkeypatch.setattr(dag_research, "_export_bundle", export_after_cleanup)
    result = await execute(
        tmp_path,
        storage,
        gateway=gateway,
        compressor=compressor,
        options=DagOptions(total_timeout_seconds=0.2),
    )
    assert compressor.entered.is_set() and compressor.drained.is_set()
    assert result["report_kind"] == "evidence_digest"
    assert result["degraded"] is True
    assert result["compression"]["status"] == "skipped_deadline"
    assert [stage for stage, _ in gateway.calls] == ["plan"]
    bundle = storage[0].get_report(result["run_id"])
    assert [item["quote"] for item in bundle["evidence"]] == LINES[:2]
    claims = bundle["report"]["sections"][0]["claims"]
    assert [item["text"] for item in claims] == ["原文摘录：" + line for line in LINES[:2]]


async def test_user_cancel_during_compression_never_exports_digest(tmp_path, storage):
    compressor, gateway = SpanCompressor(stall=True), FakeGateway()
    task = asyncio.create_task(
        execute(
            tmp_path,
            storage,
            gateway=gateway,
            compressor=compressor,
        )
    )
    try:
        await asyncio.wait_for(compressor.entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert compressor.drained.is_set()
        run_id = storage[0].last_run_id
        assert storage[0].get_run(run_id)["status"] == "cancelled"
        assert storage[0].get_report(run_id) is None
        assert len(storage[0].get_evidence(run_id)) == 2
        assert not (tmp_path / "reports").exists()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_compression_keeps_memory_ids_and_only_publishes_selected_fresh_evidence(
    tmp_path, storage
):
    old = SourceDocument(
        "https://example.org/old",
        "Historical evidence",
        "Evidence source tracking keeps historical material traceable.",
    )
    seed = await execute(tmp_path, storage, retriever=FakeRetriever({"initial": [old]}))
    store, _, memory = storage
    memory.ingest_run(seed["run_id"], scope="study")
    original_time = store.get_sources(seed["run_id"])[0]["retrieved_at"]
    gateway = FakeGateway()
    result = await execute(
        tmp_path,
        storage,
        gateway=gateway,
        compressor=SpanCompressor(),
        memory=memory,
        memory_scope="study",
    )
    bundle = store.get_report(result["run_id"])
    assert [item["evidence_id"] for item in bundle["evidence"]] == ["E1", "E2", "E3", "E4"]
    recalled = bundle["memory"]["recalled"][0]
    assert recalled["evidence_id"] == "E4"
    assert recalled["origin_run_id"] == seed["run_id"]
    assert recalled["origin_evidence_id"] == "E1"
    assert recalled["retrieved_at"] == original_time
    context = json.loads(gateway.calls[-1][1].messages[-1].content)
    assert [item["evidence_id"] for item in context["evidence"]] == ["E3", "E4"]
    assert bundle["compression"]["preserved_memory_evidence_ids"] == ["E4"]
    assert result["memory_write"]["status"] == "saved"
    assert result["memory_write"]["added_items"] == 1
    assert result["memory_write"]["added_provenance"] == 1
    assert result["memory_write"]["skipped_recalled"] == 1
    stats = memory.stats(scope="study", source_mode="local_corpus")
    assert stats["items"] == 2 and stats["provenance"] == 2
    imported_again = memory.ingest_run(result["run_id"], scope="study")
    assert imported_again["added_items"] == 0
    assert imported_again["added_provenance"] == 0
    assert imported_again["skipped_recalled"] == 1


async def test_mandatory_memory_exceeding_context_budget_fails_without_silently_dropping_it(
    tmp_path, storage
):
    paragraph = "Evidence source tracking " * 10
    old = SourceDocument(
        "https://example.org/long-history",
        "Historical paragraphs",
        paragraph + "first.\n" + paragraph + "second.",
    )
    seed = await execute(tmp_path, storage, retriever=FakeRetriever({"initial": [old]}))
    memory = storage[2]
    assert memory.ingest_run(seed["run_id"], scope="study")["added_items"] == 2
    assert len(memory.search(QUESTION, scope="study", source_mode="local_corpus")["hits"]) == 2
    gateway = FakeGateway()
    with pytest.raises(ResearchRunError) as caught:
        await execute(
            tmp_path,
            storage,
            gateway=gateway,
            compressor=SpanCompressor(budget=512),
            memory=memory,
            memory_scope="study",
        )
    assert caught.value.code == "compression_budget"
    assert [stage for stage, _ in gateway.calls] == ["plan"]
    assert storage[0].get_report(caught.value.run_id) is None
    assert memory.stats(scope="study", source_mode="local_corpus")["items"] == 2


async def test_budget_reserves_history_and_explicitly_reports_omitted_fresh_evidence(
    tmp_path, storage
):
    old = SourceDocument(
        "https://example.org/reserved-history",
        "Historical observation",
        "Evidence source tracking " * 11 + "preserved completely.😀",
    )
    seed = await execute(tmp_path, storage, retriever=FakeRetriever({"initial": [old]}))
    memory = storage[2]
    assert memory.ingest_run(seed["run_id"], scope="study")["added_items"] == 1
    gateway = FakeGateway()
    result = await execute(
        tmp_path,
        storage,
        gateway=gateway,
        compressor=SpanCompressor(budget=512),
        memory=memory,
        memory_scope="study",
    )
    bundle = storage[0].get_report(result["run_id"])
    context = json.loads(gateway.calls[-1][1].messages[-1].content)
    assert [item["evidence_id"] for item in context["evidence"]] == ["E4"]
    assert context["evidence"][0]["quote"] == old.text
    assert result["compression"]["preserved_memory_evidence_ids"] == ["E4"]
    assert result["compression"]["budget_omitted_fresh_ids"] == ["E3"]
    assert result["compression"]["report_evidence_bytes"] <= 512
    assert [item["evidence_id"] for item in bundle["evidence"]] == ["E1", "E2", "E3", "E4"]
    assert result["degraded"] is True
    assert "compression_fresh_omitted" in result["reasons"]
    assert memory.stats(scope="study", source_mode="local_corpus")["items"] == 1


async def test_compression_allows_cache_only_but_marks_it_degraded(tmp_path, storage):
    seed = await execute(tmp_path, storage)
    memory = storage[2]
    memory.ingest_run(seed["run_id"], scope="study")
    gateway, compressor = FakeGateway(), SpanCompressor()
    result = await execute(
        tmp_path,
        storage,
        gateway=gateway,
        retriever=FakeRetriever({"initial": [], "revised": []}),
        compressor=compressor,
        memory=memory,
        memory_scope="study",
    )
    assert compressor.calls == [(QUESTION, [])]
    assert [stage for stage, _ in gateway.calls] == ["plan", "replan", "report"]
    assert result["degraded"] is True and "memory_only" in result["reasons"]
    context = json.loads(gateway.calls[-1][1].messages[-1].content)
    assert {item["quote"] for item in context["evidence"]} == set(LINES[:2])
    assert result["compression"]["selected_evidence_ids"] == ["E1", "E2"]
    assert result["compression"]["preserved_memory_evidence_ids"] == ["E1", "E2"]
    assert memory.stats(scope="study", source_mode="local_corpus")["items"] == 2


async def test_audit_finishing_report_after_deadline_must_not_export_an_ordinary_report(
    tmp_path, storage, monkeypatch
):
    gateway = FakeGateway()
    journal = storage[1]
    original_finish = journal.finish_model_call
    loop = asyncio.get_running_loop()
    original_clock = loop.time
    offset = 0.0

    def shifted_clock():
        return original_clock() + offset

    def finish_after_deadline(run_id, call_id, status, details):
        nonlocal offset
        result = original_finish(run_id, call_id, status, details)
        if call_id == 2 and status == "completed":
            # The response is real, but its synchronous durable audit consumed the remaining budget.
            offset = 2.0
        return result

    monkeypatch.setattr(journal, "finish_model_call", finish_after_deadline)
    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(loop, "time", shifted_clock)
        try:
            result = await execute(
                tmp_path,
                storage,
                gateway=gateway,
                compressor=SpanCompressor(),
                options=DagOptions(total_timeout_seconds=1.0),
            )
        finally:
            offset = 0.0
    assert [stage for stage, _ in gateway.calls] == ["plan", "report"]
    assert result["report_kind"] == "evidence_digest"
    assert result["degraded"] is True
    assert "global_timeout" in result["reasons"]
    assert result["compression"]["status"] == "skipped_deadline"
    assert storage[0].get_run(result["run_id"])["status"] == "completed"


@pytest.mark.parametrize("damage", ["missing", "null"])
async def test_m4_memory_import_rejects_missing_selection_audit_instead_of_importing_baseline(
    tmp_path, storage, damage
):
    result = await execute(tmp_path, storage, compressor=SpanCompressor())
    store, _, memory = storage
    bundle = store.get_report(result["run_id"])
    assert bundle["schema_version"] == 4
    assert bundle["compression"]["selected_evidence_ids"] == ["E3"]
    if damage == "missing":
        bundle.pop("compression")
    else:
        bundle["compression"] = None
    store._db().execute(
        "UPDATE research_runs SET report_json = ? WHERE id = ?",
        (json.dumps(bundle, ensure_ascii=False), result["run_id"]),
    )
    with pytest.raises(MemoryError) as caught:
        memory.ingest_run(result["run_id"], scope="study")
    assert caught.value.code == "memory_validation"
    assert memory.stats(scope="study", source_mode="local_corpus")["items"] == 0


async def test_report_timeout_does_not_relabel_abandoned_request_bytes_as_digest_bytes(
    tmp_path, storage
):
    gateway = FakeGateway(stall_stage="report")
    result = await execute(
        tmp_path,
        storage,
        gateway=gateway,
        compressor=SpanCompressor(),
        options=DagOptions(total_timeout_seconds=0.2),
    )
    assert gateway.entered.is_set() and gateway.drained.is_set()
    assert result["report_kind"] == "evidence_digest"
    bundle = storage[0].get_report(result["run_id"])
    audit = bundle["compression"]
    assert audit["selected_evidence_ids"] == ["E1", "E2", "E3"]
    used = [
        item for item in bundle["evidence"] if item["evidence_id"] in audit["selected_evidence_ids"]
    ]
    actual_bytes = len(json.dumps(used, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    assert "report_evidence_bytes" not in audit or audit["report_evidence_bytes"] == actual_bytes
    assert [call["status"] for call in bundle["scheduler_audit"]["model_calls"]] == [
        "completed",
        "cancelled",
    ]


async def test_omitting_a_whole_fresh_source_marks_coverage_gap_and_blocks_memory_publication(
    tmp_path, storage
):
    second = SourceDocument(
        "https://example.org/second",
        "Second source",
        "Evidence source tracking offers a different observation.",
    )

    class FirstSourceOnly(SpanCompressor):
        async def compress(self, question, sources):
            return await super().compress(question, sources[:1])

    gateway = FakeGateway()
    result = await execute(
        tmp_path,
        storage,
        gateway=gateway,
        retriever=FakeRetriever({"initial": [DOCUMENT, second]}),
        compressor=FirstSourceOnly(),
        memory=storage[2],
        memory_scope="study",
    )
    assert result["source_count"] == 2
    assert result["degraded"] is True
    assert "compression_coverage" in result["reasons"]
    assert result["compression"]["report_uncovered_source_ids"] == ["S2"]
    context = json.loads(gateway.calls[-1][1].messages[-1].content)
    assert context["compression"]["uncovered_source_ids"] == ["S2"]
    assert {"stage": "compression", "code": "compression_coverage"} in context["retrieval_warnings"]
    assert result["memory_write"] == {"status": "skipped", "reason": "degraded_run"}
    assert storage[2].stats(scope="study", source_mode="local_corpus")["items"] == 0


async def test_memory_copy_failure_after_compression_deadline_cannot_become_a_digest(
    tmp_path, storage, monkeypatch
):
    old = SourceDocument(
        "https://example.org/history-copy",
        "Previous source",
        "Evidence source tracking preserves historical observations.",
    )
    seed = await execute(tmp_path, storage, retriever=FakeRetriever({"initial": [old]}))
    store, _, memory = storage
    memory.ingest_run(seed["run_id"], scope="study")
    original_save = store.save_sources

    def fail_historical_copy(run_id, sources, evidence):
        if any(source["url"] == old.url for source in sources):
            raise ResearchStoreError("store_io", "Historical evidence could not be copied.")
        return original_save(run_id, sources, evidence)

    monkeypatch.setattr(store, "save_sources", fail_historical_copy)
    compressor, gateway = SpanCompressor(stall=True), FakeGateway()
    with pytest.raises(ResearchRunError) as caught:
        await execute(
            tmp_path,
            storage,
            gateway=gateway,
            compressor=compressor,
            memory=memory,
            memory_scope="study",
            options=DagOptions(total_timeout_seconds=0.2),
        )
    assert compressor.drained.is_set()
    assert caught.value.code == "store_io"
    assert store.get_run(caught.value.run_id)["status"] == "failed"
    assert store.get_report(caught.value.run_id) is None
    assert not (tmp_path / "reports" / caught.value.run_id).exists()
    assert [stage for stage, _ in gateway.calls] == ["plan"]
    assert memory.stats(scope="study", source_mode="local_corpus")["items"] == 1


@pytest.mark.parametrize("damage", ["cite_unselected", "corrupt_unselected"])
async def test_memory_import_still_validates_archived_evidence_and_selected_citations(
    tmp_path, storage, damage
):
    result = await execute(tmp_path, storage, compressor=SpanCompressor())
    store, _, memory = storage
    if damage == "cite_unselected":
        bundle = store.get_report(result["run_id"])
        bundle["report"]["sections"][0]["claims"][0]["citations"] = ["E1"]
        store._db().execute(
            "UPDATE research_runs SET report_json = ? WHERE id = ?",
            (json.dumps(bundle, ensure_ascii=False), result["run_id"]),
        )
    else:
        store._db().execute(
            "UPDATE research_evidence SET quote = ? WHERE run_id = ? AND evidence_id = 'E1'",
            ("This archived quote was altered after the run.", result["run_id"]),
        )
    with pytest.raises(MemoryError) as caught:
        memory.ingest_run(result["run_id"], scope="study")
    assert caught.value.code == "memory_validation"
    assert memory.stats(scope="study", source_mode="local_corpus")["items"] == 0


async def test_tiny_evidence_does_not_bypass_full_request_budget_with_large_metadata(
    tmp_path, storage
):
    oversized = SourceDocument(DOCUMENT.url, "来源标题" * 6000, DOCUMENT.text)
    gateway = FakeGateway()
    with pytest.raises(ResearchRunError) as caught:
        await execute(
            tmp_path,
            storage,
            gateway=gateway,
            retriever=FakeRetriever({"initial": [oversized]}),
            compressor=SpanCompressor(budget=512),
        )
    assert caught.value.code == "compression_budget"
    assert [stage for stage, _ in gateway.calls] == ["plan"]
    assert storage[0].get_run(caught.value.run_id)["status"] == "failed"
    assert storage[0].get_report(caught.value.run_id) is None
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize("length", [650, 651, 4000])
def test_digest_shortens_only_the_display_without_mutating_archived_quotes(length):
    quote = ("原文😀" * length)[:length]
    archived = [{"evidence_id": "E1", "source_id": "S1", "start": 0, "end": length, "quote": quote}]
    before = json.loads(json.dumps(archived, ensure_ascii=False))
    report = dag_research.evidence_digest(archived)
    claim = report.sections[0].claims[0]
    assert len(claim.text) <= 700
    assert claim.citations == ["E1"]
    assert archived == before
    if length <= 650:
        assert claim.text == "原文摘录：" + quote
    else:
        assert claim.text.startswith("原文节选（仅展示前600字符，完整原文见证据）：")
        assert claim.text.endswith(quote[:600])


async def test_deadline_digest_of_long_historical_evidence_keeps_full_archive_and_provenance(
    tmp_path, storage
):
    store, _, memory = storage
    seed_id = store.create_run(QUESTION)
    quote = ("Evidence source tracking 历史😀。" * 150)[:4000]
    original_time = datetime.now(UTC).isoformat()
    source = {
        "source_id": "S1",
        "url": "https://example.org/long-archive",
        "title": "Long history",
        "text": quote,
        "sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest(),
        "retrieved_at": original_time,
        "truncated": False,
    }
    original = {
        "evidence_id": "E1",
        "source_id": "S1",
        "start": 0,
        "end": len(quote),
        "quote": quote,
    }
    report = ResearchReport(
        title="Historical report",
        sections=[
            ReportSection(
                heading="Source",
                claims=[CitedClaim(text="Historical source recorded.", citations=["E1"])],
            )
        ],
        limitations=["Fixture retains its original evidence."],
    )
    store.save_sources(seed_id, [source], [original])
    store.complete_run(
        seed_id,
        {
            "schema_version": 2,
            "run_id": seed_id,
            "retrieval_mode": "local_corpus",
            "warnings": [],
            "execution": {"degraded": False},
            "report": report.model_dump(),
            "sources": [source],
            "evidence": [original],
        },
        [],
        "fixture/seed",
    )
    assert memory.ingest_run(seed_id, scope="study")["added_items"] == 1
    gateway = FakeGateway(stall_stage="plan")
    result = await execute(
        tmp_path,
        storage,
        gateway=gateway,
        compressor=SpanCompressor(budget=512),
        memory=memory,
        memory_scope="study",
        options=DagOptions(total_timeout_seconds=0.2),
    )
    assert gateway.entered.is_set() and gateway.drained.is_set()
    assert [stage for stage, _ in gateway.calls] == ["plan"]
    assert result["report_kind"] == "evidence_digest" and result["degraded"] is True
    bundle = store.get_report(result["run_id"])
    assert bundle["evidence"] == [original]
    assert store.get_evidence(result["run_id"]) == [original]
    assert bundle["sources"][0]["text"] == quote
    assert bundle["sources"][0]["retrieved_at"] == original_time
    claim = bundle["report"]["sections"][0]["claims"][0]
    assert len(claim["text"]) <= 700
    assert "仅展示前600字符" in claim["text"] and claim["text"].endswith(quote[:600])
    assert claim["citations"] == ["E1"]
    assert bundle["memory"]["recalled"][0]["origin_run_id"] == seed_id
    assert bundle["compression"]["report_evidence_bytes"] > 512
    assert bundle["compression"]["budget_applies"] is False
    assert result["memory_write"]["status"] == "skipped"
    assert memory.stats(scope="study", source_mode="local_corpus")["items"] == 1

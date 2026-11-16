"""Offline research integration tests: real storage, fake retrieval and model calls."""

import asyncio
import hashlib
import json
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest

from deep_research import research
from deep_research.errors import ModelError, ResearchError
from deep_research.gateway import StructuredResult
from deep_research.research import (
    ResearchRunError,
    render_markdown,
    run_research_pipeline,
    select_evidence,
    validate_citations,
)
from deep_research.research_models import (
    CitedClaim,
    ReportSection,
    ResearchPlan,
    ResearchReport,
)
from deep_research.research_store import ResearchStore
from deep_research.retrieval import ResearchRetrievalError, SearchHit, SourceDocument
from deep_research.schemas import ModelRequest, ModelResponse


def report(citation: str = "E1", text: str = "Stored quotations can be traced to their source."):
    return ResearchReport(
        title="Evidence tracking",
        sections=[
            ReportSection(heading="Findings", claims=[CitedClaim(text=text, citations=[citation])])
        ],
        limitations=["A valid citation does not prove that a claim follows from its source."],
    )


class FakeGateway:
    def __init__(self, *, queries=None, result=None, fail_stage=None):
        self.queries = queries or ["evidence"]
        self.result = result or report()
        self.fail_stage = fail_stage
        self.calls: list[tuple[ModelRequest, type]] = []

    async def generate_structured(self, request, schema):
        self.calls.append((request, schema))
        if schema is self.fail_stage:
            raise ModelError("rate_limit", "Model temporarily limited.", status_code=429)
        if schema is ResearchPlan:
            data = ResearchPlan(queries=self.queries)
        else:
            assert schema is ResearchReport
            data = self.result
        return StructuredResult(
            data=data,
            response=ModelResponse(
                provider="fake",
                model="offline",
                request_id=f"mock-{len(self.calls)}",
                content=data.model_dump_json(),
                latency_ms=1.0,
            ),
        )


class FakeRetriever:
    def __init__(self, documents=None, *, searches=None, fetch_errors=None):
        self.documents = (
            list(documents)
            if documents is not None
            else [
                SourceDocument(
                    "https://example.org/evidence",
                    "Evidence reference",
                    "Evidence tracking preserves source quotations.\n"
                    "Readers can compare each quotation against the saved original text.",
                )
            ]
        )
        self.searches = searches
        self.fetch_errors = fetch_errors or {}
        self.search_calls: list[tuple[str, int]] = []
        self.fetch_calls: list[str] = []

    async def search(self, query: str, limit: int = 3):
        self.search_calls.append((query, limit))
        value = (
            self.searches[query]
            if self.searches is not None
            else [SearchHit(item.url, item.title) for item in self.documents]
        )
        if isinstance(value, Exception):
            raise value
        return value

    async def fetch(self, hit: SearchHit):
        self.fetch_calls.append(hit.url)
        if hit.url in self.fetch_errors:
            raise self.fetch_errors[hit.url]
        return next(item for item in self.documents if item.url == hit.url)


class TrackingStore(ResearchStore):
    last_run_id: str | None = None

    def create_run(self, question: str) -> str:
        self.last_run_id = super().create_run(question)
        return self.last_run_id


@pytest.fixture
def store(tmp_path: Path) -> Iterator[TrackingStore]:
    with TrackingStore(tmp_path / "research.sqlite") as opened:
        yield opened


async def execute(tmp_path, store, *, gateway=None, retriever=None, **kwargs):
    return await run_research_pipeline(
        "How does evidence tracking work?",
        gateway=gateway or FakeGateway(),
        retriever=retriever or FakeRetriever(),
        store=store,
        output_dir=tmp_path / "reports",
        **kwargs,
    )


async def test_completed_run_exports_traceable_report_and_exact_evidence(tmp_path, store):
    gateway, retriever = FakeGateway(), FakeRetriever()
    progress: list[str] = []
    result = await execute(
        tmp_path, store, gateway=gateway, retriever=retriever, on_progress=progress.append
    )

    assert result["status"] == "completed"
    assert result["source_count"] == 1
    assert result["evidence_count"] == 2
    assert result["model_call_count"] == 2
    assert result["warnings"] == []
    assert [schema for _, schema in gateway.calls] == [ResearchPlan, ResearchReport]
    assert retriever.search_calls == [("evidence", research.MAX_HITS_PER_QUERY)]
    assert retriever.fetch_calls == [retriever.documents[0].url]
    assert progress

    run_id = result["run_id"]
    record = store.get_run(run_id)
    assert record["status"] == "completed"
    assert record["finished_at"] is not None
    assert [call["stage"] for call in record["usage"]] == ["plan", "report"]
    bundle = json.loads(
        await asyncio.to_thread(Path(result["evidence_path"]).read_text, encoding="utf-8")
    )
    assert bundle == store.get_report(run_id)
    assert bundle["sources"] == store.get_sources(run_id)
    assert bundle["evidence"] == store.get_evidence(run_id)
    original = retriever.documents[0].text
    assert bundle["sources"][0]["text"] == original
    assert bundle["sources"][0]["sha256"] == hashlib.sha256(original.encode()).hexdigest()
    for item in bundle["evidence"]:
        assert original[item["start"] : item["end"]] == item["quote"]
    context = json.loads(gateway.calls[1][0].messages[-1].content)
    assert context["evidence"] == bundle["evidence"]
    markdown = await asyncio.to_thread(Path(result["report_path"]).read_text, encoding="utf-8")
    assert "[E1](#evidence-e1)" in markdown
    assert 'id="evidence-e1"' in markdown
    assert retriever.documents[0].url in markdown
    assert Path(result["report_path"]).parent == tmp_path.resolve() / "reports" / run_id


@pytest.mark.parametrize("failure", ["no_hits", "fetch_error", "empty_text"])
async def test_no_original_evidence_stops_before_report_generation(tmp_path, store, failure):
    gateway = FakeGateway()
    if failure == "no_hits":
        retriever = FakeRetriever([])
    else:
        document = SourceDocument("https://example.org/a", "Search title is not evidence", "  ")
        errors = {document.url: ResearchRetrievalError("retrieval_http", "Fetch failed.")}
        retriever = FakeRetriever(
            [document], fetch_errors=errors if failure == "fetch_error" else {}
        )

    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, store, gateway=gateway, retriever=retriever)

    assert caught.value.code == "no_evidence"
    assert caught.value.run_id == store.last_run_id
    record = store.get_run(caught.value.run_id)
    assert record["status"] == "failed"
    assert record["error_code"] == "no_evidence"
    assert len(gateway.calls) == 1
    assert store.get_report(caught.value.run_id) is None
    assert not (tmp_path / "reports").exists()


async def test_retrieval_errors_degrade_with_warnings_and_retained_originals(tmp_path, store):
    good = SourceDocument(
        "https://example.org/good", "Good evidence", "Evidence survives partial failure."
    )
    bad = SearchHit("https://example.org/bad", "Unavailable")
    gateway = FakeGateway(queries=["first", "second"])
    retriever = FakeRetriever(
        [good],
        searches={
            "first": ResearchRetrievalError("retrieval_network", "Unavailable."),
            "second": [bad, SearchHit(good.url, good.title)],
        },
        fetch_errors={bad.url: ResearchRetrievalError("source_missing", "Missing.")},
    )
    result = await execute(tmp_path, store, gateway=gateway, retriever=retriever)
    assert result["status"] == "completed"
    assert result["source_count"] == 1
    assert result["warnings"] == [
        {"stage": "search", "code": "retrieval_network"},
        {"stage": "fetch", "code": "source_missing"},
    ]
    assert store.get_sources(result["run_id"])[0]["text"] == good.text
    context = json.loads(gateway.calls[1][0].messages[-1].content)
    assert context["retrieval_warnings"] == result["warnings"]


async def test_search_results_and_fetches_are_bounded_across_queries(tmp_path, store):
    documents = [
        SourceDocument(f"https://example.org/{i}", f"Source {i}", f"Evidence item {i}.")
        for i in range(12)
    ]
    hits = [SearchHit(document.url, document.title) for document in documents]
    gateway = FakeGateway(queries=["one", "two", "three"])
    # A buggy retriever returning more than requested must not expand the run budget.
    retriever = FakeRetriever(documents, searches={"one": hits, "two": hits[3:], "three": hits[6:]})
    result = await execute(tmp_path, store, gateway=gateway, retriever=retriever)
    assert len(retriever.search_calls) == 3
    assert all(limit == 3 for _, limit in retriever.search_calls)
    assert retriever.fetch_calls == [document.url for document in documents[:4]]
    assert result["source_count"] == 4
    assert result["evidence_count"] <= 8


async def test_duplicate_urls_and_text_are_not_double_counted(tmp_path, store):
    documents = [
        SourceDocument("https://example.org/one", "One", "Repeated evidence text."),
        SourceDocument("https://example.org/two", "Two", "Repeated evidence text."),
    ]
    retriever = FakeRetriever(documents)
    result = await execute(
        tmp_path, store, gateway=FakeGateway(queries=["first", "second"]), retriever=retriever
    )
    assert retriever.fetch_calls == [document.url for document in documents]
    assert result["source_count"] == 1
    assert {"stage": "fetch", "code": "duplicate_content"} in result["warnings"]


async def test_large_utf8_source_is_bounded_without_breaking_quote_offsets(tmp_path, store):
    original = "研究证据😀" * 8000
    retriever = FakeRetriever([SourceDocument("https://example.org/long", "Long source", original)])
    result = await execute(tmp_path, store, retriever=retriever)
    source = store.get_sources(result["run_id"])[0]
    assert source["truncated"] is True
    assert len(source["text"].encode()) <= research.MAX_SOURCE_BYTES
    assert original.startswith(source["text"])
    assert "\ufffd" not in source["text"]
    for item in store.get_evidence(result["run_id"]):
        assert source["text"][item["start"] : item["end"]] == item["quote"]
    assert any("截断" in item for item in store.get_report(result["run_id"])["system_limitations"])


async def test_invented_evidence_id_fails_without_exporting_report(tmp_path, store):
    gateway = FakeGateway(result=report("E999"))
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, store, gateway=gateway)
    assert caught.value.code == "invalid_citation"
    assert len(gateway.calls) == 2
    assert store.get_run(caught.value.run_id)["status"] == "failed"
    assert store.get_sources(caught.value.run_id)
    assert store.get_report(caught.value.run_id) is None
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize("stage", [ResearchPlan, ResearchReport])
async def test_model_errors_keep_code_and_run_id_without_retry(tmp_path, store, stage):
    gateway = FakeGateway(fail_stage=stage)
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, store, gateway=gateway)
    assert caught.value.code == "rate_limit"
    assert caught.value.status_code == 429
    assert len(gateway.calls) == (1 if stage is ResearchPlan else 2)
    assert store.get_run(caught.value.run_id)["error_code"] == "rate_limit"
    assert store.get_report(caught.value.run_id) is None


async def test_unexpected_retriever_bug_is_not_hidden_as_a_retrieval_warning(tmp_path, store):
    retriever = FakeRetriever(searches={"evidence": RuntimeError("programming defect")})
    with pytest.raises(RuntimeError, match="programming defect"):
        await execute(tmp_path, store, retriever=retriever)
    assert store.get_run(store.last_run_id)["error_code"] == "unexpected"
    assert store.get_run(store.last_run_id)["status"] == "failed"


async def test_cancellation_propagates_and_marks_run_cancelled(tmp_path, store):
    entered = asyncio.Event()

    class WaitingGateway:
        async def generate_structured(self, request, schema):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(execute(tmp_path, store, gateway=WaitingGateway()))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        record = store.get_run(store.last_run_id)
        assert record["status"] == "cancelled"
        assert record["error_code"] == "cancelled"
        assert not (tmp_path / "reports").exists()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_global_deadline_marks_failed_run_and_exposes_its_id(tmp_path, store, monkeypatch):
    class WaitingGateway:
        async def generate_structured(self, request, schema):
            await asyncio.Event().wait()

    monkeypatch.setattr(research, "RUN_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(ResearchRunError) as caught:
        async with asyncio.timeout(2):
            await execute(tmp_path, store, gateway=WaitingGateway())
    assert caught.value.code == "run_timeout"
    assert store.get_run(caught.value.run_id)["status"] == "failed"
    assert store.get_run(caught.value.run_id)["error_code"] == "run_timeout"


async def test_output_directory_failure_preserves_evidence_and_records_failure(tmp_path, store):
    # An existing file cannot serve as the parent report directory.
    (tmp_path / "reports").write_text("Keep this existing file.", encoding="utf-8")
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, store)
    assert caught.value.code == "storage_io"
    assert store.get_sources(caught.value.run_id)
    assert store.get_run(caught.value.run_id)["status"] == "failed"
    assert (tmp_path / "reports").read_text(encoding="utf-8") == "Keep this existing file."


def source(text="Unrelated first paragraph.\nEvidence 中文😀 stays exact.\nLast paragraph."):
    return {
        "source_id": "S1",
        "text": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "url": "https://example.org/source",
        "title": "Original",
        "truncated": False,
        "retrieved_at": "2026-10-15T00:00:00+00:00",
    }


def test_evidence_selection_prefers_relevant_paragraph_and_preserves_unicode_offsets():
    original = source()
    evidence = select_evidence([original], "Evidence 中文")
    assert len(evidence) == 2
    assert any("中文😀" in item["quote"] for item in evidence)
    for item in evidence:
        assert original["text"][item["start"] : item["end"]] == item["quote"]
    assert [item["start"] for item in evidence] == sorted(item["start"] for item in evidence)
    validate_citations(report(), [original], evidence)


@pytest.mark.parametrize("tampering", ["quote", "offset", "hash", "source", "duplicate"])
def test_citation_validation_rejects_tampered_provenance(tampering):
    sources = [source()]
    evidence = select_evidence(sources, "Evidence")
    if tampering == "quote":
        evidence[0]["quote"] = "Invented quotation."
    elif tampering == "offset":
        evidence[0]["end"] = len(sources[0]["text"]) + 1
    elif tampering == "hash":
        sources[0]["sha256"] = "0" * 64
    elif tampering == "source":
        evidence[0]["source_id"] = "S99"
    else:
        evidence.append(deepcopy(evidence[0]))
    with pytest.raises(ResearchError) as caught:
        validate_citations(report(), sources, evidence)
    assert caught.value.code == "invalid_evidence"


def test_markdown_escapes_generated_html_and_links_while_preserving_program_citations():
    sources = [source()]
    bundle = {
        "run_id": "offline-run",
        "question": "<script>unsafe()</script>",
        "report": report(text="[unexpected](javascript:alert(1)) <img src=x>").model_dump(),
        "sources": sources,
        "evidence": select_evidence(sources, "Evidence"),
        "system_limitations": [],
        "warnings": [],
    }
    output = render_markdown(bundle)
    assert "<script>" not in output
    assert "<img " not in output
    assert "[unexpected](javascript:" not in output
    assert "[E1](#evidence-e1)" in output
    assert "https://example.org/source" in output


async def test_failed_status_write_does_not_replace_original_model_error(
    tmp_path, store, monkeypatch
):
    def cannot_save(*args, **kwargs):
        raise RuntimeError("sensitive-database-location-and-secret")

    monkeypatch.setattr(store, "fail_run", cannot_save)
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, store, gateway=FakeGateway(fail_stage=ResearchPlan))

    assert caught.value.code == "rate_limit"
    assert caught.value.status_code == 429
    assert caught.value.run_id == store.last_run_id
    assert caught.value.status_saved is False
    assert "Model temporarily limited." in str(caught.value)
    assert "状态" in str(caught.value)
    assert "sensitive-database-location-and-secret" not in str(caught.value)
    assert store.get_run(store.last_run_id)["status"] == "running"


async def test_failed_cancel_status_write_keeps_cancellation_and_reports_uncertain_state(
    tmp_path, store, monkeypatch
):
    entered = asyncio.Event()
    messages: list[str] = []

    class WaitingGateway:
        async def generate_structured(self, request, schema):
            entered.set()
            await asyncio.Event().wait()

    def cannot_save(*args, **kwargs):
        raise RuntimeError("sensitive-cancellation-storage-secret")

    monkeypatch.setattr(store, "fail_run", cannot_save)
    task = asyncio.create_task(
        execute(tmp_path, store, gateway=WaitingGateway(), on_progress=messages.append)
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert store.get_run(store.last_run_id)["status"] == "running"
        assert any("取消" in message and "无法确认" in message for message in messages)
        assert any(store.last_run_id in message for message in messages)
        assert not any("sensitive-cancellation-storage-secret" in message for message in messages)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("progress_error", [BrokenPipeError, RuntimeError])
async def test_progress_callback_failure_does_not_change_successful_result(
    tmp_path, store, progress_error
):
    messages: list[str] = []

    def broken_progress(message: str) -> None:
        messages.append(message)
        raise progress_error("The progress display is unavailable.")

    result = await execute(tmp_path, store, on_progress=broken_progress)
    assert result["status"] == "completed"
    assert messages
    assert store.get_run(result["run_id"])["status"] == "completed"
    assert store.get_report(result["run_id"])["report"] == report().model_dump()
    saved = await asyncio.to_thread(Path(result["evidence_path"]).read_text, encoding="utf-8")
    assert json.loads(saved) == store.get_report(result["run_id"])

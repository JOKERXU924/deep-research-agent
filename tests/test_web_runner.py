"""M6 uses the existing M2--M5 pipeline; every model/network provider is a fixture."""

import asyncio
from pathlib import Path

import httpx
import pytest

import deep_research.web_runner as module
from deep_research.dag_demo import DagDemoRetriever
from deep_research.errors import ResearchError
from deep_research.research import ResearchRunError
from deep_research.research_store import ResearchStore
from deep_research.review_demo import ReviewDemoProvider
from deep_research.web_models import WebRequest


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "load_settings", lambda *_: pytest.fail("Unexpected secret access"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: pytest.fail("Unexpected network"))
    return {"db": tmp_path / "web.sqlite3", "output_dir": tmp_path / "reports"}


@pytest.mark.parametrize(
    "scenario,review_status,calls",
    [
        ("passed", "passed", 3),
        ("repair", "passed", 5),
        ("oscillation", "unresolved", 6),
        ("stalled", "unresolved", 4),
        ("max-rounds", "unresolved", 7),
        ("format", "passed", 8),
        ("deadline", "interrupted", 3),
    ],
)
async def test_all_existing_review_demo_scenarios_work_without_settings(
    paths, scenario, review_status, calls
):
    bound, progress = [], []
    result = await module.make_runner(**paths)(
        WebRequest(scenario=scenario), progress.append, bound.append
    )
    assert bound == [result["run_id"]]
    assert result["review"]["status"] == review_status
    assert result["model_call_count"] == calls
    assert result["status"] == "completed" and progress
    assert "report_path" not in result and "evidence_path" not in result
    assert str(paths["output_dir"]) not in str(result)
    with ResearchStore(paths["db"]) as store:
        assert store.get_run(bound[0])["status"] == "completed"
        assert store.get_report(bound[0])["retrieval_mode"] == "offline_demo"


@pytest.mark.parametrize("review", [False, True])
@pytest.mark.parametrize("compression", [False, True])
async def test_optional_features_reuse_existing_pipeline(paths, review, compression):
    result = await module.make_runner(**paths)(
        WebRequest(review=review, compression=compression), lambda _: None, lambda _: None
    )
    assert ("review" in result) == review
    assert ("compression" in result) == compression
    with ResearchStore(paths["db"]) as store:
        bundle = store.get_report(result["run_id"])
        assert bundle["schema_version"] == (5 if review else 4 if compression else 2)


async def test_binding_failure_is_before_any_model_and_marks_run_failed(paths, monkeypatch):
    bound = []
    monkeypatch.setattr(
        ReviewDemoProvider, "complete", lambda *_: pytest.fail("Unbound model call")
    )

    def bind(run_id):
        bound.append(run_id)
        raise ResearchError("web_store_io", "Fixture durable binding failure.")

    with pytest.raises(ResearchRunError) as caught:
        await module.make_runner(**paths)(WebRequest(), lambda _: None, bind)
    assert caught.value.code == "web_store_io" and bound
    with ResearchStore(paths["db"]) as store:
        assert store.get_run(bound[0])["status"] == "failed"
        assert store.get_report(bound[0]) is None


async def test_slow_demo_can_cancel_without_real_network(paths):
    bound = []
    ready = asyncio.Event()

    def bind(run_id):
        bound.append(run_id)
        ready.set()

    task = asyncio.create_task(
        module.make_runner(**paths)(WebRequest(scenario="slow"), lambda _: None, bind)
    )
    await asyncio.wait_for(ready.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with ResearchStore(paths["db"]) as store:
        assert store.get_run(bound[0])["status"] == "cancelled"
        assert store.get_report(bound[0]) is None


async def test_live_disabled_fails_before_credentials_or_database(paths):
    with pytest.raises(ResearchError) as caught:
        await module.make_runner(**paths)(
            WebRequest(mode="live", question="真实问题", confirm_live=True),
            lambda _: None,
            lambda _: None,
        )
    assert caught.value.code == "web_live_disabled"
    assert not paths["db"].exists()


@pytest.mark.parametrize("selection", ["coverage", "multilingual"])
async def test_mock_live_uses_wikipedia_and_same_review_pipeline(paths, monkeypatch, selection):
    touched = []
    monkeypatch.setattr(module, "load_settings", lambda path: touched.append(path) or object())

    class Provider(ReviewDemoProvider):
        def __init__(self, settings):
            super().__init__("passed")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Retriever(DagDemoRetriever):
        def __init__(self, language):
            assert language == "zh"
            super().__init__()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(module, "DeepSeekProvider", Provider)
    monkeypatch.setattr(module, "WikipediaRetriever", Retriever)
    result = await module.make_runner(
        **paths, allow_live=True, env_file=Path("not-read.env"), evidence_selection=selection
    )(
        # The live topic gate must judge the actual fixture topic, not test scaffolding.
        WebRequest(mode="live", question="异步与并发限制", confirm_live=True),
        lambda _: None,
        lambda _: None,
    )
    assert touched == [Path("not-read.env")]
    assert result["retrieval_quality"]["enabled"] is True
    assert result["retrieval_quality"]["selection"]["version"] == (
        "coverage-bm25-rrf-v1" if selection == "coverage" else "multilingual-parent-v2"
    )
    assert result["gap_retrieval"]["status"] == "not_needed"
    with ResearchStore(paths["db"]) as store:
        assert store.get_report(result["run_id"])["retrieval_mode"] == "wikipedia"


async def test_demo_ignores_server_semantic_configuration_without_loading_model(paths, monkeypatch):
    build = module.build_compressor
    backends = []

    def record(*args, **kwargs):
        backends.append(kwargs["embedding_backend"])
        return build(*args, **kwargs)

    monkeypatch.setattr(module, "build_compressor", record)
    await module.make_runner(
        **paths, embedding_backend="local-e5", embedding_model_path=Path("not-read")
    )(WebRequest(compression=True), lambda _: None, lambda _: None)
    assert backends == ["lexical"]


@pytest.mark.parametrize(
    "backend,path", [("unknown", None), ("local-e5", None), ("lexical", Path("x"))]
)
def test_invalid_server_encoder_configuration_rejected(paths, backend, path):
    with pytest.raises(ResearchError):
        module.make_runner(**paths, embedding_backend=backend, embedding_model_path=path)
    assert not paths["db"].exists()
